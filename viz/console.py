"""Interactive tree view for agent execution trees.

An interactive renderer supporting:
- Collapsible/expandable agent nodes (whole subagent sessions)
- Collapsible/expandable thinking blocks and tool details
- Keyboard navigation with cursor and viewport scrolling
- Status bar with shortcuts and execution state

Keyboard controls during live execution:
    j / ↓       Move cursor down
    k / ↑       Move cursor up
    Space       Toggle expand/collapse (or collapse enclosing block)
    g           Go to top of enclosing node
    G           Go to bottom of enclosing node
    PgUp/PgDn   Scroll by page
    Ctrl+C      Cancel execution (via SIGINT handler)

Additional controls in post-completion browser (/tree):
    q / Esc     Exit browser
"""

from __future__ import annotations

import os
import re
import select
import shutil
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from multiprocessing.synchronize import Event
from typing import Any

from ..core import Function, NodeState, NodeView, ThinkingBlockPart, ToolUsePart
from .viz import Render

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ANSI Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
REVERSE = "\x1b[7m"

FG: dict[str, str] = {
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "blue": "\x1b[34m",
    "magenta": "\x1b[35m",
    "cyan": "\x1b[36m",
    "white": "\x1b[37m",
    "orange": "\x1b[38;5;208m",
}

BG_CURSOR = "\x1b[48;5;237m"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Glyphs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FOLD = "▸"
UNFOLD = "▾"
THINKING = "💭"
RESULT_GLYPH = "📤"
ARGS_GLYPH = "📋"
VERT = "│"
TEE = "├─"
ELBOW = "└─"
RAIL = "│  "
BLANK = "   "

_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_RE_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_RE_TRAILING_CSI = re.compile(r"\x1b\[[0-9;?]*$")


def _color(
    text: str,
    *,
    fg: str | None = None,
    bold: bool = False,
    dim: bool = False,
) -> str:
    """Wrap text in ANSI color/style codes."""
    codes: list[str] = []
    if bold:
        codes.append(BOLD)
    if dim:
        codes.append(DIM)
    if fg:
        codes.append(FG.get(fg, ""))
    if not codes:
        return text
    return "".join(codes) + text + RESET


_STATE_GLYPHS: dict[NodeState, tuple[str, str]] = {
    NodeState.Waiting: ("…", "yellow"),
    NodeState.Success: ("✔", "green"),
    NodeState.Error: ("✖", "red"),
    NodeState.Canceled: ("⏹", "yellow"),
}


def _state_glyph(state: NodeState, tick: int) -> tuple[str, str]:
    """Return (glyph, color) for the given node state."""
    if state is NodeState.Running:
        return (_SPINNER_FRAMES[tick % len(_SPINNER_FRAMES)], "cyan")
    return _STATE_GLYPHS.get(state, ("?", "white"))


_TERMINAL_STATES = frozenset({NodeState.Success, NodeState.Error, NodeState.Canceled})


def _has_output(nv: NodeView) -> bool:
    """Check if a node completed successfully with output."""
    return nv.state is NodeState.Success and nv.outputs is not None


def _has_error(nv: NodeView) -> bool:
    """Check if a node failed with an exception."""
    return nv.state is NodeState.Error and nv.exception is not None


def _type_glyph(fn: Function) -> str:
    """Return a type glyph for the function."""
    if fn.is_agent():
        return "✨"
    if fn.is_code():
        return "⚙️ "
    return "•"


def _preview_text(text: str, max_len: int = 60) -> str:
    """Return a short single-line preview of *text* for collapsed block headers.

    Strips leading whitespace, collapses the first meaningful line into at most
    *max_len* characters, appending '…' if truncated.
    """
    if not text:
        return ""
    # Take the first non-empty line
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if stripped:
            break
    else:
        return ""
    if len(stripped) <= max_len:
        return stripped
    return stripped[: max_len - 1] + "…"


def _short_repr(value: Any, max_len: int = 40) -> str:
    """Short string representation, truncated if needed."""
    try:
        s = repr(value)
    except Exception:
        s = str(value)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def _format_args(
    inputs: dict[str, Any],
    max_len: int = 800,
    per_val_len: int = 120,
) -> str:
    """Format function arguments as a compact string."""
    if not inputs:
        return ""
    rendered: list[tuple[str, str]] = []
    for k, v in inputs.items():
        rendered_val = _short_repr(v, per_val_len)
        rendered.append((k, rendered_val))
    rendered.sort(key=lambda kv: len(kv[1]))
    items = [f"{k}={val}" for k, val in rendered]
    s = ", ".join(items)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
    return s


def _format_elapsed(nv: NodeView) -> str | None:
    """Return a colored elapsed time string, or None if <1s or not started."""
    start = nv.started_at
    if start is None:
        return None
    end: float | None = None
    if nv.state in _TERMINAL_STATES:
        end = nv.ended_at
    elapsed = max(0.0, ((end if end is not None else time.time()) - start))
    if elapsed < 1.0:
        return None
    s = f"{elapsed:.1f}s"
    if nv.state is NodeState.Running:
        body = _color(s, fg="red")
    else:
        body = _color(s, fg="green")
    return f" [{body}]"


def _crop_line(line: str, max_cols: int) -> str:
    """Crop a line to *max_cols* visible characters, ANSI-safe."""
    if _visible_len(line) <= max_cols:
        return line
    visible = 0
    i = 0
    n = len(line)
    while i < n and visible < max_cols:
        if line[i] == "\x1b" and (i + 1) < n and line[i + 1] == "[":
            # Skip full ANSI CSI sequence: ESC [ <params> <letter>
            j = i + 2
            while j < n and not line[j].isalpha():
                j += 1
            i = min(j + 1, n)
        else:
            visible += 1
            i += 1
    chunk = line[:i]
    chunk = _RE_TRAILING_CSI.sub("", chunk)
    chunk = chunk.removesuffix("\x1b")
    return chunk + RESET


def _visible_len(s: str) -> int:
    """Length of string excluding ANSI escape sequences."""
    return len(_RE_ANSI_ESCAPE.sub("", s))


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Line Metadata
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class LineInfo:
    """Metadata for a single displayed line in the flat list."""

    key: str | None = None
    expandable: bool = False
    default_collapsed: bool = False


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Interactive Tree Renderer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ConsoleRender(Render[str]):
    """Interactive renderer for a `NodeView` tree with cursor navigation and
    collapsible sections that yields an ANSI string intended for display in a
    terminal.

    - Maintains the most recent `NodeView` it has seen; when called with
      `render(None)`, it re-renders the last view using a time-based spinner.
    - Renders into a scrollable viewport (terminal height minus one status-bar
      row) with a highlighted cursor line and automatic scroll tracking.
    - Uses ANSI colors, Unicode box-drawing connectors, and fold/unfold
      indicators suitable for modern terminal emulators.

    Collapse / expand behaviour:
    - Every node and detail section (thinking blocks, agent input/output) has
      a collapse key and a per-item default: agent nodes default *expanded*,
      code-function nodes and detail sections default *collapsed*.
    - User overrides are stored in `_collapse_overrides` and persist for the
      lifetime of the renderer.  `toggle()` flips the item under the cursor;
      if the cursor sits on non-expandable content it walks backwards to the
      nearest expandable parent and collapses it, moving the cursor to that
      header so it stays visible.

    Transcript ↔ children correlation:
    - The renderer walks `NodeView.transcript` sequentially. For each
      `ToolUsePart`, it looks up `NodeView.transcript_child_map` (keyed by
      `id(part)`) to find the corresponding child `NodeView`. If found, the
      child subtree is rendered inline; if absent (any kind of function
      invocation failure, transient gap, or AgentException), the tool use is
      silently skipped.
    - `ThinkingBlockPart` entries are rendered as collapsible sections in
      transcript order, naturally interleaved with child nodes.
    - No positional or ordinal assumptions are made between transcript entries
      and children; correlation is entirely via `tool_use_id` matching done
      by the framework when building `NodeView`.

    Thread safety:
    - All mutable state is protected by `_lock`.  The `render()` method is
      called from the netflux view-loop background thread.  Navigation methods
      (`move_up`, `move_down`, `toggle`, `go_top`, `go_bottom`,
      `page_up`, `page_down`) are called from the main thread's keyboard
      loop.

    Follow mode:
    - When `follow=True` (the default), the cursor automatically tracks the
      last line of output so new content is always visible.  Any manual
      navigation disables follow mode; `go_bottom` re-enables it when the
      cursor reaches the overall last line.

    Intended usage for a terminal UI:
    - Call `ConsoleRender.pre_console()` before starting the view loop.
    - Start the loop with `ui_driver=ConsoleRender.ui_driver`.
    - Run a keyboard-read loop on the main thread, dispatching to the
      navigation methods above.
    - On exit (e.g., in `finally:`), call `ConsoleRender.restore_console()`.
    """

    def __init__(
        self,
        cancel_event: Event | None = None,
        *,
        follow: bool = True,
    ) -> None:
        """Initialise the interactive tree renderer."""
        self._cancel_event = cancel_event
        self._last_view: NodeView | None = None
        self._t0 = time.monotonic()
        self._lock = threading.Lock()

        # Navigation state
        self._cursor: int = 0
        self._scroll_offset: int = 0
        self._follow_mode: bool = follow

        # Collapse state: per-key overrides over per-item defaults
        self._collapse_overrides: dict[str, bool] = {}

        # Cached flat-line output (rebuilt every render tick)
        self._lines: list[str] = []
        self._line_infos: list[LineInfo] = []

        # Node-scoped navigation: (start_line, end_line) per node subtree
        self._node_ranges: list[tuple[int, int]] = []
        # Terminal width cached at each render cycle (used for text wrapping)
        self._cols: int = 80

    # ── Collapse state helpers ────────────────────────────────────────────

    def _is_collapsed(self, key: str, default: bool) -> bool:
        return self._collapse_overrides.get(key, default)

    # ── Navigation (all acquire lock) ─────────────────────────────────────

    def _navigate(self, new_cursor: int) -> None:
        """Update cursor position and disable follow mode (caller must hold lock)."""
        self._follow_mode = False
        max_pos = max(0, len(self._lines) - 1)
        self._cursor = max(0, min(new_cursor, max_pos))

    def move_up(self) -> None:
        """Move cursor up one line."""
        with self._lock:
            self._navigate(self._cursor - 1)

    def move_down(self) -> None:
        """Move cursor down one line."""
        with self._lock:
            self._navigate(self._cursor + 1)

    def page_up(self) -> None:
        """Scroll up by one page."""
        with self._lock:
            page = max(1, shutil.get_terminal_size().lines - 3)
            self._navigate(self._cursor - page)

    def page_down(self) -> None:
        """Scroll down by one page."""
        with self._lock:
            page = max(1, shutil.get_terminal_size().lines - 3)
            self._navigate(self._cursor + page)

    def go_top(self) -> None:
        """Move cursor to the top of the enclosing node's subtree.

        When the cursor sits on a collapsed (single-line) element, the
        smallest enclosing range *is* that line — so we skip it and jump
        to the parent range instead.
        """
        with self._lock:
            self._follow_mode = False
            start, end = self._find_node_range(self._cursor)
            if start == end:
                start, _ = self._find_node_range(self._cursor, min_size=2)
            self._cursor = start

    def go_bottom(self) -> None:
        """Move cursor to the bottom of the enclosing node's subtree.

        See `go_top` for the collapsed-element parent-jump logic.
        """
        with self._lock:
            self._follow_mode = False
            start, end = self._find_node_range(self._cursor)
            if start == end:
                _, end = self._find_node_range(self._cursor, min_size=2)
            self._cursor = end
            # Re-enable follow if the bottom of this node is the overall bottom
            if end >= max(0, len(self._lines) - 1):
                self._follow_mode = True

    def _find_node_range(
        self,
        cursor: int,
        *,
        min_size: int = 1,
    ) -> tuple[int, int]:
        """Return the (start, end) of the smallest node subtree containing *cursor*.

        Ranges smaller than *min_size* lines are skipped, so callers can
        bypass trivial (e.g. collapsed single-line) ranges to find the
        enclosing parent range.
        """
        best_start = 0
        best_end = max(0, len(self._lines) - 1)
        best_size = best_end - best_start + 1
        for start, end in self._node_ranges:
            if start <= cursor <= end:
                size = end - start + 1
                if size < min_size:
                    continue
                if size < best_size:
                    best_start, best_end = start, end
                    best_size = size
        return best_start, best_end

    def toggle(self) -> None:
        """Toggle expand/collapse on the current line.

        If the current line is itself expandable, its collapsed state is
        toggled.  Otherwise, scan backwards to find the nearest enclosing
        expandable item and collapse it (the cursor moves to that item's
        header line so it remains visible after the content disappears).
        """
        with self._lock:
            if 0 <= self._cursor < len(self._line_infos):
                info = self._line_infos[self._cursor]
                if info.expandable and info.key is not None:
                    current = self._is_collapsed(info.key, info.default_collapsed)
                    self._collapse_overrides[info.key] = not current
                else:
                    # Inside expanded content — collapse the nearest parent.
                    for i in range(self._cursor - 1, -1, -1):
                        parent_info = self._line_infos[i]
                        if parent_info.expandable and parent_info.key is not None:
                            self._collapse_overrides[parent_info.key] = True
                            self._cursor = i
                            break

    def reset_for_browse(self) -> None:
        """Reset navigation state for post-completion interactive browsing."""
        with self._lock:
            self._follow_mode = False
            self._cursor = 0

    # ── Render (Render[str] implementation) ───────────────────────────────

    def render(self, view: NodeView | None) -> str:
        """Render the current tree view to a string."""
        with self._lock:
            return self._render_locked(view)

    def _render_locked(self, view: NodeView | None) -> str:
        if view is not None:
            self._last_view = view
        if self._last_view is None:
            return "(waiting for data...)"

        tick = self._tick()
        cancel_pending = bool(self._cancel_event and self._cancel_event.is_set())

        # Cache terminal width for content wrapping
        self._cols = max(1, shutil.get_terminal_size(fallback=(80, 24)).columns)

        # Build flat line list from tree
        lines: list[str] = []
        infos: list[LineInfo] = []
        self._node_ranges = []
        self._build_node(
            self._last_view,
            prefix="",
            is_last=True,
            tick=tick,
            cancel_pending=cancel_pending,
            lines=lines,
            infos=infos,
        )

        # Cancellation footer
        root_state = self._last_view.state
        if cancel_pending and root_state not in _TERMINAL_STATES:
            lines.append("")
            infos.append(LineInfo())
            lines.append(_color("Cancelation pending ...", fg="magenta", bold=True))
            infos.append(LineInfo())

        self._lines = lines
        self._line_infos = infos

        # Follow mode: cursor at bottom
        if self._follow_mode:
            self._cursor = max(0, len(lines) - 1)

        # Clamp cursor
        max_pos = max(0, len(lines) - 1)
        self._cursor = max(0, min(self._cursor, max_pos))

        return self._render_viewport()

    # ── Tree building (recursive) ─────────────────────────────────────────

    def _build_node(
        self,
        nv: NodeView,
        prefix: str,
        is_last: bool,
        tick: int,
        cancel_pending: bool,
        lines: list[str],
        infos: list[LineInfo],
        *,
        depth: int = 0,
    ) -> None:
        node_start_idx = len(lines)
        key = f"n:{nv.id}"
        is_agent = nv.fn.is_agent()

        # Determine expandability
        has_children = bool(nv.children)
        has_thinking = any(isinstance(p, ThinkingBlockPart) for p in nv.transcript)
        has_usage = nv.usage is not None
        has_inputs = bool(nv.inputs)
        has_result = _has_output(nv)
        has_details = (
            has_children or has_thinking or has_usage or has_inputs or has_result
        )

        # Defaults: agents expanded, code functions collapsed
        default_collapsed = not is_agent
        collapsed = self._is_collapsed(key, default_collapsed)

        # ── Header line ──────────────────────────────────────────────────

        header = self._build_header_line(
            nv,
            tick,
            cancel_pending,
            has_details,
            collapsed,
        )

        # Summary / inline content
        if collapsed:
            parts = self._build_collapsed_summary(nv, has_children, is_agent)
            if parts:
                header += f" {_color(' | '.join(parts), dim=True)}"
        elif not is_agent:
            suffix = self._build_expanded_inline(nv)
            if suffix:
                header += suffix

        # Emit header
        if depth > 0:
            connector = ELBOW + " " if is_last else TEE + " "
            line_text = f"{prefix}{connector}{header}"
        else:
            # Root node: no connector
            line_text = header

        lines.append(line_text)
        infos.append(
            LineInfo(
                key=key,
                expandable=has_details,
                default_collapsed=default_collapsed,
            )
        )

        if collapsed:
            self._node_ranges.append((node_start_idx, len(lines) - 1))
            return

        # ── Expanded content ─────────────────────────────────────────────

        # child_prefix: what prefixes children and detail lines of this node
        # Root (depth 0) uses "" so its children's connectors start at column 0.
        # Non-root nodes extend the prefix with a rail (│) or blank depending on
        # whether the node is the last sibling.
        child_prefix = "" if depth == 0 else prefix + (BLANK if is_last else RAIL)

        # Detail prefix: adds a rail connecting to content below
        detail_prefix = child_prefix + "│    "
        content_prefix = child_prefix + "│      "

        if is_agent:
            self._emit_agent_details(
                nv,
                child_prefix,
                detail_prefix,
                content_prefix,
                tick,
                cancel_pending,
                lines,
                infos,
                depth=depth,
            )
        else:
            self._emit_code_details(
                nv,
                detail_prefix,
                content_prefix,
                lines,
                infos,
            )

        self._node_ranges.append((node_start_idx, len(lines) - 1))

    # ── Header / summary helpers ──────────────────────────────────────────

    def _build_header_line(
        self,
        nv: NodeView,
        tick: int,
        cancel_pending: bool,
        has_details: bool,
        collapsed: bool,
    ) -> str:
        """Return the formatted header string for a node (glyph … elapsed)."""
        glyph, color = _state_glyph(nv.state, tick)
        if cancel_pending and nv.state in (NodeState.Waiting, NodeState.Running):
            color = "magenta"

        type_g = _type_glyph(nv.fn)

        # Collapse indicator
        indicator = (f"{FOLD} " if collapsed else f"{UNFOLD} ") if has_details else ""

        header = (
            f"{_color(glyph, fg=color, bold=True)} {indicator}"
            f"{type_g} {_color(nv.fn.name, bold=True)}"
        )

        elapsed = _format_elapsed(nv)
        if elapsed:
            header += elapsed

        return header

    def _build_collapsed_summary(
        self,
        nv: NodeView,
        has_children: bool,
        is_agent: bool,
    ) -> list[str]:
        """Return summary fragments for a collapsed node."""
        parts: list[str] = []
        if is_agent and nv.inputs.get("user_request"):
            req = str(nv.inputs["user_request"])
            if len(req) > 50:
                req = req[:47] + "..."
            req = req.replace("\n", " ")
            parts.append(f'"{req}"')
        if has_children:
            n_agents = sum(1 for c in nv.children if c.fn.is_agent())
            n_tools = len(nv.children) - n_agents
            if n_tools:
                parts.append(f"{n_tools} tool{'s' if n_tools != 1 else ''}")
            if n_agents:
                parts.append(f"{n_agents} sub{'s' if n_agents != 1 else ''}")
        if not is_agent and nv.inputs:
            args = _format_args(nv.inputs, max_len=80, per_val_len=40)
            if args:
                parts.append(args)
        if _has_output(nv):
            parts.append(f"=> {_short_repr(nv.outputs, 50)}")
        elif _has_error(nv):
            parts.append(
                f"{_color('!!', fg='red', bold=True)} "
                f"{_short_repr(str(nv.exception), 50)}"
            )
        elif nv.state is NodeState.Canceled and nv.exception is not None:
            parts.append(
                f"{_color('CANCEL', fg='yellow', bold=True)} "
                f"{_short_repr(str(nv.exception), 50)}"
            )
        return parts

    @staticmethod
    def _build_expanded_inline(nv: NodeView) -> str:
        """Return inline suffix for an expanded code-function node."""
        if _has_output(nv):
            return f" {_color('=>', dim=True)} {_short_repr(nv.outputs, 50)}"
        if _has_error(nv):
            return (
                f" {_color('!!', fg='red', bold=True)} "
                f"{_short_repr(str(nv.exception), 50)}"
            )
        return ""

    # ── Content wrapping helper ───────────────────────────────────────────

    def _append_content(
        self,
        raw_text: str,
        prefix: str,
        lines: list[str],
        infos: list[LineInfo],
        *,
        fg: str | None = None,
        dim: bool = False,
        bold: bool = False,
    ) -> None:
        """Append a content line, wrapping to fit within the terminal width.

        *raw_text* is the plain (un-styled) text.  *prefix* is the tree
        structure prefix (may contain ANSI codes).  The text is broken into
        chunks that fit within `self._cols` visible characters and each
        chunk is emitted as a separate display line with the same prefix and
        styling.
        """
        prefix_vis = _visible_len(prefix)
        avail = max(20, self._cols - prefix_vis)

        if len(raw_text) <= avail:
            lines.append(f"{prefix}{_color(raw_text, fg=fg, dim=dim, bold=bold)}")
            infos.append(LineInfo())
            return

        pos = 0
        while pos < len(raw_text):
            chunk = raw_text[pos : pos + avail]
            lines.append(f"{prefix}{_color(chunk, fg=fg, dim=dim, bold=bold)}")
            infos.append(LineInfo())
            pos += avail

    # ── Shared detail-rendering helpers ───────────────────────────────────

    def _emit_content_block(
        self,
        text_lines: list[str],
        prefix: str,
        lines: list[str],
        infos: list[LineInfo],
        *,
        max_lines: int = 200,
        fg: str | None = None,
        dim: bool = False,
    ) -> None:
        """Emit a block of text lines with optional truncation."""
        for tl in text_lines[:max_lines]:
            self._append_content(tl, prefix, lines, infos, fg=fg, dim=dim)
        if len(text_lines) > max_lines:
            lines.append(
                f"{prefix}{_color(f'... ({len(text_lines)} lines total)', dim=True)}"
            )
            infos.append(LineInfo())

    def _emit_kv_pairs(
        self,
        inputs: dict[str, Any],
        header_prefix: str,
        value_prefix: str,
        lines: list[str],
        infos: list[LineInfo],
        *,
        max_lines: int = 200,
    ) -> None:
        """Emit key-value pairs, using multi-line display for long values."""
        for k, v in inputs.items():
            val_str = str(v)
            val_lines = val_str.splitlines()
            if len(val_lines) > 1 or len(val_str) > 100:
                lines.append(
                    f"{header_prefix}{_color(ARGS_GLYPH + ' ' + k + ':', fg='cyan')}"
                )
                infos.append(LineInfo())
                self._emit_content_block(
                    val_lines,
                    value_prefix,
                    lines,
                    infos,
                    max_lines=max_lines,
                    dim=True,
                )
            else:
                lines.append(
                    f"{header_prefix}"
                    f"{_color(ARGS_GLYPH + ' ' + k + ': ', fg='cyan')}"
                    f"{_color(val_str, dim=True)}"
                )
                infos.append(LineInfo())

    def _emit_agent_details(
        self,
        nv: NodeView,
        child_prefix: str,
        detail_prefix: str,
        content_prefix: str,
        tick: int,
        cancel_pending: bool,
        lines: list[str],
        infos: list[LineInfo],
        *,
        depth: int = 0,
    ) -> None:
        """Emit expanded content for an agent node.

        Covers: input, usage, thinking, children, output.
        """
        # ── Agent input (collapsible) ────────────────────────────────────
        if nv.inputs:
            input_key = f"ai:{nv.id}"
            input_collapsed = self._is_collapsed(input_key, default=True)
            # Build a combined input string for char count
            input_text = "\n".join(f"{k}: {v}" for k, v in nv.inputs.items())
            n_chars = len(input_text)
            indicator = FOLD if input_collapsed else UNFOLD
            preview = _preview_text(input_text, 60)
            preview_part = f": {preview}" if preview else ""
            label = (
                f"{indicator} {ARGS_GLYPH}"
                f" input{preview_part} ({n_chars:,} chars)"
            )
            lines.append(
                f"{detail_prefix}{_color(label, fg='cyan')}"
            )
            infos.append(
                LineInfo(key=input_key, expandable=True, default_collapsed=True)
            )
            if not input_collapsed:
                self._emit_kv_pairs(
                    nv.inputs,
                    content_prefix,
                    content_prefix + "  ",
                    lines,
                    infos,
                )

        # Token usage
        if nv.usage:
            usage_text = self._format_usage(nv.usage)
            lines.append(f"{detail_prefix}{usage_text}")
            infos.append(LineInfo())

        # ── Transcript walk: thinking + children in transcript order ──
        # Walk nv.transcript sequentially, rendering ThinkingBlockParts inline
        # and looking up child NodeViews for each ToolUsePart via
        # nv.transcript_child_map (keyed by id(part)).  No positional or
        # ordinal assumptions between transcript entries and children;
        # correlation is entirely via tool_use_id matching done by the runtime
        # when building NodeView.
        has_trailing_output = _has_output(nv) or _has_error(nv)

        if nv.transcript:
            # Collect renderable items from transcript in order.
            render_items: list[object] = []  # ThinkingBlockPart | NodeView
            for part in nv.transcript:
                if isinstance(part, ThinkingBlockPart):
                    render_items.append(part)
                elif isinstance(part, ToolUsePart):
                    child_view = nv.transcript_child_map.get(id(part))
                    if child_view is not None:
                        render_items.append(child_view)

            # Render items in transcript order.
            n_items = len(render_items)
            thinking_seq = 0
            for ri_idx, item in enumerate(render_items):
                if isinstance(item, ThinkingBlockPart):
                    self._emit_thinking_slot(
                        item,
                        nv.id,
                        thinking_seq,
                        detail_prefix,
                        content_prefix,
                        lines,
                        infos,
                    )
                    thinking_seq += 1
                elif isinstance(item, NodeView):
                    # Last-child connector: only use └── when nothing follows
                    # (no more render items and no trailing output/error).
                    is_last_item = (ri_idx == n_items - 1) and not has_trailing_output
                    self._build_node(
                        item,
                        child_prefix,
                        is_last_item,
                        tick,
                        cancel_pending,
                        lines,
                        infos,
                        depth=depth + 1,
                    )
        else:
            # No transcript yet (early agent state): render children directly.
            n_children = len(nv.children)
            for idx, child in enumerate(nv.children):
                self._build_node(
                    child,
                    child_prefix,
                    (idx == n_children - 1) and not has_trailing_output,
                    tick,
                    cancel_pending,
                    lines,
                    infos,
                    depth=depth + 1,
                )

        # ── Agent output (collapsible) ───────────────────────────────────
        if _has_output(nv):
            output_key = f"ao:{nv.id}"
            output_collapsed = self._is_collapsed(output_key, default=True)
            result_str = str(nv.outputs)
            n_chars = len(result_str)
            indicator = FOLD if output_collapsed else UNFOLD
            preview = _preview_text(result_str, 60)
            preview_part = f": {preview}" if preview else ""
            label = (
                f"{indicator} {RESULT_GLYPH}"
                f" output{preview_part} ({n_chars:,} chars)"
            )
            lines.append(
                f"{detail_prefix}{_color(label, fg='green')}"
            )
            infos.append(
                LineInfo(key=output_key, expandable=True, default_collapsed=True)
            )
            if not output_collapsed:
                self._emit_content_block(
                    result_str.splitlines(),
                    content_prefix,
                    lines,
                    infos,
                    dim=True,
                )
        elif _has_error(nv):
            self._emit_error_block(nv, detail_prefix, content_prefix, lines, infos)

    def _emit_code_details(
        self,
        nv: NodeView,
        detail_prefix: str,
        content_prefix: str,
        lines: list[str],
        infos: list[LineInfo],
    ) -> None:
        """Emit expanded content for a code function: args and result."""
        # Arguments
        if nv.inputs:
            self._emit_kv_pairs(nv.inputs, detail_prefix, content_prefix, lines, infos)

        # Result
        if _has_output(nv):
            result_str = str(nv.outputs)
            n_chars = len(result_str)
            lines.append(
                f"{detail_prefix}"
                f"{_color(RESULT_GLYPH + f' result ({n_chars} chars):', fg='green')}"
            )
            infos.append(LineInfo())
            self._emit_content_block(
                result_str.splitlines(),
                content_prefix,
                lines,
                infos,
                dim=True,
            )

        # Error details
        elif _has_error(nv):
            self._emit_error_block(nv, detail_prefix, content_prefix, lines, infos)

    def _emit_error_block(
        self,
        nv: NodeView,
        detail_prefix: str,
        content_prefix: str,
        lines: list[str],
        infos: list[LineInfo],
    ) -> None:
        """Emit a standardised error block for a failed node."""
        lines.append(f"{detail_prefix}{_color('✖ error:', fg='red', bold=True)}")
        infos.append(LineInfo())
        self._emit_content_block(
            str(nv.exception).splitlines(),
            content_prefix,
            lines,
            infos,
            max_lines=50,
            fg="red",
        )

    def _emit_thinking_slot(
        self,
        part: ThinkingBlockPart,
        node_id: int,
        slot_idx: int,
        detail_prefix: str,
        content_prefix: str,
        lines: list[str],
        infos: list[LineInfo],
    ) -> None:
        """Emit a single thinking block for a given slot."""
        key = f"t:{node_id}:{slot_idx}:0"
        collapsed = self._is_collapsed(key, default=True)

        # Character count
        n_chars = len(part.content) if part.content else 0
        char_info = f" ({n_chars:,} chars)" if n_chars else ""

        indicator = FOLD if collapsed else UNFOLD

        if part.redacted:
            text = f"{indicator} {THINKING} thinking [redacted]"
        else:
            preview = _preview_text(part.content or "", 60)
            preview_part = f": {preview}" if preview else ""
            text = f"{indicator} {THINKING} thinking{preview_part}{char_info}"

        lines.append(f"{detail_prefix}{_color(text, dim=True)}")
        infos.append(LineInfo(key=key, expandable=True, default_collapsed=True))

        if not collapsed and part.content:
            self._emit_content_block(
                part.content.splitlines(),
                content_prefix,
                lines,
                infos,
                max_lines=500,
                dim=True,
            )

    # ── Token usage formatting ────────────────────────────────────────────

    @staticmethod
    def _format_usage(u: Any) -> str:
        in_fields = [f"cache_read={u.input_tokens_cache_read}"]
        if u.input_tokens_cache_write is not None:
            in_fields.append(f"cache_write={u.input_tokens_cache_write}")
        in_fields.append(f"regular={u.input_tokens_regular}")
        in_fields.append(f"total={u.input_tokens_total}")

        out_fields: list[str] = []
        if u.output_tokens_reasoning is not None:
            out_fields.append(f"reasoning={u.output_tokens_reasoning}")
        if u.output_tokens_text is not None:
            out_fields.append(f"text={u.output_tokens_text}")
        out_fields.append(f"total={u.output_tokens_total}")

        segs = [
            _color(f"In: {{{', '.join(in_fields)}}}", fg="cyan", bold=True),
            _color(f"Out: {{{', '.join(out_fields)}}}", fg="magenta", bold=True),
        ]

        ctx_total = u.context_window_in + u.context_window_out
        if ctx_total:
            ctx_fields = [
                f"in={u.context_window_in}",
                f"out={u.context_window_out}",
                f"total={ctx_total}",
            ]
            segs.append(
                _color(f"Ctx: {{{', '.join(ctx_fields)}}}", fg="orange", bold=True)
            )

        return ", ".join(segs)

    # ── Viewport rendering ────────────────────────────────────────────────

    def _render_viewport(self) -> str:
        sz = shutil.get_terminal_size(fallback=(80, 24))
        rows = max(1, sz.lines)
        cols = max(1, sz.columns)

        # Reserve 1 row for status bar
        tree_rows = max(1, rows - 1)

        # Adjust scroll to keep cursor visible
        if self._cursor < self._scroll_offset:
            self._scroll_offset = self._cursor
        elif self._cursor >= self._scroll_offset + tree_rows:
            self._scroll_offset = self._cursor - tree_rows + 1
        self._scroll_offset = max(0, self._scroll_offset)

        output: list[str] = []
        end = min(self._scroll_offset + tree_rows, len(self._lines))
        for i in range(self._scroll_offset, end):
            line = self._lines[i]
            if i == self._cursor:
                line = f"{BG_CURSOR}{line}{RESET}"
            output.append(_crop_line(line, cols - 1))

        # Pad to fill viewport
        while len(output) < tree_rows:
            output.append("")

        # Status bar
        output.append(self._status_bar(cols))

        return "\n".join(output)

    def _status_bar(self, cols: int) -> str:
        pos = f"{self._cursor + 1}/{len(self._lines)}"

        state_text = ""
        if self._last_view:
            s = self._last_view.state
            if s is NodeState.Running:
                tick = self._tick()
                frame = _SPINNER_FRAMES[tick % len(_SPINNER_FRAMES)]
                state_text = _color(f" {frame} Running", fg="cyan")
            elif s is NodeState.Success:
                state_text = _color(" ✔ Complete", fg="green")
            elif s is NodeState.Error:
                state_text = _color(" ✖ Error", fg="red")
            elif s is NodeState.Canceled:
                state_text = _color(" ⏹ Canceled", fg="yellow")

        # Show q:quit only when execution is complete
        is_terminal = self._last_view and self._last_view.state in _TERMINAL_STATES
        quit_hint = "  q:quit" if is_terminal else ""

        shortcuts = f"↑↓/jk:move  ␣:toggle  g/G:top/btm{quit_hint}"
        bar = f" {shortcuts} │ {pos}{state_text} "

        # Pad to terminal width (use visible length for calculation)
        vis_len = _visible_len(bar)
        if vis_len < cols:
            bar = bar + " " * (cols - vis_len)

        return f"{REVERSE}{bar}{RESET}"

    def _tick(self) -> int:
        return int((time.monotonic() - self._t0) * 10)

    # ── Static terminal management ────────────────────────────────────────

    @staticmethod
    def pre_console() -> None:
        """Enter alt screen, hide cursor, disable wrap, clear scrollback."""
        sys.stdout.write("\x1b[?1049h\x1b[?25l\x1b[?7l\x1b[3J")
        sys.stdout.flush()

    @staticmethod
    def restore_console() -> None:
        """Show cursor, re-enable wrap, leave alt screen."""
        sys.stdout.write("\x1b[?25h\x1b[?7h\x1b[?1049l")
        sys.stdout.flush()

    @staticmethod
    def ui_driver(s: str) -> None:
        """Clear screen and write the rendered frame."""
        sys.stdout.write("\x1b[3J\x1b[2J\x1b[H")
        sys.stdout.write(s)
        sys.stdout.flush()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Keyboard Input
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _read_key(fd: int, timeout: float = 0.1) -> str | None:
    """Read a keypress from *fd*, handling escape sequences."""
    try:
        r, _, _ = select.select([fd], [], [], timeout)
    except (InterruptedError, OSError):
        return None

    if not r:
        return None

    raw = os.read(fd, 1)
    if not raw:
        return None
    ch = raw.decode("utf-8", errors="ignore")

    if ch == "\x1b":
        # Possible escape sequence: ESC [ <code>
        try:
            r2, _, _ = select.select([fd], [], [], 0.05)
        except (InterruptedError, OSError):
            return "escape"
        if not r2:
            return "escape"
        ch2 = os.read(fd, 1).decode("utf-8", errors="ignore")
        if ch2 != "[":
            return None
        try:
            r3, _, _ = select.select([fd], [], [], 0.05)
        except (InterruptedError, OSError):
            return None
        if not r3:
            return None
        ch3 = os.read(fd, 1).decode("utf-8", errors="ignore")
        if ch3 == "A":
            return "up"
        if ch3 == "B":
            return "down"
        if ch3 in ("5", "6"):
            # Page Up/Down: ESC [ 5~ / ESC [ 6~
            try:
                r4, _, _ = select.select([fd], [], [], 0.05)
                if r4:
                    os.read(fd, 1)  # consume '~'
            except (InterruptedError, OSError):
                pass
            return "page_up" if ch3 == "5" else "page_down"
        return None

    return ch


_KEY_BINDINGS: dict[str, str | None] = {
    "j": "move_down",
    "down": "move_down",
    "k": "move_up",
    "up": "move_up",
    " ": "toggle",
    "\r": "toggle",
    "g": "go_top",
    "G": "go_bottom",
    "page_up": "page_up",
    "page_down": "page_down",
    "q": None,
    "escape": None,
}


def _handle_key(key: str, renderer: ConsoleRender) -> bool:
    """Process a key event, returning True if the browser should exit."""
    action = _KEY_BINDINGS.get(key)
    if action is None:
        return key in _KEY_BINDINGS  # True for quit keys, False for unmapped
    getattr(renderer, action)()
    return False


def keyboard_loop(
    renderer: ConsoleRender,
    view_thread: threading.Thread,
) -> None:
    """Handle keyboard input during live execution.

    Runs on the main thread.  Exits when the view-loop thread dies
    (i.e. execution reaches a terminal state).
    """
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while view_thread.is_alive():
            key = _read_key(fd, timeout=0.1)
            if key is None:
                continue
            _handle_key(key, renderer)
    except (KeyboardInterrupt, EOFError):
        # Ctrl+C during cbreak → SIGINT → let existing handler deal with it
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def interactive_browse(renderer: ConsoleRender, last_view: NodeView) -> None:
    """Post-completion interactive tree browser.

    Launches an interactive session where the user can navigate and
    expand/collapse the completed execution tree.  Exits on 'q' or Esc.
    """
    renderer.reset_for_browse()

    # Initial render
    output = renderer.render(last_view)
    ConsoleRender.ui_driver(output)

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while True:
            key = _read_key(fd, timeout=0.15)
            if key is not None:
                should_exit = _handle_key(key, renderer)
                if should_exit:
                    break
            # Re-render (handles viewport/cursor updates and status bar animation)
            output = renderer.render(None)
            ConsoleRender.ui_driver(output)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
