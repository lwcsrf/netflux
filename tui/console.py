from __future__ import annotations

"""Interactive tree view for agent execution trees.

An interactive renderer supporting:
- Collapsible/expandable agent nodes (whole subagent sessions)
- Collapsible/expandable thinking blocks and function details
- Keyboard navigation with cursor and viewport scrolling
- Status bar with shortcuts and execution state

Keyboard controls during live execution:
    j / ↓       Move cursor down
    k / ↑       Move cursor up
    Space       Toggle expand/collapse (or collapse enclosing block)
    n / N       Next / previous visible agent
    g           Go to top of enclosing node
    G           Go to bottom of enclosing node
    PgUp/PgDn   Scroll by page
    Ctrl+C      Cancel execution (sets `cancel_event` when provided)

Additional controls in post-completion browser:
    q / Esc     Exit browser
"""
import re
import shutil
from io import StringIO
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from multiprocessing.synchronize import Event
from typing import Any, Iterator, Mapping

from rich.console import Console
from rich.markdown import Markdown

from ._contracts import RightPaneInteractionContext, SelectedTreeStatus
from ..core import (
    Function,
    ModelTextPart,
    Node,
    NodeState,
    NodeView,
    TokenBill,
    ThinkingBlockPart,
    ToolResultPart,
    ToolUsePart,
    UserTextPart,
)
from ..providers import ModelNames, Provider

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ANSI Constants
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"

FG: dict[str, str] = {
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "blue": "\x1b[34m",
    "magenta": "\x1b[35m",
    "cyan": "\x1b[36m",
    "white": "\x1b[37m",
    "gray": "\x1b[38;5;250m",
    "orange": "\x1b[38;5;208m",
}

BG_CURSOR = "\x1b[48;5;237m"
BG_STATUS_BAR = "\x1b[48;5;238m"
BG_STATUS_STATE = "\x1b[48;5;24m"
BG_STATUS_TOKENS = "\x1b[48;5;52m"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Glyphs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FOLD = "▸"
UNFOLD = "▾"
THINKING = "➰"
AGENT_GLYPH = "✨"
CODE_GLYPH = "⚙️ "
RESULT_GLYPH = "📤"
ARGS_GLYPH = "📋"
USER_GLYPH = "👤"
FUNCTION_GLYPH = "🧰"
VERT = "│"
TEE = "├─"
ELBOW = "└─"
RAIL = "│  "
BLANK = "   "

_SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
_MOUSE_SCROLL_ROWS = 5

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Helper Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ANSI CSI grammar:
#   parameter bytes   0x30-0x3F  => [0-?]
#   intermediate      0x20-0x2F  => [ -/]
#   final byte        0x40-0x7E  => [@-~]
_RE_COMPLETE_CSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_RE_TRAILING_CSI = re.compile(r"\x1b\[[0-?]*[ -/]*$")

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


def _copy_text_to_clipboard_windows(text: str) -> bool:
    import ctypes
    from ctypes import wintypes

    CF_UNICODETEXT = 13
    GMEM_MOVEABLE = 0x0002
    GMEM_ZEROINIT = 0x0040

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32 = ctypes.WinDLL("user32", use_last_error=True)

    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = ctypes.c_void_p
    kernel32.GlobalLock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    kernel32.GlobalUnlock.restype = wintypes.BOOL
    kernel32.GlobalFree.argtypes = [ctypes.c_void_p]
    kernel32.GlobalFree.restype = ctypes.c_void_p

    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.CloseClipboard.argtypes = []
    user32.CloseClipboard.restype = wintypes.BOOL
    user32.EmptyClipboard.argtypes = []
    user32.EmptyClipboard.restype = wintypes.BOOL
    user32.SetClipboardData.argtypes = [wintypes.UINT, ctypes.c_void_p]
    user32.SetClipboardData.restype = ctypes.c_void_p

    size = (len(text) + 1) * ctypes.sizeof(ctypes.c_wchar)
    handle = kernel32.GlobalAlloc(GMEM_MOVEABLE | GMEM_ZEROINIT, size)
    if not handle:
        return False

    locked = kernel32.GlobalLock(handle)
    if not locked:
        kernel32.GlobalFree(handle)
        return False

    try:
        buffer = ctypes.create_unicode_buffer(text)
        ctypes.memmove(locked, ctypes.addressof(buffer), size)
    finally:
        kernel32.GlobalUnlock(handle)

    for _ in range(20):
        if user32.OpenClipboard(None):
            break
        time.sleep(0.01)
    else:
        kernel32.GlobalFree(handle)
        return False

    try:
        if not user32.EmptyClipboard():
            return False
        if not user32.SetClipboardData(CF_UNICODETEXT, handle):
            return False
        handle = None
        return True
    finally:
        user32.CloseClipboard()
        if handle:
            kernel32.GlobalFree(handle)


def _linux_clipboard_commands() -> list[list[str]]:
    commands: list[list[str]] = []
    if shutil.which("wl-copy"):
        commands.append(["wl-copy"])
    if shutil.which("xclip"):
        commands.append(["xclip", "-selection", "clipboard"])
    if shutil.which("xsel"):
        commands.append(["xsel", "--clipboard", "--input"])
    return commands


def _clipboard_copy_failure_message() -> str:
    if sys.platform.startswith("win"):
        return "Clipboard copy failed."
    if sys.platform == "darwin":
        if shutil.which("pbcopy") is None:
            return "Clipboard unavailable. Ensure pbcopy is installed."
        return "Clipboard copy failed."
    if not _linux_clipboard_commands():
        return "Clipboard unavailable. Install wl-copy, xclip, or xsel."
    return "Clipboard copy failed."


def _copy_text_to_clipboard(text: str) -> bool:
    if sys.platform.startswith("win"):
        return _copy_text_to_clipboard_windows(text)

    commands: list[list[str]] = []

    if sys.platform == "darwin":
        commands.append(["pbcopy"])
    else:
        commands.extend(_linux_clipboard_commands())

    for command in commands:
        try:
            subprocess.run(
                command,
                input=text,
                text=True,
                check=True,
                capture_output=True,
            )
            return True
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            continue

    return False


def _style_block(text: str, *codes: str) -> str:
    """Apply persistent ANSI styles across embedded RESET codes."""
    prefix = "".join(code for code in codes if code)
    if not prefix or not text:
        return text
    return prefix + text.replace(RESET, RESET + prefix) + RESET


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
        return AGENT_GLYPH
    if fn.is_code():
        return CODE_GLYPH
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


def _strip_ansi(text: str) -> str:
    return _RE_COMPLETE_CSI.sub("", text)


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


def _render_markdown_lines(text: str, *, width: int) -> list[_RenderedBlockLine]:
    buffer = StringIO()
    console = Console(
        file=buffer,
        force_terminal=True,
        color_system="standard",
        width=max(1, width),
        legacy_windows=False,
    )
    console.print(Markdown(text, hyperlinks=False), end="")
    rendered = buffer.getvalue()
    lines = rendered.splitlines() or [""]
    return [
        _RenderedBlockLine(
            text=_strip_ansi(line).rstrip(),
            styled_text=line,
        )
        for line in lines
    ] or [_RenderedBlockLine("")]


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


def _skip_csi_sequence(s: str, start: int) -> int:
    """Return the index after a CSI sequence starting at *start*, or *start* if none."""
    match = _RE_COMPLETE_CSI.match(s, start)
    if match is None:
        match = _RE_TRAILING_CSI.match(s, start)
        if match is None:
            return start
    return match.end()


def _iter_visible_chars(s: str) -> Iterator[tuple[int, int, str]]:
    """Yield (visible_column, slice_end, char) for each non-ANSI character in *s*."""
    visible = 0
    i = 0
    n = len(s)
    while i < n:
        next_i = _skip_csi_sequence(s, i)
        if next_i != i:
            i = next_i
            continue
        yield visible, i + 1, s[i]
        visible += 1
        i += 1


def _visible_prefix_end(s: str, max_cols: int) -> int:
    """Return the string index that spans *max_cols* visible characters."""
    visible = 0
    i = 0
    n = len(s)
    while i < n and visible < max_cols:
        next_i = _skip_csi_sequence(s, i)
        if next_i != i:
            i = next_i
            continue
        visible += 1
        i += 1
    return i


def _needs_crop(line: str, max_cols: int) -> bool:
    """Return whether cropping should run, preserving legacy handling of malformed CSI."""
    return len(_RE_COMPLETE_CSI.sub("", line)) > max_cols


def _crop_line(line: str, max_cols: int) -> str:
    """Crop a line to *max_cols* visible characters, ANSI-safe."""
    if not _needs_crop(line, max_cols):
        return line
    chunk = line[:_visible_prefix_end(line, max_cols)]
    chunk = _RE_TRAILING_CSI.sub("", chunk)
    chunk = chunk.removesuffix("\x1b")
    return chunk + RESET


def _visible_len(s: str) -> int:
    """Length of string excluding ANSI escape sequences."""
    return sum(1 for _ in _iter_visible_chars(s))


def _visible_index_of_any(s: str, targets: set[str]) -> int | None:
    """Return the visible-column index of the first matching glyph in *s*."""
    for visible, _, ch in _iter_visible_chars(s):
        if ch in targets:
            return visible
    return None


def _highlight_line(line: str, width: int) -> str:
    """Apply background highlight across the entire visible row width."""
    padded = line
    vis_len = _visible_len(padded)
    if vis_len < width:
        padded += " " * (width - vis_len)
    return f"{BG_CURSOR}{padded.replace(RESET, RESET + BG_CURSOR)}{RESET}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Line Metadata
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@dataclass
class LineInfo:
    """Metadata for a single displayed line in the flat list."""

    key: str | None = None
    expandable: bool = False
    default_collapsed: bool = False
    is_node_header: bool = False
    is_agent_header: bool = False
    toggle_col: int | None = None
    anchors: tuple[str, ...] = ()


@dataclass
class NodeRange:
    start: int
    end: int
    key: str
    is_agent: bool


@dataclass(frozen=True)
class _RenderedBlockLine:
    text: str
    fg: str | None = None
    dim: bool = False
    bold: bool = False
    styled_text: str | None = None


@dataclass(frozen=True)
class _RootResultTarget:
    key: str
    display_text: str
    copy_text: str


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Interactive Tree Renderer
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class ConsoleRender:
    """Interactive renderer for a `NodeView` tree with cursor navigation and
    collapsible sections that yields an ANSI string intended for display in a
    terminal.

    - Maintains the most recent `NodeView` snapshot and can re-render it on
      demand for spinner ticks, resize handling, and interactive navigation.
    - Renders into a scrollable viewport (terminal height minus one status-bar
      row) with a highlighted cursor line and automatic scroll tracking.
    - Uses ANSI colors, Unicode box-drawing connectors, and fold/unfold
      indicators suitable for modern terminal emulators.

    Collapse / expand behaviour:
    - Every node and detail section (thinking blocks, user/model text blocks,
      code args/results, and synthetic function rows) has a collapse key and a
      per-item default: agent nodes default *expanded*, code-function nodes
      and detail sections default *collapsed*.
    - User overrides are stored in `_collapse_overrides` and persist for the
      lifetime of the renderer. `toggle_expanded()` flips the item under the
      cursor; if the cursor sits on non-expandable content it walks backwards
      to the nearest expandable parent and collapses it, moving the cursor to
      that header so it stays visible.

    Transcript ↔ children correlation:
    - The renderer walks `NodeView.transcript` sequentially. For each
      `ToolUsePart`, it looks up `NodeView.transcript_child_map` (keyed by
      `id(part)`) to find the corresponding child `NodeView`.
    - If a child node exists, its subtree is rendered inline at the matching
      transcript position.
    - If no child node exists, the function call still renders as a synthetic
      expandable row backed by the `ToolUsePart` / `ToolResultPart` pair, so
      pending calls and failures remain visible.
    - `ThinkingBlockPart` entries are rendered as collapsible sections in
      transcript order, naturally interleaved with child nodes.
    - No positional or ordinal assumptions are made between transcript entries
      and children; correlation is entirely via `tool_use_id` matching done
      by the framework when building `NodeView`.

    Thread safety:
    - All mutable state is protected by `_lock`.
    - Navigation methods (`navigate_up`, `navigate_down`, `toggle_expanded`,
      `go_top`, `go_bottom`, `page_up`, `page_down`, `jump_prev_agent`,
      `jump_next_agent`, `collapse_enclosing_agent`, `expand_all_nodes`,
      `collapse_all_nodes`) are safe to call while the renderer is active.

    Follow mode:
    - When `follow=True` (the default), the cursor automatically tracks the
      last line of output so new content is always visible. Any manual
      navigation disables follow mode; `go_bottom()` re-enables it when the
      cursor reaches the overall last line.

    Intended usage for a terminal UI:
    - Create `ConsoleRender(...)` and call `run(node)`.
    - `run(node)` delegates terminal ownership and event-loop orchestration to
      the shared session driver, while `ConsoleRender` remains the single-tree
      viewport renderer/controller.
    """

    def __init__(
        self,
        cancel_event: Event | None = None,
        *,
        spinner_hz: float = 10.0,
        follow: bool = True,
    ) -> None:
        """Initialise the interactive tree renderer."""
        self.spinner_hz = max(1.0, float(spinner_hz))
        self._cancel_event = cancel_event
        self._last_view: NodeView | None = None
        self._root_id: int | None = None
        self._t0 = time.monotonic()
        self._lock = threading.Lock()

        # Navigation state
        self._cursor: int = 0
        self._scroll_offset: int = 0
        self._follow_mode: bool = follow
        self._selected_key: str | None = None
        self._selected_anchor: str | None = None
        self._selected_anchor_occurrence: int = 0

        # Collapse state: per-key overrides over per-item defaults
        self._collapse_overrides: dict[str, bool] = {}

        # Cached flat-line output (rebuilt every render tick)
        self._lines: list[str] = []
        self._line_infos: list[LineInfo] = []
        self._tree_rows: int = 1

        # Node-scoped navigation ranges.
        self._node_ranges: list[NodeRange] = []
        # Terminal width cached at each render cycle (used for text wrapping)
        self._cols: int = 80

    # ── Collapse state helpers ────────────────────────────────────────────

    def _is_collapsed(self, key: str, default: bool) -> bool:
        return self._collapse_overrides.get(key, default)

    def _terminal_root_result_target_locked(self) -> _RootResultTarget | None:
        view = self._last_view
        if view is None or view.state is not NodeState.Success:
            return None

        if view.fn.is_agent():
            for idx in range(len(view.transcript) - 1, -1, -1):
                part = view.transcript[idx]
                if isinstance(part, ModelTextPart):
                    copy_text = str(view.outputs) if view.outputs is not None else part.text
                    return _RootResultTarget(
                        key=f"tp:{view.id}:{idx}:model",
                        display_text=part.text,
                        copy_text=copy_text,
                    )
            if view.outputs is None:
                return None
            rendered = str(view.outputs)
            return _RootResultTarget(
                key=f"ao:{view.id}",
                display_text=rendered,
                copy_text=rendered,
            )

        if view.outputs is None:
            return None

        rendered = str(view.outputs)
        return _RootResultTarget(
            key=f"cr:{view.id}",
            display_text=rendered,
            copy_text=rendered,
        )

    def _rendered_root_result_lines_locked(
        self,
        key: str,
        text: str,
        content_prefix: str,
    ) -> list[_RenderedBlockLine] | None:
        target = self._terminal_root_result_target_locked()
        if target is None or target.key != key:
            return None
        width = max(1, self._cols - _visible_len(content_prefix))
        return _render_markdown_lines(text, width=width)

    def copy_terminal_result(self) -> bool:
        success, _ = self.copy_terminal_result_with_feedback()
        return success

    def copy_terminal_result_with_feedback(self) -> tuple[bool, str | None]:
        with self._lock:
            target = self._terminal_root_result_target_locked()
        if target is None:
            return False, None
        if _copy_text_to_clipboard(target.copy_text):
            return True, None
        return False, _clipboard_copy_failure_message()

    def focus_terminal_result(self) -> bool:
        with self._lock:
            target = self._terminal_root_result_target_locked()
            if target is None:
                return False

            if self._root_id is not None:
                self._collapse_overrides[f"n:{self._root_id}"] = False
            self._collapse_overrides[target.key] = False
            self._follow_mode = False
            self._selected_key = None
            self._selected_anchor = target.key
            self._selected_anchor_occurrence = len(self._lines) + 1024

            matches = [
                idx
                for idx, info in enumerate(self._line_infos)
                if info.key == target.key or target.key in info.anchors
            ]
            if matches:
                self._cursor = matches[-1]
            return True

    # ── Navigation (all acquire lock) ─────────────────────────────────────

    def _remember_selection(self) -> None:
        if 0 <= self._cursor < len(self._line_infos):
            info = self._line_infos[self._cursor]
            self._selected_key = info.key
            self._selected_anchor = info.key or (info.anchors[0] if info.anchors else None)
            self._selected_anchor_occurrence = 0
            if self._selected_anchor is not None:
                for idx in range(self._cursor + 1):
                    candidate = self._line_infos[idx]
                    if (
                        candidate.key == self._selected_anchor
                        or self._selected_anchor in candidate.anchors
                    ):
                        self._selected_anchor_occurrence += 1
                self._selected_anchor_occurrence = max(
                    0, self._selected_anchor_occurrence - 1
                )
        else:
            self._selected_key = None
            self._selected_anchor = None
            self._selected_anchor_occurrence = 0

    def _iter_node_keys(self, nv: NodeView) -> list[str]:
        keys = [f"n:{nv.id}"]
        for child in nv.children:
            keys.extend(self._iter_node_keys(child))
        return keys

    def _set_cursor(self, new_cursor: int, *, disable_follow: bool = True) -> None:
        if disable_follow:
            self._follow_mode = False
        max_pos = max(0, len(self._lines) - 1)
        self._cursor = max(0, min(new_cursor, max_pos))
        self._remember_selection()

    def _navigate(self, new_cursor: int) -> None:
        self._set_cursor(new_cursor, disable_follow=True)

    def _toggle_line_locked(self, line_idx: int) -> bool:
        if not (0 <= line_idx < len(self._line_infos)):
            return False
        info = self._line_infos[line_idx]
        if not info.expandable or info.key is None:
            return False
        current = self._is_collapsed(info.key, info.default_collapsed)
        self._collapse_overrides[info.key] = not current
        self._remember_selection()
        return True

    def _toggle_expanded_locked(self) -> None:
        if not (0 <= self._cursor < len(self._line_infos)):
            return

        if self._toggle_line_locked(self._cursor):
            return

        for i in range(self._cursor - 1, -1, -1):
            parent_info = self._line_infos[i]
            if parent_info.expandable and parent_info.key is not None:
                self._collapse_overrides[parent_info.key] = True
                self._set_cursor(i, disable_follow=True)
                return

    def handle_click(self, x: int, y: int, *, button: str = "left") -> None:
        with self._lock:
            if button == "wheel_up":
                self._navigate(self._cursor - _MOUSE_SCROLL_ROWS)
                return
            if button == "wheel_down":
                self._navigate(self._cursor + _MOUSE_SCROLL_ROWS)
                return
            if button != "left":
                return
            if y < 0 or y >= self._tree_rows:
                return
            line_idx = self._scroll_offset + y
            if not (0 <= line_idx < len(self._line_infos)):
                return

            info = self._line_infos[line_idx]
            if line_idx == self._cursor:
                self._toggle_expanded_locked()
                return

            self._set_cursor(line_idx, disable_follow=True)
            if info.toggle_col is not None and x == info.toggle_col:
                self._toggle_line_locked(line_idx)

    def navigate_up(self) -> None:
        with self._lock:
            self._navigate(self._cursor - 1)

    def navigate_down(self) -> None:
        with self._lock:
            self._navigate(self._cursor + 1)

    def move_up(self) -> None:
        self.navigate_up()

    def move_down(self) -> None:
        self.navigate_down()

    def page_up(self) -> None:
        with self._lock:
            page = max(1, self._tree_rows - 2)
            self._navigate(self._cursor - page)

    def page_down(self) -> None:
        with self._lock:
            page = max(1, self._tree_rows - 2)
            self._navigate(self._cursor + page)

    def _find_node_range(
        self,
        cursor: int,
        *,
        min_size: int = 1,
        agent_only: bool = False,
    ) -> NodeRange | None:
        best: NodeRange | None = None
        best_size = (max(0, len(self._lines) - 1) + 1) + 1
        for rng in self._node_ranges:
            if rng.start <= cursor <= rng.end:
                if agent_only and not rng.is_agent:
                    continue
                size = rng.end - rng.start + 1
                if size < min_size:
                    continue
                if size < best_size:
                    best = rng
                    best_size = size
        return best

    def go_top(self) -> None:
        with self._lock:
            rng = self._find_node_range(self._cursor)
            if rng is None:
                return
            if rng.start == rng.end:
                larger = self._find_node_range(self._cursor, min_size=2)
                if larger is not None:
                    rng = larger
            self._set_cursor(rng.start, disable_follow=True)

    def go_bottom(self) -> None:
        with self._lock:
            rng = self._find_node_range(self._cursor)
            if rng is None:
                return
            if rng.start == rng.end:
                larger = self._find_node_range(self._cursor, min_size=2)
                if larger is not None:
                    rng = larger
            self._set_cursor(rng.end, disable_follow=True)
            if rng.end >= max(0, len(self._lines) - 1):
                self._follow_mode = True

    def jump_prev_agent(self) -> None:
        with self._lock:
            targets = [
                idx
                for idx, info in enumerate(self._line_infos)
                if info.is_agent_header
            ]
            if not targets:
                return
            prev = [idx for idx in targets if idx < self._cursor]
            target = prev[-1] if prev else targets[0]
            self._set_cursor(target, disable_follow=True)

    def jump_next_agent(self) -> None:
        with self._lock:
            targets = [
                idx
                for idx, info in enumerate(self._line_infos)
                if info.is_agent_header
            ]
            if not targets:
                return
            nxt = [idx for idx in targets if idx > self._cursor]
            target = nxt[0] if nxt else targets[-1]
            self._set_cursor(target, disable_follow=True)

    def toggle_expanded(self) -> None:
        with self._lock:
            self._toggle_expanded_locked()

    def toggle(self) -> None:
        self.toggle_expanded()

    def expand_all_nodes(self) -> None:
        with self._lock:
            if self._last_view is None:
                return
            for key in self._iter_node_keys(self._last_view):
                self._collapse_overrides[key] = False

    def collapse_all_nodes(self) -> None:
        with self._lock:
            if self._last_view is None:
                return
            for key in self._iter_node_keys(self._last_view):
                self._collapse_overrides[key] = True
            self._set_cursor(0, disable_follow=True)

    def collapse_enclosing_agent(self) -> None:
        with self._lock:
            rng = self._find_node_range(self._cursor, min_size=1, agent_only=True)
            if rng is None:
                return
            self._collapse_overrides[rng.key] = True
            self._set_cursor(rng.start, disable_follow=True)

    def reset_for_browse(self) -> None:
        with self._lock:
            self._follow_mode = False
            self._set_cursor(0, disable_follow=False)

    # ── Render state update and frame building ────────────────────────────

    def _adopt_root_view_locked(self, view: NodeView) -> None:
        if self._root_id is not None and view.id != self._root_id:
            self._collapse_overrides.clear()
            self._cursor = 0
            self._scroll_offset = 0
            self._selected_key = None
            self._selected_anchor = None
            self._selected_anchor_occurrence = 0
            self._follow_mode = True
            self._lines = []
            self._line_infos = []
            self._node_ranges = []
        self._root_id = view.id
        self._last_view = view

    def assign_view(self, view: NodeView | None) -> None:
        with self._lock:
            if view is None:
                return
            if (
                self._last_view is not None
                and view.id == self._last_view.id
                and view.update_seqnum == self._last_view.update_seqnum
            ):
                return
            self._adopt_root_view_locked(view)

    def set_cancel_event(self, cancel_event: Event | None) -> None:
        with self._lock:
            self._cancel_event = cancel_event

    def cancel_event(self) -> Event | None:
        with self._lock:
            return self._cancel_event

    def apply_action(self, action: str) -> None:
        action_map = {
            "move_up": self.navigate_up,
            "move_down": self.navigate_down,
            "page_up": self.page_up,
            "page_down": self.page_down,
            "toggle": self.toggle_expanded,
            "go_top": self.go_top,
            "go_bottom": self.go_bottom,
            "next_agent": self.jump_next_agent,
            "prev_agent": self.jump_prev_agent,
            "collapse_agent": self.collapse_enclosing_agent,
            "expand_all": self.expand_all_nodes,
            "collapse_all": self.collapse_all_nodes,
            "copy_result": self.copy_terminal_result,
            "focus_result": self.focus_terminal_result,
        }
        handler = action_map.get(action)
        if handler is None:
            raise ValueError(f"Unknown ConsoleRender action: {action}")
        handler()

    def handle_mouse_event(self, x: int, y: int, *, button: str = "left") -> None:
        self.handle_click(x, y, button=button)

    def _selected_tree_status_locked(self) -> SelectedTreeStatus:
        token_bill = {}
        state = None
        can_cancel = False
        cancel_pending = False
        if self._last_view is not None:
            token_bill = self._last_view.total_tree_token_bill()
            state = self._last_view.state
            can_cancel = (
                self._cancel_event is not None
                and self._last_view.state not in _TERMINAL_STATES
            )
            cancel_pending = bool(self._cancel_event and self._cancel_event.is_set())

        return SelectedTreeStatus(
            cursor_line=(self._cursor + 1) if self._lines else 0,
            total_lines=len(self._lines),
            state=state,
            cancel_pending=cancel_pending,
            can_cancel=can_cancel,
            token_bill=token_bill,
        )

    def selected_tree_status(self) -> SelectedTreeStatus:
        with self._lock:
            return self._selected_tree_status_locked()

    def _right_pane_context_locked(self) -> RightPaneInteractionContext:
        has_lines = bool(self._lines)
        can_expand = any(info.expandable for info in self._line_infos)
        can_jump = sum(1 for info in self._line_infos if info.is_agent_header) > 1
        result_target = self._terminal_root_result_target_locked()
        is_terminal = bool(
            self._last_view and self._last_view.state in _TERMINAL_STATES
        )
        return RightPaneInteractionContext(
            has_lines=has_lines,
            can_expand_collapse=can_expand,
            can_jump_agents=can_jump,
            follow_mode=self._follow_mode,
            is_terminal=is_terminal,
            can_copy_root_result=result_target is not None,
            can_focus_root_result=result_target is not None,
        )

    def right_pane_context(self) -> RightPaneInteractionContext:
        with self._lock:
            return self._right_pane_context_locked()

    def render_body(
        self,
        *,
        width: int,
        height: int,
        view: NodeView | None = None,
        tick: int | None = None,
    ) -> str:
        with self._lock:
            return self._render_locked(
                view,
                width=width,
                height=height,
                tick=tick,
            )

    def _render_locked(
        self,
        view: NodeView | None,
        *,
        width: int,
        height: int,
        tick: int | None = None,
    ) -> str:
        if view is not None:
            self._adopt_root_view_locked(view)
        if self._last_view is None:
            cols = max(1, width)
            rows = max(1, height)
            self._cols = cols
            self._tree_rows = rows
            placeholder = _crop_line("(waiting for data...)", cols)
            if _visible_len(placeholder) < cols:
                placeholder = placeholder + " " * (cols - _visible_len(placeholder))
            output = [placeholder]
            while len(output) < rows:
                output.append(" " * cols)
            return "\n".join(output)

        tick = self._tick() if tick is None else tick
        cancel_pending = bool(self._cancel_event and self._cancel_event.is_set())

        self._cols = max(1, width)

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

        if self._follow_mode:
            self._cursor = max(0, len(lines) - 1)
        else:
            resolved: int | None = None
            if self._selected_key is not None:
                for idx, info in enumerate(infos):
                    if info.key == self._selected_key or self._selected_key in info.anchors:
                        resolved = idx
                        break
            if resolved is None and self._selected_anchor is not None:
                matching = [
                    idx
                    for idx, info in enumerate(infos)
                    if info.key == self._selected_anchor
                    or self._selected_anchor in info.anchors
                ]
                if matching:
                    resolved = matching[
                        min(self._selected_anchor_occurrence, len(matching) - 1)
                    ]
            if resolved is not None:
                self._cursor = resolved

        max_pos = max(0, len(lines) - 1)
        self._cursor = max(0, min(self._cursor, max_pos))
        self._remember_selection()

        return self._render_viewport(
            rows=height,
            cols=self._cols,
        )

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
        anchors: tuple[str, ...] = (),
    ) -> None:
        node_start_idx = len(lines)
        key = f"n:{nv.id}"
        is_agent = nv.fn.is_agent()

        # Determine expandability
        has_children = bool(nv.children)
        has_transcript_rows = any(
            isinstance(
                p,
                (
                    ThinkingBlockPart,
                    UserTextPart,
                    ModelTextPart,
                    ToolUsePart,
                    ToolResultPart,
                ),
            )
            for p in nv.transcript
        )
        has_usage = nv.usage is not None
        has_inputs = bool(nv.inputs)
        has_result = _has_output(nv)
        has_error = _has_error(nv) or (
            nv.state is NodeState.Canceled and nv.exception is not None
        )
        has_details = (
            has_children
            or has_transcript_rows
            or has_usage
            or has_result
            or has_error
            or (not is_agent and has_inputs)
        )
        if nv.fn.is_code():
            has_details = True

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
        elif is_agent and nv.inputs:
            args_inline = _format_args(nv.inputs, max_len=140, per_val_len=50)
            if args_inline:
                header += f"({_color(args_inline, dim=True)})"

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
                is_node_header=True,
                is_agent_header=is_agent,
                toggle_col=_visible_index_of_any(line_text, {FOLD, UNFOLD}),
                anchors=anchors,
            )
        )

        if collapsed:
            self._node_ranges.append(
                NodeRange(
                    start=node_start_idx,
                    end=len(lines) - 1,
                    key=key,
                    is_agent=is_agent,
                )
            )
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
            n_children = len(nv.children)
            for idx, child in enumerate(nv.children):
                self._build_node(
                    child,
                    child_prefix,
                    idx == (n_children - 1),
                    tick,
                    cancel_pending,
                    lines,
                    infos,
                    depth=depth + 1,
                )

        self._node_ranges.append(
            NodeRange(
                start=node_start_idx,
                end=len(lines) - 1,
                key=key,
                is_agent=is_agent,
            )
        )

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
        indicator = (f"{FOLD} " if collapsed else f"{UNFOLD} ") if has_details else "  "

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
        if is_agent and nv.inputs:
            args = _format_args(nv.inputs, max_len=100, per_val_len=40)
            if args:
                parts.append(args)
        if has_children:
            n_agent_fns = sum(1 for c in nv.children if c.fn.is_agent())
            n_code_fns = len(nv.children) - n_agent_fns
            if n_code_fns:
                parts.append(f"{n_code_fns} CodeFn")
            if n_agent_fns:
                parts.append(f"{n_agent_fns} AgentFn")
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
        anchors: tuple[str, ...] = (),
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
            infos.append(LineInfo(anchors=anchors))
            return

        pos = 0
        while pos < len(raw_text):
            chunk = raw_text[pos : pos + avail]
            lines.append(f"{prefix}{_color(chunk, fg=fg, dim=dim, bold=bold)}")
            infos.append(LineInfo(anchors=anchors))
            pos += avail

    # ── Shared detail-rendering helpers ───────────────────────────────────

    def _emit_content_block(
        self,
        text_lines: list[str],
        prefix: str,
        lines: list[str],
        infos: list[LineInfo],
        *,
        max_lines: int | None = None,
        fg: str | None = None,
        dim: bool = False,
        anchors: tuple[str, ...] = (),
    ) -> None:
        """Emit a block of text lines with optional truncation."""
        emitted = text_lines if max_lines is None else text_lines[:max_lines]
        for tl in emitted:
            self._append_content(tl, prefix, lines, infos, fg=fg, dim=dim, anchors=anchors)
        if max_lines is not None and len(text_lines) > max_lines:
            lines.append(
                f"{prefix}{_color(f'... ({len(text_lines)} lines total)', dim=True)}"
            )
            infos.append(LineInfo(anchors=anchors))

    def _emit_rendered_block(
        self,
        rendered_lines: list[_RenderedBlockLine],
        prefix: str,
        lines: list[str],
        infos: list[LineInfo],
        *,
        anchors: tuple[str, ...] = (),
    ) -> None:
        for rendered_line in rendered_lines:
            if rendered_line.styled_text is not None:
                avail = max(20, self._cols - _visible_len(prefix))
                styled_text = rendered_line.styled_text
                if _visible_len(styled_text) > avail:
                    styled_text = _crop_line(styled_text, avail)
                lines.append(f"{prefix}{styled_text}")
                infos.append(LineInfo(anchors=anchors))
                continue
            self._append_content(
                rendered_line.text,
                prefix,
                lines,
                infos,
                fg=rendered_line.fg,
                dim=rendered_line.dim,
                bold=rendered_line.bold,
                anchors=anchors,
            )

    def _emit_kv_pairs(
        self,
        inputs: dict[str, Any],
        header_prefix: str,
        value_prefix: str,
        lines: list[str],
        infos: list[LineInfo],
        *,
        max_lines: int | None = None,
        anchors: tuple[str, ...] = (),
    ) -> None:
        """Emit key-value pairs, using multi-line display for long values."""
        for k, v in inputs.items():
            val_str = str(v)
            val_lines = val_str.splitlines()
            if len(val_lines) > 1 or len(val_str) > 100:
                lines.append(
                    f"{header_prefix}{_color(ARGS_GLYPH + ' ' + k + ':', fg='cyan')}"
                )
                infos.append(LineInfo(anchors=anchors))
                self._emit_content_block(
                    val_lines,
                    value_prefix,
                    lines,
                    infos,
                    max_lines=max_lines,
                    dim=True,
                    anchors=anchors,
                )
            else:
                lines.append(
                    f"{header_prefix}"
                    f"{_color(ARGS_GLYPH + ' ' + k + ': ', fg='cyan')}"
                    f"{_color(val_str, dim=True)}"
                )
                infos.append(LineInfo(anchors=anchors))

    def _preview_for_header(
        self,
        text: str,
        header_prefix: str,
        header_plain: str,
    ) -> str:
        max_len = max(20, self._cols - _visible_len(header_prefix) - len(header_plain) - 4)
        return _preview_text(text, max_len)

    def _preview_for_rendered_lines(
        self,
        rendered_lines: list[_RenderedBlockLine],
        header_prefix: str,
        header_plain: str,
    ) -> str:
        for rendered_line in rendered_lines:
            preview = self._preview_for_header(
                rendered_line.text,
                header_prefix,
                header_plain,
            )
            if preview:
                return preview
        return ""

    def _emit_text_part(
        self,
        *,
        key: str,
        title: str,
        glyph: str,
        text: str,
        detail_prefix: str,
        content_prefix: str,
        lines: list[str],
        infos: list[LineInfo],
        fg: str | None = None,
        dim: bool = False,
        rendered_lines: list[_RenderedBlockLine] | None = None,
    ) -> None:
        collapsed = self._is_collapsed(key, default=True)
        indicator = FOLD if collapsed else UNFOLD
        n_chars = len(text)
        header_plain = f"{indicator} {glyph} {title} ({n_chars:,} chars)"
        label = f"{detail_prefix}{_color(header_plain, fg=fg, dim=dim)}"
        if collapsed:
            preview = (
                self._preview_for_rendered_lines(
                    rendered_lines,
                    detail_prefix,
                    header_plain,
                )
                if rendered_lines is not None
                else self._preview_for_header(text, detail_prefix, header_plain)
            )
            if preview:
                label += f" {_color(preview, dim=True)}"
        lines.append(label)
        infos.append(
            LineInfo(
                key=key,
                expandable=True,
                default_collapsed=True,
                toggle_col=_visible_index_of_any(label, {FOLD, UNFOLD}),
                anchors=(key,),
            )
        )
        if not collapsed:
            if rendered_lines is None:
                block = text.splitlines() or [""]
                self._emit_content_block(
                    block,
                    content_prefix,
                    lines,
                    infos,
                    max_lines=None,
                    dim=True,
                    anchors=(key,),
                )
            else:
                self._emit_rendered_block(
                    rendered_lines,
                    content_prefix,
                    lines,
                    infos,
                    anchors=(key,),
                )

    def _emit_value_part(
        self,
        *,
        key: str,
        title: str,
        glyph: str,
        text: str,
        detail_prefix: str,
        content_prefix: str,
        lines: list[str],
        infos: list[LineInfo],
        fg: str | None = None,
        rendered_lines: list[_RenderedBlockLine] | None = None,
    ) -> None:
        collapsed = self._is_collapsed(key, default=True)
        indicator = FOLD if collapsed else UNFOLD
        n_chars = len(text)
        header_plain = f"{indicator} {glyph} {title} ({n_chars:,} chars)"
        line = f"{detail_prefix}{_color(header_plain, fg=fg)}"
        if collapsed:
            preview = (
                self._preview_for_rendered_lines(
                    rendered_lines,
                    detail_prefix,
                    header_plain,
                )
                if rendered_lines is not None
                else self._preview_for_header(text, detail_prefix, header_plain)
            )
            if preview:
                line += f" {_color(preview, dim=True)}"
        lines.append(line)
        infos.append(
            LineInfo(
                key=key,
                expandable=True,
                default_collapsed=True,
                toggle_col=_visible_index_of_any(line, {FOLD, UNFOLD}),
                anchors=(key,),
            )
        )
        if not collapsed:
            if rendered_lines is None:
                self._emit_content_block(
                    text.splitlines() or [""],
                    content_prefix,
                    lines,
                    infos,
                    max_lines=None,
                    dim=True,
                    anchors=(key,),
                )
            else:
                self._emit_rendered_block(
                    rendered_lines,
                    content_prefix,
                    lines,
                    infos,
                    anchors=(key,),
                )

    def _emit_synthetic_function_row(
        self,
        *,
        node_id: int,
        invocation_id: str,
        function_name: str,
        args: dict[str, Any] | None,
        result_part: ToolResultPart | None,
        detail_prefix: str,
        content_prefix: str,
        lines: list[str],
        infos: list[LineInfo],
    ) -> None:
        key = f"fncall:{node_id}:{invocation_id}"
        collapsed = self._is_collapsed(key, default=True)
        indicator = FOLD if collapsed else UNFOLD

        if result_part is None:
            status = "pending"
            status_fg = "yellow"
        elif result_part.is_error:
            status = "failed"
            status_fg = "red"
        else:
            status = "completed"
            status_fg = "green"

        args_preview = _format_args(args or {}, max_len=90, per_val_len=40)
        suffix = f" ({args_preview})" if args_preview else ""
        if result_part is None:
            result_preview = ""
        elif result_part.is_error:
            result_preview = f" {_color('!!', fg='red', bold=True)} {_short_repr(result_part.outputs, 60)}"
        else:
            result_preview = f" {_color('=>', dim=True)} {_short_repr(result_part.outputs, 60)}"

        header = f"{indicator} {FUNCTION_GLYPH} {function_name} [{status}]{suffix}{result_preview}"
        line = f"{detail_prefix}{_color(header, fg=status_fg)}"
        lines.append(line)
        infos.append(
            LineInfo(
                key=key,
                expandable=True,
                default_collapsed=True,
                toggle_col=_visible_index_of_any(line, {FOLD, UNFOLD}),
                anchors=(key,),
            )
        )

        if collapsed:
            return

        if args:
            self._emit_kv_pairs(
                args,
                content_prefix,
                content_prefix + "  ",
                lines,
                infos,
                max_lines=None,
                anchors=(key,),
            )
        else:
            lines.append(f"{content_prefix}{_color('(no args)', dim=True)}")
            infos.append(LineInfo(anchors=(key,)))

        if result_part is None:
            lines.append(f"{content_prefix}{_color('waiting for function result...', fg='yellow')}")
            infos.append(LineInfo(anchors=(key,)))
            return

        label = "error" if result_part.is_error else "result"
        color = "red" if result_part.is_error else "green"
        lines.append(f"{content_prefix}{_color(f'{RESULT_GLYPH} {label}:', fg=color)}")
        infos.append(LineInfo(anchors=(key,)))
        self._emit_content_block(
            str(result_part.outputs).splitlines() or [""],
            content_prefix + "  ",
            lines,
            infos,
            max_lines=None,
            fg="red" if result_part.is_error else None,
            dim=not result_part.is_error,
            anchors=(key,),
        )

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
        if nv.usage:
            usage_text = self._format_usage(nv.usage)
            lines.append(f"{detail_prefix}{usage_text}")
            infos.append(LineInfo(anchors=(f"n:{nv.id}",)))

        tool_use_by_id: dict[str, ToolUsePart] = {}
        tool_result_by_id: dict[str, ToolResultPart] = {}
        for part in nv.transcript:
            if isinstance(part, ToolUsePart):
                tool_use_by_id.setdefault(part.tool_use_id, part)
            elif isinstance(part, ToolResultPart):
                tool_result_by_id[part.tool_use_id] = part

        render_entries: list[tuple[str, int, Any]] = []
        seen_invocation_ids: set[str] = set()
        for tx_idx, part in enumerate(nv.transcript):
            if isinstance(part, UserTextPart):
                render_entries.append(("user", tx_idx, part))
            elif isinstance(part, ModelTextPart):
                render_entries.append(("model", tx_idx, part))
            elif isinstance(part, ThinkingBlockPart):
                render_entries.append(("thinking", tx_idx, part))
            elif isinstance(part, ToolUsePart):
                if part.tool_use_id in seen_invocation_ids:
                    continue
                seen_invocation_ids.add(part.tool_use_id)
                render_entries.append(("call", tx_idx, part))
            elif isinstance(part, ToolResultPart):
                if part.tool_use_id in seen_invocation_ids:
                    continue
                seen_invocation_ids.add(part.tool_use_id)
                render_entries.append(("call_result_only", tx_idx, part))

        rendered_child_ids: set[int] = set()
        for child in nv.children:
            if child.tool_use_id and child.tool_use_id in seen_invocation_ids:
                continue
            render_entries.append(("child_only", len(nv.transcript), child))

        has_model_text = any(isinstance(p, ModelTextPart) for p in nv.transcript)
        has_error_outcome = _has_error(nv) or (
            nv.state is NodeState.Canceled and nv.exception is not None
        )
        has_trailing_output = has_error_outcome or (_has_output(nv) and not has_model_text)

        for idx, (kind, tx_idx, payload) in enumerate(render_entries):
            is_last_item = (idx == len(render_entries) - 1) and not has_trailing_output

            if kind == "user":
                part = payload
                assert isinstance(part, UserTextPart)
                self._emit_text_part(
                    key=f"tp:{nv.id}:{tx_idx}:user",
                    title="user",
                    glyph=USER_GLYPH,
                    text=part.text,
                    detail_prefix=detail_prefix,
                    content_prefix=content_prefix,
                    lines=lines,
                    infos=infos,
                    fg="blue",
                )
                continue

            if kind == "model":
                part = payload
                assert isinstance(part, ModelTextPart)
                model_key = f"tp:{nv.id}:{tx_idx}:model"
                self._emit_text_part(
                    key=model_key,
                    title="result",
                    glyph=RESULT_GLYPH,
                    text=part.text,
                    detail_prefix=detail_prefix,
                    content_prefix=content_prefix,
                    lines=lines,
                    infos=infos,
                    fg="green",
                    rendered_lines=self._rendered_root_result_lines_locked(
                        model_key,
                        part.text,
                        content_prefix,
                    ),
                )
                continue

            if kind == "thinking":
                part = payload
                assert isinstance(part, ThinkingBlockPart)
                self._emit_thinking_slot(
                    part,
                    key=f"tp:{nv.id}:{tx_idx}:thinking",
                    detail_prefix=detail_prefix,
                    content_prefix=content_prefix,
                    lines=lines,
                    infos=infos,
                )
                continue

            if kind == "child_only":
                child = payload
                assert isinstance(child, NodeView)
                rendered_child_ids.add(child.id)
                self._build_node(
                    child,
                    child_prefix,
                    is_last_item,
                    tick,
                    cancel_pending,
                    lines,
                    infos,
                    depth=depth + 1,
                )
                continue

            if kind == "call":
                part = payload
                assert isinstance(part, ToolUsePart)
                tool_use_id = part.tool_use_id
                tool_name = part.tool_name
                child_view = nv.transcript_child_map.get(id(part))
            else:
                part = payload
                assert isinstance(part, ToolResultPart)
                tool_use_id = part.tool_use_id
                tool_name = part.tool_name
                child_view = nv.transcript_child_map.get(id(part))

            use_part = tool_use_by_id.get(tool_use_id)
            result_part = tool_result_by_id.get(tool_use_id)
            if child_view is None and use_part is not None:
                child_view = nv.transcript_child_map.get(id(use_part))
            if child_view is None and result_part is not None:
                child_view = nv.transcript_child_map.get(id(result_part))

            if child_view is not None:
                rendered_child_ids.add(child_view.id)
                self._build_node(
                    child_view,
                    child_prefix,
                    is_last_item,
                    tick,
                    cancel_pending,
                    lines,
                    infos,
                    depth=depth + 1,
                    anchors=(f"fncall:{nv.id}:{tool_use_id}",),
                )
                continue

            if kind == "call":
                assert isinstance(part, ToolUsePart)
                args = dict(part.args)
                tool_name = part.tool_name
            elif use_part is not None:
                args = dict(use_part.args)
                tool_name = use_part.tool_name
            else:
                args = {}

            self._emit_synthetic_function_row(
                node_id=nv.id,
                invocation_id=tool_use_id,
                function_name=tool_name,
                args=args,
                result_part=result_part,
                detail_prefix=detail_prefix,
                content_prefix=content_prefix,
                lines=lines,
                infos=infos,
            )

        orphan_children = [child for child in nv.children if child.id not in rendered_child_ids]
        for orphan_idx, child in enumerate(orphan_children):
            self._build_node(
                child,
                child_prefix,
                (orphan_idx == len(orphan_children) - 1) and not has_trailing_output,
                tick,
                cancel_pending,
                lines,
                infos,
                depth=depth + 1,
            )

        if _has_output(nv) and not has_model_text:
            result_key = f"ao:{nv.id}"
            self._emit_text_part(
                key=result_key,
                title="result",
                glyph=RESULT_GLYPH,
                text=str(nv.outputs),
                detail_prefix=detail_prefix,
                content_prefix=content_prefix,
                lines=lines,
                infos=infos,
                fg="green",
                rendered_lines=self._rendered_root_result_lines_locked(
                    result_key,
                    str(nv.outputs),
                    content_prefix,
                ),
            )
        elif has_error_outcome:
            self._emit_error_block(
                nv,
                detail_prefix,
                content_prefix,
                lines,
                infos,
                anchors=(f"ae:{nv.id}",),
            )

    def _emit_code_details(
        self,
        nv: NodeView,
        detail_prefix: str,
        content_prefix: str,
        lines: list[str],
        infos: list[LineInfo],
    ) -> None:
        """Emit expanded content for a code function: args and result."""
        if nv.inputs:
            for arg_name, arg_value in nv.inputs.items():
                self._emit_value_part(
                    key=f"ca:{nv.id}:{arg_name}",
                    title=arg_name,
                    glyph=ARGS_GLYPH,
                    text=str(arg_value),
                    detail_prefix=detail_prefix,
                    content_prefix=content_prefix,
                    lines=lines,
                    infos=infos,
                    fg="cyan",
                )

        if _has_output(nv):
            result_key = f"cr:{nv.id}"
            self._emit_value_part(
                key=result_key,
                title="result",
                glyph=RESULT_GLYPH,
                text=str(nv.outputs),
                detail_prefix=detail_prefix,
                content_prefix=content_prefix,
                lines=lines,
                infos=infos,
                fg="green",
                rendered_lines=self._rendered_root_result_lines_locked(
                    result_key,
                    str(nv.outputs),
                    content_prefix,
                ),
            )
        elif _has_error(nv) or (nv.state is NodeState.Canceled and nv.exception is not None):
            self._emit_error_block(
                nv,
                detail_prefix,
                content_prefix,
                lines,
                infos,
                anchors=(f"n:{nv.id}",),
            )

    def _emit_error_block(
        self,
        nv: NodeView,
        detail_prefix: str,
        content_prefix: str,
        lines: list[str],
        infos: list[LineInfo],
        *,
        anchors: tuple[str, ...] = (),
    ) -> None:
        is_cancel = nv.state is NodeState.Canceled
        label = "canceled" if is_cancel else "error"
        color = "yellow" if is_cancel else "red"
        lines.append(f"{detail_prefix}{_color(f'✖ {label}:', fg=color, bold=True)}")
        infos.append(LineInfo(anchors=anchors))
        self._emit_content_block(
            str(nv.exception).splitlines() if nv.exception is not None else [""],
            content_prefix,
            lines,
            infos,
            max_lines=None,
            fg=color,
            anchors=anchors,
        )

    def _emit_thinking_slot(
        self,
        part: ThinkingBlockPart,
        *,
        key: str,
        detail_prefix: str,
        content_prefix: str,
        lines: list[str],
        infos: list[LineInfo],
    ) -> None:
        collapsed = self._is_collapsed(key, default=True)

        # Character count
        n_chars = len(part.content) if part.content else 0
        char_info = f" ({n_chars:,} chars)" if n_chars else ""

        indicator = FOLD if collapsed else UNFOLD

        if part.redacted:
            text = f"{indicator} {THINKING} thinking [redacted]"
        else:
            preview_part = ""
            if collapsed:
                preview = _preview_text(part.content or "", 60)
                preview_part = f": {preview}" if preview else ""
            text = f"{indicator} {THINKING} thinking{preview_part}{char_info}"

        line = f"{detail_prefix}{_color(text, dim=True)}"
        lines.append(line)
        infos.append(
            LineInfo(
                key=key,
                expandable=True,
                default_collapsed=True,
                toggle_col=_visible_index_of_any(line, {FOLD, UNFOLD}),
                anchors=(key,),
            )
        )

        if not collapsed and part.content:
            self._emit_content_block(
                part.content.splitlines(),
                content_prefix,
                lines,
                infos,
                max_lines=None,
                dim=True,
                anchors=(key,),
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

    @staticmethod
    def _provider_model_abbrev(provider: Provider) -> str:
        model_name = ModelNames[provider].replace(".", "-")
        return "".join(part[0] for part in model_name.split("-") if part)

    @staticmethod
    def _format_tokens_in_k(value: int, *, decimals: int) -> str:
        if decimals == 0:
            return f"{(value + 500) // 1000}k"

        scale = 10**decimals
        rounded = (value * scale + 500) // 1000
        whole, frac = divmod(rounded, scale)
        return f"{whole}.{frac:0{decimals}d}k"

    @staticmethod
    def _format_token_bill_fields(bill: TokenBill) -> str:
        fields: list[str] = []
        if bill.input_tokens_cache_read:
            fields.append(f"CR:{ConsoleRender._format_tokens_in_k(bill.input_tokens_cache_read, decimals=0)}")
        if bill.input_tokens_cache_write:
            fields.append(
                f"CW:{ConsoleRender._format_tokens_in_k(bill.input_tokens_cache_write, decimals=0)}"
            )
        if bill.input_tokens_regular:
            fields.append(f"Reg:{ConsoleRender._format_tokens_in_k(bill.input_tokens_regular, decimals=0)}")
        if bill.output_tokens_total:
            fields.append(f"Out:{ConsoleRender._format_tokens_in_k(bill.output_tokens_total, decimals=1)}")
        return " ".join(fields)

    @classmethod
    def _format_total_token_bill(
        cls,
        bills: Mapping[Provider, TokenBill],
    ) -> str:
        rendered: list[str] = []
        for provider, bill in bills.items():
            fields = cls._format_token_bill_fields(bill)
            if not fields:
                continue
            rendered.append(f"{cls._provider_model_abbrev(provider)}[{fields}]")
        return "  ".join(rendered)

    # ── Viewport rendering ────────────────────────────────────────────────

    def _render_viewport(
        self,
        *,
        rows: int,
        cols: int,
    ) -> str:
        rows = max(1, rows)
        cols = max(1, cols)
        tree_rows = max(1, rows)
        self._tree_rows = tree_rows

        # Adjust scroll to keep cursor visible
        if self._cursor < self._scroll_offset:
            self._scroll_offset = self._cursor
        elif self._cursor >= self._scroll_offset + tree_rows:
            self._scroll_offset = self._cursor - tree_rows + 1
        self._scroll_offset = max(0, self._scroll_offset)

        output: list[str] = []
        end = min(self._scroll_offset + tree_rows, len(self._lines))
        for i in range(self._scroll_offset, end):
            line = _crop_line(self._lines[i], cols)
            if i == self._cursor:
                line = _highlight_line(line, cols)
            if _visible_len(line) < cols:
                line = line + " " * (cols - _visible_len(line))
            output.append(line)

        # Pad to fill viewport
        while len(output) < tree_rows:
            output.append(" " * cols)

        return "\n".join(output)


    def _tick(self) -> int:
        return int((time.monotonic() - self._t0) * self.spinner_hz)

    def run(
        self,
        node: Node,
    ) -> None:
        """Run the standard interactive console session for *node*."""
        from ._controllers import SingleTreeConsoleController
        from ._driver import ConsoleSessionDriver

        if node.cancel_event is None:
            raise ValueError(
                "ConsoleRender.run(node) requires the node to have a cancel_event. "
                "Pass cancel_event=... when invoking the top-level function."
            )

        driver = ConsoleSessionDriver(spinner_hz=self.spinner_hz)
        controller = SingleTreeConsoleController(self, node)
        driver.run(controller)
