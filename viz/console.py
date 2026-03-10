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

from __future__ import annotations

import os
import re
import signal
import shutil
import sys
import threading
import time
import ctypes
from dataclasses import dataclass
from multiprocessing.synchronize import Event
from typing import Any, Iterator, Mapping

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

if os.name != "nt":
    import select
    import termios
    import tty
else:  # pragma: no cover - runtime guarded use on non-POSIX terminals only
    select = None
    termios = None
    tty = None

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

_WIN_STD_OUTPUT_HANDLE = -11
_WIN_STD_INPUT_HANDLE = -10
_WIN_ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
_WIN_ENABLE_EXTENDED_FLAGS = 0x0080
_WIN_ENABLE_QUICK_EDIT_MODE = 0x0040
_WIN_ENABLE_MOUSE_INPUT = 0x0010
_WIN_KEY_EVENT = 0x0001
_WIN_MOUSE_EVENT = 0x0002
_WIN_DOUBLE_CLICK = 0x0002
_WIN_MOUSE_WHEELED = 0x0004
_WIN_FROM_LEFT_1ST_BUTTON_PRESSED = 0x0001
_WIN_RIGHTMOST_BUTTON_PRESSED = 0x0002
_WIN_FROM_LEFT_2ND_BUTTON_PRESSED = 0x0004


class _WinCoord(ctypes.Structure):
    _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]


class _WinKeyEventRecord(ctypes.Structure):
    _fields_ = [
        ("bKeyDown", ctypes.c_int),
        ("wRepeatCount", ctypes.c_ushort),
        ("wVirtualKeyCode", ctypes.c_ushort),
        ("wVirtualScanCode", ctypes.c_ushort),
        ("uChar", ctypes.c_wchar),
        ("dwControlKeyState", ctypes.c_uint),
    ]


class _WinMouseEventRecord(ctypes.Structure):
    _fields_ = [
        ("dwMousePosition", _WinCoord),
        ("dwButtonState", ctypes.c_uint),
        ("dwControlKeyState", ctypes.c_uint),
        ("dwEventFlags", ctypes.c_uint),
    ]


class _WinEventUnion(ctypes.Union):
    _fields_ = [
        ("KeyEvent", _WinKeyEventRecord),
        ("MouseEvent", _WinMouseEventRecord),
        ("_padding", ctypes.c_byte * 16),
    ]


class _WinInputRecord(ctypes.Structure):
    _fields_ = [("EventType", ctypes.c_ushort), ("Event", _WinEventUnion)]


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
class MouseEvent:
    x: int
    y: int
    button: str


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
    - `ConsoleRender.run(node)` owns the terminal session and manages both the
      background watch/render loop and the foreground keyboard loop.
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
    - `run(node)` manages terminal ownership, the live watch/render loop,
      keyboard handling, and the post-completion browser.
    """

    _console_lock = threading.Lock()
    _console_ready = False

    def __init__(
        self,
        cancel_event: Event | None = None,
        *,
        width: int | None = None,
        spinner_hz: float = 10.0,
        follow: bool = True,
    ) -> None:
        """Initialise the interactive tree renderer."""
        self.width = width
        self.spinner_hz = max(1.0, float(spinner_hz))
        self._cancel_event = cancel_event
        self._last_view: NodeView | None = None
        self._root_id: int | None = None
        self._t0 = time.monotonic()
        self._lock = threading.Lock()
        self._loop_wake = threading.Condition()
        self._loop_pending_view: NodeView | None = None
        self._loop_pending_redraw = False
        self._loop_stop = False
        self._sigint_handler_installed = False
        self._exit_after_terminal = False
        self._console_session_active = False

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

    def request_redraw(self) -> None:
        with self._lock:
            self._signal_redraw_locked()

    def _signal_redraw_locked(self) -> None:
        with self._loop_wake:
            self._loop_pending_redraw = True
            self._loop_wake.notify()

    def _stop_view_loop(self) -> None:
        with self._loop_wake:
            self._loop_stop = True
            self._loop_wake.notify_all()

    def _ui_driver(self, s: str) -> None:
        if not self._console_session_active:
            return
        ConsoleRender.ui_driver(s)

    def _request_cancel(self) -> bool:
        if self._cancel_event is None:
            return False
        already_set = self._cancel_event.is_set()
        self._cancel_event.set()
        self._exit_after_terminal = True
        with self._lock:
            self._signal_redraw_locked()
        return not already_set

    def _install_sigint_handler(
        self,
        view_thread: threading.Thread,
    ) -> signal.Handlers | None:
        if self._cancel_event is None:
            return None
        if threading.current_thread() is not threading.main_thread():
            return None

        previous_handler = signal.getsignal(signal.SIGINT)

        def _handle_sigint(signum: int, frame: Any) -> None:
            del signum, frame
            if view_thread.is_alive() and self._request_cancel():
                return
            self._stop_view_loop()
            self._console_session_active = False
            ConsoleRender.restore_console()
            if callable(previous_handler):
                previous_handler(signal.SIGINT, None)
                return
            if previous_handler == signal.SIG_IGN:
                return
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, _handle_sigint)
        self._sigint_handler_installed = True
        return previous_handler

    def _restore_sigint_handler(
        self,
        previous_handler: signal.Handlers | None,
    ) -> None:
        if not self._sigint_handler_installed:
            return
        signal.signal(signal.SIGINT, previous_handler)
        self._sigint_handler_installed = False

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
        self._signal_redraw_locked()

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
        self._signal_redraw_locked()
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
            self._signal_redraw_locked()

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

    def render(self, view: NodeView | None) -> str:
        """Render the current tree view to a string."""
        with self._lock:
            return self._render_locked(view)

    def _render_locked(self, view: NodeView | None) -> str:
        if view is not None:
            if self._root_id is not None and view.id != self._root_id:
                self._collapse_overrides.clear()
                self._cursor = 0
                self._scroll_offset = 0
                self._selected_key = None
                self._selected_anchor = None
                self._selected_anchor_occurrence = 0
                self._follow_mode = True
            self._root_id = view.id
            self._last_view = view
        if self._last_view is None:
            return "(waiting for data...)"

        tick = self._tick()
        cancel_pending = bool(self._cancel_event and self._cancel_event.is_set())

        sz = shutil.get_terminal_size(fallback=(80, 24))
        self._cols = max(1, self.width if self.width is not None else sz.columns)
        self._tree_rows = max(1, sz.lines - 1)

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
    ) -> None:
        collapsed = self._is_collapsed(key, default=True)
        indicator = FOLD if collapsed else UNFOLD
        n_chars = len(text)
        header_plain = f"{indicator} {glyph} {title} ({n_chars:,} chars)"
        label = f"{detail_prefix}{_color(header_plain, fg=fg, dim=dim)}"
        if collapsed:
            preview = self._preview_for_header(text, detail_prefix, header_plain)
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
    ) -> None:
        collapsed = self._is_collapsed(key, default=True)
        indicator = FOLD if collapsed else UNFOLD
        n_chars = len(text)
        header_plain = f"{indicator} {glyph} {title} ({n_chars:,} chars)"
        line = f"{detail_prefix}{_color(header_plain, fg=fg)}"
        if collapsed:
            preview = self._preview_for_header(text, detail_prefix, header_plain)
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
            self._emit_content_block(
                text.splitlines() or [""],
                content_prefix,
                lines,
                infos,
                max_lines=None,
                dim=True,
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
                self._emit_text_part(
                    key=f"tp:{nv.id}:{tx_idx}:model",
                    title="result",
                    glyph=RESULT_GLYPH,
                    text=part.text,
                    detail_prefix=detail_prefix,
                    content_prefix=content_prefix,
                    lines=lines,
                    infos=infos,
                    fg="green",
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
            self._emit_text_part(
                key=f"ao:{nv.id}",
                title="result",
                glyph=RESULT_GLYPH,
                text=str(nv.outputs),
                detail_prefix=detail_prefix,
                content_prefix=content_prefix,
                lines=lines,
                infos=infos,
                fg="green",
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
            self._emit_value_part(
                key=f"cr:{nv.id}",
                title="result",
                glyph=RESULT_GLYPH,
                text=str(nv.outputs),
                detail_prefix=detail_prefix,
                content_prefix=content_prefix,
                lines=lines,
                infos=infos,
                fg="green",
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
    def _format_token_bill_fields(bill: TokenBill) -> str:
        fields: list[str] = []
        if bill.input_tokens_cache_read:
            fields.append(f"CR:{bill.input_tokens_cache_read}")
        if bill.input_tokens_cache_write:
            fields.append(f"CW:{bill.input_tokens_cache_write}")
        if bill.input_tokens_regular:
            fields.append(f"Reg:{bill.input_tokens_regular}")
        if bill.output_tokens_total:
            fields.append(f"Out:{bill.output_tokens_total}")
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

    def _render_viewport(self) -> str:
        sz = shutil.get_terminal_size(fallback=(80, 24))
        rows = max(1, sz.lines)
        cols = max(1, self.width if self.width is not None else sz.columns)
        tree_rows = max(1, rows - 1)
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
            line = _crop_line(self._lines[i], cols - 1)
            if i == self._cursor:
                line = _highlight_line(line, cols - 1)
            output.append(f"{line}\x1b[K")

        # Pad to fill viewport
        while len(output) < tree_rows:
            output.append("\x1b[K")

        # Status bar
        output.append(f"{self._status_bar(cols)}\x1b[K")

        return "\n".join(output)

    def _status_bar(self, cols: int) -> str:
        pos = f"{self._cursor + 1}/{len(self._lines)}"

        state_text = ""
        if self._last_view:
            s = self._last_view.state
            if s is NodeState.Running:
                tick = self._tick()
                frame = _SPINNER_FRAMES[tick % len(_SPINNER_FRAMES)]
                state_text = _color(f"{frame} Running", fg="cyan")
            elif s is NodeState.Success:
                state_text = _color("✔ Complete", fg="green")
            elif s is NodeState.Error:
                state_text = _color("✖ Error", fg="red")
            elif s is NodeState.Canceled:
                state_text = _color("⏹ Canceled", fg="yellow")

        can_cancel = (
            self._cancel_event is not None
            and self._last_view is not None
            and self._last_view.state not in _TERMINAL_STATES
        )
        is_terminal = self._last_view and self._last_view.state in _TERMINAL_STATES

        quit_hint = "  q:quit" if is_terminal else ""
        cancel_hint = "  ^C:cancel" if can_cancel else ""
        status_text = pos if not state_text else f"{pos}  {state_text}"
        token_text = ""
        if self._last_view is not None:
            token_text = self._format_total_token_bill(
                self._last_view.total_tree_token_bill()
            )

        shortcut_variants = [
            "↑↓/jk:move  ␣:toggle  PgUp/PgDn:page  n/N:next/prev  c:agent  g/G:top/btm  E/C:all",
            "↑↓/jk:move  ␣:toggle  Pg:page  n/N:next/prev  c:agent  g/G:top/btm  E/C:all",
            "jk/↑↓:move  ␣:toggle  Pg:page  n/N:next/prev  c:agent  g/G:top/btm  E/C:all",
            "jk:move  ␣:toggle  Pg:page  n/N:next/prev  c:agent  g/G:top/btm  E/C:all",
            "jk:move  ␣:toggle  Pg:page  n/N:next/prev  c:agent  E/C:all",
            "jk ␣ Pg n/N c g/G E/C",
            "jk ␣ Pg n/N c E/C",
            "jk ␣ n/N c E/C",
            "jk ␣ n/N",
        ]

        status_raw = f" {status_text} "
        token_raw = f" {token_text} " if token_text else ""

        status_len = _visible_len(status_raw)
        token_len = _visible_len(token_raw)
        if status_len + token_len > cols:
            remaining = max(0, cols - status_len)
            if remaining <= 0:
                status_raw = _crop_line(status_raw, cols)
                token_raw = ""
            elif token_raw:
                token_raw = _crop_line(token_raw, remaining) if remaining >= 8 else ""

        right_len = _visible_len(status_raw) + _visible_len(token_raw)
        left_budget = max(0, cols - right_len)
        left_text_budget = max(0, left_budget - 1)

        left = f"{shortcut_variants[-1]}{cancel_hint}{quit_hint}"
        for candidate in shortcut_variants:
            candidate_left = f"{candidate}{cancel_hint}{quit_hint}"
            if _visible_len(candidate_left) <= left_text_budget:
                left = candidate_left
                break

        if _visible_len(left) > left_text_budget:
            if left_text_budget <= 1:
                left = ""
            elif left_text_budget == 2:
                left = "…"
            else:
                left = left[: left_text_budget - 1] + "…"

        left_raw = ""
        if left_budget > 0:
            left_raw = f" {left}"
            vis_len = _visible_len(left_raw)
            if vis_len < left_budget:
                left_raw = left_raw + " " * (left_budget - vis_len)
            elif vis_len > left_budget:
                left_raw = _crop_line(left_raw, left_budget)

        segments: list[str] = []
        if left_raw:
            segments.append(_style_block(left_raw, FG["white"], BG_STATUS_BAR))
        segments.append(_style_block(status_raw, BOLD, FG["white"], BG_STATUS_STATE))
        if token_raw:
            segments.append(_style_block(token_raw, BOLD, FG["white"], BG_STATUS_TOKENS))
        return "".join(segments)

    def _tick(self) -> int:
        return int((time.monotonic() - self._t0) * self.spinner_hz)

    def run(
        self,
        node: Node,
    ) -> None:
        """Run the standard interactive console session for *node*."""
        self._exit_after_terminal = False
        self._console_session_active = True
        self.pre_console()
        view_thread = self._start_view_thread(node)
        try:
            self.interact(view_thread)
        finally:
            self._stop_view_loop()
            if view_thread.is_alive():
                view_thread.join(timeout=1.0)
            self._console_session_active = False
            self.restore_console()

    def _start_view_thread(self, node: Node) -> threading.Thread:
        tick_interval = 1.0 / self.spinner_hz

        with self._loop_wake:
            self._loop_stop = False
            self._loop_pending_view = None
            self._loop_pending_redraw = True

        def _watch_loop() -> None:
            prev_seq = 0
            while True:
                with self._loop_wake:
                    if self._loop_stop:
                        return
                view = node.watch(as_of_seq=prev_seq, timeout=0.1)
                if view is None:
                    continue
                prev_seq = view.update_seqnum
                with self._loop_wake:
                    self._loop_pending_view = view
                    self._loop_wake.notify()
                if view.state in _TERMINAL_STATES:
                    return

        def _loop() -> None:
            last_view: NodeView | None = None
            next_at = time.monotonic()
            watch_thread = threading.Thread(
                target=_watch_loop,
                name="netflux-view-watch",
                daemon=True,
            )
            watch_thread.start()
            try:
                while True:
                    current = last_view
                    timeout: float | None = None
                    if current is None or current.state not in _TERMINAL_STATES:
                        timeout = max(0.0, next_at - time.monotonic())

                    with self._loop_wake:
                        if self._loop_pending_view is None and not self._loop_pending_redraw:
                            self._loop_wake.wait(timeout)
                        if self._loop_stop:
                            break
                        view = self._loop_pending_view
                        self._loop_pending_view = None
                        self._loop_pending_redraw = False

                    payload = self.render(view)
                    self._ui_driver(payload)

                    if view is not None:
                        last_view = view

                    current = view if view is not None else last_view
                    if current is not None and current.state in _TERMINAL_STATES:
                        break

                    now = time.monotonic()
                    if next_at <= now:
                        next_at = now + tick_interval
            finally:
                with self._loop_wake:
                    self._loop_stop = True
                    self._loop_wake.notify_all()

        view_thread = threading.Thread(
            target=_loop,
            name="netflux-view-loop",
            daemon=True,
        )
        view_thread.start()
        return view_thread

    def interact(self, view_thread: threading.Thread) -> None:
        """
        Handle live keyboard interaction for *view_thread* and keep the
        completed tree open until the user quits.
        """
        self.pre_console()
        previous_sigint_handler = self._install_sigint_handler(view_thread)
        try:
            if not sys.stdin.isatty():
                self._wait_non_interactive(view_thread)
                return
            if os.name == "nt":
                self._interactive_loop_windows(view_thread)
            else:
                self._interactive_loop_posix(view_thread)
        finally:
            self._restore_sigint_handler(previous_sigint_handler)

    def _wait_non_interactive(self, view_thread: threading.Thread) -> None:
        while view_thread.is_alive():
            try:
                view_thread.join(timeout=0.1)
            except KeyboardInterrupt:
                if self._cancel_event is not None and not self._sigint_handler_installed:
                    self._request_cancel()
                    continue
                raise

    def _interactive_loop_windows(self, view_thread: threading.Thread) -> None:
        while view_thread.is_alive():
            try:
                event = _read_key_windows(timeout=0.1)
            except KeyboardInterrupt:
                if self._cancel_event is not None and not self._sigint_handler_installed:
                    self._request_cancel()
                    continue
                raise
            if event is None:
                continue
            _handle_input_event(event, self, allow_quit=False)
        if self._exit_after_terminal:
            return
        self._browse_until_quit_windows()

    def _interactive_loop_posix(self, view_thread: threading.Thread) -> None:
        if termios is None or tty is None:
            self._wait_non_interactive(view_thread)
            return

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while view_thread.is_alive():
                try:
                    event = _read_key(fd, timeout=0.1)
                except KeyboardInterrupt:
                    if self._cancel_event is not None and not self._sigint_handler_installed:
                        self._request_cancel()
                        continue
                    raise
                if event is None:
                    continue
                _handle_input_event(event, self, allow_quit=False)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        if self._exit_after_terminal:
            return
        self._browse_until_quit_posix()

    def _browse_until_quit_windows(self) -> None:
        self._ui_driver(self.render(None))
        last_size = _terminal_size_token()
        while True:
            try:
                event = _read_key_windows(timeout=0.1)
            except KeyboardInterrupt:
                return
            if event is not None:
                should_exit = _handle_input_event(event, self, allow_quit=True)
                self._ui_driver(self.render(None))
                last_size = _terminal_size_token()
                if should_exit:
                    return
                continue
            size = _terminal_size_token()
            if size != last_size:
                last_size = size
                self._ui_driver(self.render(None))

    def _browse_until_quit_posix(self) -> None:
        if termios is None or tty is None:
            return

        self._ui_driver(self.render(None))
        last_size = _terminal_size_token()
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while True:
                try:
                    event = _read_key(fd, timeout=0.1)
                except KeyboardInterrupt:
                    return
                if event is not None:
                    should_exit = _handle_input_event(event, self, allow_quit=True)
                    self._ui_driver(self.render(None))
                    last_size = _terminal_size_token()
                    if should_exit:
                        return
                    continue
                size = _terminal_size_token()
                if size != last_size:
                    last_size = size
                    self._ui_driver(self.render(None))
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    # ── Static terminal management ────────────────────────────────────────

    @staticmethod
    def pre_console() -> None:
        """Enter alt screen, hide cursor, disable wrap, clear scrollback."""
        if not sys.stdout.isatty():
            return
        with ConsoleRender._console_lock:
            if ConsoleRender._console_ready:
                return
            _enable_vt_if_windows()
            _configure_windows_console_input(enable_mouse=True)
            sys.stdout.write(
                "\x1b[?1049h\x1b[?25l\x1b[?7l\x1b[3J"
                "\x1b[?1000h\x1b[?1006h"
            )
            sys.stdout.flush()
            ConsoleRender._console_ready = True

    @staticmethod
    def restore_console() -> None:
        """Show cursor, re-enable wrap, leave alt screen."""
        if not sys.stdout.isatty():
            return
        with ConsoleRender._console_lock:
            if not ConsoleRender._console_ready:
                return
            _configure_windows_console_input(enable_mouse=False)
            sys.stdout.write("\x1b[?1000l\x1b[?1006l\x1b[?25h\x1b[?7h\x1b[?1049l")
            sys.stdout.flush()
            ConsoleRender._console_ready = False

    @staticmethod
    def ui_driver(s: str) -> None:
        """Home the cursor and write the rendered frame."""
        if not sys.stdout.isatty():
            return
        ConsoleRender.pre_console()
        sys.stdout.write("\x1b[H")
        sys.stdout.write(s)
        sys.stdout.flush()


def _terminal_size_token() -> tuple[int, int]:
    sz = shutil.get_terminal_size(fallback=(80, 24))
    return (sz.columns, sz.lines)


def _enable_vt_if_windows() -> None:
    if os.name != "nt":
        return

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.GetStdHandle(_WIN_STD_OUTPUT_HANDLE)
    mode = ctypes.c_uint()
    if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        kernel32.SetConsoleMode(handle, mode.value | _WIN_ENABLE_VIRTUAL_TERMINAL_PROCESSING)


_WINDOWS_INPUT_MODE_SAVED: int | None = None


def _configure_windows_console_input(*, enable_mouse: bool) -> None:
    global _WINDOWS_INPUT_MODE_SAVED

    if os.name != "nt":
        return

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.GetStdHandle(_WIN_STD_INPUT_HANDLE)
    mode = ctypes.c_uint()
    if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        return

    if enable_mouse:
        if _WINDOWS_INPUT_MODE_SAVED is None:
            _WINDOWS_INPUT_MODE_SAVED = mode.value
        new_mode = mode.value | _WIN_ENABLE_EXTENDED_FLAGS | _WIN_ENABLE_MOUSE_INPUT
        new_mode &= ~_WIN_ENABLE_QUICK_EDIT_MODE
        kernel32.SetConsoleMode(handle, new_mode)
        return

    if _WINDOWS_INPUT_MODE_SAVED is not None:
        kernel32.SetConsoleMode(handle, _WINDOWS_INPUT_MODE_SAVED)
        _WINDOWS_INPUT_MODE_SAVED = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Keyboard / Mouse Input
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _read_key(fd: int, timeout: float = 0.1) -> str | MouseEvent | None:
    """Read a keypress or mouse event from *fd*."""
    if select is None:
        return None
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
        seq = ""
        while True:
            try:
                ready, _, _ = select.select([fd], [], [], 0.05)
            except (InterruptedError, OSError):
                return "escape" if not seq else None
            if not ready:
                return "escape" if not seq else None
            piece = os.read(fd, 1).decode("utf-8", errors="ignore")
            if not piece:
                return "escape" if not seq else None
            seq += piece
            if piece.isalpha() or piece in ("~", "m"):
                break

        if seq == "[A":
            return "up"
        if seq == "[B":
            return "down"
        if seq == "[5~":
            return "page_up"
        if seq == "[6~":
            return "page_down"
        if seq.startswith("[<") and seq[-1] in ("M", "m"):
            try:
                cb_s, cx_s, cy_s = seq[2:-1].split(";")
                cb = int(cb_s)
                cx = int(cx_s)
                cy = int(cy_s)
            except (TypeError, ValueError):
                return None
            x = max(0, cx - 1)
            y = max(0, cy - 1)
            if cb & 64:
                if cb & 1:
                    return MouseEvent(x=x, y=y, button="wheel_down")
                return MouseEvent(x=x, y=y, button="wheel_up")
            if seq[-1] == "m" or (cb & 32):
                return None
            button_code = cb & 3
            if button_code == 0:
                return MouseEvent(x=x, y=y, button="left")
            if button_code == 1:
                return MouseEvent(x=x, y=y, button="middle")
            if button_code == 2:
                return MouseEvent(x=x, y=y, button="right")
        return None

    return ch


def _decode_windows_mouse_event(event: _WinMouseEventRecord) -> MouseEvent | None:
    x = max(0, int(event.dwMousePosition.X))
    y = max(0, int(event.dwMousePosition.Y))
    if event.dwEventFlags == _WIN_MOUSE_WHEELED:
        delta = ctypes.c_short((event.dwButtonState >> 16) & 0xFFFF).value
        return MouseEvent(
            x=x,
            y=y,
            button="wheel_up" if delta > 0 else "wheel_down",
        )
    # Treat double-click as a second press; ignore other flagged mouse events.
    if event.dwEventFlags not in (0, _WIN_DOUBLE_CLICK):
        return None
    if event.dwButtonState & _WIN_FROM_LEFT_1ST_BUTTON_PRESSED:
        return MouseEvent(x=x, y=y, button="left")
    if event.dwButtonState & _WIN_RIGHTMOST_BUTTON_PRESSED:
        return MouseEvent(x=x, y=y, button="right")
    if event.dwButtonState & _WIN_FROM_LEFT_2ND_BUTTON_PRESSED:
        return MouseEvent(x=x, y=y, button="middle")
    return None


def _read_key_windows(timeout: float = 0.1) -> str | MouseEvent | None:
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.GetStdHandle(_WIN_STD_INPUT_HANDLE)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = _WinInputRecord()
        count = ctypes.c_uint()
        if not kernel32.PeekConsoleInputW(handle, ctypes.byref(record), 1, ctypes.byref(count)):
            time.sleep(0.01)
            continue
        if count.value == 0:
            time.sleep(0.01)
            continue
        if not kernel32.ReadConsoleInputW(handle, ctypes.byref(record), 1, ctypes.byref(count)):
            time.sleep(0.01)
            continue
        if record.EventType == _WIN_KEY_EVENT:
            event = record.Event.KeyEvent
            if not event.bKeyDown:
                continue
            if event.uChar == "\x03":
                raise KeyboardInterrupt
            vk = event.wVirtualKeyCode
            if vk == 0x26:
                return "up"
            if vk == 0x28:
                return "down"
            if vk == 0x21:
                return "page_up"
            if vk == 0x22:
                return "page_down"
            if vk == 0x1B:
                return "escape"
            if vk == 0x0D:
                return "\r"
            if event.uChar not in ("", "\x00"):
                return event.uChar
            continue
        if record.EventType == _WIN_MOUSE_EVENT:
            decoded = _decode_windows_mouse_event(record.Event.MouseEvent)
            if decoded is not None:
                return decoded
        time.sleep(0.01)
    return None


_KEY_BINDINGS: dict[str, str | None] = {
    "j": "navigate_down",
    "down": "navigate_down",
    "k": "navigate_up",
    "up": "navigate_up",
    " ": "toggle_expanded",
    "\r": "toggle_expanded",
    "g": "go_top",
    "G": "go_bottom",
    "page_up": "page_up",
    "page_down": "page_down",
    "n": "jump_next_agent",
    "N": "jump_prev_agent",
    "c": "collapse_enclosing_agent",
    "E": "expand_all_nodes",
    "C": "collapse_all_nodes",
    "q": None,
    "escape": None,
}


def _handle_key(key: str, renderer: ConsoleRender, *, allow_quit: bool) -> bool:
    """Process a key event, returning True if the browser should exit."""
    action = _KEY_BINDINGS.get(key)
    if action is None:
        return allow_quit and key in _KEY_BINDINGS
    getattr(renderer, action)()
    return False


def _handle_input_event(
    event: str | MouseEvent,
    renderer: ConsoleRender,
    *,
    allow_quit: bool,
) -> bool:
    if isinstance(event, MouseEvent):
        renderer.handle_click(event.x, event.y, button=event.button)
        return False
    return _handle_key(event, renderer, allow_quit=allow_quit)


def keyboard_loop(
    renderer: ConsoleRender,
    view_thread: threading.Thread,
) -> None:
    """Run the renderer-managed interactive session for an existing view loop."""
    renderer._console_session_active = True
    renderer.pre_console()
    try:
        if os.name == "nt":
            renderer._interactive_loop_windows(view_thread)
        else:
            renderer._interactive_loop_posix(view_thread)
    finally:
        renderer._console_session_active = False
        renderer.restore_console()


def interactive_browse(renderer: ConsoleRender, last_view: NodeView) -> None:
    """Post-completion interactive tree browser.

    Launches an interactive session where the user can navigate and
    expand/collapse the completed execution tree.  Exits on 'q' or Esc.
    """
    renderer.reset_for_browse()
    renderer._console_session_active = True
    renderer.pre_console()
    try:
        renderer._ui_driver(renderer.render(last_view))
        if os.name == "nt":
            renderer._browse_until_quit_windows()
        else:
            renderer._browse_until_quit_posix()
    finally:
        renderer._console_session_active = False
        renderer.restore_console()
