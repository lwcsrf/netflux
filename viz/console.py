import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..core import NodeState, NodeView
from .viz import Render

# ANSI helpers for console rendering
RESET = "\x1b[0m"
BOLD = "\x1b[1m"
DIM = "\x1b[2m"
FG = {
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "blue": "\x1b[34m",
    "magenta": "\x1b[35m",
    "cyan": "\x1b[36m",
    "white": "\x1b[37m",
}


def _color(text: str, *, fg: Optional[str] = None, bold: bool = False, dim: bool = False) -> str:
    parts: List[str] = []
    if bold:
        parts.append(BOLD)
    if dim:
        parts.append(DIM)
    if fg:
        parts.append(FG.get(fg, ""))
    parts.append(text)
    parts.append(RESET)
    return "".join(parts)


_SPINNER_FRAMES = [
    "⠋",
    "⠙",
    "⠹",
    "⠸",
    "⠼",
    "⠴",
    "⠦",
    "⠧",
    "⠇",
    "⠏",
]


def _short_repr(value, max_len: int = 40) -> str:
    try:
        s = repr(value)
    except Exception:
        s = str(value)
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


def _format_args(inputs: dict, max_len: int = 60) -> str:
    if not inputs:
        return ""
    items = []
    for k, v in inputs.items():
        items.append(f"{k}={_short_repr(v, 20)}")
    s = ", ".join(items)
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s


def _state_glyph(state: NodeState, tick: int) -> Tuple[str, str]:
    """Return (glyph, color) for the given state; glyph may animate with tick."""
    if state is NodeState.Waiting:
        return ("…", "yellow")
    if state is NodeState.Running:
        frame = _SPINNER_FRAMES[tick % len(_SPINNER_FRAMES)]
        return (frame, "cyan")
    if state is NodeState.Success:
        return ("✔", "green")
    if state is NodeState.Error:
        return ("✖", "red")
    if state is NodeState.Canceled:
        return ("⏹", "yellow")
    return ("?", "white")


@dataclass
class ConsoleRender(Render[str]):
    """Renderer for a `NodeView` tree with lightweight animation that yields
    an ANSI string intended for display in a terminal.

    - Maintains the most recent `NodeView` it has seen; when called with
      `render(None)`, it re-renders the last view using a time-based spinner.
    - Uses simple ANSI colors and symbols suitable for terminal UIs.
    - The output is a full frame (no cursor control). The caller typically
      clears the screen in their `ui_driver` before writing the frame.
    """

    width: Optional[int] = None
    spinner_hz: float = 10.0

    def __post_init__(self) -> None:
        self._last_view: Optional[NodeView] = None
        self._t0 = time.monotonic()

    def _tick(self) -> int:
        # time-based tick independent from update interval
        dt = time.monotonic() - self._t0
        return int(dt * max(self.spinner_hz, 1.0))

    def render(self, view: Optional[NodeView]) -> str:
        if view is not None:
            self._last_view = view
        if self._last_view is None:
            return "(no data)"

        tick = self._tick()
        lines: List[str] = []

        def add_node(nv: NodeView, prefix: str, is_last: bool) -> None:
            glyph, color = _state_glyph(nv.state, tick)
            args = _format_args(nv.inputs)
            header = f"{_color(glyph, fg=color, bold=True)} {_color(nv.fn.name, bold=True)}"
            if args:
                header += f"({_color(args, dim=True)})"

            # attach short result/exception to header to keep compact
            if nv.state is NodeState.Success and nv.outputs is not None:
                header += f" {_color('=>', dim=True)} {_short_repr(nv.outputs, 50)}"
            elif nv.state is NodeState.Error and nv.exception is not None:
                try:
                    msg = str(nv.exception)
                except Exception:
                    msg = nv.exception.__class__.__name__
                header += f" {_color('!!', fg='red', bold=True)} {_short_repr(msg, 50)}"
            elif nv.state is NodeState.Canceled and nv.exception is not None:
                try:
                    msg = str(nv.exception)
                except Exception:
                    msg = nv.exception.__class__.__name__
                header += f" {_color('CANCEL', fg='yellow', bold=True)} {_short_repr(msg, 50)}"

            branch = "└─ " if is_last else "├─ "
            lines.append(prefix + branch + header if prefix else header)

            child_prefix = prefix + ("   " if is_last else "│  ")
            count = len(nv.children)
            for idx, child in enumerate(nv.children):
                add_node(child, child_prefix, idx == count - 1)

        add_node(self._last_view, prefix="", is_last=True)
        return "\n".join(lines)
