import sys
import threading
import time
import queue
from typing import Callable, Optional, Tuple, List

from ..core import Node, NodeView, NodeState

# ANSI Helpers.
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
        return ("·", "yellow")
    if state is NodeState.Running:
        frame = _SPINNER_FRAMES[tick % len(_SPINNER_FRAMES)]
        return (frame, "cyan")
    if state is NodeState.Success:
        return ("✔", "green")
    if state is NodeState.Error:
        return ("✖", "red")
    return ("?", "white")

def render_tree_str(view: NodeView, tick: int = 0, *, width: Optional[int] = None) -> str:
    """Render the NodeView tree into a full ANSI string (no cursor control)."""

    lines: List[str] = []

    def add_node(nv: NodeView, prefix: str, is_last: bool) -> None:
        glyph, color = _state_glyph(nv.state, tick)
        args = _format_args(nv.inputs)
        header = f"{_color(glyph, fg=color, bold=True)} {_color(nv.fn.name, bold=True)}"
        if args:
            header += f"({_color(args, dim=True)})"

        # attach short result/exception to header to keep compact
        if nv.state is NodeState.Success and nv.outputs is not None:
            header += f" {_color('→', dim=True)} {_short_repr(nv.outputs, 50)}"
        elif nv.state is NodeState.Error and nv.exception is not None:
            try:
                msg = str(nv.exception)
            except Exception:
                msg = nv.exception.__class__.__name__
            header += f" {_color('!!', fg='red', bold=True)} {_short_repr(msg, 50)}"

        branch = "└─ " if is_last else "├─ "
        lines.append(prefix + branch + header if prefix else header)

        child_prefix = prefix + ("   " if is_last else "│  ")
        count = len(nv.children)
        for idx, child in enumerate(nv.children):
            add_node(child, child_prefix, idx == count - 1)

    add_node(view, prefix="", is_last=True)
    return "\n".join(lines)

def start_tree_str_view(
    stop_event: threading.Event,
    node: Node,
    render_callback: Callable[[NodeView, int], str] = render_tree_str,
    writer_callback: Callable[[str], None] = lambda s: (sys.stdout.write(s + "\n"), sys.stdout.flush()),
    *,
    hz: float = 10.0,
) -> tuple[threading.Thread, threading.Thread]:
    """Start watcher and ticker threads for string-based tree visualization.

    - stop_event: when set, both threads exit promptly.
    - node: the Node to watch.
    - render_callback(view, tick) -> str: returns the full string to display (may include ANSI colors).
    - writer_callback(text): emits the string to terminal (caller decides clearing/overwriting strategy).
    - hz: target frames per second for the ticker.

    Returns (watcher_thread, ticker_thread). Threads are started as daemons.
    """

    q: queue.Queue[NodeView] = queue.Queue(maxsize=1)

    def _watcher() -> None:
        prev = 0
        try:
            while not stop_event.is_set():
                view = node.watch(as_of_seq=prev)
                prev = view.update_seqnum
                # replace any stale snapshot; latest wins
                try:
                    q.put_nowait(view)
                except queue.Full:
                    try:
                        _ = q.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        q.put_nowait(view)
                    except queue.Full:
                        pass
                if view.state in (NodeState.Success, NodeState.Error):
                    # allow renderer to continue until stop_event; watcher can stop
                    break
        except Exception:
            # In demos, swallow to avoid noisy tracebacks on shutdown races
            pass

    def _ticker() -> None:
        period = 1.0 / hz if hz > 0 else 0.1
        next_t = time.monotonic()
        tick = 0
        last_view: Optional[NodeView] = None
        last_str: Optional[str] = None
        try:
            while not stop_event.is_set():
                timeout = max(0.0, next_t - time.monotonic())
                try:
                    last_view = q.get(timeout=timeout)
                except queue.Empty:
                    pass

                if last_view is not None:
                    try:
                        s = render_callback(last_view, tick)
                    except Exception:
                        # Don't kill the loop if the renderer fails once
                        s = "(render error)"
                    if s != last_str:
                        try:
                            writer_callback(s)
                        except Exception:
                            pass
                        last_str = s

                tick += 1
                next_t += period
        except Exception:
            pass

    watcher = threading.Thread(target=_watcher, name="tree-str-watcher", daemon=True)
    ticker = threading.Thread(target=_ticker, name="tree-str-ticker", daemon=True)
    watcher.start()
    ticker.start()
    return watcher, ticker
