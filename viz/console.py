import time
import sys
import shutil
import re
from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing.synchronize import Event
from typing import List, Optional, Tuple

from ..core import NodeState, NodeView, ThinkingBlockPart, ToolUsePart
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
    "orange": "\x1b[38;5;208m",
}

THOUGHT_GLYPH = "\u27B0"  # ➰
VERT_GLYPH = "\u2502"     # │


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


# Cute Unicode spinner frames (braille)
_SPINNER_FRAMES = [
    "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏",
]


def _short_repr(value, max_len: int = 40) -> str:
    try:
        s = repr(value)
    except Exception:
        s = str(value)
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def _format_args(inputs: dict, max_len: int = 800, per_val_len: int = 120) -> str:
    if not inputs:
        return ""
    # Render values first to sort by their displayed length (ascending).
    rendered: List[tuple[str, str]] = []
    for k, v in inputs.items():
        rendered_val = _short_repr(v, per_val_len)
        rendered.append((k, rendered_val))
    rendered.sort(key=lambda kv: len(kv[1]))  # stable sort by value length

    items = [f"{k}={val}" for k, val in rendered]
    s = ", ".join(items)
    if len(s) > max_len:
        s = s[: max_len - 3] + "..."
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


def _type_glyph_for_fn(fn) -> str:
    if fn.is_agent():
        return "✨"
    if fn.is_code():
        return "⚙️ "
    return "•"


def _format_elapsed(nv: NodeView) -> Optional[str]:
    """Return a colored elapsed time in square brackets.

    - Running: red [Xs]
    - Terminal: green [Xs]
    - Waiting/no start: None

    Keeps string short (1 decimal). For durations <1s, hide it entirely to
    avoid visual noise. The ANSI wrapper is self-contained so cropping logic
    that trims trailing partial CSI remains safe.
    """
    start = nv.started_at
    if start is None:
        return None
    end: Optional[float] = None
    if nv.state in (NodeState.Success, NodeState.Error, NodeState.Canceled):
        end = nv.ended_at
    tnow = time.time()
    elapsed = max(0.0, (end if end is not None else tnow) - start)
    # Hide sub-second durations entirely
    if elapsed < 1.0:
        return None
    s = f"{elapsed:.1f}s"
    # color by state (running=red, done=green)
    if nv.state is NodeState.Running:
        body = _color(s, fg="red")
    else:
        body = _color(s, fg="green")
    return f" [{body}]"


@dataclass
class ConsoleRender(Render[str]):
    """Renderer for a `NodeView` tree with lightweight animation that yields
    an ANSI string intended for display in a terminal.

    - Maintains the most recent `NodeView` it has seen; when called with
      `render(None)`, it re-renders the last view using a time-based spinner.
    - Uses simple ANSI colors and symbols suitable for terminal UIs.
    - The output is a full frame (no cursor control). The caller typically
      clears the screen in their `ui_driver` before writing the frame.

    Intended usage for a terminal UI:
    - Call `ConsoleRender.pre_console()` before starting the view loop.
    - Start the loop with `ui_driver=ConsoleRender.ui_driver`.
    - On exit (e.g., in `finally:`), call `ConsoleRender.restore_console()`.
    """

    width: Optional[int] = None
    spinner_hz: float = 10.0
    cancel_event: Optional[Event] = None

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
        cancel_pending = bool(self.cancel_event and self.cancel_event.is_set())

        def add_node(nv: NodeView, prefix: str, is_last: bool) -> None:
            def format_thinking(part: ThinkingBlockPart) -> Optional[str]:
                content = part.content or ""
                signature = part.signature or ""

                if part.redacted or content == "":
                    content_display = ""
                else:
                    content_display = _short_repr(content, 200)
                sig_prefix = signature[:5]

                if content_display:
                    return f"{THOUGHT_GLYPH} thought: {content_display}"
                return f"{THOUGHT_GLYPH} thought"

            # Precompute thinking slots relative to tool uses for this node.
            thinking_slots = {}
            if nv.transcript:
                tool_uses_seen = 0
                for part in nv.transcript:
                    if isinstance(part, ThinkingBlockPart):
                        thinking_slots.setdefault(tool_uses_seen, []).append(part)
                    elif isinstance(part, ToolUsePart):
                        tool_uses_seen += 1

            def emit_thinking(slot_index: int) -> None:
                if not thinking_slots:
                    return
                parts = thinking_slots.get(slot_index)
                if not parts:
                    return
                detail_prefix = prefix + ("   " if is_last else "│  ")
                for tb in parts:
                    msg = format_thinking(tb)
                    if msg:
                        lines.append(detail_prefix + f"{VERT_GLYPH}    " + _color(msg, dim=True))

            glyph, color = _state_glyph(nv.state, tick)
            # If cancellation is pending, visually mark non-terminal states distinctly
            # to indicate "cancel requested" overlay without implying terminal state.
            if cancel_pending and nv.state in (NodeState.Waiting, NodeState.Running):
                color = "magenta"
            args = _format_args(nv.inputs)
            type_g = _type_glyph_for_fn(nv.fn)
            # Base prefix + function name
            header = f"{_color(glyph, fg=color, bold=True)} {_color(type_g)} {_color(nv.fn.name, bold=True)}"
            # Elapsed time immediately after function name, to reduce truncation risk
            et = _format_elapsed(nv)
            if et:
                header += et
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

            # Unicode box-drawing tree connectors
            branch = ("└─ " if is_last else "├─ ")
            lines.append(prefix + branch + header if prefix else header)

            # Print TokenUsage directly under the node header (agents only),
            # preserving tree rails but without a branch marker.
            if nv.usage:
                u = nv.usage
                in_fields = []
                in_fields.append(f"cache_read={u.input_tokens_cache_read}")
                if u.input_tokens_cache_write is not None:
                    in_fields.append(f"cache_write={u.input_tokens_cache_write}")
                in_fields.append(f"regular={u.input_tokens_regular}")
                in_fields.append(f"total={u.input_tokens_total}")

                out_fields = []
                if u.output_tokens_reasoning is not None:
                    out_fields.append(f"reasoning={u.output_tokens_reasoning}")
                if u.output_tokens_text is not None:
                    out_fields.append(f"text={u.output_tokens_text}")
                out_fields.append(f"total={u.output_tokens_total}")

                segs = []
                if in_fields:
                    segs.append(_color(f"In: {{{', '.join(in_fields)}}}", fg="cyan", bold=True))
                if out_fields:
                    segs.append(_color(f"Out: {{{', '.join(out_fields)}}}", fg="magenta", bold=True))

                ctx_total = u.context_window_in + u.context_window_out
                if ctx_total:
                    ctx_fields = [
                        f"in={u.context_window_in}",
                        f"out={u.context_window_out}",
                        f"total={ctx_total}",
                    ]
                    segs.append(_color(f"Ctx: {{{', '.join(ctx_fields)}}}", fg="orange", bold=True))

                if segs:
                    detail_prefix = prefix + ("   " if is_last else "│  ")
                    lines.append(detail_prefix + f"{VERT_GLYPH}    " + ", ".join(segs))

            # Thinking blocks associated with this node appear as their own lines
            # under the node header (and any usage line), aligned with tree rails.
            emit_thinking(0)

            child_prefix = prefix + ("   " if is_last else "│  ")
            count = len(nv.children)
            for idx, child in enumerate(nv.children):
                has_trailing_thought = bool(thinking_slots.get(idx + 1))
                is_last_child = (idx == count - 1) and not has_trailing_thought
                add_node(child, child_prefix, is_last_child)
                emit_thinking(idx + 1)

        add_node(self._last_view, prefix="", is_last=True)

        # Footer: show cancellation overlay only while tree is non-terminal (root state).
        root_state = self._last_view.state
        if cancel_pending and root_state not in (NodeState.Success, NodeState.Error, NodeState.Canceled):
            lines.append("")
            lines.append(_color("Cancelation pending ...", fg="magenta", bold=True))

        return "\n".join(lines)

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
        """Simple console UI driver: clear and write cropped frame.

        - Clears scrollback and screen and homes cursor before writing.
        - Crops to a safe viewport (rows-1, cols-1) to avoid scroll/wrap.
        - Pads with blank lines to overwrite remnants from previous frames.
        """
        # Clear scrollback + screen, then home
        sys.stdout.write("\x1b[3J\x1b[2J\x1b[H")

        # Compute safe viewport and crop to avoid implicit scroll/wrap
        sz = shutil.get_terminal_size(fallback=(80, 24))
        rows = max(1, sz.lines)
        cols = max(1, sz.columns)
        safe_rows = max(1, rows - 1)
        safe_cols = max(1, cols - 1)

        def _crop_tail_safe(line: str, max_cols: int, tail_allow: int = 18) -> str:
            """Crop allowing a small tail to avoid cutting ANSI codes mid-seq.

            Heuristic: overslice by `tail_allow`, then strip any trailing
            partial CSI (e.g., ESC[ ... with no final byte). Always append
            RESET to avoid style bleed if we did cut styling.
            """
            if len(line) <= max_cols:
                return line
            end = min(len(line), max_cols + max(0, tail_allow))
            chunk = line[:end]
            # Strip trailing partial CSI sequences like '\x1b[31;1' (no final letter)
            chunk = re.sub(r"\x1b\[[0-9;?]*$", "", chunk)
            # Also strip a bare trailing ESC, if any
            if chunk.endswith("\x1b"):
                chunk = chunk[:-1]
            # Add RESET to ensure attributes are closed
            return chunk + RESET

        lines = s.splitlines()
        # Take the last safe_rows lines to "tail" the output (show bottom when full)
        visible_lines = lines[-safe_rows:] if len(lines) > safe_rows else lines
        cropped = [_crop_tail_safe(ln, safe_cols) for ln in visible_lines]
        # Pad with blanks to overwrite remnants from previous longer frames
        if len(cropped) < safe_rows:
            cropped.extend([""] * (safe_rows - len(cropped)))
        payload = "\n".join(cropped)
        sys.stdout.write(payload)
        sys.stdout.flush()
