from __future__ import annotations

from ._contracts import TerminalSize
from .console import _color, _crop_line, _highlight_line, _visible_len


class CancelConfirmation:
    """Shared modal selection; only a navigation key followed by Enter confirms."""

    def __init__(self, *, title: str) -> None:
        self._title = title
        self._confirm_selected = False
        self._too_small = False

    def handle_key(self, key: str) -> bool | None:
        """Return True to confirm, False to dismiss, or None to stay open."""
        if key == "escape":
            return False
        if key in ("\r", "\n"):
            return self._confirm_selected and not self._too_small
        if not self._too_small:
            if key in ("down", "j"):
                self._confirm_selected = True
            elif key in ("up", "k"):
                self._confirm_selected = False
        return None

    def render_frame(self, size: TerminalSize) -> str:
        columns, rows = max(1, size.columns), max(1, size.lines)
        self._too_small = columns < 24 or rows < 8
        if self._too_small:
            # Never allow an invisible selection to authorize cancellation.
            self._confirm_selected = False
            lines = [
                "Resize to confirm quit.",
                "Need at least 24x8.",
                "Esc/Enter: back",
                "Ctrl-C ignored here.",
            ]
        else:
            lines = [
                _color(self._title, fg="yellow", bold=True),
                "",
                self._option("Cancel", selected=not self._confirm_selected),
                self._option("Confirm", selected=self._confirm_selected),
                "",
                "Up/Down or j/k: select",
                "Enter: choose  Esc: back",
                "Ctrl-C ignored here.",
            ]
            if rows >= 10:
                lines[1:1] = ["Work continues until you confirm.", ""]

        width = min(columns, max(_visible_len(line) for line in lines))
        margin = " " * max(0, (columns - width) // 2)
        output = [""] * max(0, (rows - len(lines)) // 2)
        output.extend(margin + _crop_line(line, width) for line in lines)
        output.extend([""] * max(0, rows - len(output)))
        return "\n".join(f"{line}\x1b[K" for line in output[:rows])

    @staticmethod
    def _option(label: str, *, selected: bool) -> str:
        text = f"{'>' if selected else ' '} {label}"
        return _highlight_line(text, len(text)) if selected else text
