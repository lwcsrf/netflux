from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable

from ..core import Node, NodeState, NodeView, TerminalNodeStates
from ._contracts import SelectedTreeStatus, SessionController, TerminalSize
from ._controller_helpers import (
    compose_bottom_bar,
    render_too_small_frame,
    standalone_shortcut_variants,
    standalone_too_small,
)
from .console import ConsoleRender


_STANDALONE_ACTION_KEYS = {
    "j": "move_down",
    "down": "move_down",
    "k": "move_up",
    "up": "move_up",
    " ": "toggle",
    "\r": "toggle",
    "\n": "toggle",
    "g": "go_top",
    "G": "go_bottom",
    "page_up": "page_up",
    "page_down": "page_down",
    "n": "next_agent",
    "N": "prev_agent",
    "c": "collapse_agent",
    "E": "expand_all",
    "C": "collapse_all",
}


@dataclass(frozen=True)
class _ViewEvent:
    view: NodeView


class SingleTreeConsoleController(SessionController):
    def __init__(self, renderer: ConsoleRender, node: Node) -> None:
        self._renderer = renderer
        self._node = node
        self._renderer.set_cancel_event(node.cancel_event)
        self._interactive = True
        self._should_exit = False
        self._mode = "live"
        self._exit_after_terminal = False
        self._exit_after_render = False
        self._latest_view: NodeView | None = node.watch(as_of_seq=0, timeout=0)
        self._queue: queue.Queue[_ViewEvent] = queue.Queue()
        self._stop_event = threading.Event()
        self._wakeup: Callable[[], None] = lambda: None
        self._watch_thread = threading.Thread(
            target=self._watch_root,
            name=f"netflux-standalone-watch-{node.id}",
            daemon=True,
        )
        self._too_small = False

    def set_wakeup(self, wakeup: Callable[[], None]) -> None:
        self._wakeup = wakeup

    def _watch_root(self) -> None:
        prev_seq = 0
        while not self._stop_event.is_set():
            view = self._node.watch(as_of_seq=prev_seq)
            if view is None:
                continue
            prev_seq = view.update_seqnum
            self._queue.put(_ViewEvent(view=view))
            self._wakeup()
            if view.state in TerminalNodeStates:
                return

    def on_session_start(self, *, interactive: bool) -> None:
        self._interactive = interactive
        if self._watch_thread.ident is None:
            self._watch_thread.start()
        if self._latest_view is not None:
            self._renderer.assign_view(self._latest_view)
        self._apply_latest_terminal_state()

    def on_session_stop(self) -> None:
        self._stop_event.set()

    def pump_events(self) -> bool:
        changed = False
        while True:
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                break
            changed = True
            self._latest_view = event.view
            self._renderer.assign_view(event.view)
            self._apply_latest_terminal_state()
        return changed

    def _apply_latest_terminal_state(self) -> None:
        if (
            self._latest_view is None
            or self._latest_view.state not in TerminalNodeStates
            or self._mode != "live"
        ):
            return
        if not self._interactive or self._exit_after_terminal:
            self._exit_after_render = True
            return
        self._mode = "browse"
        self._renderer.reset_for_browse()

    def _status_without_position(self) -> SelectedTreeStatus:
        status = self._renderer.selected_tree_status()
        return SelectedTreeStatus(
            state=status.state,
            cancel_pending=status.cancel_pending,
            can_cancel=status.can_cancel,
            token_bill=status.token_bill,
        )

    @staticmethod
    def _mandatory_shortcuts(status: SelectedTreeStatus) -> list[str]:
        if status.state in TerminalNodeStates:
            return ["q:quit"]
        if status.can_cancel:
            return ["^C:cancel"]
        return []

    def _finalize_frame(self, frame: str) -> str:
        if self._exit_after_render:
            self._exit_after_render = False
            self._should_exit = True
        return frame

    def wants_animation_ticks(self) -> bool:
        view = self._latest_view
        return bool(
            view is None
            or (
                self._mode == "live"
                and view.state not in TerminalNodeStates
            )
        )

    def render_frame(self, size: TerminalSize, tick: int) -> str:
        self._too_small = standalone_too_small(size)

        self._apply_latest_terminal_state()

        if self._too_small:
            status = self._status_without_position()
            mandatory = self._mandatory_shortcuts(status)
            bottom_bar = compose_bottom_bar(
                size.columns,
                shortcut_variants=[
                    ["Resize terminal to continue"],
                    ["Resize to continue"],
                    ["Resize"],
                ],
                status=status,
                tick=tick,
                mandatory_shortcuts=mandatory,
            )
            return self._finalize_frame(render_too_small_frame(
                size,
                message="Terminal too small for ConsoleRender.",
                hint="Resize to at least 40x6.",
                bottom_bar=bottom_bar,
            ))

        body = self._renderer.render_body(
            width=size.columns,
            height=max(1, size.lines - 1),
            tick=tick,
        )
        status = self._renderer.selected_tree_status()
        ctx = self._renderer.right_pane_context()
        mandatory = self._mandatory_shortcuts(status)
        bottom_bar = compose_bottom_bar(
            size.columns,
            shortcut_variants=standalone_shortcut_variants(ctx),
            status=status,
            tick=tick,
            mandatory_shortcuts=mandatory,
        )
        return self._finalize_frame("\n".join([body, f"{bottom_bar}\x1b[K"]))

    def handle_key(self, key: str) -> bool:
        if key in ("q", "escape"):
            return self._mode == "browse"

        if self._too_small:
            return False

        action = _STANDALONE_ACTION_KEYS.get(key)
        if action is None:
            return False
        self._renderer.apply_action(action)
        return False

    def handle_mouse(self, event: object) -> bool:
        if self._too_small:
            return False
        button = getattr(event, "button", None)
        if button is None:
            return False
        self._renderer.handle_mouse_event(
            getattr(event, "x", 0),
            getattr(event, "y", 0),
            button=button,
        )
        return False

    def handle_interrupt(self) -> bool:
        if self._mode != "live":
            return True

        cancel_event = self._renderer.cancel_event()
        if cancel_event is not None and not cancel_event.is_set():
            cancel_event.set()
            self._exit_after_terminal = True
            return False
        raise KeyboardInterrupt

    def should_exit(self) -> bool:
        return self._should_exit
