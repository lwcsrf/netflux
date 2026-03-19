from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol

from ..core import NodeState, TokenBill
from ..providers import Provider


@dataclass(frozen=True)
class TerminalSize:
    columns: int
    lines: int


@dataclass(frozen=True)
class SelectedTreeStatus:
    cursor_line: int = 0
    total_lines: int = 0
    state: NodeState | None = None
    cancel_pending: bool = False
    can_cancel: bool = False
    token_bill: Mapping[Provider, TokenBill] = field(default_factory=dict)


@dataclass(frozen=True)
class RightPaneInteractionContext:
    has_lines: bool = False
    can_expand_collapse: bool = False
    can_jump_agents: bool = False
    follow_mode: bool = False
    is_terminal: bool = False
    can_copy_root_result: bool = False
    can_focus_root_result: bool = False


class SessionController(Protocol):
    def set_wakeup(self, wakeup: Callable[[], None]) -> None:
        ...

    def on_session_start(self, *, interactive: bool) -> None:
        ...

    def on_session_stop(self) -> None:
        ...

    def pump_events(self) -> bool:
        ...

    def wants_animation_ticks(self) -> bool:
        ...

    def render_frame(self, size: TerminalSize, tick: int) -> str:
        ...

    def handle_key(self, key: str) -> bool:
        ...

    def handle_mouse(self, event: object) -> bool:
        ...

    def handle_interrupt(self) -> bool:
        ...

    def should_exit(self) -> bool:
        ...
