from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import re
import threading
import textwrap
from dataclasses import dataclass, field
from multiprocessing.synchronize import Event as MpEvent
from typing import Callable, Mapping

from ..core import AgentFunction, Function, FunctionArg, Node, NodeState, NodeView, TerminalNodeStates, TokenBill
from ..providers import Provider
from ..runtime import Runtime
from ._contracts import RightPaneInteractionContext, SelectedTreeStatus, SessionController, TerminalSize
from ._controller_helpers import (
    compose_bottom_bar,
    multi_pane_shortcut_variants,
    multi_pane_too_small,
    preferred_left_pane_width,
    render_too_small_frame,
)
from .console import (
    FG,
    RESET,
    ConsoleRender,
    _color,
    _crop_line,
    _format_args,
    _highlight_line,
    _state_glyph,
    _visible_len,
)
from ._driver import ConsoleSessionDriver
from ._terminal_io import restore_console


_MULTI_ACTION_KEYS = {
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
    "e": "expand_all",
    "r": "collapse_all",
}


_PANE_SEPARATOR = f"{FG['white']}║{RESET}"
_PANE_SEPARATOR_COLS = 1
_RIGHT_PANE_MARGIN_COLS = 2

TokenBills = Mapping[Provider, TokenBill]
TerminalCallback = Callable[[TokenBills], None]


@dataclass
class _RunRecord:
    name: str
    fn: Function
    node: Node
    renderer: ConsoleRender
    cancel_event: MpEvent
    latest_view: NodeView | None = None
    terminal_callback_invoked: bool = False
    terminal_browse_applied: bool = False
    terminal_browse_pending: bool = False
    manual_unread: bool = False
    auto_unread: bool = False
    watcher_stop: threading.Event = field(default_factory=threading.Event)
    watcher_thread: threading.Thread | None = None

    @property
    def unread(self) -> bool:
        return self.manual_unread or self.auto_unread


@dataclass(frozen=True)
class _RunUpdateEvent:
    run_index: int
    view: NodeView


@dataclass(frozen=True)
class _RunPaneEntry:
    kind: str
    run_index: int | None = None
    label: str = ""


@dataclass
class _LaunchField:
    label: str
    value: str = ""
    arg: FunctionArg | None = None
    kind: str = "arg"

    @property
    def is_name(self) -> bool:
        return self.kind == "name"

    @property
    def is_provider(self) -> bool:
        return self.kind == "provider"

    @property
    def is_provider_options(self) -> bool:
        return self.kind == "provider_options"

    @property
    def type_label(self) -> str:
        if self.is_name:
            return "str"
        if self.is_provider:
            return "Provider"
        if self.is_provider_options:
            return ""
        assert self.arg is not None
        return self.arg.argtype.__name__

    @property
    def optional(self) -> bool:
        if self.is_provider:
            return False
        if self.is_provider_options:
            return False
        return bool(self.arg and self.arg.optional)


@dataclass
class _LaunchFormState:
    fn_index: int
    fields: list[_LaunchField]
    cursor: int = 0
    error: str = ""


@dataclass(frozen=True)
class _LaunchHistoryEntry:
    name: str
    inputs: Mapping[str, object]
    provider: Provider | None = None


@dataclass(frozen=True)
class _LaunchFormLayout:
    header_lines: tuple[str, ...]
    item_start: int
    item_end: int
    header_rows: int
    header_clipped: bool = False


class TUI(SessionController):
    _HISTORY_NAME_SUFFIX_RE = re.compile(r"^(.*)\s\((\d+)\)$")

    def __init__(
        self,
        runtime: Runtime,
        *,
        spinner_hz: float = 10.0,
    ) -> None:
        self.runtime = runtime
        self.spinner_hz = max(1.0, float(spinner_hz))
        self.invocable_functions = runtime.invocable_functions
        if len(self.invocable_functions) > 10:
            raise ValueError("TUI supports at most 10 invocable functions in v1.")

        self._interactive = True
        self._should_exit = False
        self._global_cancel_requested = False
        self._selected_run: int | None = None
        self._visible_run: int | None = None
        self._runs: list[_RunRecord] = []
        self._run_scroll = 0
        self._event_queue: queue.Queue[_RunUpdateEvent] = queue.Queue()
        self._last_size = TerminalSize(columns=80, lines=24)
        self._too_small = False
        self._form_state: _LaunchFormState | None = None
        self._exit_after_render = False
        self._terminal_callback: TerminalCallback | None = None
        self._wakeup: Callable[[], None] = lambda: None

    def run(self) -> None:
        ConsoleSessionDriver(spinner_hz=self.spinner_hz).run(self)

    def register_terminal_callback(
        self,
        callback: TerminalCallback | None,
    ) -> None:
        """Register or clear the callback used for terminal top-level runs."""
        if callback is not None and not callable(callback):
            raise TypeError("callback must be callable")
        self._terminal_callback = callback
        if callback is None:
            return
        for run in self._runs:
            self._invoke_terminal_callback_if_needed(run)

    def set_wakeup(self, wakeup: Callable[[], None]) -> None:
        self._wakeup = wakeup

    def on_session_start(self, *, interactive: bool) -> None:
        self._interactive = interactive
        if not interactive:
            self._should_exit = True

    def on_session_stop(self) -> None:
        for run in self._runs:
            run.watcher_stop.set()
        self._attempt_terminal_callbacks_for_all_runs(refresh_from_runtime=True)

    def pump_events(self) -> bool:
        changed = False
        while True:
            try:
                event = self._event_queue.get_nowait()
            except queue.Empty:
                break
            changed = True
            run = self._runs[event.run_index]
            prev_view = run.latest_view
            run.latest_view = event.view
            if self._selected_run is None:
                self._set_selected_run(event.run_index)

            transitioned_to_terminal = (
                event.view.state in TerminalNodeStates
                and (prev_view is None or prev_view.state not in TerminalNodeStates)
            )
            if not transitioned_to_terminal:
                continue

            if self._current_visible_run() == event.run_index:
                if not run.terminal_browse_applied:
                    run.terminal_browse_pending = True
            else:
                run.auto_unread = True
            self._invoke_terminal_callback_if_needed(run, event.view)
        self._request_exit_after_graceful_cancel_if_complete()
        return changed

    def wants_animation_ticks(self) -> bool:
        if self._form_state is not None:
            return any(run.latest_view and run.latest_view.state not in TerminalNodeStates for run in self._runs)
        return any(
            run.latest_view is None or run.latest_view.state not in TerminalNodeStates
            for run in self._runs
        )

    def render_frame(self, size: TerminalSize, tick: int) -> str:
        self._last_size = size
        self._too_small = multi_pane_too_small(size)
        self._sync_visible_run()
        if self._form_state is not None:
            self._clear_selected_terminal_browse_pending()
            frame = self._render_launch_form(size, tick)
            return self._finalize_frame(frame)
        if self._too_small:
            self._clear_selected_terminal_browse_pending()
            status = self._selected_status(include_position=False)
            interrupt_hint = self._global_interrupt_hint()
            bottom_bar = compose_bottom_bar(
                size.columns,
                shortcut_variants=[
                    ["Resize terminal to continue"],
                    ["Resize to continue"],
                    [],
                ],
                status=status,
                tick=tick,
                mandatory_shortcuts=[interrupt_hint],
            )
            frame = render_too_small_frame(
                size,
                message="Terminal too small for TUI.",
                hint="Resize to at least 67x15.",
                bottom_bar=bottom_bar,
            )
            return self._finalize_frame(frame)

        left_width = preferred_left_pane_width(size.columns)
        right_width = self._right_pane_width(size.columns, left_width)
        body_rows = max(1, size.lines - 1)
        top_rows, bottom_rows = self._left_pane_heights(body_rows)

        left_top = self._render_runs_pane(left_width, top_rows, tick)
        left_bottom = self._render_functions_pane(left_width, bottom_rows)
        left_rows = left_top + left_bottom
        while len(left_rows) < body_rows:
            left_rows.append(" " * left_width)

        self._apply_selected_terminal_browse_if_visible()
        self._sync_selected_renderer_view()
        right_rows = self._render_right_pane(right_width, body_rows, tick)
        while len(right_rows) < body_rows:
            right_rows.append(" " * right_width)

        frame_rows: list[str] = []
        right_margin = " " * _RIGHT_PANE_MARGIN_COLS
        for idx in range(body_rows):
            left_line = self._pad_visible(left_rows[idx], left_width)
            right_line = self._pad_visible(right_rows[idx], right_width)
            frame_rows.append(f"{left_line}{_PANE_SEPARATOR}{right_margin}{right_line}")

        selected_renderer = self._selected_renderer()
        status = (
            selected_renderer.selected_tree_status()
            if selected_renderer is not None
            else SelectedTreeStatus()
        )
        ctx = selected_renderer.right_pane_context() if selected_renderer is not None else RightPaneInteractionContext()
        bottom_bar = compose_bottom_bar(
            size.columns,
            shortcut_variants=multi_pane_shortcut_variants(
                ctx,
                can_cancel=status.can_cancel,
                can_mark_unread=selected_renderer is not None,
                interrupt_hint=self._global_interrupt_hint(),
            ),
            status=status,
            tick=tick,
        )
        frame_rows.append(f"{bottom_bar}\x1b[K")
        return self._finalize_frame("\n".join(frame_rows))

    def handle_key(self, key: str) -> bool:
        if self._form_state is not None:
            if self._too_small:
                return False
            return self._handle_form_key(key)

        if self._too_small:
            return False

        if key == "\t":
            self._select_next_run()
            return False
        if key == "shift_tab":
            self._select_prev_run()
            return False
        if key == "C":
            self._cancel_selected_run()
            return False
        if key == "u":
            self._toggle_selected_run_unread()
            return False
        if key.isdigit():
            idx = int(key)
            if idx < len(self.invocable_functions):
                self._open_launch_form(idx)
            return False

        action = _MULTI_ACTION_KEYS.get(key)
        renderer = self._selected_renderer()
        if action is not None and renderer is not None:
            renderer.apply_action(action)
        return False

    def handle_mouse(self, event: object) -> bool:
        if self._form_state is not None:
            if self._too_small:
                return False
            return self._handle_form_mouse(event)

        if self._too_small:
            return False

        x = int(getattr(event, "x", 0))
        y = int(getattr(event, "y", 0))
        button = getattr(event, "button", "")

        body_rows = max(1, self._last_size.lines - 1)
        if y >= body_rows:
            return False

        left_width = preferred_left_pane_width(self._last_size.columns)
        right_origin = self._right_pane_origin(left_width)
        if x < left_width:
            if button != "left":
                return False
            top_rows, bottom_rows = self._left_pane_heights(body_rows)
            if y < top_rows:
                self._select_run_row(y)
            else:
                self._launch_function_hit(x, y - top_rows, left_width, bottom_rows)
            return False

        if x < right_origin:
            return False

        renderer = self._selected_renderer()
        if renderer is None:
            return False
        renderer.handle_mouse_event(x - right_origin, y, button=button)
        return False

    def handle_interrupt(self) -> bool:
        if self._global_cancel_requested:
            self._attempt_terminal_callbacks_for_all_runs(refresh_from_runtime=True)
            raise KeyboardInterrupt

        for run in self._runs:
            run.cancel_event.set()
        self._global_cancel_requested = True
        self._attempt_terminal_callbacks_for_all_runs(refresh_from_runtime=True)
        self._request_exit_after_graceful_cancel_if_complete()
        return False

    def should_exit(self) -> bool:
        return self._should_exit

    def _open_launch_form(self, fn_index: int) -> None:
        fn = self.invocable_functions[fn_index]
        fields = [_LaunchField(label="run_name", kind="name")]
        if isinstance(fn, AgentFunction):
            fields.append(_LaunchField(label="provider", value=fn.default_model.value, kind="provider"))
            fields.append(_LaunchField(label="provider_options", kind="provider_options"))
        fields.extend(_LaunchField(label=arg.name, arg=arg) for arg in fn.args)
        self._form_state = _LaunchFormState(fn_index=fn_index, fields=fields)
        self._sync_visible_run()

    def _launch_form_history(self) -> list[_LaunchHistoryEntry]:
        assert self._form_state is not None
        fn = self.invocable_functions[self._form_state.fn_index]
        history: list[_LaunchHistoryEntry] = []
        for run in reversed(self._runs):
            if run.fn is not fn:
                continue
            inputs: Mapping[str, object] = {}
            if run.latest_view is not None:
                inputs = run.latest_view.inputs
            else:
                raw_inputs = getattr(run.node, "inputs", {})
                if isinstance(raw_inputs, Mapping):
                    inputs = raw_inputs
            provider: Provider | None = None
            if isinstance(fn, AgentFunction):
                if run.latest_view is not None:
                    provider = run.latest_view.provider
                if provider is None:
                    raw_provider = getattr(run.node, "provider", None)
                    if isinstance(raw_provider, Provider):
                        provider = raw_provider
            history.append(_LaunchHistoryEntry(name=run.name, inputs=dict(inputs), provider=provider))
            if len(history) >= 20:
                break
        return history

    def _launch_form_item_count(self) -> int:
        assert self._form_state is not None
        return len(self._form_state.fields) + 2 + len(self._launch_form_history())

    def _launch_form_submit_index(self) -> int:
        assert self._form_state is not None
        return len(self._form_state.fields)

    def _launch_form_cancel_index(self) -> int:
        return self._launch_form_submit_index() + 1

    def _launch_form_history_start_index(self) -> int:
        return self._launch_form_cancel_index() + 1

    def _clamp_form_cursor(self) -> None:
        assert self._form_state is not None
        focusable = self._focusable_form_indices()
        if not focusable:
            self._form_state.cursor = 0
            return
        max_cursor = focusable[-1]
        self._form_state.cursor = max(0, min(self._form_state.cursor, max_cursor))
        if self._form_state.cursor not in focusable:
            eligible = [index for index in focusable if index <= self._form_state.cursor]
            self._form_state.cursor = eligible[-1] if eligible else focusable[0]

    def _first_editable_field_cursor(self) -> int:
        assert self._form_state is not None
        if len(self._form_state.fields) > 1:
            return 1
        return 0

    def _focusable_form_indices(self) -> list[int]:
        assert self._form_state is not None
        indices = [
            idx for idx, field in enumerate(self._form_state.fields) if not field.is_provider_options
        ]
        indices.append(self._launch_form_submit_index())
        indices.append(self._launch_form_cancel_index())
        indices.extend(range(self._launch_form_history_start_index(), self._launch_form_item_count()))
        return indices

    def _move_form_cursor(self, step: int) -> None:
        assert self._form_state is not None
        focusable = self._focusable_form_indices()
        if not focusable:
            self._form_state.cursor = 0
            return
        current = self._form_state.cursor
        if current not in focusable:
            self._form_state.cursor = focusable[0]
            return
        position = focusable.index(current)
        position = max(0, min(position + step, len(focusable) - 1))
        self._form_state.cursor = focusable[position]

    @staticmethod
    def _launch_form_field_value(arg: FunctionArg, value: object) -> str:
        if value is None:
            return ""
        if arg.argtype is bool:
            return "true" if bool(value) else "false"
        if arg.argtype is str:
            return str(value)
        return str(value)

    @classmethod
    def _history_restore_name(cls, name: str) -> str:
        match = cls._HISTORY_NAME_SUFFIX_RE.fullmatch(name)
        if match is not None:
            base, suffix = match.groups()
            return f"{base} ({int(suffix) + 1})"
        if name:
            return f"{name} (1)"
        return "(1)"

    def _apply_launch_history(self, history_index: int) -> None:
        assert self._form_state is not None
        fn = self.invocable_functions[self._form_state.fn_index]
        history = self._launch_form_history()
        if not (0 <= history_index < len(history)):
            return

        entry = history[history_index]
        self._form_state.fields[0].value = self._history_restore_name(entry.name)
        for field in self._form_state.fields[1:]:
            field.value = ""
            if field.is_provider:
                assert isinstance(fn, AgentFunction)
                provider = entry.provider or fn.default_model
                field.value = provider.value
                continue
            if field.is_provider_options:
                continue
            assert field.arg is not None
            if field.arg.name in entry.inputs:
                field.value = self._launch_form_field_value(
                    field.arg,
                    entry.inputs[field.arg.name],
                )

        self._form_state.error = ""
        self._form_state.cursor = self._first_editable_field_cursor()

    def _handle_form_key(self, key: str) -> bool:
        assert self._form_state is not None
        self._clamp_form_cursor()
        field = self._current_form_field()

        if key in ("escape",):
            self._form_state = None
            self._sync_visible_run()
            return False
        if key == "\t":
            self._move_form_cursor(1)
            return False
        if key == "shift_tab":
            self._move_form_cursor(-1)
            return False
        if key == "up":
            self._move_form_cursor(-1)
            return False
        if key == "down":
            self._move_form_cursor(1)
            return False
        if key == " " and field is not None and field.is_provider:
            self._cycle_provider_field()
            return False
        if key in ("\r", "\n"):
            return self._submit_or_move_form()
        if key in ("\x08", "\x7f"):
            if field is not None and (field.is_provider or field.is_provider_options):
                return False
            self._edit_form_text(backspace=True, ch="")
            return False
        if len(key) == 1:
            if field is not None and (field.is_provider or field.is_provider_options):
                return False
            self._edit_form_text(backspace=False, ch=key)
            return False
        return False

    def _submit_or_move_form(self) -> bool:
        assert self._form_state is not None
        self._clamp_form_cursor()
        submit_index = self._launch_form_submit_index()
        cancel_index = self._launch_form_cancel_index()
        history_start = self._launch_form_history_start_index()
        if self._form_state.cursor == cancel_index:
            self._form_state = None
            self._sync_visible_run()
            return False
        if self._form_state.cursor == submit_index:
            self._submit_form()
            return False
        if self._form_state.cursor >= history_start:
            self._apply_launch_history(self._form_state.cursor - history_start)
            return False
        self._move_form_cursor(1)
        return False

    def _edit_form_text(self, *, backspace: bool, ch: str) -> None:
        assert self._form_state is not None
        if self._form_state.cursor >= len(self._form_state.fields):
            return
        field = self._form_state.fields[self._form_state.cursor]
        if backspace:
            field.value = field.value[:-1]
        else:
            field.value += ch

    def _current_form_field(self) -> _LaunchField | None:
        assert self._form_state is not None
        if self._form_state.cursor >= len(self._form_state.fields):
            return None
        return self._form_state.fields[self._form_state.cursor]

    def _selected_provider_for_field(self, fn: AgentFunction, field: _LaunchField) -> Provider:
        raw_value = field.value.strip()
        if raw_value:
            try:
                return self._parse_provider(raw_value)
            except ValueError:
                pass
        return fn.default_model

    def _cycle_provider_field(self) -> None:
        assert self._form_state is not None
        field = self._current_form_field()
        if field is None or not field.is_provider:
            return
        fn = self.invocable_functions[self._form_state.fn_index]
        assert isinstance(fn, AgentFunction)
        providers = list(Provider)
        current = self._selected_provider_for_field(fn, field)
        current_index = providers.index(current)
        field.value = providers[(current_index + 1) % len(providers)].value
        self._form_state.error = ""

    def _submit_form(self) -> None:
        assert self._form_state is not None
        fn = self.invocable_functions[self._form_state.fn_index]
        parsed_args: dict[str, object] = {}
        provider_override: Provider | None = None
        try:
            for field in self._form_state.fields[1:]:
                if field.is_provider:
                    assert isinstance(fn, AgentFunction)
                    raw_provider = field.value.strip()
                    if raw_provider:
                        selected_provider = self._parse_provider(raw_provider)
                        if selected_provider != fn.default_model:
                            provider_override = selected_provider
                    continue
                if field.is_provider_options:
                    continue
                assert field.arg is not None
                raw = field.value
                if raw.strip() == "":
                    if field.optional:
                        parsed_args[field.arg.name] = None
                    continue
                parsed_args[field.arg.name] = self._parse_arg(field.arg, raw)
            cancel_event = mp.Event()
            node = self.runtime.invoke(
                None,
                fn,
                parsed_args,
                provider=provider_override,
                cancel_event=cancel_event,
            )
        except Exception as exc:
            self._form_state.error = str(exc)
            return

        try:
            run_name = self._form_state.fields[0].value.strip() or f"{fn.name} #{len(self._runs)}"
            renderer = ConsoleRender(cancel_event=cancel_event, spinner_hz=self.spinner_hz)
            run = _RunRecord(
                name=run_name,
                fn=fn,
                node=node,
                renderer=renderer,
                cancel_event=cancel_event,
                latest_view=self.runtime.get_view(node.id),
            )
            watcher_index = len(self._runs)
            run.watcher_thread = threading.Thread(
                target=self._watch_run,
                args=(watcher_index, node, run.watcher_stop),
                name=f"netflux-tui-watch-{node.id}",
                daemon=True,
            )
            run.watcher_thread.start()
        except Exception:
            self._fatal_after_launch(cancel_event)
        self._runs.append(run)
        run_index = len(self._runs) - 1
        self._set_selected_run(run_index)
        self._invoke_terminal_callback_if_needed(run)
        self._ensure_selected_run_post_terminal_browse(run)
        self._form_state = None
        self._sync_visible_run()

    @staticmethod
    def _fatal_after_launch(cancel_event: MpEvent) -> None:
        cancel_event.set()
        # Continuing would leave a real launched root running without tracked UI
        # state, so this path requests cancellation and fails fast.
        restore_console()
        os._exit(1)

    @staticmethod
    def _parse_arg(arg: FunctionArg, raw: str) -> object:
        text = raw.strip()
        if arg.argtype is str:
            value: object = raw
        elif arg.argtype is int:
            value = int(text)
        elif arg.argtype is float:
            value = float(text)
        elif arg.argtype is bool:
            normalized = text.lower()
            if normalized == "true":
                value = True
            elif normalized == "false":
                value = False
            else:
                raise ValueError(f"Arg '{arg.name}' expects bool text 'true' or 'false'.")
        else:
            raise ValueError(f"Unsupported arg type: {arg.argtype}")

        if arg.enum is not None and value not in arg.enum:
            allowed = ", ".join(sorted(arg.enum))
            raise ValueError(f"Arg '{arg.name}' must be one of: {allowed}")
        return value

    @staticmethod
    def _parse_provider(raw: str) -> Provider:
        normalized = raw.strip().lower()
        for provider in Provider:
            if normalized == provider.value.lower() or normalized == provider.name.lower():
                return provider
        allowed = ", ".join(provider.value for provider in Provider)
        raise ValueError(f"Provider must be one of: {allowed}")

    def _watch_run(self, run_index: int, node: Node, stop_event: threading.Event) -> None:
        prev_seq = 0
        while not stop_event.is_set():
            view = node.watch(as_of_seq=prev_seq)
            if view is None:
                continue
            prev_seq = view.update_seqnum
            self._event_queue.put(_RunUpdateEvent(run_index=run_index, view=view))
            self._wakeup()
            if view.state in TerminalNodeStates:
                return

    def _attempt_terminal_callbacks_for_all_runs(
        self,
        *,
        refresh_from_runtime: bool,
    ) -> None:
        for run in self._runs:
            if refresh_from_runtime:
                self._refresh_run_view_from_runtime(run)
            self._invoke_terminal_callback_if_needed(run)

    def _refresh_run_view_from_runtime(self, run: _RunRecord) -> None:
        node_id: object = getattr(run.node, "id", None)
        if not isinstance(node_id, int):
            return
        try:
            run.latest_view = self.runtime.get_view(node_id)
        except KeyError:
            return

    def _invoke_terminal_callback_if_needed(
        self,
        run: _RunRecord,
        view: NodeView | None = None,
    ) -> None:
        if run.terminal_callback_invoked:
            return
        final_view: NodeView | None = view or run.latest_view
        if final_view is None or final_view.state not in TerminalNodeStates:
            return
        callback: TerminalCallback | None = self._terminal_callback
        if callback is None:
            return

        run.terminal_callback_invoked = True
        try:
            callback(final_view.total_tree_token_bill())
        except Exception:
            logging.exception(
                "TUI terminal callback failed for top-level run '%s'.",
                run.name,
            )

    def _all_runs_terminal(self) -> bool:
        return all(
            run.latest_view is not None and run.latest_view.state in TerminalNodeStates
            for run in self._runs
        )

    def _request_exit_after_graceful_cancel_if_complete(self) -> None:
        if not self._global_cancel_requested:
            return
        if self._all_runs_terminal():
            self._exit_after_render = True

    def _ensure_selected_run_post_terminal_browse(self, run: _RunRecord) -> None:
        if (
            run.terminal_browse_applied
            or run.latest_view is None
            or run.latest_view.state not in TerminalNodeStates
        ):
            return
        self._sync_run_renderer_view(run)
        run.renderer.reset_for_browse()
        run.terminal_browse_applied = True
        run.terminal_browse_pending = False

    def _clear_selected_terminal_browse_pending(self) -> None:
        if self._selected_run is None or not (0 <= self._selected_run < len(self._runs)):
            return
        self._runs[self._selected_run].terminal_browse_pending = False

    def _apply_selected_terminal_browse_if_visible(self) -> None:
        if self._selected_run is None or not (0 <= self._selected_run < len(self._runs)):
            return
        run = self._runs[self._selected_run]
        if not run.terminal_browse_pending:
            return
        self._ensure_selected_run_post_terminal_browse(run)

    def _selected_renderer(self) -> ConsoleRender | None:
        if self._selected_run is None or not (0 <= self._selected_run < len(self._runs)):
            return None
        return self._runs[self._selected_run].renderer

    @staticmethod
    def _sync_run_renderer_view(run: _RunRecord) -> None:
        if run.latest_view is not None:
            run.renderer.assign_view(run.latest_view)

    def _sync_selected_renderer_view(self) -> None:
        if self._selected_run is None or not (0 <= self._selected_run < len(self._runs)):
            return
        self._sync_run_renderer_view(self._runs[self._selected_run])

    def _selected_status(self, *, include_position: bool = True) -> SelectedTreeStatus:
        if self._selected_run is None or not (0 <= self._selected_run < len(self._runs)):
            return SelectedTreeStatus()

        run = self._runs[self._selected_run]
        self._sync_run_renderer_view(run)
        status = run.renderer.selected_tree_status()
        if include_position:
            return status
        return SelectedTreeStatus(
            state=status.state,
            cancel_pending=status.cancel_pending,
            can_cancel=status.can_cancel,
            token_bill=status.token_bill,
        )

    def _left_pane_heights(self, body_rows: int) -> tuple[int, int]:
        bottom = min(max(4, len(self.invocable_functions) + 2), max(4, body_rows - 3))
        top = max(3, body_rows - bottom)
        bottom = max(1, body_rows - top)
        return top, bottom

    def _render_runs_pane(self, width: int, rows: int, tick: int) -> list[str]:
        if rows <= 0:
            return []

        entries = self._run_pane_entries()
        available = rows
        if entries:
            self._sync_run_scroll(entries, available)
        else:
            self._run_scroll = 0

        lines: list[str] = []
        end = min(len(entries), self._run_scroll + available)
        for entry in entries[self._run_scroll:end]:
            rendered = self._render_run_pane_entry(entry, width, tick)
            lines.append(rendered)

        while len(lines) < rows:
            lines.append(" " * width)
        return lines[:rows]

    def _render_functions_pane(self, width: int, rows: int) -> list[str]:
        lines = [self._pad_visible(_color("Functions", fg="cyan", bold=True), width)]
        available = max(0, rows - 1)
        if available <= 0:
            return lines[:rows]

        for idx, fn in enumerate(self.invocable_functions[:available]):
            text = f"({idx}) {fn.name}"
            lines.append(self._pad_visible(text, width))

        while len(lines) < rows:
            lines.append(" " * width)
        return lines[:rows]

    def _render_right_pane(self, width: int, rows: int, tick: int) -> list[str]:
        renderer = self._selected_renderer()
        if renderer is None:
            placeholder = [
                self._pad_visible(_color("No run selected.", dim=True), width),
                self._pad_visible(_color("Launch a function with 0-9.", dim=True), width),
            ]
            while len(placeholder) < rows:
                placeholder.append(" " * width)
            return placeholder[:rows]

        body = renderer.render_body(
            width=width,
            height=rows,
            tick=tick,
        )
        split = body.splitlines()
        return [self._pad_visible(line, width) for line in split[:rows]]

    def _render_launch_form(self, size: TerminalSize, tick: int) -> str:
        assert self._form_state is not None
        self._clamp_form_cursor()
        if self._too_small:
            interrupt_hint = self._global_interrupt_hint()
            bottom_bar = compose_bottom_bar(
                size.columns,
                shortcut_variants=[
                    ["Resize terminal to continue"],
                    ["Resize to continue"],
                    [],
                ],
                status=self._selected_status(include_position=False),
                tick=tick,
                mandatory_shortcuts=[interrupt_hint],
            )
            return render_too_small_frame(
                size,
                message="Terminal too small for launch form.",
                hint="Resize to continue.",
                bottom_bar=bottom_bar,
            )

        rows = max(1, size.lines - 1)
        layout = self._launch_form_layout(size)
        history = self._launch_form_history()
        submit_index = self._launch_form_submit_index()
        cancel_index = self._launch_form_cancel_index()
        history_start = self._launch_form_history_start_index()
        item_count = self._launch_form_item_count()
        lines = list(layout.header_lines)
        hint_parts: list[str] = []
        if layout.header_clipped:
            hint_parts.append("desc clipped")
        if layout.item_start > 0:
            hint_parts.append("↑ more")
        if layout.item_end < item_count:
            hint_parts.append("↓ more")
        hint = _color("  ".join(hint_parts), dim=True) if hint_parts else ""
        lines.append(self._pad_visible(hint, size.columns))

        for item_index in range(layout.item_start, layout.item_end):
            if item_index < len(self._form_state.fields):
                field = self._form_state.fields[item_index]
                if field.is_provider_options:
                    provider_field = self._form_state.fields[item_index - 1]
                    fn = self.invocable_functions[self._form_state.fn_index]
                    assert isinstance(fn, AgentFunction)
                    selected_provider = self._selected_provider_for_field(fn, provider_field)
                    option_tokens: list[str] = []
                    for provider in Provider:
                        if provider == selected_provider:
                            option_tokens.append(_color(f"[{provider.value}]", fg="green", bold=True))
                        else:
                            option_tokens.append(_color(provider.value, fg="gray", dim=True))
                    rendered = self._pad_visible(f"  {'  '.join(option_tokens)}", size.columns)
                else:
                    label_color = "orange" if field.is_name else "cyan"
                    label = _color(field.label, fg=label_color, bold=True)
                    meta_parts: list[str] = []
                    if not field.is_name:
                        meta_parts.append(_color(f"({field.type_label})", fg="gray"))
                        if field.optional:
                            meta_parts.append(_color("[optional]", fg="yellow", bold=True))
                    meta = f" {' '.join(meta_parts)}" if meta_parts else ""
                    rendered = self._pad_visible(f"{label}{meta}: {field.value}", size.columns)
            elif item_index == submit_index:
                rendered = self._pad_visible(_color("[Submit]", fg="green", bold=True), size.columns)
            elif item_index == cancel_index:
                rendered = self._pad_visible(_color("[Cancel]", fg="red", bold=True), size.columns)
            else:
                history_entry = history[item_index - history_start]
                args_preview = _format_args(
                    dict(history_entry.inputs),
                    max_len=max(12, size.columns - max(18, len(history_entry.name) + 4)),
                    per_val_len=32,
                ) or "(no args)"
                rendered = self._pad_visible(
                    f"{_color('↺', fg='yellow', bold=True)} "
                    f"{_color(history_entry.name, fg='white', bold=True)} "
                    f"{_color(args_preview, fg='gray', dim=True)}",
                    size.columns,
                )
            highlight_index = item_index
            if item_index < len(self._form_state.fields) and self._form_state.fields[item_index].is_provider_options:
                highlight_index = item_index - 1
            if highlight_index == self._form_state.cursor:
                rendered = _highlight_line(rendered, size.columns)
            lines.append(rendered)

        if self._form_state.error:
            lines.append(self._pad_visible(_color(self._form_state.error, fg="red"), size.columns))

        while len(lines) < rows:
            lines.append(" " * size.columns)

        current_field = self._current_form_field()
        provider_selected = current_field is not None and current_field.is_provider
        bottom_bar = compose_bottom_bar(
            size.columns,
            shortcut_variants=[
                [
                    "Tab/Shift+Tab:item",
                    "Enter:next/use/submit",
                    "Space:toggle provider" if provider_selected else "Esc:cancel",
                    "Esc:cancel" if provider_selected else "Backspace:delete",
                ],
                [
                    "Tab/S-Tab:item",
                    "Enter:next/use/submit",
                    "Space:toggle" if provider_selected else "Esc:cancel",
                    "Esc" if provider_selected else "Backspace",
                ],
                ["Tab", "Enter", "Space" if provider_selected else "Esc", "Esc" if provider_selected else "Backspace"],
            ],
            status=self._selected_status(include_position=False),
            tick=tick,
            mandatory_shortcuts=[self._global_interrupt_hint()],
        )
        lines = lines[:rows]
        lines.append(f"{bottom_bar}\x1b[K")
        return "\n".join(lines)

    def _global_interrupt_hint(self) -> str:
        if self._global_cancel_requested:
            return "^C:force quit"
        return "^C:cancel all"

    def _finalize_frame(self, frame: str) -> str:
        if self._exit_after_render:
            self._exit_after_render = False
            self._should_exit = True
        return frame

    def _handle_form_mouse(self, event: object) -> bool:
        assert self._form_state is not None
        self._clamp_form_cursor()
        y = int(getattr(event, "y", 0))
        button = getattr(event, "button", "")
        if button != "left":
            return False
        layout = self._launch_form_layout(self._last_size)
        submit_index = self._launch_form_submit_index()
        cancel_index = self._launch_form_cancel_index()
        history_start = self._launch_form_history_start_index()
        if y < layout.header_rows:
            return False
        item_index = layout.item_start + (y - layout.header_rows)
        if item_index >= layout.item_end:
            return False
        if item_index < len(self._form_state.fields):
            if self._form_state.fields[item_index].is_provider_options:
                self._form_state.cursor = max(0, item_index - 1)
            else:
                self._form_state.cursor = item_index
            return False
        if item_index == submit_index:
            self._form_state.cursor = item_index
            self._submit_form()
            return False
        if item_index == cancel_index:
            self._form_state = None
            self._sync_visible_run()
            return False
        if item_index >= history_start:
            self._form_state.cursor = item_index
            self._apply_launch_history(item_index - history_start)
            return False
        return False

    def _form_item_window(self, size: TerminalSize) -> tuple[int, int, int]:
        layout = self._launch_form_layout(size)
        return layout.item_start, layout.item_end, layout.header_rows

    def _launch_form_layout(self, size: TerminalSize) -> _LaunchFormLayout:
        assert self._form_state is not None
        self._clamp_form_cursor()
        fn = self.invocable_functions[self._form_state.fn_index]
        rows = max(1, size.lines - 1)
        error_rows = 1 if self._form_state.error else 0

        full_header_lines = [
            self._pad_visible(_color(f"Launch {fn.name}", fg="cyan", bold=True), size.columns)
        ]
        full_header_lines.extend(
            self._pad_visible(_color(desc_line, fg="gray", dim=True), size.columns)
            for desc_line in self._launch_form_desc_lines(fn, size.columns)
        )
        arg_lines = self._launch_form_arg_lines(fn, size.columns)
        for idx, arg_line in enumerate(arg_lines):
            is_header = idx == 0
            full_header_lines.append(
                self._pad_visible(
                    _color(arg_line, fg="white" if is_header else "gray", bold=is_header, dim=not is_header),
                    size.columns,
                )
            )

        # Reserve one row for the scroll hint line and one row for a visible item.
        min_item_rows = 1
        hint_rows = 1
        header_budget = max(0, rows - error_rows - min_item_rows - hint_rows)
        header_clipped = len(full_header_lines) > header_budget
        if header_budget <= 0:
            header_lines: list[str] = []
        elif len(full_header_lines) <= header_budget:
            header_lines = full_header_lines
        else:
            header_lines = list(full_header_lines[:header_budget])

        header_rows = len(header_lines) + hint_rows
        available_rows = max(1, rows - header_rows - error_rows)
        item_count = self._launch_form_item_count()
        max_start = max(0, item_count - available_rows)
        cursor = min(self._form_state.cursor, item_count - 1)
        start = min(max(0, cursor - available_rows + 1), max_start)
        end = min(item_count, start + available_rows)
        return _LaunchFormLayout(
            header_lines=tuple(header_lines),
            item_start=start,
            item_end=end,
            header_rows=header_rows,
            header_clipped=header_clipped,
        )

    def _launch_form_desc_lines(self, fn: Function, width: int) -> list[str]:
        wrapped_lines: list[str] = []
        wrap_width = max(1, width)
        wrapper = textwrap.TextWrapper(
            width=wrap_width,
            replace_whitespace=False,
            drop_whitespace=True,
            break_long_words=True,
            break_on_hyphens=False,
        )

        for desc_line in fn.desc.splitlines() or [""]:
            if desc_line.strip() == "":
                wrapped_lines.append("")
                continue
            wrapped_lines.extend(wrapper.wrap(desc_line) or [""])

        return wrapped_lines or [""]

    def _launch_form_arg_lines(self, fn: Function, width: int) -> list[str]:
        if not fn.args:
            return []

        wrap_width = max(1, width)
        wrapper = textwrap.TextWrapper(
            width=wrap_width,
            replace_whitespace=False,
            drop_whitespace=True,
            break_long_words=True,
            break_on_hyphens=False,
        )

        lines = ["Arguments"]
        for arg in fn.args:
            optional = " [optional]" if arg.optional else ""
            prefix = f"{arg.name} ({arg.argtype.__name__}){optional}"
            body = arg.desc if arg.desc else ""
            text = f"{prefix}: {body}" if body else prefix
            wrapped = wrapper.wrap(text) or [text]
            lines.extend(wrapped)

        return lines

    def _select_next_run(self) -> None:
        ordered = self._grouped_run_indices()
        if not ordered:
            return
        if self._selected_run not in ordered:
            self._set_selected_run(ordered[0])
            return
        current = ordered.index(self._selected_run)
        self._set_selected_run(ordered[min(len(ordered) - 1, current + 1)])

    def _select_prev_run(self) -> None:
        ordered = self._grouped_run_indices()
        if not ordered:
            return
        if self._selected_run not in ordered:
            self._set_selected_run(ordered[0])
            return
        current = ordered.index(self._selected_run)
        self._set_selected_run(ordered[max(0, current - 1)])

    def _cancel_selected_run(self) -> None:
        if not self._selected_status(include_position=False).can_cancel:
            return
        self._runs[self._selected_run].cancel_event.set()

    def _select_run_row(self, row: int) -> None:
        if row < 0:
            return
        entries = self._run_pane_entries()
        entry_index = self._run_scroll + row
        if not (0 <= entry_index < len(entries)):
            return
        run_index = entries[entry_index].run_index
        if run_index is not None:
            self._set_selected_run(run_index)

    def _launch_function_hit(self, local_x: int, row: int, width: int, bottom_rows: int) -> None:
        if row == 0:
            return

        available = max(0, bottom_rows - 1)
        if available <= 0:
            return

        entry_row = row - 1
        if entry_row >= available:
            return

        fn_index = entry_row
        if not (0 <= fn_index < len(self.invocable_functions)):
            return

        label = f"({fn_index}) {self.invocable_functions[fn_index].name}"
        clickable_cols = min(width, len(label))
        if not (0 <= local_x < clickable_cols):
            return

        self._open_launch_form(fn_index)

    @staticmethod
    def _pad_visible(text: str, width: int) -> str:
        visible = _visible_len(text)
        if visible > width:
            return _crop_line(text, width)
        if visible < width:
            return text + " " * (width - visible)
        return text

    @staticmethod
    def _right_pane_origin(left_width: int) -> int:
        return left_width + _PANE_SEPARATOR_COLS + _RIGHT_PANE_MARGIN_COLS

    def _right_pane_width(self, columns: int, left_width: int) -> int:
        return max(1, columns - self._right_pane_origin(left_width))

    def _grouped_run_indices(self) -> list[int]:
        return [entry.run_index for entry in self._run_pane_entries() if entry.run_index is not None]

    def _run_pane_entries(self) -> list[_RunPaneEntry]:
        if not self._runs:
            return [_RunPaneEntry(kind="placeholder", label="(no runs yet)")]

        runs_by_fn: dict[Function, list[int]] = {}
        for idx, run in enumerate(self._runs):
            runs_by_fn.setdefault(run.fn, []).append(idx)

        entries: list[_RunPaneEntry] = []
        seen_runs: set[int] = set()
        for fn in self.invocable_functions:
            run_indices = runs_by_fn.get(fn)
            if not run_indices:
                continue
            if entries:
                entries.append(_RunPaneEntry(kind="spacer"))
            entries.append(_RunPaneEntry(kind="group_header", label=fn.name))
            for run_index in run_indices:
                entries.append(_RunPaneEntry(kind="run", run_index=run_index))
                seen_runs.add(run_index)

        for run_index, run in enumerate(self._runs):
            if run_index in seen_runs:
                continue
            if entries:
                entries.append(_RunPaneEntry(kind="spacer"))
            entries.append(_RunPaneEntry(kind="group_header", label=run.fn.name))
            entries.append(_RunPaneEntry(kind="run", run_index=run_index))

        return entries

    def _sync_run_scroll(self, entries: list[_RunPaneEntry], available: int) -> None:
        max_scroll = max(0, len(entries) - available)
        self._run_scroll = max(0, min(self._run_scroll, max_scroll))
        if self._selected_run is None:
            return

        selected_row = next(
            (idx for idx, entry in enumerate(entries) if entry.run_index == self._selected_run),
            None,
        )
        if selected_row is None:
            return

        group_header_row = self._group_header_row(entries, selected_row)
        if selected_row < self._run_scroll:
            target = group_header_row if group_header_row is not None else selected_row
            self._run_scroll = max(0, min(target, max_scroll))
            return

        if selected_row >= self._run_scroll + available:
            target = selected_row - available + 1
            if (
                group_header_row is not None
                and group_header_row <= selected_row
                and selected_row - group_header_row < available
            ):
                target = group_header_row
            self._run_scroll = max(0, min(target, max_scroll))

    @staticmethod
    def _group_header_row(entries: list[_RunPaneEntry], selected_row: int) -> int | None:
        for idx in range(selected_row, -1, -1):
            if entries[idx].kind == "group_header":
                return idx
        return None

    def _render_run_pane_entry(self, entry: _RunPaneEntry, width: int, tick: int) -> str:
        if entry.kind == "placeholder":
            return self._pad_visible(_color(entry.label, dim=True), width)
        if entry.kind == "group_header":
            return self._pad_visible(_color(entry.label, fg="gray", bold=True), width)
        if entry.kind == "spacer":
            return " " * width

        assert entry.run_index is not None
        run = self._runs[entry.run_index]
        state = run.latest_view.state if run.latest_view is not None else NodeState.Waiting
        glyph, color = _state_glyph(state, tick)
        if run.cancel_event.is_set() and state not in TerminalNodeStates:
            color = "magenta"
        run_name = _color(run.name, bold=run.unread) if run.unread else run.name
        text = f"{_color(glyph, fg=color, bold=True)} {run_name}"
        rendered = self._pad_visible(text, width)
        if entry.run_index == self._selected_run:
            rendered = _highlight_line(rendered, width)
        return rendered

    def _set_selected_run(self, run_index: int | None) -> None:
        if run_index is not None and not (0 <= run_index < len(self._runs)):
            return
        selection_changed = run_index != self._selected_run
        self._selected_run = run_index
        if selection_changed and run_index is not None:
            self._clear_run_unread(run_index, clear_manual=True)
        self._sync_visible_run()

    def _current_visible_run(self) -> int | None:
        if self._form_state is not None or self._too_small:
            return None
        if self._selected_run is None or not (0 <= self._selected_run < len(self._runs)):
            return None
        return self._selected_run

    def _sync_visible_run(self) -> None:
        visible_run = self._current_visible_run()
        if visible_run == self._visible_run:
            return
        self._visible_run = visible_run
        if visible_run is not None:
            self._clear_run_unread(visible_run, clear_manual=False)

    def _clear_run_unread(self, run_index: int, *, clear_manual: bool) -> None:
        run = self._runs[run_index]
        run.auto_unread = False
        if clear_manual:
            run.manual_unread = False

    def _toggle_selected_run_unread(self) -> None:
        if self._selected_run is None or not (0 <= self._selected_run < len(self._runs)):
            return
        run = self._runs[self._selected_run]
        if run.unread:
            run.auto_unread = False
            run.manual_unread = False
            return
        run.manual_unread = True
