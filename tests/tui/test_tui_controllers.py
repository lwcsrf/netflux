from __future__ import annotations

import ctypes
import io
import multiprocessing as mp
import os
import re
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ...core import (
    AgentFunction,
    CancellationException,
    CodeFunction,
    FunctionArg,
    ModelTextPart,
    NodeState,
    TokenUsage,
    NodeView,
    RunContext,
    TokenBill,
)
from ...providers import Provider
from ...runtime import Runtime
from ...tui import ConsoleRender
from ...tui._controller_helpers import (
    compose_bottom_bar,
    multi_pane_shortcut_variants,
    preferred_left_pane_width,
    render_too_small_frame,
)
from ...tui._controllers import SingleTreeConsoleController
from ...tui._contracts import SelectedTreeStatus, SessionController, TerminalSize
from ...tui._driver import ConsoleSessionDriver
from ...tui._terminal_io import (
    _WIN_ENABLE_MOUSE_INPUT,
    _WIN_ENABLE_WINDOW_INPUT,
    _WIN_KEY_EVENT,
    _WinInputRecord,
    _configure_windows_console_input,
    read_key,
    read_key_windows,
)
from ...tui import _terminal_io as terminal_io
from ...tui.console import FG
from ...tui.tui import TUI, TokenBills, _RunRecord, _RunUpdateEvent


_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _make_code_function(name: str) -> CodeFunction:
    def _callable(ctx: RunContext) -> str:
        return name

    return CodeFunction(
        name=name,
        desc=f"code fn {name}",
        args=[],
        callable=_callable,
        uses=[],
    )


def _make_agent_function(name: str) -> AgentFunction:
    return AgentFunction(
        name=name,
        desc=f"agent fn {name}",
        args=[],
        system_prompt="system",
        user_prompt_template="",
        uses=[],
    )


def _make_view(
    fn: CodeFunction,
    *,
    state: NodeState,
    update_seqnum: int,
    inputs: dict[str, object] | None = None,
) -> NodeView:
    return NodeView(
        id=update_seqnum,
        fn=fn,
        inputs=inputs or {},
        state=state,
        outputs=None,
        exception=None,
        children=(),
        usage=None,
        transcript=(),
        started_at=0.0,
        ended_at=0.0 if state in {NodeState.Success, NodeState.Error, NodeState.Canceled} else None,
        update_seqnum=update_seqnum,
    )


def _make_output_view(
    fn: CodeFunction,
    *,
    output: object,
    update_seqnum: int,
) -> NodeView:
    return NodeView(
        id=update_seqnum,
        fn=fn,
        inputs={},
        state=NodeState.Success,
        outputs=output,
        exception=None,
        children=(),
        usage=None,
        transcript=(),
        started_at=0.0,
        ended_at=0.0,
        update_seqnum=update_seqnum,
    )


def _make_agent_view(
    fn: AgentFunction,
    *,
    state: NodeState,
    update_seqnum: int,
    provider: Provider | None = None,
) -> NodeView:
    return NodeView(
        id=update_seqnum,
        fn=fn,
        inputs={},
        state=state,
        outputs=None,
        exception=None,
        children=(),
        usage=None,
        transcript=(ModelTextPart(text=f"model output {update_seqnum}"),),
        started_at=0.0,
        ended_at=0.0 if state in {NodeState.Success, NodeState.Error, NodeState.Canceled} else None,
        update_seqnum=update_seqnum,
        provider=provider,
    )


def _make_billed_agent_view(
    fn: AgentFunction,
    *,
    state: NodeState,
    update_seqnum: int,
    provider: Provider,
    usage: TokenUsage,
    children: tuple[NodeView, ...] = (),
) -> NodeView:
    return NodeView(
        id=update_seqnum,
        fn=fn,
        inputs={},
        state=state,
        outputs=None,
        exception=None,
        children=children,
        usage=usage,
        transcript=(ModelTextPart(text=f"model output {update_seqnum}"),),
        started_at=0.0,
        ended_at=0.0 if state in {NodeState.Success, NodeState.Error, NodeState.Canceled} else None,
        update_seqnum=update_seqnum,
        provider=provider,
    )


def _make_windows_key_record(*, key_down: bool, char: str = "\x00", vk: int = 0) -> _WinInputRecord:
    record = _WinInputRecord()
    record.EventType = _WIN_KEY_EVENT
    record.Event.KeyEvent.bKeyDown = int(key_down)
    record.Event.KeyEvent.uChar = char
    record.Event.KeyEvent.wVirtualKeyCode = vk
    return record


class _FakeWindowsKernel32:
    def __init__(self, records: list[_WinInputRecord]) -> None:
        self._records = list(records)

    def GetStdHandle(self, which: int) -> int:
        del which
        return 1

    def PeekConsoleInputW(self, handle, record_ptr, length, count_ptr) -> int:
        del handle, length
        count = ctypes.cast(count_ptr, ctypes.POINTER(ctypes.c_uint))
        if not self._records:
            count.contents.value = 0
            return 1
        ctypes.memmove(record_ptr, ctypes.byref(self._records[0]), ctypes.sizeof(_WinInputRecord))
        count.contents.value = 1
        return 1

    def ReadConsoleInputW(self, handle, record_ptr, length, count_ptr) -> int:
        del handle, length
        if not self._records:
            raise AssertionError("ReadConsoleInputW called after the queue was drained")
        record = self._records.pop(0)
        ctypes.memmove(record_ptr, ctypes.byref(record), ctypes.sizeof(_WinInputRecord))
        count = ctypes.cast(count_ptr, ctypes.POINTER(ctypes.c_uint))
        count.contents.value = 1
        return 1


class _FakeWindowsModeKernel32:
    def __init__(self, mode: int) -> None:
        self.mode = mode
        self.set_modes: list[int] = []

    def GetStdHandle(self, which: int) -> int:
        del which
        return 1

    def GetConsoleMode(self, handle, mode_ptr) -> int:
        del handle
        mode = ctypes.cast(mode_ptr, ctypes.POINTER(ctypes.c_uint))
        mode.contents.value = self.mode
        return 1

    def SetConsoleMode(self, handle, mode: int) -> int:
        del handle
        self.mode = int(mode)
        self.set_modes.append(self.mode)
        return 1


class TestSingleTreeConsoleController(unittest.TestCase):
    def test_console_render_run_requires_node_cancel_event(self) -> None:
        fn = _make_code_function("done")
        runtime = Runtime([fn], client_factories={})
        node = runtime.invoke(None, fn, {})
        renderer = ConsoleRender()

        with patch("netflux.tui._driver.ConsoleSessionDriver.run") as driver_run:
            with self.assertRaisesRegex(ValueError, "requires the node to have a cancel_event"):
                renderer.run(node)

        driver_run.assert_not_called()

    def test_console_render_run_accepts_node_cancel_event_without_renderer_cancel_event(self) -> None:
        fn = _make_code_function("done")
        runtime = Runtime([fn], client_factories={})
        cancel_event = mp.Event()
        node = runtime.invoke(None, fn, {}, cancel_event=cancel_event)
        renderer = ConsoleRender()

        with patch("netflux.tui._driver.ConsoleSessionDriver.run") as driver_run:
            renderer.run(node)

        driver_run.assert_called_once()

    def test_wants_animation_ticks_has_intended_truth_table(self) -> None:
        fn = _make_code_function("done")
        runtime = Runtime([fn], client_factories={})
        node = runtime.invoke(None, fn, {})
        self.assertEqual(node.result(), "done")

        renderer = ConsoleRender()
        controller = SingleTreeConsoleController(renderer, node)
        self.addCleanup(controller.on_session_stop)

        controller._latest_view = None
        self.assertTrue(controller.wants_animation_ticks())

        controller._latest_view = _make_view(fn, state=NodeState.Running, update_seqnum=1)
        controller._mode = "live"
        self.assertTrue(controller.wants_animation_ticks())

        controller._mode = "browse"
        self.assertFalse(controller.wants_animation_ticks())

        controller._latest_view = _make_view(fn, state=NodeState.Success, update_seqnum=2)
        controller._mode = "live"
        self.assertFalse(controller.wants_animation_ticks())

    def test_ctrl_c_cancel_exits_after_terminal_update(self) -> None:
        cancel_event = mp.Event()
        started = threading.Event()

        def blocking(ctx: RunContext) -> str:
            started.set()
            while not ctx.cancel_requested():
                time.sleep(0.01)
            raise CancellationException()

        fn = CodeFunction(
            name="blocking",
            desc="blocking test fn",
            args=[],
            callable=blocking,
            uses=[],
        )
        runtime = Runtime([fn], client_factories={})
        node = runtime.invoke(None, fn, {}, cancel_event=cancel_event)
        self.assertTrue(started.wait(timeout=1), "blocking callable did not start")

        renderer = ConsoleRender(cancel_event=cancel_event)
        controller = SingleTreeConsoleController(renderer, node)
        self.addCleanup(controller.on_session_stop)
        controller.on_session_start(interactive=True)

        self.assertFalse(controller.handle_interrupt())
        with self.assertRaises(CancellationException):
            node.result()

        deadline = time.time() + 1.0
        while time.time() < deadline:
            controller.pump_events()
            frame = controller.render_frame(TerminalSize(columns=80, lines=10), tick=0)
            if controller.should_exit():
                self.assertIn("Canceled", _strip_ansi(frame))
                self.assertIn("q:quit", _strip_ansi(frame))
                break
            time.sleep(0.01)

        self.assertNotEqual(controller._mode, "browse")
        self.assertTrue(controller.should_exit())
        self.assertEqual(renderer.selected_tree_status().state, NodeState.Canceled)

    def test_already_terminal_root_enters_browse_before_first_frame(self) -> None:
        fn = _make_code_function("done")
        runtime = Runtime([fn], client_factories={})
        node = runtime.invoke(None, fn, {})
        self.assertEqual(node.result(), "done")

        renderer = ConsoleRender()
        controller = SingleTreeConsoleController(renderer, node)
        self.addCleanup(controller.on_session_stop)
        controller.on_session_start(interactive=True)

        frame = _strip_ansi(controller.render_frame(TerminalSize(columns=80, lines=10), tick=0))

        self.assertEqual(controller._mode, "browse")
        self.assertIn("q:quit", frame)

    def test_noninteractive_terminal_root_final_frame_shows_quit_hint(self) -> None:
        fn = _make_code_function("done")
        runtime = Runtime([fn], client_factories={})
        node = runtime.invoke(None, fn, {})
        self.assertEqual(node.result(), "done")

        renderer = ConsoleRender()
        controller = SingleTreeConsoleController(renderer, node)
        self.addCleanup(controller.on_session_stop)
        controller.on_session_start(interactive=False)

        frame = _strip_ansi(controller.render_frame(TerminalSize(columns=80, lines=10), tick=0))

        self.assertIn("q:quit", frame)
        self.assertTrue(controller.should_exit())

    def test_copy_result_failure_shows_clipboard_install_hint(self) -> None:
        fn = _make_code_function("done")
        runtime = Runtime([fn], client_factories={})
        cancel_event = mp.Event()
        node = runtime.invoke(None, fn, {}, cancel_event=cancel_event)
        self.assertEqual(node.result(), "done")

        renderer = ConsoleRender()
        controller = SingleTreeConsoleController(renderer, node)
        self.addCleanup(controller.on_session_stop)
        controller.on_session_start(interactive=True)

        with patch.object(
            renderer,
            "copy_terminal_result_with_feedback",
            return_value=(False, "Clipboard unavailable. Install wl-copy, xclip, or xsel."),
        ):
            self.assertFalse(controller.handle_key("c"))

        frame = _strip_ansi(controller.render_frame(TerminalSize(columns=140, lines=10), tick=0))
        self.assertIn("Install wl-copy, xclip, or xsel", frame.splitlines()[-1])

    def test_standalone_key_remap_routes_agent_and_all_actions(self) -> None:
        fn = _make_code_function("done")
        runtime = Runtime([fn], client_factories={})
        cancel_event = mp.Event()
        node = runtime.invoke(None, fn, {}, cancel_event=cancel_event)
        self.assertEqual(node.result(), "done")

        renderer = ConsoleRender()
        renderer.apply_action = Mock()  # type: ignore[method-assign]
        controller = SingleTreeConsoleController(renderer, node)
        self.addCleanup(controller.on_session_stop)
        controller.on_session_start(interactive=True)

        controller.handle_key("a")
        controller.handle_key("e")
        controller.handle_key("E")
        controller.handle_key("r")

        self.assertEqual(
            [call.args[0] for call in renderer.apply_action.call_args_list],
            ["collapse_agent", "expand_all", "collapse_all", "focus_result"],
        )

    def test_interrupt_uses_node_cancel_event_when_renderer_has_none(self) -> None:
        cancel_event = mp.Event()
        started = threading.Event()

        def blocking(ctx: RunContext) -> str:
            started.set()
            while not ctx.cancel_requested():
                time.sleep(0.01)
            raise CancellationException()

        fn = CodeFunction(
            name="blocking",
            desc="blocking test fn",
            args=[],
            callable=blocking,
            uses=[],
        )
        runtime = Runtime([fn], client_factories={})
        node = runtime.invoke(None, fn, {}, cancel_event=cancel_event)
        self.assertTrue(started.wait(timeout=1), "blocking callable did not start")

        renderer = ConsoleRender()
        controller = SingleTreeConsoleController(renderer, node)
        self.addCleanup(controller.on_session_stop)
        controller.on_session_start(interactive=True)

        self.assertFalse(controller.handle_interrupt())
        self.assertTrue(cancel_event.is_set())
        with self.assertRaises(CancellationException):
            node.result()

    def test_on_session_stop_does_not_join_blocked_watcher(self) -> None:
        cancel_event = mp.Event()
        started = threading.Event()

        def blocking(ctx: RunContext) -> str:
            started.set()
            while not ctx.cancel_requested():
                time.sleep(0.01)
            raise CancellationException()

        fn = CodeFunction(
            name="blocking",
            desc="blocking test fn",
            args=[],
            callable=blocking,
            uses=[],
        )
        runtime = Runtime([fn], client_factories={})
        node = runtime.invoke(None, fn, {}, cancel_event=cancel_event)
        self.assertTrue(started.wait(timeout=1), "blocking callable did not start")

        renderer = ConsoleRender()
        controller = SingleTreeConsoleController(renderer, node)
        self.addCleanup(controller.on_session_stop)
        controller.on_session_start(interactive=True)

        controller._watch_thread.join = Mock(side_effect=AssertionError("join should not be called"))  # type: ignore[method-assign]

        controller.on_session_stop()

        controller._watch_thread.join.assert_not_called()
        cancel_event.set()
        with self.assertRaises(CancellationException):
            node.result()

    def test_too_small_browse_frame_only_shows_resize_and_quit(self) -> None:
        fn = _make_code_function("done")
        runtime = Runtime([fn], client_factories={})
        node = runtime.invoke(None, fn, {})
        self.assertEqual(node.result(), "done")

        renderer = ConsoleRender()
        controller = SingleTreeConsoleController(renderer, node)
        self.addCleanup(controller.on_session_stop)
        controller.on_session_start(interactive=True)

        rendered = _strip_ansi(controller.render_frame(TerminalSize(columns=39, lines=5), tick=0))

        self.assertIn("Resize", rendered)
        self.assertIn("q:quit", rendered)
        self.assertNotIn("jk:move", rendered)
        self.assertNotIn("Pg", rendered)


class TestBottomBarFormatting(unittest.TestCase):
    def test_compose_bottom_bar_shows_cancel_pending(self) -> None:
        bar = compose_bottom_bar(
            120,
            shortcut_variants=[["jk:move"]],
            status=SelectedTreeStatus(
                cursor_line=3,
                total_lines=9,
                state=NodeState.Running,
                cancel_pending=True,
            ),
            tick=0,
        )

        plain = _strip_ansi(bar)
        self.assertIn("3/9", plain)
        self.assertIn("Running", plain)
        self.assertIn("Cancel pending", plain)

    def test_compose_bottom_bar_keeps_tokens_ahead_of_mandatory_shortcuts(self) -> None:
        status = SelectedTreeStatus(
            state=NodeState.Running,
            can_cancel=True,
            token_bill={
                Provider.Anthropic: TokenBill(
                    input_tokens_regular=12_000,
                    output_tokens_total=8_000,
                )
            },
        )
        token_text = ConsoleRender._format_total_token_bill(status.token_bill)
        bar = compose_bottom_bar(
            40,
            shortcut_variants=[["jk:move", "Pg:page"], ["jk", "Pg"]],
            status=status,
            tick=0,
            mandatory_shortcuts=["^C:cancel"],
        )

        plain = _strip_ansi(bar)
        self.assertIn(token_text, plain)
        self.assertNotIn("^C:cancel", plain)

    def test_multi_pane_bottom_bar_keeps_global_interrupt_hint_ahead_of_tree_shortcuts(self) -> None:
        bar = compose_bottom_bar(
            68,
            shortcut_variants=multi_pane_shortcut_variants(
                SimpleNamespace(
                    has_lines=True,
                    can_expand_collapse=True,
                    can_jump_agents=True,
                ),
                can_cancel=True,
                interrupt_hint="^C:cancel all",
            ),
            status=SelectedTreeStatus(state=NodeState.Running, can_cancel=True),
            tick=0,
        )

        plain = _strip_ansi(bar)
        self.assertIn("^C:cancel all", plain)
        self.assertNotIn("e/E:all", plain)

    def test_render_too_small_frame_respects_single_line_height(self) -> None:
        rendered = render_too_small_frame(
            TerminalSize(columns=12, lines=1),
            message="too small",
            hint="resize",
            bottom_bar="bottom",
        )

        plain = ""
        self.assertEqual(rendered.splitlines(), ["bottom      "])
        self.assertNotIn("^C:can…", plain)

class TestTUIState(unittest.TestCase):
    def test_terminal_callback_receives_total_tree_bill(self) -> None:
        fn = _make_agent_function("billed")
        child_fn = _make_agent_function("child")
        runtime = Runtime([fn, child_fn], client_factories={})
        received: list[TokenBills] = []
        tui = TUI(runtime)

        def _capture(bills: TokenBills) -> None:
            received.append(bills)

        tui.register_terminal_callback(_capture)
        tui._runs = [
            _RunRecord(
                name="run",
                fn=fn,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=mp.Event()),
                cancel_event=mp.Event(),
                latest_view=_make_agent_view(fn, state=NodeState.Running, update_seqnum=1),
            )
        ]
        terminal_view = _make_billed_agent_view(
            fn,
            state=NodeState.Success,
            update_seqnum=2,
            provider=Provider.Anthropic,
            usage=TokenUsage(
                input_tokens_cache_read=10,
                input_tokens_total=13,
                input_tokens_regular=3,
                output_tokens_total=7,
            ),
            children=(
                _make_billed_agent_view(
                    child_fn,
                    state=NodeState.Success,
                    update_seqnum=3,
                    provider=Provider.Gemini,
                    usage=TokenUsage(
                        input_tokens_cache_write=5,
                        input_tokens_total=5,
                        output_tokens_total=11,
                    ),
                ),
            ),
        )

        tui._event_queue.put(_RunUpdateEvent(run_index=0, view=terminal_view))
        tui.pump_events()

        self.assertEqual(
            received,
            [
                {
                    Provider.Anthropic: TokenBill(
                        input_tokens_cache_read=10,
                        input_tokens_regular=3,
                        output_tokens_total=7,
                    ),
                    Provider.Gemini: TokenBill(
                        input_tokens_cache_write=5,
                        output_tokens_total=11,
                    ),
                }
            ],
        )
        self.assertTrue(tui._runs[0].terminal_callback_invoked)

    def test_register_terminal_callback_applies_to_immediately_terminal_launch(self) -> None:
        fn = _make_code_function("done")
        runtime = Runtime([fn], client_factories={})
        received: list[TokenBills] = []
        tui = TUI(runtime)

        def _capture(bills: TokenBills) -> None:
            received.append(bills)

        tui.register_terminal_callback(_capture)
        tui._open_launch_form(0)
        terminal_view = _make_view(fn, state=NodeState.Success, update_seqnum=1)
        node = SimpleNamespace(id=123, watch=Mock(return_value=terminal_view))

        with patch.object(runtime, "invoke", return_value=node), patch.object(
            runtime,
            "get_view",
            return_value=terminal_view,
        ):
            tui._submit_form()

        self.assertEqual(len(tui._runs), 1)
        self.assertEqual(received, [{}])
        self.assertTrue(tui._runs[0].terminal_callback_invoked)

        latest_view = tui._runs[0].latest_view
        assert latest_view is not None
        tui._event_queue.put(_RunUpdateEvent(run_index=0, view=latest_view))
        tui.pump_events()

        self.assertEqual(received, [{}])

    def test_register_terminal_callback_replays_existing_terminal_runs(self) -> None:
        fn = _make_code_function("done")
        runtime = Runtime([fn], client_factories={})
        received: list[TokenBills] = []
        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="run",
                fn=fn,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=mp.Event()),
                cancel_event=mp.Event(),
                latest_view=_make_view(fn, state=NodeState.Success, update_seqnum=1),
            )
        ]

        def _capture(bills: TokenBills) -> None:
            received.append(bills)

        tui.register_terminal_callback(_capture)

        self.assertEqual(received, [{}])
        self.assertTrue(tui._runs[0].terminal_callback_invoked)

    def test_selected_status_uses_latest_cached_view(self) -> None:
        fn = _make_code_function("cached")
        runtime = Runtime([fn], client_factories={})
        renderer = ConsoleRender(cancel_event=mp.Event())
        renderer.assign_view(_make_view(fn, state=NodeState.Waiting, update_seqnum=1))

        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="run",
                fn=fn,
                node=object(),  # type: ignore[arg-type]
                renderer=renderer,
                cancel_event=mp.Event(),
                latest_view=_make_view(fn, state=NodeState.Success, update_seqnum=2),
            )
        ]
        tui._selected_run = 0

        status = tui._selected_status()

        self.assertEqual(status.state, NodeState.Success)
        self.assertEqual(renderer.selected_tree_status().state, NodeState.Success)

    def test_assign_view_resets_cached_state_for_new_root(self) -> None:
        fn_a = _make_agent_function("agent_a")
        fn_b = _make_agent_function("agent_b")
        renderer = ConsoleRender(cancel_event=mp.Event(), follow=False)
        renderer.render_body(width=80, height=8, view=_make_agent_view(fn_a, state=NodeState.Running, update_seqnum=1), tick=0)
        renderer.navigate_down()

        renderer.assign_view(_make_agent_view(fn_b, state=NodeState.Running, update_seqnum=2))

        status = renderer.selected_tree_status()
        self.assertEqual(status.cursor_line, 0)
        self.assertEqual(status.total_lines, 0)

    def test_functions_pane_keeps_single_column_order_when_clipped(self) -> None:
        functions = [_make_code_function(f"fn{i}") for i in range(6)]
        runtime = Runtime(functions, client_factories={})
        tui = TUI(runtime)

        rendered = tui._render_functions_pane(width=40, rows=4)
        plain_rows = [_strip_ansi(line).rstrip() for line in rendered]

        self.assertEqual(plain_rows[1], "(0) fn0")
        self.assertEqual(plain_rows[2], "(1) fn1")
        self.assertEqual(plain_rows[3], "(2) fn2")
        self.assertNotIn("(3) fn3", plain_rows[1])

    def test_preferred_left_pane_width_targets_wider_layout(self) -> None:
        self.assertEqual(preferred_left_pane_width(100), 48)
        self.assertEqual(preferred_left_pane_width(80), 37)

    def test_runs_pane_groups_runs_by_function_order(self) -> None:
        fn0 = _make_code_function("fn0")
        fn1 = _make_code_function("fn1")
        fn2 = _make_code_function("fn2")
        runtime = Runtime([fn0, fn1, fn2], client_factories={})
        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="name1",
                fn=fn1,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=mp.Event()),
                cancel_event=mp.Event(),
                latest_view=_make_view(fn1, state=NodeState.Success, update_seqnum=1),
            ),
            _RunRecord(
                name="name2",
                fn=fn2,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=mp.Event()),
                cancel_event=mp.Event(),
                latest_view=_make_view(fn2, state=NodeState.Success, update_seqnum=2),
            ),
            _RunRecord(
                name="name0",
                fn=fn0,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=mp.Event()),
                cancel_event=mp.Event(),
                latest_view=_make_view(fn0, state=NodeState.Success, update_seqnum=3),
            ),
        ]
        tui._selected_run = 2

        rendered = tui._render_runs_pane(width=40, rows=8, tick=0)
        plain_rows = [_strip_ansi(line).rstrip() for line in rendered]

        self.assertIn(FG["gray"], rendered[0])
        self.assertEqual(plain_rows[0], "fn0")
        self.assertIn("name0", plain_rows[1])
        self.assertEqual(plain_rows[2], "")
        self.assertEqual(plain_rows[3], "fn1")
        self.assertIn("name1", plain_rows[4])
        self.assertEqual(plain_rows[5], "")
        self.assertEqual(plain_rows[6], "fn2")
        self.assertIn("name2", plain_rows[7])

    def test_run_navigation_uses_grouped_visible_order(self) -> None:
        fn0 = _make_code_function("fn0")
        fn1 = _make_code_function("fn1")
        fn2 = _make_code_function("fn2")
        runtime = Runtime([fn0, fn1, fn2], client_factories={})
        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="name1",
                fn=fn1,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=mp.Event()),
                cancel_event=mp.Event(),
                latest_view=_make_view(fn1, state=NodeState.Success, update_seqnum=1),
            ),
            _RunRecord(
                name="name2",
                fn=fn2,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=mp.Event()),
                cancel_event=mp.Event(),
                latest_view=_make_view(fn2, state=NodeState.Success, update_seqnum=2),
            ),
            _RunRecord(
                name="name0",
                fn=fn0,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=mp.Event()),
                cancel_event=mp.Event(),
                latest_view=_make_view(fn0, state=NodeState.Success, update_seqnum=3),
            ),
        ]
        tui._selected_run = 0

        tui.handle_key("shift_tab")
        self.assertEqual(tui._selected_run, 2)

        tui.handle_key("\t")
        self.assertEqual(tui._selected_run, 0)

        tui.handle_key("\t")
        self.assertEqual(tui._selected_run, 1)

    def test_run_header_click_does_not_change_selection_but_run_click_uses_grouped_order(self) -> None:
        fn0 = _make_code_function("fn0")
        fn1 = _make_code_function("fn1")
        runtime = Runtime([fn0, fn1], client_factories={})
        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="name1",
                fn=fn1,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=mp.Event()),
                cancel_event=mp.Event(),
                latest_view=_make_view(fn1, state=NodeState.Success, update_seqnum=1),
            ),
            _RunRecord(
                name="name0",
                fn=fn0,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=mp.Event()),
                cancel_event=mp.Event(),
                latest_view=_make_view(fn0, state=NodeState.Success, update_seqnum=2),
            ),
        ]
        tui._selected_run = 0
        tui.render_frame(TerminalSize(columns=100, lines=15), tick=0)

        tui.handle_mouse(SimpleNamespace(x=1, y=0, button="left"))
        self.assertEqual(tui._selected_run, 0)

        tui.handle_mouse(SimpleNamespace(x=1, y=1, button="left"))
        self.assertEqual(tui._selected_run, 1)

    def test_multi_pane_frame_uses_double_separator_and_right_margin(self) -> None:
        runtime = Runtime([_make_code_function("fn0")], client_factories={})
        tui = TUI(runtime)

        frame = _strip_ansi(tui.render_frame(TerminalSize(columns=80, lines=15), tick=0))
        first_line = frame.splitlines()[0]
        left_width = preferred_left_pane_width(80)

        self.assertEqual(first_line[left_width], "║")
        self.assertEqual(first_line[left_width + 1:left_width + 3], "  ")

    def test_right_pane_margin_blocks_mouse_until_content_origin(self) -> None:
        fn = _make_code_function("fn0")
        runtime = Runtime([fn], client_factories={})
        renderer = ConsoleRender(cancel_event=mp.Event())
        renderer.handle_mouse_event = Mock()  # type: ignore[method-assign]
        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="run",
                fn=fn,
                node=object(),  # type: ignore[arg-type]
                renderer=renderer,
                cancel_event=mp.Event(),
                latest_view=_make_view(fn, state=NodeState.Success, update_seqnum=1),
            )
        ]
        tui._selected_run = 0
        size = TerminalSize(columns=80, lines=15)
        left_width = preferred_left_pane_width(size.columns)
        right_origin = left_width + 3

        tui.render_frame(size, tick=0)
        tui.handle_mouse(SimpleNamespace(x=left_width + 1, y=0, button="left"))
        renderer.handle_mouse_event.assert_not_called()

        tui.handle_mouse(SimpleNamespace(x=right_origin, y=0, button="left"))
        renderer.handle_mouse_event.assert_called_once_with(0, 0, button="left")

    def test_launch_hit_uses_row_order_within_visible_function_label(self) -> None:
        functions = [_make_code_function(f"fn{i}") for i in range(6)]
        runtime = Runtime(functions, client_factories={})
        tui = TUI(runtime)

        tui._launch_function_hit(local_x=3, row=2, width=40, bottom_rows=4)

        self.assertIsNotNone(tui._form_state)
        assert tui._form_state is not None
        self.assertEqual(tui._form_state.fn_index, 1)

    def test_launch_hit_ignores_padding_after_function_label(self) -> None:
        functions = [_make_code_function(f"fn{i}") for i in range(6)]
        runtime = Runtime(functions, client_factories={})
        tui = TUI(runtime)

        tui._launch_function_hit(local_x=39, row=2, width=40, bottom_rows=4)

        self.assertIsNone(tui._form_state)

    def test_first_interrupt_requests_global_cancel_and_waits_for_terminal_update_before_exit(self) -> None:
        fn = _make_code_function("cancel_me")
        runtime = Runtime([fn], client_factories={})
        tui = TUI(runtime)
        cancel_event = mp.Event()
        received: list[TokenBills] = []

        def _capture(bills: TokenBills) -> None:
            received.append(bills)

        tui.register_terminal_callback(_capture)
        tui._runs = [
            _RunRecord(
                name="run",
                fn=fn,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=cancel_event),
                cancel_event=cancel_event,
                latest_view=_make_view(fn, state=NodeState.Running, update_seqnum=1),
            )
        ]
        tui._selected_run = 0

        self.assertFalse(tui.handle_interrupt())
        self.assertTrue(cancel_event.is_set())
        self.assertFalse(tui.should_exit())

        rendered = tui.render_frame(TerminalSize(columns=100, lines=12), tick=0)

        self.assertIn("^C:force quit", _strip_ansi(rendered))
        self.assertFalse(tui.should_exit())

        tui._event_queue.put(
            _RunUpdateEvent(run_index=0, view=_make_view(fn, state=NodeState.Canceled, update_seqnum=2))
        )
        tui.pump_events()

        rendered = tui.render_frame(TerminalSize(columns=100, lines=12), tick=1)

        self.assertEqual(received, [{}])
        self.assertTrue(tui.should_exit())

        with self.assertRaises(KeyboardInterrupt):
            tui.handle_interrupt()

    def test_second_interrupt_makes_final_billing_attempt_before_forced_exit(self) -> None:
        fn = _make_code_function("cancel_me")
        runtime = Runtime([fn], client_factories={})
        tui = TUI(runtime)
        cancel_event = mp.Event()
        received: list[TokenBills] = []
        node = SimpleNamespace(id=7)

        def _capture(bills: TokenBills) -> None:
            received.append(bills)

        tui.register_terminal_callback(_capture)
        tui._runs = [
            _RunRecord(
                name="run",
                fn=fn,
                node=node,  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=cancel_event),
                cancel_event=cancel_event,
                latest_view=_make_view(fn, state=NodeState.Running, update_seqnum=1),
            )
        ]

        self.assertFalse(tui.handle_interrupt())

        with patch.object(runtime, "get_view", return_value=_make_view(fn, state=NodeState.Canceled, update_seqnum=2)):
            with self.assertRaises(KeyboardInterrupt):
                tui.handle_interrupt()

        self.assertEqual(received, [{}])
        self.assertTrue(tui._runs[0].terminal_callback_invoked)

    def test_launch_form_too_small_keeps_bottom_bar(self) -> None:
        fn = _make_code_function("form_fn")
        runtime = Runtime([fn], client_factories={})
        tui = TUI(runtime)
        tui._open_launch_form(0)

        rendered = tui.render_frame(TerminalSize(columns=60, lines=8), tick=0)
        lines = [_strip_ansi(line).rstrip() for line in rendered.splitlines()]

        self.assertIn("Terminal too small for launch form.", lines[0])
        self.assertIn("Resize to continue.", lines[1])
        self.assertTrue(any("^C:cancel all" in line for line in lines))

    def test_empty_tui_omits_fake_tree_position_and_tree_shortcuts(self) -> None:
        runtime = Runtime([_make_code_function("fn0")], client_factories={})
        tui = TUI(runtime)

        rendered = tui.render_frame(TerminalSize(columns=80, lines=15), tick=0)
        bottom_bar = _strip_ansi(rendered.splitlines()[-1])

        self.assertNotIn("0/0", bottom_bar)
        self.assertIn("0-9:launch", bottom_bar)
        self.assertNotIn("jk:tree", bottom_bar)
        self.assertNotIn("C:cancel tree", bottom_bar)

    def test_multi_pane_requires_height_for_full_function_list(self) -> None:
        functions = [_make_code_function(f"fn{i}") for i in range(10)]
        runtime = Runtime(functions, client_factories={})
        tui = TUI(runtime)

        rendered = tui.render_frame(TerminalSize(columns=65, lines=10), tick=0)
        lines = [_strip_ansi(line).rstrip() for line in rendered.splitlines()]

        self.assertIn("Terminal too small for TUI.", lines[0])
        self.assertIn("Resize to at least 67x15.", lines[1])

    def test_launch_form_bottom_bar_drops_stale_tree_position(self) -> None:
        fn = _make_code_function("form_fn")
        runtime = Runtime([fn], client_factories={})
        renderer = ConsoleRender(cancel_event=mp.Event())
        renderer.render_body(
            width=40,
            height=8,
            view=NodeView(
                id=1,
                fn=fn,
                inputs={},
                state=NodeState.Running,
                outputs=None,
                exception=None,
                children=(),
                usage=None,
                transcript=(),
                started_at=0.0,
                ended_at=None,
                update_seqnum=1,
            ),
            tick=0,
        )

        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="run",
                fn=fn,
                node=object(),  # type: ignore[arg-type]
                renderer=renderer,
                cancel_event=mp.Event(),
                latest_view=_make_view(fn, state=NodeState.Success, update_seqnum=2),
            )
        ]
        tui._selected_run = 0
        tui._open_launch_form(0)

        rendered = tui.render_frame(TerminalSize(columns=80, lines=15), tick=0)
        bottom_bar = _strip_ansi(rendered.splitlines()[-1])

        self.assertNotIn("0/0", bottom_bar)
        self.assertNotIn("1/1", bottom_bar)

    def test_launch_form_too_small_bottom_bar_omits_tree_position(self) -> None:
        fn = _make_code_function("form_fn")
        runtime = Runtime([fn], client_factories={})
        renderer = ConsoleRender(cancel_event=mp.Event())
        renderer.render_body(
            width=40,
            height=8,
            view=_make_view(fn, state=NodeState.Success, update_seqnum=1),
            tick=0,
        )

        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="run",
                fn=fn,
                node=object(),  # type: ignore[arg-type]
                renderer=renderer,
                cancel_event=mp.Event(),
                latest_view=_make_view(fn, state=NodeState.Success, update_seqnum=2),
            )
        ]
        tui._selected_run = 0
        tui._open_launch_form(0)

        rendered = tui.render_frame(TerminalSize(columns=60, lines=8), tick=0)
        bottom_bar = _strip_ansi(rendered.splitlines()[-1])

        self.assertNotIn("0/0", bottom_bar)
        self.assertNotIn("1/1", bottom_bar)

    def test_cancel_selected_run_only_sets_selected_tree_event(self) -> None:
        fn = _make_code_function("cancel_me")
        runtime = Runtime([fn], client_factories={})
        cancel_a = mp.Event()
        cancel_b = mp.Event()
        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="run-a",
                fn=fn,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=cancel_a),
                cancel_event=cancel_a,
                latest_view=_make_view(fn, state=NodeState.Running, update_seqnum=1),
            ),
            _RunRecord(
                name="run-b",
                fn=fn,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=cancel_b),
                cancel_event=cancel_b,
                latest_view=_make_view(fn, state=NodeState.Running, update_seqnum=2),
            ),
        ]
        tui._selected_run = 1

        self.assertFalse(tui.handle_key("C"))
        self.assertFalse(cancel_a.is_set())
        self.assertTrue(cancel_b.is_set())

    def test_runs_pane_uses_cancel_pending_color_for_nonterminal_run(self) -> None:
        fn = _make_code_function("cancel_me")
        runtime = Runtime([fn], client_factories={})
        cancel_event = mp.Event()
        cancel_event.set()
        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="run",
                fn=fn,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=cancel_event),
                cancel_event=cancel_event,
                latest_view=_make_view(fn, state=NodeState.Running, update_seqnum=1),
            )
        ]

        rendered = tui._render_runs_pane(width=40, rows=4, tick=0)

        self.assertIn(FG["magenta"], rendered[1])

    def test_hidden_terminal_update_marks_run_unread_and_bolds_row(self) -> None:
        fn_hidden = _make_code_function("hidden_fn")
        fn_selected = _make_code_function("selected_fn")
        runtime = Runtime([fn_hidden, fn_selected], client_factories={})
        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="hidden",
                fn=fn_hidden,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=mp.Event()),
                cancel_event=mp.Event(),
                latest_view=_make_view(fn_hidden, state=NodeState.Running, update_seqnum=1),
            ),
            _RunRecord(
                name="selected",
                fn=fn_selected,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=mp.Event()),
                cancel_event=mp.Event(),
                latest_view=_make_view(fn_selected, state=NodeState.Running, update_seqnum=2),
            ),
        ]
        tui._selected_run = 1
        tui.render_frame(TerminalSize(columns=80, lines=15), tick=0)
        tui._event_queue.put(
            _RunUpdateEvent(run_index=0, view=_make_view(fn_hidden, state=NodeState.Success, update_seqnum=3))
        )

        tui.pump_events()
        rendered = tui._render_runs_pane(width=40, rows=6, tick=0)

        self.assertTrue(tui._runs[0].unread)
        self.assertIn("\x1b[1mhidden", rendered[1])

    def test_hidden_terminal_update_invokes_callback(self) -> None:
        fn_hidden = _make_code_function("hidden_fn")
        fn_selected = _make_code_function("selected_fn")
        runtime = Runtime([fn_hidden, fn_selected], client_factories={})
        received: list[TokenBills] = []
        tui = TUI(runtime)

        def _capture(bills: TokenBills) -> None:
            received.append(bills)

        tui.register_terminal_callback(_capture)
        tui._runs = [
            _RunRecord(
                name="hidden",
                fn=fn_hidden,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=mp.Event()),
                cancel_event=mp.Event(),
                latest_view=_make_view(fn_hidden, state=NodeState.Running, update_seqnum=1),
            ),
            _RunRecord(
                name="selected",
                fn=fn_selected,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=mp.Event()),
                cancel_event=mp.Event(),
                latest_view=_make_view(fn_selected, state=NodeState.Running, update_seqnum=2),
            ),
        ]
        tui._selected_run = 1
        tui.render_frame(TerminalSize(columns=80, lines=15), tick=0)

        tui._event_queue.put(
            _RunUpdateEvent(run_index=0, view=_make_view(fn_hidden, state=NodeState.Success, update_seqnum=3))
        )
        tui.pump_events()

        self.assertEqual(received, [{}])
        self.assertTrue(tui._runs[0].unread)

    def test_terminal_callback_failure_is_logged_without_breaking_pump(self) -> None:
        fn = _make_code_function("boom")
        runtime = Runtime([fn], client_factories={})
        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="run",
                fn=fn,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=mp.Event()),
                cancel_event=mp.Event(),
                latest_view=_make_view(fn, state=NodeState.Running, update_seqnum=1),
            )
        ]

        def _boom(bills: TokenBills) -> None:
            del bills
            raise RuntimeError("boom")

        tui.register_terminal_callback(_boom)
        tui._event_queue.put(
            _RunUpdateEvent(run_index=0, view=_make_view(fn, state=NodeState.Success, update_seqnum=2))
        )

        with patch("netflux.tui.tui.logging.exception") as log_exception:
            self.assertTrue(tui.pump_events())

        log_exception.assert_called_once()
        self.assertTrue(tui._runs[0].terminal_callback_invoked)

    def test_navigating_to_unread_run_marks_it_read(self) -> None:
        fn0 = _make_code_function("fn0")
        fn1 = _make_code_function("fn1")
        runtime = Runtime([fn0, fn1], client_factories={})
        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="run-0",
                fn=fn0,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=mp.Event()),
                cancel_event=mp.Event(),
                latest_view=_make_view(fn0, state=NodeState.Success, update_seqnum=1),
                manual_unread=True,
            ),
            _RunRecord(
                name="run-1",
                fn=fn1,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=mp.Event()),
                cancel_event=mp.Event(),
                latest_view=_make_view(fn1, state=NodeState.Success, update_seqnum=2),
            ),
        ]
        tui._selected_run = 1
        tui.render_frame(TerminalSize(columns=80, lines=15), tick=0)

        tui.handle_key("shift_tab")

        self.assertEqual(tui._selected_run, 0)
        self.assertFalse(tui._runs[0].unread)

    def test_mark_unread_key_toggles_selected_run(self) -> None:
        fn = _make_code_function("toggle")
        runtime = Runtime([fn], client_factories={})
        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="run",
                fn=fn,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=mp.Event()),
                cancel_event=mp.Event(),
                latest_view=_make_view(fn, state=NodeState.Running, update_seqnum=1),
            )
        ]
        tui._selected_run = 0
        tui.render_frame(TerminalSize(columns=80, lines=15), tick=0)

        tui.handle_key("u")
        self.assertTrue(tui._runs[0].manual_unread)
        self.assertTrue(tui._runs[0].unread)

        tui.handle_key("u")
        self.assertFalse(tui._runs[0].manual_unread)
        self.assertFalse(tui._runs[0].auto_unread)
        self.assertFalse(tui._runs[0].unread)

    def test_manual_unread_survives_terminal_transition_in_view(self) -> None:
        fn = _make_agent_function("agent")
        runtime = Runtime([fn], client_factories={})
        running = _make_agent_view(fn, state=NodeState.Running, update_seqnum=1)
        terminal = _make_agent_view(fn, state=NodeState.Success, update_seqnum=2)
        renderer = ConsoleRender(cancel_event=mp.Event(), follow=False)
        renderer.render_body(width=80, height=8, view=running, tick=0)
        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="run",
                fn=fn,
                node=object(),  # type: ignore[arg-type]
                renderer=renderer,
                cancel_event=mp.Event(),
                latest_view=running,
            )
        ]
        tui._selected_run = 0
        tui.render_frame(TerminalSize(columns=80, lines=15), tick=0)

        tui.handle_key("u")
        tui._event_queue.put(_RunUpdateEvent(run_index=0, view=terminal))

        tui.pump_events()
        tui.render_frame(TerminalSize(columns=80, lines=15), tick=0)

        self.assertTrue(tui._runs[0].manual_unread)
        self.assertTrue(tui._runs[0].unread)
        self.assertTrue(tui._runs[0].terminal_browse_applied)

    def test_auto_unread_clears_when_selected_tree_returns_after_launch_form(self) -> None:
        fn = _make_code_function("form_hidden")
        runtime = Runtime([fn], client_factories={})
        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="run",
                fn=fn,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=mp.Event()),
                cancel_event=mp.Event(),
                latest_view=_make_view(fn, state=NodeState.Running, update_seqnum=1),
            )
        ]
        tui._selected_run = 0
        tui.render_frame(TerminalSize(columns=80, lines=15), tick=0)
        tui._open_launch_form(0)
        tui._event_queue.put(_RunUpdateEvent(run_index=0, view=_make_view(fn, state=NodeState.Success, update_seqnum=2)))

        tui.pump_events()
        self.assertTrue(tui._runs[0].auto_unread)
        self.assertTrue(tui._runs[0].unread)

        tui.handle_key("escape")

        self.assertFalse(tui._runs[0].auto_unread)
        self.assertFalse(tui._runs[0].unread)

    def test_manual_unread_survives_launch_form_visibility_change(self) -> None:
        fn = _make_code_function("manual_form")
        runtime = Runtime([fn], client_factories={})
        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="run",
                fn=fn,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=mp.Event()),
                cancel_event=mp.Event(),
                latest_view=_make_view(fn, state=NodeState.Success, update_seqnum=1),
            )
        ]
        tui._selected_run = 0
        tui.render_frame(TerminalSize(columns=80, lines=15), tick=0)

        tui.handle_key("u")
        tui._open_launch_form(0)
        tui.handle_key("escape")

        self.assertTrue(tui._runs[0].manual_unread)
        self.assertTrue(tui._runs[0].unread)

    def test_selected_tree_bottom_bar_shows_mark_unread_shortcut(self) -> None:
        fn = _make_code_function("shortcut")
        runtime = Runtime([fn], client_factories={})
        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="run",
                fn=fn,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=mp.Event()),
                cancel_event=mp.Event(),
                latest_view=_make_view(fn, state=NodeState.Success, update_seqnum=1),
            )
        ]
        tui._selected_run = 0

        rendered = tui.render_frame(TerminalSize(columns=80, lines=15), tick=0)

        self.assertRegex(_strip_ansi(rendered.splitlines()[-1]), r"(^|  )u(?:(:unread)|  )")

    def test_terminal_run_hides_cancel_tree_shortcut(self) -> None:
        fn = _make_code_function("done")
        runtime = Runtime([fn], client_factories={})
        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="run",
                fn=fn,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=mp.Event()),
                cancel_event=mp.Event(),
                latest_view=_make_view(fn, state=NodeState.Success, update_seqnum=1),
            )
        ]
        tui._selected_run = 0

        rendered = tui.render_frame(TerminalSize(columns=80, lines=15), tick=0)
        bottom_bar = _strip_ansi(rendered.splitlines()[-1])

        self.assertNotIn("C:cancel tree", bottom_bar)

    def test_terminal_run_shows_result_shortcuts_when_root_result_available(self) -> None:
        fn = _make_code_function("done")
        runtime = Runtime([fn], client_factories={})
        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="run",
                fn=fn,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=mp.Event()),
                cancel_event=mp.Event(),
                latest_view=_make_output_view(
                    fn,
                    output="# Summary\n\n- first item",
                    update_seqnum=1,
                ),
            )
        ]
        tui._selected_run = 0

        rendered = tui.render_frame(TerminalSize(columns=100, lines=15), tick=0)
        bottom_bar = _strip_ansi(rendered.splitlines()[-1])

        self.assertRegex(bottom_bar, r"(^|  )c(?:(:copy result|:copy)|  )")
        self.assertRegex(bottom_bar, r"(^|  )r(?:(:show result|:result)|  )")

    def test_copy_result_failure_shows_clipboard_install_hint_in_bottom_bar(self) -> None:
        fn = _make_code_function("done")
        runtime = Runtime([fn], client_factories={})
        renderer = ConsoleRender(cancel_event=mp.Event())
        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="run",
                fn=fn,
                node=object(),  # type: ignore[arg-type]
                renderer=renderer,
                cancel_event=mp.Event(),
                latest_view=_make_output_view(
                    fn,
                    output="# Summary\n\n- first item",
                    update_seqnum=1,
                ),
            )
        ]
        tui._selected_run = 0

        with patch.object(
            renderer,
            "copy_terminal_result_with_feedback",
            return_value=(False, "Clipboard unavailable. Install wl-copy, xclip, or xsel."),
        ):
            self.assertFalse(tui.handle_key("c"))

        rendered = tui.render_frame(TerminalSize(columns=140, lines=15), tick=0)
        bottom_bar = _strip_ansi(rendered.splitlines()[-1])

        self.assertIn("Install wl-copy, xclip, or xsel", bottom_bar)

    def test_multi_pane_key_remap_routes_agent_and_all_actions(self) -> None:
        fn = _make_code_function("done")
        runtime = Runtime([fn], client_factories={})
        renderer = ConsoleRender(cancel_event=mp.Event())
        renderer.apply_action = Mock()  # type: ignore[method-assign]
        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="run",
                fn=fn,
                node=object(),  # type: ignore[arg-type]
                renderer=renderer,
                cancel_event=mp.Event(),
                latest_view=_make_view(fn, state=NodeState.Success, update_seqnum=1),
            )
        ]
        tui._selected_run = 0

        tui.handle_key("a")
        tui.handle_key("e")
        tui.handle_key("E")
        tui.handle_key("r")

        self.assertEqual(
            [call.args[0] for call in renderer.apply_action.call_args_list],
            ["collapse_agent", "expand_all", "collapse_all", "focus_result"],
        )

    def test_terminal_run_ignores_cancel_tree_key(self) -> None:
        fn = _make_code_function("done")
        runtime = Runtime([fn], client_factories={})
        cancel_event = mp.Event()
        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="run",
                fn=fn,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=cancel_event),
                cancel_event=cancel_event,
                latest_view=_make_view(fn, state=NodeState.Success, update_seqnum=1),
            )
        ]
        tui._selected_run = 0

        self.assertFalse(tui.handle_key("C"))
        self.assertFalse(cancel_event.is_set())

    def test_launch_form_parses_declared_types_before_invoke(self) -> None:
        def typed(
            ctx: RunContext,
            *,
            count: int,
            flag: bool,
            ratio: float | None = None,
        ) -> str:
            return (
                f"{count}|{flag}|{ratio}|"
                f"{type(count).__name__}|{type(flag).__name__}|{type(ratio).__name__}"
            )

        fn = CodeFunction(
            name="typed",
            desc="typed launch target",
            args=[
                FunctionArg("count", int),
                FunctionArg("flag", bool),
                FunctionArg("ratio", float, optional=True),
            ],
            callable=typed,
            uses=[],
        )
        runtime = Runtime([fn], client_factories={})
        tui = TUI(runtime)
        tui._open_launch_form(0)
        assert tui._form_state is not None
        tui._form_state.fields[0].value = "typed run"
        tui._form_state.fields[1].value = "7"
        tui._form_state.fields[2].value = "false"
        tui._form_state.fields[3].value = "1.5"

        tui._submit_form()

        self.assertIsNone(tui._form_state)
        self.assertEqual(len(tui._runs), 1)
        self.assertEqual(
            tui._runs[0].node.result(),
            "7|False|1.5|int|bool|float",
        )

    def test_launch_form_strips_blank_optional_fields(self) -> None:
        fn = CodeFunction(
            name="blank_optional",
            desc="blank optional",
            args=[FunctionArg("value", str, optional=True)],
            callable=lambda ctx, *, value=None: value,
            uses=[],
        )
        runtime = Runtime([fn], client_factories={})
        tui = TUI(runtime)
        tui._open_launch_form(0)
        assert tui._form_state is not None
        tui._form_state.fields[1].value = "   "

        tui._submit_form()

        self.assertEqual(len(tui._runs), 1)
        self.assertEqual(tui._runs[0].node.inputs, {"value": None})
        self.assertIsNone(tui._runs[0].node.result())

    def test_launch_form_accepts_literal_j_and_k(self) -> None:
        fn = CodeFunction(
            name="texty",
            desc="text input target",
            args=[FunctionArg("value", str)],
            callable=lambda ctx, *, value: value,
            uses=[],
        )
        runtime = Runtime([fn], client_factories={})
        tui = TUI(runtime)
        tui._open_launch_form(0)
        assert tui._form_state is not None

        tui.handle_key("j")
        tui.handle_key("k")

        self.assertEqual(tui._form_state.cursor, 0)
        self.assertEqual(tui._form_state.fields[0].value, "jk")

    def test_launch_form_newline_submits(self) -> None:
        fn = _make_code_function("newline_submit")
        runtime = Runtime([fn], client_factories={})
        tui = TUI(runtime)
        tui._open_launch_form(0)
        assert tui._form_state is not None
        tui._form_state.cursor = len(tui._form_state.fields)

        tui.handle_key("\n")

        self.assertIsNone(tui._form_state)
        self.assertEqual(len(tui._runs), 1)

    def test_launch_form_missing_required_arg_shows_error_and_does_not_launch(self) -> None:
        def typed(ctx: RunContext, *, count: int, flag: bool) -> str:
            return f"{count}:{flag}"

        fn = CodeFunction(
            name="typed",
            desc="typed launch target",
            args=[
                FunctionArg("count", int),
                FunctionArg("flag", bool),
            ],
            callable=typed,
            uses=[],
        )
        runtime = Runtime([fn], client_factories={})
        tui = TUI(runtime)
        tui._open_launch_form(0)
        assert tui._form_state is not None
        tui._form_state.fields[0].value = "typed run"
        tui._form_state.fields[2].value = "true"

        tui._submit_form()

        self.assertEqual(len(tui._runs), 0)
        self.assertIsNotNone(tui._form_state)
        assert tui._form_state is not None
        self.assertIn("Missing required arg(s)", tui._form_state.error)

    def test_switching_runs_preserves_each_renderer_state(self) -> None:
        runtime = Runtime([_make_code_function("root")], client_factories={})
        fn_a = _make_agent_function("agent_a")
        fn_b = _make_agent_function("agent_b")
        view_a = _make_agent_view(fn_a, state=NodeState.Running, update_seqnum=1)
        view_b = _make_agent_view(fn_b, state=NodeState.Running, update_seqnum=2)
        renderer_a = ConsoleRender(cancel_event=mp.Event(), follow=False)
        renderer_b = ConsoleRender(cancel_event=mp.Event(), follow=False)
        renderer_a.render_body(width=80, height=8, view=view_a, tick=0)
        renderer_b.render_body(width=80, height=8, view=view_b, tick=0)
        renderer_a.navigate_down()

        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="run-a",
                fn=fn_a,
                node=object(),  # type: ignore[arg-type]
                renderer=renderer_a,
                cancel_event=mp.Event(),
                latest_view=view_a,
            ),
            _RunRecord(
                name="run-b",
                fn=fn_b,
                node=object(),  # type: ignore[arg-type]
                renderer=renderer_b,
                cancel_event=mp.Event(),
                latest_view=view_b,
            ),
        ]
        tui._selected_run = 0

        tui.render_frame(TerminalSize(columns=80, lines=15), tick=0)
        tui.handle_key("\t")
        tui.render_frame(TerminalSize(columns=80, lines=15), tick=0)
        tui.handle_key("shift_tab")
        tui.render_frame(TerminalSize(columns=80, lines=15), tick=0)

        self.assertEqual(tui._selected_status().cursor_line, 2)

    def test_terminal_update_resets_run_to_post_completion_browse_state(self) -> None:
        runtime = Runtime([_make_code_function("root")], client_factories={})
        fn = _make_agent_function("agent")
        running = NodeView(
            id=1,
            fn=fn,
            inputs={},
            state=NodeState.Running,
            outputs=None,
            exception=None,
            children=(),
            usage=None,
            transcript=(ModelTextPart(text="model output"),),
            started_at=0.0,
            ended_at=None,
            update_seqnum=1,
        )
        terminal = NodeView(
            id=1,
            fn=fn,
            inputs={},
            state=NodeState.Success,
            outputs=None,
            exception=None,
            children=(),
            usage=None,
            transcript=(ModelTextPart(text="model output"),),
            started_at=0.0,
            ended_at=0.0,
            update_seqnum=2,
        )
        renderer = ConsoleRender(cancel_event=mp.Event(), follow=False)
        renderer.render_body(width=80, height=8, view=running, tick=0)
        renderer.navigate_down()

        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="run",
                fn=fn,
                node=object(),  # type: ignore[arg-type]
                renderer=renderer,
                cancel_event=mp.Event(),
                latest_view=running,
            )
        ]
        tui._selected_run = 0
        tui._event_queue.put(_RunUpdateEvent(run_index=0, view=terminal))

        tui.pump_events()
        tui.render_frame(TerminalSize(columns=80, lines=15), tick=0)

        self.assertEqual(renderer.selected_tree_status().cursor_line, 1)
        self.assertTrue(tui._runs[0].terminal_browse_applied)

    def test_hidden_terminal_update_preserves_renderer_state(self) -> None:
        runtime = Runtime([_make_code_function("root")], client_factories={})
        fn = _make_agent_function("agent")
        running = _make_agent_view(fn, state=NodeState.Running, update_seqnum=1)
        terminal = _make_agent_view(fn, state=NodeState.Success, update_seqnum=2)
        hidden_renderer = ConsoleRender(cancel_event=mp.Event(), follow=False)
        selected_renderer = ConsoleRender(cancel_event=mp.Event(), follow=False)
        hidden_renderer.render_body(width=80, height=8, view=running, tick=0)
        hidden_renderer.navigate_down()

        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="hidden",
                fn=fn,
                node=object(),  # type: ignore[arg-type]
                renderer=hidden_renderer,
                cancel_event=mp.Event(),
                latest_view=running,
            ),
            _RunRecord(
                name="selected",
                fn=fn,
                node=object(),  # type: ignore[arg-type]
                renderer=selected_renderer,
                cancel_event=mp.Event(),
                latest_view=running,
            ),
        ]
        tui._selected_run = 1
        tui._event_queue.put(_RunUpdateEvent(run_index=0, view=terminal))

        tui.pump_events()
        tui.render_frame(TerminalSize(columns=80, lines=15), tick=0)

        self.assertEqual(hidden_renderer.selected_tree_status().cursor_line, 2)
        self.assertFalse(tui._runs[0].terminal_browse_applied)

    def test_selected_terminal_update_during_launch_form_preserves_renderer_state(self) -> None:
        runtime = Runtime([_make_code_function("root")], client_factories={})
        fn = _make_agent_function("agent")
        running = NodeView(
            id=1,
            fn=fn,
            inputs={},
            state=NodeState.Running,
            outputs=None,
            exception=None,
            children=(),
            usage=None,
            transcript=(ModelTextPart(text="model output"),),
            started_at=0.0,
            ended_at=None,
            update_seqnum=1,
        )
        terminal = NodeView(
            id=1,
            fn=fn,
            inputs={},
            state=NodeState.Success,
            outputs=None,
            exception=None,
            children=(),
            usage=None,
            transcript=(ModelTextPart(text="model output"),),
            started_at=0.0,
            ended_at=0.0,
            update_seqnum=2,
        )
        renderer = ConsoleRender(cancel_event=mp.Event(), follow=False)
        renderer.render_body(width=80, height=8, view=running, tick=0)
        renderer.navigate_down()

        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="run",
                fn=fn,
                node=object(),  # type: ignore[arg-type]
                renderer=renderer,
                cancel_event=mp.Event(),
                latest_view=running,
            )
        ]
        tui._selected_run = 0
        tui._open_launch_form(0)
        tui._event_queue.put(_RunUpdateEvent(run_index=0, view=terminal))

        tui.pump_events()
        tui.render_frame(TerminalSize(columns=80, lines=15), tick=0)
        tui._form_state = None
        tui.render_frame(TerminalSize(columns=80, lines=15), tick=0)

        self.assertEqual(renderer.selected_tree_status().cursor_line, 2)
        self.assertFalse(tui._runs[0].terminal_browse_applied)

    def test_selected_terminal_update_before_too_small_render_preserves_renderer_state(self) -> None:
        runtime = Runtime([_make_code_function("root")], client_factories={})
        fn = _make_agent_function("agent")
        running = NodeView(
            id=1,
            fn=fn,
            inputs={},
            state=NodeState.Running,
            outputs=None,
            exception=None,
            children=(),
            usage=None,
            transcript=(ModelTextPart(text="model output"),),
            started_at=0.0,
            ended_at=None,
            update_seqnum=1,
        )
        terminal = NodeView(
            id=1,
            fn=fn,
            inputs={},
            state=NodeState.Success,
            outputs=None,
            exception=None,
            children=(),
            usage=None,
            transcript=(ModelTextPart(text="model output"),),
            started_at=0.0,
            ended_at=0.0,
            update_seqnum=2,
        )
        renderer = ConsoleRender(cancel_event=mp.Event(), follow=False)
        renderer.render_body(width=80, height=8, view=running, tick=0)
        renderer.navigate_down()

        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="run",
                fn=fn,
                node=object(),  # type: ignore[arg-type]
                renderer=renderer,
                cancel_event=mp.Event(),
                latest_view=running,
            )
        ]
        tui._selected_run = 0
        tui.render_frame(TerminalSize(columns=80, lines=15), tick=0)
        tui._event_queue.put(_RunUpdateEvent(run_index=0, view=terminal))

        tui.pump_events()
        tui.render_frame(TerminalSize(columns=64, lines=14), tick=0)
        tui.render_frame(TerminalSize(columns=80, lines=15), tick=0)

        self.assertEqual(renderer.selected_tree_status().cursor_line, 2)
        self.assertFalse(tui._runs[0].terminal_browse_applied)

    def test_left_pane_ignores_non_left_mouse_buttons(self) -> None:
        runtime = Runtime([_make_code_function("fn0")], client_factories={})
        tui = TUI(runtime)
        tui.render_frame(TerminalSize(columns=80, lines=15), tick=0)

        tui.handle_mouse(SimpleNamespace(x=1, y=11, button="wheel_down"))
        self.assertIsNone(tui._form_state)

        tui.handle_mouse(SimpleNamespace(x=1, y=11, button="left"))
        self.assertIsNotNone(tui._form_state)

    def test_launch_form_scrolls_to_submit_when_form_is_tall(self) -> None:
        args = [FunctionArg(f"a{i}", str) for i in range(12)]
        signature = ", ".join(f"a{i}" for i in range(12))
        namespace: dict[str, object] = {}
        exec(
            "def _call(ctx, *, " + signature + "):\n"
            "    return 'ok'\n",
            namespace,
        )
        fn = CodeFunction(
            name="many_args",
            desc="many args target",
            args=args,
            callable=namespace["_call"],  # type: ignore[arg-type]
            uses=[],
        )
        runtime = Runtime([fn], client_factories={})
        tui = TUI(runtime)
        tui._open_launch_form(0)
        assert tui._form_state is not None
        tui._form_state.cursor = len(tui._form_state.fields)

        rendered = tui.render_frame(TerminalSize(columns=80, lines=15), tick=0)
        plain_lines = [_strip_ansi(line) for line in rendered.splitlines()]

        self.assertTrue(any("[Submit]" in line for line in plain_lines))

    def test_launch_form_respects_multiline_descriptions(self) -> None:
        fn = CodeFunction(
            name="multiline",
            desc="first line\nsecond line",
            args=[FunctionArg("value", str)],
            callable=lambda ctx, *, value: value,
            uses=[],
        )
        runtime = Runtime([fn], client_factories={})
        tui = TUI(runtime)
        tui._open_launch_form(0)

        rendered = tui.render_frame(TerminalSize(columns=80, lines=15), tick=0)
        plain_lines = [_strip_ansi(line).rstrip() for line in rendered.splitlines()]
        _, _, header_rows = tui._form_item_window(TerminalSize(columns=80, lines=15))

        self.assertEqual(
            header_rows,
            len(tui._launch_form_desc_lines(fn, 80)) + len(tui._launch_form_arg_lines(fn, 80)) + 2,
        )
        self.assertIn("first line", plain_lines[1])
        self.assertIn("second line", plain_lines[2])
        self.assertIn("Arguments", plain_lines[3])
        self.assertIn("value (str)", plain_lines[4])

    def test_launch_form_wraps_long_single_line_descriptions(self) -> None:
        fn = CodeFunction(
            name="wrapped_desc",
            desc=" ".join(["alpha"] * 20),
            args=[FunctionArg("value", str)],
            callable=lambda ctx, *, value: value,
            uses=[],
        )
        runtime = Runtime([fn], client_factories={})
        tui = TUI(runtime)
        tui._open_launch_form(0)

        size = TerminalSize(columns=67, lines=15)
        wrapped_desc = tui._launch_form_desc_lines(fn, size.columns)
        rendered = tui.render_frame(size, tick=0)
        plain_lines = [_strip_ansi(line).rstrip() for line in rendered.splitlines()]
        _, _, header_rows = tui._form_item_window(size)

        self.assertGreater(len(wrapped_desc), 1)
        self.assertEqual(
            header_rows,
            len(wrapped_desc) + len(tui._launch_form_arg_lines(fn, size.columns)) + 2,
        )
        for idx, desc_line in enumerate(wrapped_desc, start=1):
            self.assertEqual(plain_lines[idx].strip(), desc_line)

    def test_launch_form_shows_arg_descriptions_and_optional_marker(self) -> None:
        fn = CodeFunction(
            name="metadata",
            desc="launch metadata target",
            args=[
                FunctionArg("count", int, "How many items to process."),
                FunctionArg("ratio", float, "Blend ratio for the run.", optional=True),
            ],
            callable=lambda ctx, *, count, ratio=None: f"{count}:{ratio}",
            uses=[],
        )
        runtime = Runtime([fn], client_factories={})
        tui = TUI(runtime)
        tui._open_launch_form(0)

        rendered = tui.render_frame(TerminalSize(columns=80, lines=18), tick=0)
        plain_lines = [_strip_ansi(line).rstrip() for line in rendered.splitlines()]

        self.assertTrue(any("Arguments" == line.strip() for line in plain_lines))
        self.assertTrue(any("count (int): How many items to process." in line for line in plain_lines))
        self.assertTrue(any("ratio (float) [optional]: Blend ratio for the run." in line for line in plain_lines))

    def test_launch_form_history_uses_most_recent_20_runs_for_same_function(self) -> None:
        fn = CodeFunction(
            name="history_target",
            desc="history target",
            args=[FunctionArg("value", int)],
            callable=lambda ctx, *, value: value,
            uses=[],
        )
        other = _make_code_function("other")
        runtime = Runtime([fn, other], client_factories={})
        tui = TUI(runtime)

        for idx in range(22):
            tui._runs.append(
                _RunRecord(
                    name=f"run{idx}",
                    fn=fn,
                    node=object(),  # type: ignore[arg-type]
                    renderer=ConsoleRender(cancel_event=mp.Event()),
                    cancel_event=mp.Event(),
                    latest_view=_make_view(
                        fn,
                        state=NodeState.Success,
                        update_seqnum=idx + 1,
                        inputs={"value": idx},
                    ),
                )
            )
        tui._runs.append(
            _RunRecord(
                name="other-run",
                fn=other,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=mp.Event()),
                cancel_event=mp.Event(),
                latest_view=_make_view(other, state=NodeState.Success, update_seqnum=99),
            )
        )
        tui._open_launch_form(0)

        history = tui._launch_form_history()

        self.assertEqual(len(history), 20)
        self.assertEqual(history[0].name, "run21")
        self.assertEqual(history[0].inputs["value"], 21)
        self.assertEqual(history[-1].name, "run2")

    def test_launch_form_recent_history_enter_repopulates_args(self) -> None:
        fn = CodeFunction(
            name="reuse_args",
            desc="reuse args target",
            args=[
                FunctionArg("count", int),
                FunctionArg("flag", bool),
                FunctionArg("label", str, optional=True),
            ],
            callable=lambda ctx, *, count, flag, label=None: f"{count}:{flag}:{label}",
            uses=[],
        )
        runtime = Runtime([fn], client_factories={})
        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="recent run",
                fn=fn,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=mp.Event()),
                cancel_event=mp.Event(),
                latest_view=_make_view(
                    fn,
                    state=NodeState.Success,
                    update_seqnum=1,
                    inputs={"count": 7, "flag": False, "label": None},
                ),
            )
        ]
        tui._open_launch_form(0)
        assert tui._form_state is not None
        tui._form_state.fields[0].value = "keep name"
        tui._form_state.fields[1].value = "999"
        tui._form_state.error = "bad input"
        tui._form_state.cursor = tui._launch_form_history_start_index()

        tui.handle_key("\n")

        self.assertEqual(tui._form_state.fields[0].value, "recent run (1)")
        self.assertEqual(tui._form_state.fields[1].value, "7")
        self.assertEqual(tui._form_state.fields[2].value, "false")
        self.assertEqual(tui._form_state.fields[3].value, "")
        self.assertEqual(tui._form_state.cursor, 1)
        self.assertEqual(tui._form_state.error, "")

    def test_launch_form_last_arg_enter_still_reaches_submit_before_history(self) -> None:
        fn = CodeFunction(
            name="submit_before_history",
            desc="submit ordering target",
            args=[FunctionArg("value", str)],
            callable=lambda ctx, *, value: value,
            uses=[],
        )
        runtime = Runtime([fn], client_factories={})
        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="recent run",
                fn=fn,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=mp.Event()),
                cancel_event=mp.Event(),
                latest_view=_make_view(
                    fn,
                    state=NodeState.Success,
                    update_seqnum=1,
                    inputs={"value": "history"},
                ),
            )
        ]
        tui._open_launch_form(0)
        assert tui._form_state is not None
        tui._form_state.fields[1].value = "typed"
        tui._form_state.cursor = 1

        tui.handle_key("\n")

        assert tui._form_state is not None
        self.assertEqual(tui._form_state.cursor, len(tui._form_state.fields))
        self.assertEqual(tui._form_state.fields[1].value, "typed")

        tui.handle_key("\n")

        self.assertIsNone(tui._form_state)
        self.assertEqual(len(tui._runs), 2)
        self.assertEqual(tui._runs[-1].node.result(), "typed")

    def test_launch_form_recent_history_mouse_click_repopulates_args(self) -> None:
        fn = CodeFunction(
            name="reuse_mouse",
            desc="reuse args target",
            args=[FunctionArg("value", str)],
            callable=lambda ctx, *, value: value,
            uses=[],
        )
        runtime = Runtime([fn], client_factories={})
        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="recent click",
                fn=fn,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=mp.Event()),
                cancel_event=mp.Event(),
                latest_view=_make_view(
                    fn,
                    state=NodeState.Success,
                    update_seqnum=1,
                    inputs={"value": "from history"},
                ),
            )
        ]
        tui._open_launch_form(0)

        rendered = tui.render_frame(TerminalSize(columns=90, lines=18), tick=0)
        plain_lines = [_strip_ansi(line).rstrip() for line in rendered.splitlines()]
        history_row = next(i for i, line in enumerate(plain_lines) if "recent click" in line)

        tui.handle_mouse(SimpleNamespace(x=2, y=history_row, button="left"))

        assert tui._form_state is not None
        self.assertEqual(tui._form_state.fields[0].value, "recent click (1)")
        self.assertEqual(tui._form_state.fields[1].value, "from history")
        self.assertEqual(tui._form_state.cursor, 1)

    def test_launch_form_recent_history_increments_existing_suffix_in_name(self) -> None:
        fn = CodeFunction(
            name="reuse_suffix",
            desc="reuse suffix target",
            args=[FunctionArg("value", str)],
            callable=lambda ctx, *, value: value,
            uses=[],
        )
        runtime = Runtime([fn], client_factories={})
        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="recent click (7)",
                fn=fn,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=mp.Event()),
                cancel_event=mp.Event(),
                latest_view=_make_view(
                    fn,
                    state=NodeState.Success,
                    update_seqnum=1,
                    inputs={"value": "from history"},
                ),
            )
        ]
        tui._open_launch_form(0)
        assert tui._form_state is not None
        tui._form_state.cursor = tui._launch_form_history_start_index()

        tui.handle_key("\n")

        self.assertEqual(tui._form_state.fields[0].value, "recent click (8)")
        self.assertEqual(tui._form_state.fields[1].value, "from history")

    def test_launch_form_agent_provider_field_starts_from_default_model(self) -> None:
        fn = AgentFunction(
            name="agent_launch",
            desc="agent launch target",
            args=[FunctionArg("prompt", str)],
            system_prompt="system",
            user_prompt_template="{prompt}",
            uses=[],
            default_model=Provider.Anthropic,
        )
        runtime = Runtime(
            [fn],
            client_factories={
                Provider.Anthropic: lambda: None,
                Provider.Gemini: lambda: None,
            },
        )
        tui = TUI(runtime)

        tui._open_launch_form(0)

        assert tui._form_state is not None
        self.assertEqual(tui._form_state.fields[0].label, "run_name")
        self.assertEqual(tui._form_state.fields[1].label, "provider")
        self.assertTrue(tui._form_state.fields[2].is_provider_options)
        self.assertEqual(tui._form_state.fields[1].value, Provider.Anthropic.value)
        self.assertEqual(tui._form_state.fields[3].label, "prompt")

    def test_launch_form_agent_default_provider_submits_without_override(self) -> None:
        fn = AgentFunction(
            name="agent_default_submit",
            desc="agent default submit target",
            args=[FunctionArg("prompt", str)],
            system_prompt="system",
            user_prompt_template="{prompt}",
            uses=[],
            default_model=Provider.Anthropic,
        )
        runtime = Runtime(
            [fn],
            client_factories={
                Provider.Anthropic: lambda: None,
                Provider.Gemini: lambda: None,
            },
        )
        tui = TUI(runtime)
        tui._open_launch_form(0)
        assert tui._form_state is not None
        tui._form_state.fields[0].value = "agent run"
        tui._form_state.fields[3].value = "hello"
        terminal_view = _make_agent_view(
            fn,
            state=NodeState.Success,
            update_seqnum=1,
            provider=Provider.Anthropic,
        )
        node = SimpleNamespace(id=123, watch=Mock(return_value=terminal_view))

        with patch.object(runtime, "invoke", return_value=node) as invoke_mock, patch.object(
            runtime,
            "get_view",
            return_value=terminal_view,
        ):
            tui._submit_form()

        invoke_mock.assert_called_once_with(
            None,
            fn,
            {"prompt": "hello"},
            provider=None,
            cancel_event=unittest.mock.ANY,
        )

    def test_launch_form_agent_provider_override_is_forwarded_only_for_root_invoke(self) -> None:
        fn = AgentFunction(
            name="agent_override_submit",
            desc="agent override submit target",
            args=[FunctionArg("prompt", str)],
            system_prompt="system",
            user_prompt_template="{prompt}",
            uses=[],
            default_model=Provider.Anthropic,
        )
        runtime = Runtime(
            [fn],
            client_factories={
                Provider.Anthropic: lambda: None,
                Provider.Gemini: lambda: None,
            },
        )
        tui = TUI(runtime)
        tui._open_launch_form(0)
        assert tui._form_state is not None
        tui._form_state.fields[0].value = "agent run"
        tui._form_state.fields[1].value = "gemini"
        tui._form_state.fields[3].value = "hello"
        terminal_view = _make_agent_view(
            fn,
            state=NodeState.Success,
            update_seqnum=1,
            provider=Provider.Gemini,
        )
        node = SimpleNamespace(id=123, watch=Mock(return_value=terminal_view))

        with patch.object(runtime, "invoke", return_value=node) as invoke_mock, patch.object(
            runtime,
            "get_view",
            return_value=terminal_view,
        ):
            tui._submit_form()

        invoke_mock.assert_called_once_with(
            None,
            fn,
            {"prompt": "hello"},
            provider=Provider.Gemini,
            cancel_event=unittest.mock.ANY,
        )

    def test_launch_form_invalid_provider_shows_error_and_does_not_launch(self) -> None:
        fn = AgentFunction(
            name="agent_invalid_provider",
            desc="agent invalid provider target",
            args=[FunctionArg("prompt", str)],
            system_prompt="system",
            user_prompt_template="{prompt}",
            uses=[],
            default_model=Provider.Anthropic,
        )
        runtime = Runtime(
            [fn],
            client_factories={
                Provider.Anthropic: lambda: None,
                Provider.Gemini: lambda: None,
            },
        )
        tui = TUI(runtime)
        tui._open_launch_form(0)
        assert tui._form_state is not None
        tui._form_state.fields[1].value = "not-a-provider"
        tui._form_state.fields[3].value = "hello"

        tui._submit_form()

        self.assertEqual(len(tui._runs), 0)
        assert tui._form_state is not None
        self.assertIn("Provider must be one of", tui._form_state.error)

    def test_launch_form_recent_history_repopulates_provider_for_agent_runs(self) -> None:
        fn = AgentFunction(
            name="agent_history",
            desc="agent history target",
            args=[FunctionArg("prompt", str)],
            system_prompt="system",
            user_prompt_template="{prompt}",
            uses=[],
            default_model=Provider.Anthropic,
        )
        runtime = Runtime(
            [fn],
            client_factories={
                Provider.Anthropic: lambda: None,
                Provider.Gemini: lambda: None,
            },
        )
        tui = TUI(runtime)
        tui._runs = [
            _RunRecord(
                name="recent agent",
                fn=fn,
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=mp.Event()),
                cancel_event=mp.Event(),
                latest_view=_make_agent_view(
                    fn,
                    state=NodeState.Success,
                    update_seqnum=1,
                    provider=Provider.Gemini,
                ),
            )
        ]
        tui._open_launch_form(0)
        assert tui._form_state is not None
        tui._form_state.fields[0].value = "keep name"
        tui._form_state.fields[1].value = Provider.Anthropic.value
        tui._form_state.fields[3].value = "stale"
        tui._form_state.cursor = tui._launch_form_history_start_index()

        tui.handle_key("\n")

        assert tui._form_state is not None
        self.assertEqual(tui._form_state.fields[0].value, "recent agent (1)")
        self.assertEqual(tui._form_state.fields[1].value, Provider.Gemini.value)
        self.assertEqual(tui._form_state.fields[3].value, "")
        self.assertEqual(tui._form_state.cursor, 1)

    def test_launch_form_provider_row_renders_enum_and_highlights_selected_option(self) -> None:
        fn = AgentFunction(
            name="agent_provider_enum",
            desc="agent provider enum target",
            args=[],
            system_prompt="system",
            user_prompt_template="",
            uses=[],
            default_model=Provider.Anthropic,
        )
        runtime = Runtime(
            [fn],
            client_factories={
                Provider.Anthropic: lambda: None,
                Provider.Gemini: lambda: None,
            },
        )
        tui = TUI(runtime)
        tui._open_launch_form(0)

        rendered = tui.render_frame(TerminalSize(columns=90, lines=18), tick=0)
        plain_lines = [_strip_ansi(line).rstrip() for line in rendered.splitlines()]
        provider_options = next(
            line for line in plain_lines if "Anthropic" in line and "Gemini" in line and "OpenAI" in line and "xAI" in line
        )

        self.assertIn("[Anthropic]", provider_options)
        self.assertIn("Gemini", provider_options)
        self.assertIn("OpenAI", provider_options)
        self.assertIn("xAI", provider_options)
        self.assertIn(FG["green"], rendered)

    def test_launch_form_space_toggles_provider_when_provider_field_selected(self) -> None:
        fn = AgentFunction(
            name="agent_provider_toggle",
            desc="agent provider toggle target",
            args=[],
            system_prompt="system",
            user_prompt_template="",
            uses=[],
            default_model=Provider.Anthropic,
        )
        runtime = Runtime(
            [fn],
            client_factories={
                Provider.Anthropic: lambda: None,
                Provider.Gemini: lambda: None,
            },
        )
        tui = TUI(runtime)
        tui._open_launch_form(0)
        assert tui._form_state is not None
        tui._form_state.cursor = 1

        tui.handle_key(" ")

        self.assertEqual(tui._form_state.fields[1].value, Provider.Gemini.value)

    def test_launch_form_space_still_inserts_literal_space_for_text_fields(self) -> None:
        fn = CodeFunction(
            name="space_text",
            desc="space text target",
            args=[FunctionArg("value", str)],
            callable=lambda ctx, *, value: value,
            uses=[],
        )
        runtime = Runtime([fn], client_factories={})
        tui = TUI(runtime)
        tui._open_launch_form(0)
        assert tui._form_state is not None
        tui._form_state.cursor = 1

        tui.handle_key(" ")

        self.assertEqual(tui._form_state.fields[1].value, " ")

    def test_launch_form_clicking_provider_options_row_selects_provider_field(self) -> None:
        fn = AgentFunction(
            name="agent_provider_click",
            desc="agent provider click target",
            args=[],
            system_prompt="system",
            user_prompt_template="",
            uses=[],
            default_model=Provider.Anthropic,
        )
        runtime = Runtime(
            [fn],
            client_factories={
                Provider.Anthropic: lambda: None,
                Provider.Gemini: lambda: None,
            },
        )
        tui = TUI(runtime)
        tui._open_launch_form(0)

        rendered = tui.render_frame(TerminalSize(columns=90, lines=18), tick=0)
        plain_lines = [_strip_ansi(line).rstrip() for line in rendered.splitlines()]
        provider_options_row = next(
            i for i, line in enumerate(plain_lines) if "Anthropic" in line and "Gemini" in line and "OpenAI" in line and "xAI" in line
        )

        tui.handle_mouse(SimpleNamespace(x=2, y=provider_options_row, button="left"))

        assert tui._form_state is not None
        self.assertEqual(tui._form_state.cursor, 1)

    def test_launch_form_keeps_fields_visible_when_description_exceeds_viewport(self) -> None:
        fn = CodeFunction(
            name="long_desc",
            desc="\n".join(f"line {idx}" for idx in range(20)),
            args=[FunctionArg("value", str)],
            callable=lambda ctx, *, value: value,
            uses=[],
        )
        runtime = Runtime([fn], client_factories={})
        tui = TUI(runtime)
        tui._open_launch_form(0)

        rendered = tui.render_frame(TerminalSize(columns=80, lines=15), tick=0)
        plain_lines = [_strip_ansi(line).rstrip() for line in rendered.splitlines()]

        self.assertTrue(any("run_name:" in line for line in plain_lines))
        self.assertTrue(any("desc clipped" in line for line in plain_lines))

    def test_launch_form_mouse_hit_uses_visible_header_rows_when_description_is_clipped(self) -> None:
        fn = CodeFunction(
            name="long_desc",
            desc="\n".join(f"line {idx}" for idx in range(20)),
            args=[FunctionArg("value", str)],
            callable=lambda ctx, *, value: value,
            uses=[],
        )
        runtime = Runtime([fn], client_factories={})
        tui = TUI(runtime)
        tui._open_launch_form(0)
        assert tui._form_state is not None
        tui._form_state.cursor = len(tui._form_state.fields)

        rendered = tui.render_frame(TerminalSize(columns=80, lines=15), tick=0)
        plain_lines = [_strip_ansi(line).rstrip() for line in rendered.splitlines()]
        submit_row = next(i for i, line in enumerate(plain_lines) if "[Submit]" in line)

        tui.handle_mouse(SimpleNamespace(x=2, y=submit_row, button="left"))

        assert tui._form_state is not None
        self.assertIn("Missing required arg(s)", tui._form_state.error)

    def test_submit_does_not_admit_run_if_watcher_start_fails(self) -> None:
        fn = _make_code_function("watcher_fail")
        runtime = Runtime([fn], client_factories={})
        tui = TUI(runtime)
        tui._open_launch_form(0)
        assert tui._form_state is not None

        original_start = threading.Thread.start

        def fail_watcher_start(thread_self, *args, **kwargs):
            if thread_self.name.startswith("netflux-tui-watch-"):
                raise RuntimeError("watcher boom")
            return original_start(thread_self, *args, **kwargs)

        with patch.object(threading.Thread, "start", autospec=True, side_effect=fail_watcher_start):
            with patch("netflux.tui.tui.restore_console"), patch(
                "netflux.tui.tui.os._exit",
                side_effect=SystemExit(1),
            ):
                with self.assertRaises(SystemExit):
                    tui._submit_form()

        self.assertEqual(len(tui._runs), 0)

    def test_submit_requests_cancellation_if_watcher_start_fails_after_launch(self) -> None:
        started = threading.Event()

        def blocking(ctx: RunContext) -> str:
            started.set()
            while not ctx.cancel_requested():
                time.sleep(0.01)
            raise CancellationException()

        fn = CodeFunction(
            name="watcher_fail",
            desc="watcher failure target",
            args=[],
            callable=blocking,
            uses=[],
        )
        runtime = Runtime([fn], client_factories={})
        tui = TUI(runtime)
        tui._open_launch_form(0)
        assert tui._form_state is not None

        original_start = threading.Thread.start

        def fail_watcher_start(thread_self, *args, **kwargs):
            if thread_self.name.startswith("netflux-tui-watch-"):
                raise RuntimeError("watcher boom")
            return original_start(thread_self, *args, **kwargs)

        with patch.object(threading.Thread, "start", autospec=True, side_effect=fail_watcher_start):
            with patch("netflux.tui.tui.restore_console"), patch(
                "netflux.tui.tui.os._exit",
                side_effect=SystemExit(1),
            ):
                with self.assertRaises(SystemExit):
                    tui._submit_form()

        self.assertTrue(started.wait(timeout=1), "blocking callable did not start")
        self.assertEqual(len(tui._runs), 0)
        self.assertTrue(runtime._roots[0].cancel_event.is_set())  # type: ignore[union-attr]
        with self.assertRaises(CancellationException):
            runtime._roots[0].result()

    def test_submit_requests_cancellation_if_setup_fails_after_launch(self) -> None:
        started = threading.Event()

        def blocking(ctx: RunContext) -> str:
            started.set()
            while not ctx.cancel_requested():
                time.sleep(0.01)
            raise CancellationException()

        fn = CodeFunction(
            name="setup_fail",
            desc="post-invoke setup failure target",
            args=[],
            callable=blocking,
            uses=[],
        )
        runtime = Runtime([fn], client_factories={})
        tui = TUI(runtime)
        tui._open_launch_form(0)
        assert tui._form_state is not None

        class _BoomRender:
            def __init__(self, *args, **kwargs) -> None:
                del args, kwargs
                raise RuntimeError("renderer boom")

        with patch("netflux.tui.tui.ConsoleRender", _BoomRender):
            with patch("netflux.tui.tui.restore_console"), patch(
                "netflux.tui.tui.os._exit",
                side_effect=SystemExit(1),
            ):
                with self.assertRaises(SystemExit):
                    tui._submit_form()

        self.assertTrue(started.wait(timeout=1), "blocking callable did not start")
        self.assertEqual(len(tui._runs), 0)
        self.assertTrue(runtime._roots[0].cancel_event.is_set())  # type: ignore[union-attr]
        with self.assertRaises(CancellationException):
            runtime._roots[0].result()

    def test_render_body_returns_full_width_viewport_without_el(self) -> None:
        fn = _make_agent_function("viewport")
        renderer = ConsoleRender(cancel_event=mp.Event())
        body = renderer.render_body(
            width=20,
            height=4,
            view=_make_agent_view(fn, state=NodeState.Running, update_seqnum=1),
            tick=0,
        )

        self.assertNotIn("\x1b[K", body)
        for line in body.splitlines():
            self.assertEqual(len(_strip_ansi(line)), 20)

    def test_render_body_placeholder_respects_requested_viewport(self) -> None:
        renderer = ConsoleRender(cancel_event=mp.Event())

        body = renderer.render_body(width=24, height=4, tick=0)

        lines = body.splitlines()
        self.assertEqual(len(lines), 4)
        self.assertIn("(waiting for data...)", _strip_ansi(lines[0]))
        for line in lines:
            self.assertEqual(len(_strip_ansi(line)), 24)

    def test_tui_shutdown_does_not_join_each_watcher_thread(self) -> None:
        runtime = Runtime([_make_code_function("fn0")], client_factories={})
        tui = TUI(runtime)
        stop_event = threading.Event()
        fake_thread = Mock()
        fake_thread.is_alive.return_value = True
        tui._runs = [
            _RunRecord(
                name="run",
                fn=_make_code_function("fn0"),
                node=object(),  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=mp.Event()),
                cancel_event=mp.Event(),
                latest_view=None,
                watcher_stop=stop_event,
                watcher_thread=fake_thread,
            )
        ]

        tui.on_session_stop()

        self.assertTrue(stop_event.is_set())
        fake_thread.join.assert_not_called()

    def test_on_session_stop_makes_final_billing_attempt(self) -> None:
        fn = _make_code_function("fn0")
        runtime = Runtime([fn], client_factories={})
        tui = TUI(runtime)
        stop_event = threading.Event()
        received: list[TokenBills] = []
        node = SimpleNamespace(id=7)

        def _capture(bills: TokenBills) -> None:
            received.append(bills)

        tui.register_terminal_callback(_capture)
        tui._runs = [
            _RunRecord(
                name="run",
                fn=fn,
                node=node,  # type: ignore[arg-type]
                renderer=ConsoleRender(cancel_event=mp.Event()),
                cancel_event=mp.Event(),
                latest_view=_make_view(fn, state=NodeState.Running, update_seqnum=1),
                watcher_stop=stop_event,
                watcher_thread=Mock(),
            )
        ]

        with patch.object(runtime, "get_view", return_value=_make_view(fn, state=NodeState.Canceled, update_seqnum=2)):
            tui.on_session_stop()

        self.assertTrue(stop_event.is_set())
        self.assertEqual(received, [{}])
        self.assertTrue(tui._runs[0].terminal_callback_invoked)


class _RecordingController(SessionController):
    def __init__(self) -> None:
        self.interactive: bool | None = None
        self.started = False
        self.stopped = False

    def set_wakeup(self, wakeup) -> None:
        del wakeup

    def on_session_start(self, *, interactive: bool) -> None:
        self.started = True
        self.interactive = interactive

    def on_session_stop(self) -> None:
        self.stopped = True

    def pump_events(self) -> bool:
        return False

    def wants_animation_ticks(self) -> bool:
        return False

    def render_frame(self, size: TerminalSize, tick: int) -> str:
        return ""

    def handle_key(self, key: str) -> bool:
        return False

    def handle_mouse(self, event: object) -> bool:
        return False

    def handle_interrupt(self) -> bool:
        return True

    def should_exit(self) -> bool:
        return True


class _RenderInterruptController(SessionController):
    def __init__(self) -> None:
        self.interrupts = 0
        self.rendered_after_interrupt = False
        self._stop = False

    def set_wakeup(self, wakeup) -> None:
        del wakeup

    def on_session_start(self, *, interactive: bool) -> None:
        del interactive

    def on_session_stop(self) -> None:
        pass

    def pump_events(self) -> bool:
        return False

    def wants_animation_ticks(self) -> bool:
        return False

    def render_frame(self, size: TerminalSize, tick: int) -> str:
        del size, tick
        if self.interrupts == 0:
            raise KeyboardInterrupt
        self.rendered_after_interrupt = True
        self._stop = True
        return ""

    def handle_key(self, key: str) -> bool:
        del key
        return False

    def handle_mouse(self, event: object) -> bool:
        del event
        return False

    def handle_interrupt(self) -> bool:
        self.interrupts += 1
        return False

    def should_exit(self) -> bool:
        return self._stop


class _StartFailureController(SessionController):
    def __init__(self) -> None:
        self.stop_called = False

    def set_wakeup(self, wakeup) -> None:
        del wakeup

    def on_session_start(self, *, interactive: bool) -> None:
        del interactive
        raise RuntimeError("start boom")

    def on_session_stop(self) -> None:
        self.stop_called = True

    def pump_events(self) -> bool:
        return False

    def wants_animation_ticks(self) -> bool:
        return False

    def render_frame(self, size: TerminalSize, tick: int) -> str:
        del size, tick
        return ""

    def handle_key(self, key: str) -> bool:
        del key
        return False

    def handle_mouse(self, event: object) -> bool:
        del event
        return False

    def handle_interrupt(self) -> bool:
        return True

    def should_exit(self) -> bool:
        return True


class _StopFailureController(_RecordingController):
    def __init__(self) -> None:
        super().__init__()
        self._stop = False

    def on_session_start(self, *, interactive: bool) -> None:
        self.started = True
        self.interactive = interactive
        self._stop = False

    def on_session_stop(self) -> None:
        self.stopped = True
        raise RuntimeError("stop boom")

    def render_frame(self, size: TerminalSize, tick: int) -> str:
        del size, tick
        self._stop = True
        return ""

    def should_exit(self) -> bool:
        return self._stop


class _VetoController(_RecordingController):
    def should_exit(self) -> bool:
        return True


class _SingleRenderController(_RecordingController):
    def __init__(self) -> None:
        super().__init__()
        self.render_calls = 0

    def render_frame(self, size: TerminalSize, tick: int) -> str:
        del size, tick
        self.render_calls += 1
        return "frame"

    def should_exit(self) -> bool:
        return self.render_calls > 0


class _SingleKeyExitController(_RecordingController):
    def __init__(self) -> None:
        super().__init__()
        self.keys: list[str] = []

    def should_exit(self) -> bool:
        return False

    def handle_key(self, key: str) -> bool:
        self.keys.append(key)
        return True


class _TTYWithoutFileno:
    def isatty(self) -> bool:
        return True

    def fileno(self) -> int:
        raise io.UnsupportedOperation("redirected stdin is pseudofile, has no fileno()")


class TestTerminalIO(unittest.TestCase):
    def test_read_key_buffers_split_posix_mouse_sequence_until_complete(self) -> None:
        select_results = iter([
            ([7], [], []),  # initial byte available
            ([7], [], []),  # [
            ([7], [], []),  # <
            ([7], [], []),  # 0
            ([7], [], []),  # ;
            ([7], [], []),  # 5
            ([7], [], []),  # 0
            ([], [], []),   # partial sequence timeout
            ([7], [], []),  # ;
            ([7], [], []),  # 1
            ([7], [], []),  # 0
            ([7], [], []),  # M
        ])
        read_results = iter([
            b"\x1b",
            b"[",
            b"<",
            b"0",
            b";",
            b"5",
            b"0",
            b";",
            b"1",
            b"0",
            b"M",
        ])

        def _fake_select(_readers, _writers, _errors, _timeout):
            return next(select_results)

        def _fake_read(_fd: int, _count: int) -> bytes:
            return next(read_results)

        self.addCleanup(setattr, terminal_io, "_POSIX_PENDING_ESCAPE_SEQ", None)
        self.addCleanup(setattr, terminal_io, "_POSIX_PENDING_ESCAPE_DEADLINE", None)
        self.addCleanup(setattr, terminal_io, "_POSIX_PENDING_ESCAPE_DISCARD", False)
        self.addCleanup(setattr, terminal_io, "_POSIX_ORPHAN_ESCAPE_DEADLINE", None)
        with patch.object(terminal_io, "_POSIX_PENDING_ESCAPE_SEQ", None), patch.object(
            terminal_io,
            "_POSIX_PENDING_ESCAPE_DEADLINE",
            None,
        ), patch.object(
            terminal_io,
            "_POSIX_PENDING_ESCAPE_DISCARD",
            False,
        ), patch.object(
            terminal_io,
            "_POSIX_ORPHAN_ESCAPE_DEADLINE",
            None,
        ), patch.object(
            terminal_io,
            "select",
            SimpleNamespace(select=_fake_select),
        ), patch.object(terminal_io.os, "read", side_effect=_fake_read):
            self.assertIsNone(read_key(7, timeout=0))
            self.assertEqual(read_key(7, timeout=0), terminal_io.MouseEvent(x=49, y=9, button="left"))
            self.assertIsNone(terminal_io._POSIX_PENDING_ESCAPE_SEQ)

    def test_read_key_buffers_bare_escape_until_following_mouse_sequence_starts(self) -> None:
        select_results = iter([
            ([7], [], []),  # initial ESC available
            ([], [], []),   # no next byte yet
            ([7], [], []),  # [
            ([7], [], []),  # <
            ([7], [], []),  # 0
            ([7], [], []),  # ;
            ([7], [], []),  # 5
            ([7], [], []),  # 0
            ([7], [], []),  # ;
            ([7], [], []),  # 1
            ([7], [], []),  # 0
            ([7], [], []),  # M
        ])
        read_results = iter([
            b"\x1b",
            b"[",
            b"<",
            b"0",
            b";",
            b"5",
            b"0",
            b";",
            b"1",
            b"0",
            b"M",
        ])
        monotonic_results = iter([10.0, 10.01] + [10.01] * 20)

        def _fake_select(_readers, _writers, _errors, _timeout):
            return next(select_results)

        def _fake_read(_fd: int, _count: int) -> bytes:
            return next(read_results)

        def _fake_monotonic() -> float:
            return next(monotonic_results)

        self.addCleanup(setattr, terminal_io, "_POSIX_PENDING_ESCAPE_SEQ", None)
        self.addCleanup(setattr, terminal_io, "_POSIX_PENDING_ESCAPE_DEADLINE", None)
        self.addCleanup(setattr, terminal_io, "_POSIX_PENDING_ESCAPE_DISCARD", False)
        self.addCleanup(setattr, terminal_io, "_POSIX_ORPHAN_ESCAPE_DEADLINE", None)
        with patch.object(terminal_io, "_POSIX_PENDING_ESCAPE_SEQ", None), patch.object(
            terminal_io,
            "_POSIX_PENDING_ESCAPE_DEADLINE",
            None,
        ), patch.object(
            terminal_io,
            "_POSIX_PENDING_ESCAPE_DISCARD",
            False,
        ), patch.object(
            terminal_io,
            "_POSIX_ORPHAN_ESCAPE_DEADLINE",
            None,
        ), patch.object(
            terminal_io,
            "select",
            SimpleNamespace(select=_fake_select),
        ), patch.object(terminal_io.os, "read", side_effect=_fake_read), patch.object(
            terminal_io.time,
            "monotonic",
            side_effect=_fake_monotonic,
        ):
            self.assertIsNone(read_key(7, timeout=0))
            self.assertEqual(terminal_io._POSIX_PENDING_ESCAPE_SEQ, "")
            self.assertIsNotNone(terminal_io._POSIX_PENDING_ESCAPE_DEADLINE)
            self.assertEqual(read_key(7, timeout=0), terminal_io.MouseEvent(x=49, y=9, button="left"))
            self.assertIsNone(terminal_io._POSIX_PENDING_ESCAPE_SEQ)
            self.assertIsNone(terminal_io._POSIX_PENDING_ESCAPE_DEADLINE)

    def test_read_key_emits_escape_after_bare_escape_grace_expires(self) -> None:
        select_results = iter([
            ([7], [], []),  # initial ESC available
            ([], [], []),   # no next byte yet
            ([], [], []),   # grace deadline expired, still no bytes
        ])
        read_results = iter([b"\x1b"])
        monotonic_results = iter([20.0, 20.2, 20.2, 20.2])

        def _fake_select(_readers, _writers, _errors, _timeout):
            return next(select_results)

        def _fake_read(_fd: int, _count: int) -> bytes:
            return next(read_results)

        def _fake_monotonic() -> float:
            return next(monotonic_results)

        self.addCleanup(setattr, terminal_io, "_POSIX_PENDING_ESCAPE_SEQ", None)
        self.addCleanup(setattr, terminal_io, "_POSIX_PENDING_ESCAPE_DEADLINE", None)
        self.addCleanup(setattr, terminal_io, "_POSIX_PENDING_ESCAPE_DISCARD", False)
        self.addCleanup(setattr, terminal_io, "_POSIX_ORPHAN_ESCAPE_DEADLINE", None)
        with patch.object(terminal_io, "_POSIX_PENDING_ESCAPE_SEQ", None), patch.object(
            terminal_io,
            "_POSIX_PENDING_ESCAPE_DEADLINE",
            None,
        ), patch.object(
            terminal_io,
            "_POSIX_PENDING_ESCAPE_DISCARD",
            False,
        ), patch.object(
            terminal_io,
            "_POSIX_ORPHAN_ESCAPE_DEADLINE",
            None,
        ), patch.object(
            terminal_io,
            "select",
            SimpleNamespace(select=_fake_select),
        ), patch.object(terminal_io.os, "read", side_effect=_fake_read), patch.object(
            terminal_io.time,
            "monotonic",
            side_effect=_fake_monotonic,
        ):
            self.assertIsNone(read_key(7, timeout=0))
            self.assertEqual(read_key(7, timeout=0), "escape")
            self.assertIsNone(terminal_io._POSIX_PENDING_ESCAPE_SEQ)
            self.assertIsNone(terminal_io._POSIX_PENDING_ESCAPE_DEADLINE)

    def test_read_key_discards_late_mouse_tail_after_escape_already_emitted(self) -> None:
        select_results = iter([
            ([7], [], []),  # initial ESC available
            ([], [], []),   # no next byte yet
            ([], [], []),   # grace deadline expired, still no bytes
            ([7], [], []),  # orphaned [
            ([7], [], []),  # <
            ([7], [], []),  # 6
            ([7], [], []),  # 4
            ([7], [], []),  # ;
            ([7], [], []),  # 5
            ([7], [], []),  # 0
            ([7], [], []),  # ;
            ([7], [], []),  # 1
            ([7], [], []),  # 0
            ([7], [], []),  # M
        ])
        read_results = iter([
            b"\x1b",
            b"[",
            b"<",
            b"6",
            b"4",
            b";",
            b"5",
            b"0",
            b";",
            b"1",
            b"0",
            b"M",
        ])
        monotonic_results = iter([20.0, 20.2, 20.2, 20.21] + [20.21] * 40)

        def _fake_select(_readers, _writers, _errors, _timeout):
            return next(select_results)

        def _fake_read(_fd: int, _count: int) -> bytes:
            return next(read_results)

        def _fake_monotonic() -> float:
            return next(monotonic_results)

        self.addCleanup(setattr, terminal_io, "_POSIX_PENDING_ESCAPE_SEQ", None)
        self.addCleanup(setattr, terminal_io, "_POSIX_PENDING_ESCAPE_DEADLINE", None)
        self.addCleanup(setattr, terminal_io, "_POSIX_PENDING_ESCAPE_DISCARD", False)
        self.addCleanup(setattr, terminal_io, "_POSIX_ORPHAN_ESCAPE_DEADLINE", None)
        with patch.object(terminal_io, "_POSIX_PENDING_ESCAPE_SEQ", None), patch.object(
            terminal_io,
            "_POSIX_PENDING_ESCAPE_DEADLINE",
            None,
        ), patch.object(
            terminal_io,
            "_POSIX_PENDING_ESCAPE_DISCARD",
            False,
        ), patch.object(
            terminal_io,
            "_POSIX_ORPHAN_ESCAPE_DEADLINE",
            None,
        ), patch.object(
            terminal_io,
            "select",
            SimpleNamespace(select=_fake_select),
        ), patch.object(terminal_io.os, "read", side_effect=_fake_read), patch.object(
            terminal_io.time,
            "monotonic",
            side_effect=_fake_monotonic,
        ):
            events = [read_key(7, timeout=0) for _ in range(3)]

        self.assertEqual([event for event in events if event is not None], ["escape"])
        self.assertIsNone(terminal_io._POSIX_PENDING_ESCAPE_SEQ)
        self.assertIsNone(terminal_io._POSIX_PENDING_ESCAPE_DEADLINE)
        self.assertFalse(terminal_io._POSIX_PENDING_ESCAPE_DISCARD)

        runtime = Runtime([_make_code_function(f"fn{i}") for i in range(7)], client_factories={})
        tui = TUI(runtime)
        for event in events:
            if isinstance(event, str):
                tui.handle_key(event)

        self.assertIsNone(tui._form_state)

    def test_read_key_decodes_legacy_csi_m_mouse_sequence(self) -> None:
        select_results = iter([
            ([7], [], []),  # initial ESC available
            ([7], [], []),  # [
            ([7], [], []),  # M
            ([7], [], []),  # cb
            ([7], [], []),  # cx
            ([7], [], []),  # cy
        ])
        read_results = iter([
            b"\x1b",
            b"[",
            b"M",
            b" ",
            b"!",
            b"!",
        ])

        def _fake_select(_readers, _writers, _errors, _timeout):
            return next(select_results)

        def _fake_read(_fd: int, _count: int) -> bytes:
            return next(read_results)

        self.addCleanup(setattr, terminal_io, "_POSIX_PENDING_ESCAPE_SEQ", None)
        self.addCleanup(setattr, terminal_io, "_POSIX_PENDING_ESCAPE_DEADLINE", None)
        self.addCleanup(setattr, terminal_io, "_POSIX_PENDING_ESCAPE_DISCARD", False)
        self.addCleanup(setattr, terminal_io, "_POSIX_ORPHAN_ESCAPE_DEADLINE", None)
        with patch.object(terminal_io, "_POSIX_PENDING_ESCAPE_SEQ", None), patch.object(
            terminal_io,
            "_POSIX_PENDING_ESCAPE_DEADLINE",
            None,
        ), patch.object(
            terminal_io,
            "_POSIX_PENDING_ESCAPE_DISCARD",
            False,
        ), patch.object(
            terminal_io,
            "_POSIX_ORPHAN_ESCAPE_DEADLINE",
            None,
        ), patch.object(
            terminal_io,
            "select",
            SimpleNamespace(select=_fake_select),
        ), patch.object(terminal_io.os, "read", side_effect=_fake_read):
            self.assertEqual(read_key(7, timeout=0), terminal_io.MouseEvent(x=0, y=0, button="left"))

    def test_read_key_windows_returns_none_after_ignored_record_drains_queue(self) -> None:
        kernel32 = _FakeWindowsKernel32([
            _make_windows_key_record(key_down=False),
        ])

        with patch(
            "netflux.tui._terminal_io.ctypes.windll",
            SimpleNamespace(kernel32=kernel32),
            create=True,
        ):
            self.assertIsNone(read_key_windows())

    def test_configure_windows_console_input_enables_resize_notifications(self) -> None:
        kernel32 = _FakeWindowsModeKernel32(mode=0)

        with patch("netflux.tui._terminal_io.os.name", "nt"), patch(
            "netflux.tui._terminal_io.ctypes.windll",
            SimpleNamespace(kernel32=kernel32),
            create=True,
        ), patch("netflux.tui._terminal_io._WINDOWS_INPUT_MODE_SAVED", None):
            _configure_windows_console_input(enable_mouse=True)

        self.assertEqual(len(kernel32.set_modes), 1)
        self.assertTrue(kernel32.set_modes[0] & _WIN_ENABLE_MOUSE_INPUT)
        self.assertTrue(kernel32.set_modes[0] & _WIN_ENABLE_WINDOW_INPUT)


class TestConsoleSessionDriver(unittest.TestCase):
    def test_standalone_controller_does_not_start_watcher_before_driver_startup_succeeds(self) -> None:
        fn = _make_code_function("done")
        runtime = Runtime([fn], client_factories={})
        node = runtime.invoke(None, fn, {})
        self.assertEqual(node.result(), "done")

        controller = SingleTreeConsoleController(ConsoleRender(), node)
        driver = ConsoleSessionDriver()
        fake_current = object()
        fake_main = object()

        self.assertIsNone(controller._watch_thread.ident)

        with patch("sys.stdin.isatty", return_value=True), patch(
            "sys.stdout.isatty", return_value=True
        ), patch("netflux.tui._driver.os.name", "posix"), patch(
            "netflux.tui._driver.termios",
            SimpleNamespace(tcgetattr=Mock(), tcsetattr=Mock(), TCSADRAIN=0),
        ), patch(
            "netflux.tui._driver.tty",
            SimpleNamespace(setcbreak=Mock()),
        ), patch.object(
            ConsoleSessionDriver,
            "_stdin_fileno",
            return_value=0,
        ), patch(
            "netflux.tui._driver.threading.current_thread", return_value=fake_current
        ), patch(
            "netflux.tui._driver.threading.main_thread", return_value=fake_main
        ):
            with self.assertRaisesRegex(RuntimeError, "main thread on POSIX"):
                driver.run(controller)

        self.assertIsNone(controller._watch_thread.ident)
        self.assertFalse(controller._watch_thread.is_alive())

    def test_interactive_posix_requires_main_thread(self) -> None:
        controller = _RecordingController()
        driver = ConsoleSessionDriver()
        fake_current = object()
        fake_main = object()

        with patch("sys.stdin.isatty", return_value=True), patch(
            "sys.stdout.isatty", return_value=True
        ), patch("netflux.tui._driver.os.name", "posix"), patch(
            "netflux.tui._driver.termios",
            SimpleNamespace(tcgetattr=Mock(), tcsetattr=Mock(), TCSADRAIN=0),
        ), patch(
            "netflux.tui._driver.tty",
            SimpleNamespace(setcbreak=Mock()),
        ), patch.object(
            ConsoleSessionDriver,
            "_stdin_fileno",
            return_value=0,
        ), patch(
            "netflux.tui._driver.threading.current_thread", return_value=fake_current
        ), patch(
            "netflux.tui._driver.threading.main_thread", return_value=fake_main
        ), patch("netflux.tui._driver.pre_console") as pre_console:
            with self.assertRaisesRegex(RuntimeError, "main thread on POSIX"):
                driver.run(controller)

        self.assertFalse(controller.started)
        self.assertFalse(controller.stopped)
        self.assertFalse(pre_console.called)

    def test_interactive_posix_requires_default_sigint_handler(self) -> None:
        controller = _RecordingController()
        driver = ConsoleSessionDriver()
        fake_main = object()
        custom_handler = lambda _signum, _frame: None

        with patch("sys.stdin.isatty", return_value=True), patch(
            "sys.stdout.isatty", return_value=True
        ), patch("netflux.tui._driver.os.name", "posix"), patch(
            "netflux.tui._driver.termios",
            SimpleNamespace(tcgetattr=Mock(), tcsetattr=Mock(), TCSADRAIN=0),
        ), patch(
            "netflux.tui._driver.tty",
            SimpleNamespace(setcbreak=Mock()),
        ), patch.object(
            ConsoleSessionDriver,
            "_stdin_fileno",
            return_value=0,
        ), patch(
            "netflux.tui._driver.threading.current_thread", return_value=fake_main
        ), patch(
            "netflux.tui._driver.threading.main_thread", return_value=fake_main
        ), patch(
            "netflux.tui._driver.signal.getsignal", return_value=custom_handler
        ), patch("netflux.tui._driver.pre_console") as pre_console:
            with self.assertRaisesRegex(RuntimeError, "default Python SIGINT handler"):
                driver.run(controller)

        self.assertFalse(controller.started)
        self.assertFalse(controller.stopped)
        self.assertFalse(pre_console.called)

    def test_stdout_non_tty_forces_noninteractive_mode(self) -> None:
        controller = _RecordingController()
        driver = ConsoleSessionDriver()

        with patch("sys.stdin.isatty", return_value=True), patch(
            "sys.stdout.isatty", return_value=False
        ), patch("netflux.tui._driver.pre_console"), patch(
            "netflux.tui._driver.restore_console"
        ):
            driver.run(controller)

        self.assertTrue(controller.started)
        self.assertTrue(controller.stopped)
        self.assertFalse(controller.interactive)

    def test_stdin_non_tty_stdout_tty_restores_console_after_noninteractive_render(self) -> None:
        controller = _SingleRenderController()
        driver = ConsoleSessionDriver()

        with patch("sys.stdin.isatty", return_value=False), patch(
            "sys.stdout.isatty", return_value=True
        ), patch("netflux.tui._driver.pre_console") as pre_console, patch(
            "netflux.tui._driver.restore_console"
        ) as restore_console, patch(
            "netflux.tui._driver.ui_driver",
            side_effect=lambda frame: pre_console(),
        ):
            driver.run(controller)

        self.assertFalse(controller.interactive)
        self.assertEqual(controller.render_calls, 1)
        self.assertTrue(pre_console.called)
        self.assertTrue(restore_console.called)

    def test_keyboard_interrupt_during_render_still_routes_through_controller(self) -> None:
        controller = _RenderInterruptController()
        driver = ConsoleSessionDriver()

        with patch("sys.stdin.isatty", return_value=False), patch(
            "sys.stdout.isatty", return_value=False
        ), patch("netflux.tui._driver.pre_console"), patch(
            "netflux.tui._driver.restore_console"
        ), patch("netflux.tui._driver.ui_driver"):
            driver.run(controller)

        self.assertEqual(controller.interrupts, 1)
        self.assertTrue(controller.rendered_after_interrupt)

    def test_noninteractive_terminal_root_renders_once_before_exit(self) -> None:
        fn = _make_code_function("done")
        runtime = Runtime([fn], client_factories={})
        node = runtime.invoke(None, fn, {})
        self.assertEqual(node.result(), "done")

        controller = SingleTreeConsoleController(ConsoleRender(), node)
        driver = ConsoleSessionDriver()

        with patch("sys.stdin.isatty", return_value=False), patch(
            "sys.stdout.isatty", return_value=False
        ), patch("netflux.tui._driver.ui_driver") as ui_driver:
            driver.run(controller)

        self.assertEqual(ui_driver.call_count, 1)

    def test_start_failure_still_calls_on_session_stop(self) -> None:
        controller = _StartFailureController()
        driver = ConsoleSessionDriver()

        with self.assertRaisesRegex(RuntimeError, "start boom"):
            driver.run(controller)

        self.assertTrue(controller.stop_called)

    def test_stop_failure_still_restores_console(self) -> None:
        controller = _StopFailureController()
        driver = ConsoleSessionDriver()

        loop_name = "_loop_windows" if os.name == "nt" else "_loop_posix"
        loop_args = {"return_value": None}

        with patch("sys.stdin.isatty", return_value=True), patch(
            "sys.stdout.isatty", return_value=True
        ), patch("netflux.tui._driver.pre_console"), patch(
            "netflux.tui._driver.restore_console"
        ) as restore_console, patch("netflux.tui._driver.ui_driver"), patch.object(
            ConsoleSessionDriver, loop_name, **loop_args
        ):
            with self.assertRaisesRegex(RuntimeError, "stop boom"):
                driver.run(controller)

        self.assertTrue(restore_console.called)

    def test_posix_stdin_without_fileno_falls_back_to_noninteractive(self) -> None:
        controller = _StopFailureController()
        driver = ConsoleSessionDriver()
        fake_current = object()
        fake_main = object()

        with patch.object(sys, "stdin", _TTYWithoutFileno()), patch(
            "sys.stdout.isatty", return_value=True
        ), patch("netflux.tui._driver.os.name", "posix"), patch(
            "netflux.tui._driver.termios",
            SimpleNamespace(tcgetattr=Mock(), tcsetattr=Mock(), TCSADRAIN=0),
        ), patch(
            "netflux.tui._driver.tty",
            SimpleNamespace(setcbreak=Mock()),
        ), patch(
            "netflux.tui._driver.threading.current_thread", return_value=fake_current
        ), patch(
            "netflux.tui._driver.threading.main_thread", return_value=fake_main
        ), patch("netflux.tui._driver.pre_console") as pre_console, patch(
            "netflux.tui._driver.restore_console"
        ) as restore_console, patch("netflux.tui._driver.ui_driver"):
            with self.assertRaisesRegex(RuntimeError, "stop boom"):
                driver.run(controller)

        self.assertFalse(controller.interactive)
        self.assertFalse(pre_console.called)
        self.assertTrue(restore_console.called)

    def test_controller_veto_skips_console_setup(self) -> None:
        controller = _VetoController()
        driver = ConsoleSessionDriver()

        with patch("sys.stdin.isatty", return_value=True), patch(
            "sys.stdout.isatty", return_value=True
        ), patch("netflux.tui._driver.pre_console") as pre_console, patch(
            "netflux.tui._driver.restore_console"
        ):
            driver.run(controller)

        self.assertFalse(pre_console.called)

    def test_posix_loop_resolves_pending_bare_escape_without_waiting_for_more_fd_bytes(self) -> None:
        controller = _SingleKeyExitController()
        driver = ConsoleSessionDriver()
        driver._wake_pipe_read = 11

        with patch(
            "netflux.tui._driver.select",
            SimpleNamespace(select=Mock(return_value=([], [], []))),
        ), patch(
            "netflux.tui._driver.posix_pending_input_timeout",
            return_value=0.0,
        ), patch(
            "netflux.tui._driver.read_key",
            return_value="escape",
        ) as read_key_mock, patch(
            "netflux.tui._driver.ui_driver",
        ):
            driver._loop_posix(controller, 7)

        read_key_mock.assert_called_once_with(7, timeout=0)
        self.assertEqual(controller.keys, ["escape"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
