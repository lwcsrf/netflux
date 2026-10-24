from __future__ import annotations

from contextlib import contextmanager
import signal
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ...core import CodeFunction, NodeState, NodeView
from ...tui import ConsoleRender
from ...tui._contracts import TerminalSize
from ...tui._controllers import SingleTreeConsoleController
from ...tui._driver import ConsoleSessionDriver


class TestInterruptDriver(unittest.TestCase):
    @contextmanager
    def _session(self):
        fn = CodeFunction(name="work", desc="", args=[], callable=lambda _ctx: "done", uses=[])
        cancel = threading.Event()
        node = Mock(id=1, cancel_event=cancel)
        node.watch.return_value = NodeView(
            id=1, fn=fn, inputs={}, state=NodeState.Running, outputs=None,
            exception=None, children=(), usage=None, transcript=(),
            started_at=1.0, ended_at=None, update_seqnum=1,
        )
        controller = SingleTreeConsoleController(ConsoleRender(), node)
        with patch.object(controller._watch_thread, "start"):
            controller.on_session_start(interactive=True)
        driver = ConsoleSessionDriver()
        controller.set_interrupt_mode_callback(driver._set_interrupt_mode)
        driver._install_sigint()
        try:
            with patch.object(driver, "_enable_terminal_interrupt") as rearm:
                yield controller, driver, cancel, rearm
        finally:
            driver._restore_sigint()
            controller.on_session_stop()

    def _run_loop(self, driver, controller, *, read, render=None):
        driver._terminal_fd, driver._wake_pipe_read = 7, 11
        with patch.object(driver, "_current_size", return_value=TerminalSize(100, 24)), patch(
            "netflux.tui._driver.ui_driver", side_effect=render,
        ), patch(
            "netflux.tui._driver.select", SimpleNamespace(select=Mock(return_value=([7], [], []))),
        ), patch("netflux.tui._driver.posix_pending_input_timeout", return_value=None), patch(
            "netflux.tui._driver.read_key", side_effect=read,
        ):
            driver._loop_posix(controller, 7)

    def test_interrupts_during_input_are_deferred_and_coalesced(self) -> None:
        with self._session() as (controller, driver, cancel, _rearm):
            controller.render_frame(TerminalSize(100, 24), 0)
            completed = []

            def navigate(*_args, **_kwargs):
                for _ in range(3):
                    signal.raise_signal(signal.SIGINT)
                completed.append(True)

            with patch.object(controller._renderer, "apply_action", side_effect=navigate):
                driver._dispatch_input(controller, "j")
            self.assertEqual(completed, [True])
            self.assertIsNone(controller._cancel_confirmation)
            with patch.object(controller, "handle_key") as key:
                driver._dispatch_input(controller, "\r")
                key.assert_not_called()
            with patch.object(controller, "handle_interrupt", wraps=controller.handle_interrupt) as interrupt:
                driver._dispatch_pending_interrupt(controller)
                self.assertIsNone(driver._dispatch_pending_interrupt(controller))
                interrupt.assert_called_once_with()
            self.assertIsNotNone(controller._cancel_confirmation)
            self.assertFalse(cancel.is_set())

    def test_confirmation_discards_typeahead_on_entry_and_exit(self) -> None:
        for confirm in (False, True):
            with self.subTest(confirm=confirm), self._session() as (controller, driver, cancel, rearm):
                pending = ["\x03", "down", "\r"]
                modal_rendered = False
                discards = []

                def render(_frame):
                    nonlocal modal_rendered
                    if controller._cancel_confirmation is not None and not modal_rendered:
                        modal_rendered = True
                        # Input received during the first write cannot confirm
                        # a choice the user has not seen yet.
                        pending.extend(["down", "\r", "\x03"])
                        signal.raise_signal(signal.SIGINT)

                def discard(fd):
                    discards.append(fd)
                    self.assertTrue(modal_rendered)
                    self.assertEqual(driver._interrupt_mode, "ignore")
                    committed = confirm and len(discards) == 2
                    self.assertEqual(cancel.is_set(), committed)
                    self.assertEqual(rearm.call_count, int(committed))
                    pending.clear()
                    if len(discards) == 1:
                        pending.extend((["down"] if confirm else []) + ["\r", "\x03"])

                def read(*_args, **_kwargs):
                    if not pending:
                        raise RuntimeError("input drained")
                    return pending.pop(0)

                # Leaving the modal must discard queued Ctrl-C after rearming.
                rearm.side_effect = lambda: pending.append("\x03")
                with patch("netflux.tui._driver.discard_pending_input", side_effect=discard):
                    with self.assertRaisesRegex(RuntimeError, "input drained"):
                        self._run_loop(driver, controller, read=read, render=render)
                self.assertEqual(discards, [7, 7])
                self.assertIsNone(controller._cancel_confirmation)
                self.assertEqual(driver._interrupt_mode, "force" if confirm else "defer")

    def test_confirmed_cancel_can_interrupt_render_without_redispatch(self) -> None:
        with self._session() as (controller, driver, cancel, rearm):
            controller.handle_interrupt()
            controller.handle_key("down")
            controller.handle_key("\r")
            self.assertTrue(cancel.is_set())
            rearm.assert_called_once_with()
            with patch.object(controller, "handle_interrupt") as interrupt:
                with self.assertRaises(KeyboardInterrupt):
                    self._run_loop(
                        driver, controller, read=Mock(side_effect=AssertionError("force interrupt was swallowed")),
                        render=lambda _frame: signal.raise_signal(signal.SIGINT),
                    )
                interrupt.assert_not_called()
            # Repeated signals must not interrupt the subsequent cleanup.
            signal.raise_signal(signal.SIGINT)

    def test_resize_before_enter_prevents_confirming_a_hidden_selection(self) -> None:
        with self._session() as (controller, driver, cancel, _rearm):
            controller.handle_interrupt()
            old_size = TerminalSize(100, 24)
            controller.render_frame(old_size, 0)
            controller.handle_key("down")
            with patch.object(driver, "_current_size", return_value=TerminalSize(10, 3)), patch(
                "netflux.tui._driver.ui_driver",
            ) as render:
                self.assertFalse(driver._dispatch_input(controller, "\r", rendered_size=old_size))
            render.assert_called_once()
            self.assertFalse(cancel.is_set())
            self.assertIsNone(controller._cancel_confirmation)
            self.assertFalse(controller.should_exit())
