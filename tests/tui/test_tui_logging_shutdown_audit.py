from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import re
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

from ...core import CancellationException, CodeFunction, NodeState, RunContext
from ...runtime import Runtime
from ...tui._contracts import TerminalSize
from ...tui._driver import ConsoleSessionDriver
from ...tui._logging import close_tui_logging
from ...tui.tui import TUI


class TestTUILoggingShutdownAudit(unittest.TestCase):
    """Exercise completion and shutdown with real runtime and watcher threads."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.controllers: list[TUI] = []
        self.releases: list[threading.Event] = []
        # A watcher defect must fail this test, never terminate the test runner.
        self.fatal = patch.object(TUI, "_fatal", side_effect=AssertionError("unexpected fatal TUI shutdown"))
        self.fatal_mock = self.fatal.start()
        self.addCleanup(self.fatal.stop)
        self.addCleanup(self._cleanup_runs)

    def _cleanup_runs(self) -> None:
        for release in self.releases:
            release.set()
        for tui in self.controllers:
            for run in tui._runs:
                run.cancel_event.set()
                if run.node.thread is not None:
                    run.node.thread.join(timeout=5)
                if run.watcher_thread is not None:
                    run.watcher_thread.join(timeout=5)
            tui.on_session_stop()
            close_tui_logging(tui.log_path)

    def _function(self, name: str, callable, *, uses=()) -> CodeFunction:
        return CodeFunction(name=name, desc=name, args=[], callable=callable, uses=list(uses))

    def _tui(self, *functions: CodeFunction) -> TUI:
        runtime = Runtime(functions, client_factories={})
        tui = TUI(runtime, log_path=Path(self.temporary.name) / f"audit-{len(self.controllers)}.log")
        self.controllers.append(tui)
        return tui

    def _launch(self, tui: TUI, name: str, fn_index: int = 0):
        tui._open_launch_form(fn_index)
        assert tui._form_state is not None
        tui._form_state.fields[0].value = name
        count = len(tui._runs)
        tui._submit_form()
        self.assertEqual(len(tui._runs), count + 1)
        self.assertIsNone(tui._form_state)
        return tui._runs[-1]

    def _complete(self, run) -> None:
        self.assertTrue(run.node.done.wait(timeout=5), "runtime node did not complete")
        if run.node.thread is not None:
            run.node.thread.join(timeout=5)
            self.assertFalse(run.node.thread.is_alive())
        assert run.watcher_thread is not None
        run.watcher_thread.join(timeout=5)
        self.assertFalse(run.watcher_thread.is_alive(), "TUI watcher did not finish")
        self.fatal_mock.assert_not_called()

    def _results(self, tui: TUI) -> list[ET.Element]:
        text = tui.log_path.read_text(encoding="utf-8")
        return [ET.fromstring(fragment) for fragment in re.findall(
            r"<session_result>.*?</session_result>", text, re.DOTALL
        )]

    def _handler(self, tui: TUI) -> logging.FileHandler:
        return next(handler for handler in logging.getLogger("netflux").handlers
                    if isinstance(handler, logging.FileHandler)
                    and Path(handler.baseFilename) == tui.log_path)

    def _blocked_function(self, *, cooperative: bool):
        entered = threading.Event()
        release = threading.Event()
        self.releases.append(release)

        def work(ctx: RunContext) -> str:
            entered.set()
            if cooperative:
                assert ctx.cancel_event is not None
                if not ctx.cancel_event.wait(timeout=5):
                    raise AssertionError("cancel was not requested")
                raise CancellationException("stopped through real runtime")
            if not release.wait(timeout=5):
                raise AssertionError("test did not release blocked runtime")
            return "released"

        return self._function("blocked", work), entered, release

    def test_real_watchers_log_each_root_once_and_exclude_successful_children(self) -> None:
        child = self._function("child", lambda ctx: "child-only output")

        def parent(ctx: RunContext) -> str:
            self.assertEqual(ctx.invoke(child, {}).result(), "child-only output")
            return "root output <ok> & complete"

        tui = self._tui(self._function("parent", parent, uses=[child]))
        def failed_callback(_bill) -> None:
            raise RuntimeError("callback failed deliberately")

        callback = Mock(side_effect=failed_callback)
        tui.register_terminal_callback(callback)
        before = time.time()
        runs = [self._launch(tui, f"root {index}") for index in range(18)]
        for run in runs:
            self._complete(run)
        tui.pump_events()
        tui.pump_events()
        tui.on_session_stop()
        results = self._results(tui)
        self.assertEqual(len(results), len(runs))
        self.assertEqual(callback.call_count, len(runs))
        self.assertEqual({entry.findtext("name") for entry in results}, {run.name for run in runs})
        self.assertTrue(all(entry.findtext("content") == "root output <ok> & complete" for entry in results))
        self.assertNotIn("child-only output", tui.log_path.read_text(encoding="utf-8"))
        by_name = {entry.findtext("name"): entry for entry in results}
        for run in runs:
            self.assertEqual(run.latest_view.state, NodeState.Success)
            self.assertEqual(len(run.latest_view.children), 1)
            self.assertEqual(run.latest_view.children[0].state, NodeState.Success)
            created = datetime.fromisoformat(by_name[run.name].findtext("created_at")).timestamp()
            finished = datetime.fromisoformat(by_name[run.name].findtext("finished_at")).timestamp()
            # ISO serialization rounds the float clock to whole microseconds.
            self.assertLessEqual(before, created + 0.000001)
            self.assertLessEqual(created, run.node.started_at + 0.000001)
            self.assertAlmostEqual(finished, run.node.ended_at, places=5)

    def test_real_cooperative_cancel_logs_before_graceful_exit_frame(self) -> None:
        fn, entered, _ = self._blocked_function(cooperative=True)
        tui = self._tui(fn)
        run = self._launch(tui, "cancel me")
        self.assertTrue(entered.wait(timeout=5))
        callback = Mock(side_effect=RuntimeError("billing callback failed"))
        tui.register_terminal_callback(callback)
        self.assertFalse(tui.handle_interrupt())
        self._complete(run)
        tui.pump_events()
        self.assertFalse(tui.should_exit())
        self.assertTrue(tui._exit_after_render)
        self.assertEqual(run.latest_view.state, NodeState.Canceled)
        self.assertEqual(self._results(tui)[0].findtext("content"),
                         "CancellationException: stopped through real runtime")
        callback.assert_called_once_with({})
        tui.render_frame(TerminalSize(columns=80, lines=24), tick=0)
        self.assertTrue(tui.should_exit())
        tui.on_session_stop()
        self.assertEqual(len(self._results(tui)), 1)

    def test_tui_run_flushes_unpumped_completion_and_closes_file_handler(self) -> None:
        fn, entered, release = self._blocked_function(cooperative=False)
        tui = self._tui(fn)
        run = self._launch(tui, "shutdown snapshot")
        self.assertTrue(entered.wait(timeout=5))
        self.assertEqual(self._results(tui), [])
        release.set()
        self._complete(run)
        self.assertFalse(tui._event_queue.empty())
        handler = self._handler(tui)
        driver = ConsoleSessionDriver()
        with patch("sys.stdin.isatty", return_value=False), patch("sys.stdout.isatty", return_value=True), patch(
            "netflux.tui._driver.restore_console"
        ) as restore, patch("netflux.tui.tui.ConsoleSessionDriver", return_value=driver):
            tui.run()
        restore.assert_called_once_with()
        self.assertEqual([entry.findtext("content") for entry in self._results(tui)], ["released"])
        self.assertIsNone(handler.stream)
        self.assertNotIn(handler, logging.getLogger("netflux").handlers)
        self.assertTrue(run.watcher_stop.is_set())
        self.assertFalse(driver._thread_wakeup.is_set())

    def test_real_driver_gracefully_cancels_and_renders_final_state(self) -> None:
        fn, entered, _ = self._blocked_function(cooperative=True)
        tui = self._tui(fn)
        run = self._launch(tui, "driver cooperative cancel")
        self.assertTrue(entered.wait(timeout=5))
        handler = self._handler(tui)
        driver = ConsoleSessionDriver()
        frames: list[str] = []
        original_start = tui.on_session_start

        def virtual_start(*, interactive: bool) -> None:
            original_start(interactive=interactive)
            tui._should_exit = False

        def interrupt_first_frame(frame: str) -> None:
            frames.append(frame)
            if len(frames) == 1:
                raise KeyboardInterrupt
            self._complete(run)
            self.assertLess(len(frames), 10, "driver did not exit after cancellation")

        with patch("sys.stdin.isatty", return_value=False), patch("sys.stdout.isatty", return_value=True), patch(
            "netflux.tui._driver.restore_console"
        ) as restore, patch("netflux.tui._driver.ui_driver", side_effect=interrupt_first_frame), patch.object(
            tui, "on_session_start", side_effect=virtual_start
        ), patch("netflux.tui.tui.ConsoleSessionDriver", return_value=driver):
            tui.run()
        self.assertGreaterEqual(len(frames), 2)
        self.assertTrue(tui.should_exit())
        self.assertEqual(run.latest_view.state, NodeState.Canceled)
        self.assertEqual(len(self._results(tui)), 1)
        self.assertIn("Canceled", frames[-1])
        self.assertIsNone(handler.stream)
        restore.assert_called_once_with()

    def test_real_failed_result_remains_visible_and_logs_once_after_stop(self) -> None:
        def fail(ctx: RunContext) -> str:
            raise ValueError("invalid <argument> & data")

        tui = self._tui(self._function("failure", fail))
        run = self._launch(tui, "failed root")
        self._complete(run)
        with self.assertRaisesRegex(ValueError, "invalid <argument> & data"):
            run.node.result()
        tui.pump_events()
        tui.on_session_stop()
        tui.on_session_stop()
        self.assertEqual(run.latest_view.state, NodeState.Error)
        frame = tui.render_frame(TerminalSize(columns=100, lines=30), tick=0)
        self.assertIn("invalid <argument> & data", frame)
        results = self._results(tui)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].findtext("content"), "ValueError: invalid <argument> & data")

    def test_real_driver_forced_interrupt_restores_terminal_and_preserves_completed_result(self) -> None:
        fn, entered, release = self._blocked_function(cooperative=False)
        done = self._function("done", lambda ctx: "already complete")
        tui = self._tui(fn, done)
        complete_run = self._launch(tui, "finished", fn_index=1)
        self._complete(complete_run)
        pending_run = self._launch(tui, "still running")
        self.assertTrue(entered.wait(timeout=5))
        handler = self._handler(tui)
        driver = ConsoleSessionDriver()
        original_start = tui.on_session_start

        def virtual_start(*, interactive: bool) -> None:
            original_start(interactive=interactive)
            tui._should_exit = False

        with patch("sys.stdin.isatty", return_value=False), patch("sys.stdout.isatty", return_value=True), patch(
            "netflux.tui._driver.restore_console"
        ) as restore, patch("netflux.tui._driver.ui_driver", side_effect=KeyboardInterrupt) as render, patch.object(
            tui, "on_session_start", side_effect=virtual_start
        ), patch("netflux.tui.tui.ConsoleSessionDriver", return_value=driver):
            with self.assertRaises(KeyboardInterrupt):
                tui.run()
        self.assertEqual(render.call_count, 2)
        restore.assert_called_once_with()
        self.assertTrue(tui._global_cancel_requested)
        self.assertTrue(pending_run.cancel_event.is_set())
        self.assertTrue(pending_run.watcher_stop.is_set())
        self.assertIsNone(handler.stream)
        self.assertEqual([entry.findtext("name") for entry in self._results(tui)], ["finished"])
        release.set()
        self._complete(pending_run)

    def test_render_failure_still_logs_latest_result_and_closes_driver_resources(self) -> None:
        fn, entered, release = self._blocked_function(cooperative=False)
        tui = self._tui(fn)
        run = self._launch(tui, "finishes during failed render")
        self.assertTrue(entered.wait(timeout=5))
        handler = self._handler(tui)
        driver = ConsoleSessionDriver()
        original_start = tui.on_session_start

        def virtual_start(*, interactive: bool) -> None:
            original_start(interactive=interactive)
            tui._should_exit = False

        def failed_render(frame: str) -> None:
            release.set()
            self._complete(run)
            raise ValueError("render driver failure")

        with patch("sys.stdin.isatty", return_value=False), patch("sys.stdout.isatty", return_value=True), patch(
            "netflux.tui._driver.restore_console"
        ) as restore, patch("netflux.tui._driver.ui_driver", side_effect=failed_render), patch.object(
            tui, "on_session_start", side_effect=virtual_start
        ), patch("netflux.tui.tui.ConsoleSessionDriver", return_value=driver):
            with self.assertRaisesRegex(ValueError, "render driver failure"):
                tui.run()
        restore.assert_called_once_with()
        self.assertEqual([entry.findtext("content") for entry in self._results(tui)], ["released"])
        text = tui.log_path.read_text(encoding="utf-8")
        self.assertIn("Console session driver failed during ui_driver", text)
        self.assertIsNone(handler.stream)
        self.assertFalse(driver._thread_wakeup.is_set())

    def test_repeated_real_sessions_leave_no_worker_watchers_or_open_handlers(self) -> None:
        initial_threads = set(threading.enumerate())
        fn = self._function("repeated", lambda ctx: "done")
        for index in range(12):
            tui = self._tui(fn)
            run = self._launch(tui, f"repeat {index}")
            self._complete(run)
            handler = self._handler(tui)
            with patch("sys.stdin.isatty", return_value=False), patch("sys.stdout.isatty", return_value=False):
                tui.run()
            self.assertEqual(len(self._results(tui)), 1)
            self.assertIsNone(handler.stream)
            self.assertNotIn(handler, logging.getLogger("netflux").handlers)
            # On Windows rename also proves the FileHandler released its OS handle.
            renamed = tui.log_path.with_suffix(".closed")
            tui.log_path.rename(renamed)
            renamed.rename(tui.log_path)
        leaked = [thread for thread in threading.enumerate() if thread not in initial_threads
                  and thread.name.startswith(("netflux-node-", "netflux-tui-watch-"))]
        self.assertEqual(leaked, [])
