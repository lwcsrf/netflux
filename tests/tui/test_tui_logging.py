from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import datetime
import io
import logging
from pathlib import Path
import re
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

from ...core import CancellationException, CodeFunction, NodeState, NodeView, RunContext
from ...func_lib import ImageResult
from ...runtime import Runtime
from ...tui._contracts import TerminalSize
from ...tui._logging import close_tui_logging, configure_tui_logging, log_tui_result
from ...tui.tui import TUI, _RunRecord, _RunUpdateEvent
from .test_tui_controllers import _make_code_function, _make_output_view, _make_view


def _read_results(path: Path) -> list[ET.Element]:
    # Universal-newline conversion must not hide carriage-return corruption.
    text = path.read_bytes().decode("utf-8")
    return [
        ET.fromstring(fragment)
        for fragment in re.findall(r"<session_result>.*?</session_result>", text, re.DOTALL)
    ]


class _UnprintableResult:
    def __str__(self) -> str:
        raise ValueError("result cannot be stringified")


class _RecordingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class TestTUIResultLogging(unittest.TestCase):
    def setUp(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        self.fn = _make_code_function("root")
        self.runtime = Runtime([self.fn], client_factories={})
        self.tui = TUI(self.runtime, log_path=Path(tmpdir.name) / "tui.log")
        self.addCleanup(close_tui_logging, self.tui.log_path)

    def _launch(self, view: NodeView, *, name: str = "named session") -> None:
        self.tui._open_launch_form(0)
        assert self.tui._form_state is not None
        self.tui._form_state.fields[0].value = name
        node = SimpleNamespace(id=view.id)
        with patch.object(self.runtime, "invoke", return_value=node), patch.object(
            self.runtime, "get_view", return_value=view
        ), patch("netflux.tui.tui.threading.Thread.start"):
            self.tui._submit_form()

    def _results(self) -> list[ET.Element]:
        return _read_results(self.tui.log_path)

    def test_immediate_success_logs_named_xml_result_without_callback(self) -> None:
        created_at = 1_700_000_000.125
        finished_at = created_at + 2.25
        name = 'Research <one> & "two" — café'
        content = "First line: <answer> & café\nSecond line: </session_result> \x1b[31mred"
        view = replace(
            _make_output_view(self.fn, output=content, update_seqnum=1),
            started_at=created_at + 1,
            ended_at=finished_at,
        )

        with patch("netflux.tui.tui.time.time", return_value=created_at):
            self._launch(view, name=name)

        results = self._results()
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(
            [child.tag for child in result], ["created_at", "finished_at", "name", "content"]
        )
        self.assertEqual(result.findtext("name"), name)
        self.assertEqual(result.findtext("content"), content.replace("\x1b", r"\u001b"))
        for tag, expected in (("created_at", created_at), ("finished_at", finished_at)):
            parsed = datetime.fromisoformat(result.findtext(tag) or "")
            self.assertIsNotNone(parsed.utcoffset())
            self.assertEqual(parsed.timestamp(), expected)

    def test_queued_success_logs_root_once_across_callback_reentry_and_shutdown(self) -> None:
        child = _make_output_view(
            _make_code_function("child"), output="child-only result", update_seqnum=3
        )
        running = replace(
            _make_view(self.fn, state=NodeState.Running, update_seqnum=1),
            children=(child,),
        )
        terminal = replace(
            _make_output_view(self.fn, output={"answer": 42}, update_seqnum=2),
            children=(child,),
        )
        self._launch(running)
        self.assertEqual(self._results(), [])

        self.tui._event_queue.put(_RunUpdateEvent(run_index=0, view=terminal))
        self.tui.pump_events()
        self.assertEqual(len(self._results()), 1)

        def reenter(_bill: object) -> None:
            self.tui.register_terminal_callback(callback)
            self.tui._event_queue.put(_RunUpdateEvent(run_index=0, view=terminal))
            self.tui.pump_events()
            with patch.object(self.runtime, "get_view", return_value=terminal):
                self.tui.on_session_stop()

        callback = Mock(side_effect=reenter)
        self.tui.register_terminal_callback(callback)
        self.tui._event_queue.put(_RunUpdateEvent(run_index=0, view=terminal))
        self.tui.pump_events()
        with patch.object(self.runtime, "get_view", return_value=terminal):
            self.tui.on_session_stop()
            self.tui.on_session_stop()

        callback.assert_called_once_with({})
        results = self._results()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].findtext("content"), str(terminal.outputs))
        self.assertNotIn("child-only result", self.tui.log_path.read_text(encoding="utf-8"))

    def test_shutdown_refresh_logs_errors_and_cancellations_without_queued_updates(self) -> None:
        cases = (
            (NodeState.Error, ValueError("bad <input> & value"), "ValueError: bad <input> & value"),
            (NodeState.Canceled, CancellationException("user stopped"), "CancellationException: user stopped"),
            (NodeState.Canceled, None, "Canceled"),
        )
        terminal_views = {}
        for node_id, (state, exception, _) in enumerate(cases, start=1):
            running = _make_view(self.fn, state=NodeState.Running, update_seqnum=node_id)
            self._launch(running, name=f"session {node_id}")
            terminal_views[node_id] = replace(
                running, state=state, exception=exception, ended_at=1_700_000_005.0
            )
        self.assertEqual(self._results(), [])

        with patch.object(self.runtime, "get_view", side_effect=terminal_views.__getitem__):
            self.tui.on_session_stop()
            self.tui.on_session_stop()

        results = self._results()
        self.assertEqual([result.findtext("name") for result in results], ["session 1", "session 2", "session 3"])
        self.assertEqual([result.findtext("content") for result in results], [case[2] for case in cases])

    def test_non_string_outputs_and_image_status_use_text_representation(self) -> None:
        class UnprintableMedia:
            def __str__(self) -> str:
                raise AssertionError("Image media must not be stringified")

            def __repr__(self) -> str:
                raise AssertionError("Image media must not be represented")

        status = "image/png, 100x100, 1000 bytes; unchanged."
        image = ImageResult(
            source_path="/image.png", mime_type="image/png", width=100,
            height=100, status=status, data=UnprintableMedia(), base64_data=UnprintableMedia(),
        )
        values = [None, False, 0, {"answer": [1, 2, "<ok>&"]}, image]
        for index, output in enumerate(values, start=1):
            self._launch(_make_output_view(self.fn, output=output, update_seqnum=index))

        self.assertEqual(
            [entry.findtext("content") for entry in self._results()],
            ["None", "False", "0", str(values[3]), status],
        )

    def test_log_write_failure_does_not_abort_immediate_launch_or_billing(self) -> None:
        callback = Mock()
        self.tui.register_terminal_callback(callback)
        terminal = _make_output_view(self.fn, output="answer", update_seqnum=1)

        with patch("netflux.tui.tui.log_tui_result", side_effect=OSError("log unavailable")) as write_result:
            self._launch(terminal, name="immediate")
            self.tui.register_terminal_callback(callback)
            with patch.object(self.runtime, "get_view", return_value=terminal):
                self.tui.on_session_stop()

        write_result.assert_called_once()
        self.assertEqual(self.tui.log_path.read_text(encoding="utf-8").count("TUI could not log the result"), 1)
        self.assertIsNone(self.tui._form_state)
        callback.assert_called_once_with({})

    def test_unprintable_hidden_result_preserves_event_drain_and_callbacks(self) -> None:
        callback = Mock()
        self.tui.register_terminal_callback(callback)
        first = _make_view(self.fn, state=NodeState.Running, update_seqnum=1)
        second = _make_view(self.fn, state=NodeState.Running, update_seqnum=2)
        self._launch(first, name="unprintable")
        self._launch(second, name="healthy")
        first_done = replace(first, state=NodeState.Success, outputs=_UnprintableResult(), ended_at=3.0)
        second_done = replace(second, state=NodeState.Success, outputs="healthy output", ended_at=4.0)
        self.tui._event_queue.put(_RunUpdateEvent(0, first_done))
        self.tui._event_queue.put(_RunUpdateEvent(1, second_done))

        self.assertTrue(self.tui.pump_events())

        self.assertTrue(self.tui._event_queue.empty())
        self.assertEqual(callback.call_count, 2)
        with patch.object(self.runtime, "get_view", side_effect={1: first_done, 2: second_done}.__getitem__):
            self.tui.on_session_stop()
        self.assertEqual(callback.call_count, 2)
        self.assertEqual([entry.findtext("name") for entry in self._results()], ["healthy"])
        self.assertEqual(self.tui.log_path.read_text(encoding="utf-8").count("TUI could not log the result"), 1)

    def test_stop_continues_to_other_runs_when_one_result_cannot_be_logged(self) -> None:
        callback = Mock()
        self.tui.register_terminal_callback(callback)
        first = _make_view(self.fn, state=NodeState.Running, update_seqnum=1)
        second = _make_view(self.fn, state=NodeState.Running, update_seqnum=2)
        self._launch(first, name="first")
        self._launch(second, name="second")
        terminal_views = {
            1: replace(first, state=NodeState.Success, outputs="first output", ended_at=3.0),
            2: replace(second, state=NodeState.Success, outputs="second output", ended_at=4.0),
        }

        def write_result(*args: object, **kwargs: object) -> None:
            if kwargs["name"] == "first":
                raise OSError("first result cannot be written")
            log_tui_result(*args, **kwargs)

        with patch.object(self.runtime, "get_view", side_effect=terminal_views.__getitem__), patch(
            "netflux.tui.tui.log_tui_result", side_effect=write_result
        ) as write_mock:
            self.tui.on_session_stop()
            self.tui.on_session_stop()

        self.assertEqual(write_mock.call_count, 2)
        self.assertEqual(callback.call_count, 2)
        self.assertTrue(all(run.watcher_stop.is_set() for run in self.tui._runs))
        self.assertEqual(self.tui._runs[1].latest_view, terminal_views[2])
        self.assertEqual([entry.findtext("name") for entry in self._results()], ["second"])

    def test_log_failure_does_not_prevent_graceful_cancel_exit(self) -> None:
        callback = Mock()
        self.tui.register_terminal_callback(callback)
        running = _make_view(self.fn, state=NodeState.Running, update_seqnum=1)
        self._launch(running, name="canceled")
        terminal = replace(running, state=NodeState.Canceled, ended_at=3.0)

        with patch.object(self.runtime, "get_view", return_value=terminal), patch(
            "netflux.tui.tui.log_tui_result", side_effect=OSError("log unavailable")
        ):
            self.assertFalse(self.tui.handle_interrupt())

        callback.assert_called_once_with({})
        self.assertTrue(self.tui._runs[0].cancel_event.is_set())
        self.assertTrue(self.tui._exit_after_render)
        self.tui.render_frame(TerminalSize(columns=80, lines=24), tick=0)
        self.assertTrue(self.tui.should_exit())


class TestTUILogFile(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.logger = logging.getLogger("netflux")
        self.addCleanup(self.logger.setLevel, self.logger.level)
        self.logger.setLevel(logging.DEBUG)
        self.path = configure_tui_logging(Path(self.tmpdir.name) / "session.log")
        self.addCleanup(close_tui_logging, self.path)

    def _result(self, name: str = "session", content: str = "successful result", *, path: Path | None = None) -> None:
        log_tui_result(
            self.path if path is None else path,
            created_at=1_700_000_000.125,
            finished_at=1_700_000_002.375,
            name=name,
            content=content,
        )

    def _text(self, path: Path | None = None) -> str:
        return (self.path if path is None else path).read_text(encoding="utf-8")

    def _results(self, path: Path | None = None) -> list[ET.Element]:
        return _read_results(self.path if path is None else path)

    def _handler(self) -> logging.FileHandler:
        return next(
            handler for handler in self.logger.handlers
            if handler.name == "netflux.tui.file" and isinstance(handler, logging.FileHandler)
        )

    def test_xml_preserves_whitespace_line_endings_and_unicode(self) -> None:
        values = [
            "", " ", "\t", "\n", " \t\n  ",
            "progress 1\rprogress 2\rdone\r",
            "first\r\nsecond\r\n\r\nlast\r\n",
            '  <session_result>\n    <content attr="x">&lt;answer&gt; & value</content>\n'
            "  </session_result>\n\t<!-- comment --> <![CDATA[body]]> ]]></content>\n",
            "café 漢字 🎉 \ud7ff\ue000\ufffd\U00010000\U0010ffff",
        ]
        for index, content in enumerate(values):
            with self.subTest(content=repr(content)):
                self._result(name=content, content=content)
                entries = self._results()
                self.assertEqual(len(entries), index + 1)
                self.assertEqual(entries[-1].findtext("name"), content)
                self.assertEqual(entries[-1].findtext("content"), content)

    def test_xml_invalid_characters_are_visible_in_name_and_content(self) -> None:
        # Cover control ranges, surrogate boundaries, and forbidden BMP noncharacters.
        points = [0, 8, 11, 12, 14, 27, 31, 0xD800, 0xDFFF, 0xFFFE, 0xFFFF]
        content = "|".join(chr(point) for point in points)
        expected = "|".join(f"\\u{point:04x}" for point in points)
        self._result(name=content, content=content)
        entry = self._results()[0]
        self.assertEqual(entry.findtext("name"), expected)
        self.assertEqual(entry.findtext("content"), expected)

    def test_result_only_reaches_matching_file_not_other_handlers_or_console(self) -> None:
        root = logging.getLogger()
        child = logging.getLogger("netflux.tui.result")
        recorders = [_RecordingHandler() for _ in range(3)]
        for logger, recorder in zip((root, self.logger, child), recorders):
            logger.addHandler(recorder)
            self.addCleanup(logger.removeHandler, recorder)
            self.addCleanup(recorder.close)
        unrelated_path = Path(self.tmpdir.name) / "unrelated.log"
        unrelated_handler = logging.FileHandler(unrelated_path, encoding="utf-8")
        self.logger.addHandler(unrelated_handler)
        self.addCleanup(unrelated_handler.close)
        self.addCleanup(self.logger.removeHandler, unrelated_handler)

        self._handler().setFormatter(logging.Formatter("CUSTOM %(levelname)s: %(message)s"))
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            self._result()

        self.assertEqual(len(self._results()), 1)
        self.assertTrue(self._text().startswith("CUSTOM INFO: \n<session_result>"))
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")
        self.assertTrue(all(not recorder.records for recorder in recorders))
        self.assertEqual(unrelated_path.read_text(encoding="utf-8"), "")

    def test_result_does_not_change_ordinary_levels_or_exception_tracebacks(self) -> None:
        child = logging.getLogger("netflux.test_tui_logging")
        old_level = child.level
        self.addCleanup(child.setLevel, old_level)
        child.setLevel(logging.DEBUG)
        for stage in ("before", "after"):
            for level in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL):
                child.log(level, "%s ordinary %s", stage, logging.getLevelName(level))
            if stage == "before":
                self._result()
        try:
            raise ValueError("traceback preserved")
        except ValueError:
            child.exception("original error context")

        text = self._text()
        for stage in ("before", "after"):
            for level in ("DEBUG", "INFO", "WARNING"):
                self.assertNotIn(f"{stage} ordinary {level}", text)
            for level in ("ERROR", "CRITICAL"):
                self.assertIn(f"{stage} ordinary {level}", text)
        self.assertIn("Traceback (most recent call last):", text)
        self.assertIn("ValueError: traceback preserved", text)
        self.assertIn("original error context", text)
        self.assertEqual(self._handler().level, logging.ERROR)
        self.assertEqual(self.logger.level, logging.DEBUG)
        self.assertEqual(child.level, logging.DEBUG)

    def test_shared_handler_lock_prevents_interleaving_with_error_threads(self) -> None:
        workers, writes = 4, 3
        start = threading.Barrier(workers)
        child = logging.getLogger("netflux.test_tui_logging.concurrent")
        old_level = child.level
        self.addCleanup(child.setLevel, old_level)
        child.setLevel(logging.ERROR)

        def write_batch(worker: int) -> None:
            start.wait(timeout=10)
            for item in range(writes):
                marker = f"worker-{worker}-item-{item}"
                if worker % 2:
                    child.error("%s error", marker)
                else:
                    self._result(marker, "\n".join([f"{marker} <payload>&"] * 2))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(write_batch, worker) for worker in range(workers)]
            for future in futures:
                future.result(timeout=30)

        text = self._text()
        results = self._results()
        self.assertEqual(len(results), workers // 2 * writes)
        expected_names = {f"worker-{worker}-item-{item}" for worker in range(0, workers, 2) for item in range(writes)}
        self.assertEqual({entry.findtext("name") for entry in results}, expected_names)
        for entry in results:
            name = entry.findtext("name")
            self.assertEqual(entry.findtext("content"), "\n".join([f"{name} <payload>&"] * 2))
        for worker in range(1, workers, 2):
            for item in range(writes):
                self.assertEqual(text.count(f"worker-{worker}-item-{item} error\n"), 1)

    def test_reconfiguration_appends_without_duplicates_and_rejects_stale_paths(self) -> None:
        self._result("first")
        configure_tui_logging(self.path)
        alias = self.path.parent / "unused-directory" / ".." / self.path.name
        self._result("same file", path=alias)
        self.assertEqual(sum(handler.name == "netflux.tui.file" for handler in self.logger.handlers), 1)
        first_handler = self._handler()

        second_path = configure_tui_logging(Path(self.tmpdir.name) / "second.log")
        self.addCleanup(close_tui_logging, second_path)
        self.assertIsNone(first_handler.stream)
        self.assertNotIn(first_handler, self.logger.handlers)
        self._result("stale first", path=self.path)
        close_tui_logging(self.path)
        self._result("second remains open", path=second_path)

        self.assertEqual([entry.findtext("name") for entry in self._results()], ["first", "same file"])
        self.assertEqual(
            [entry.findtext("name") for entry in self._results(second_path)], ["second remains open"]
        )

    def test_close_is_idempotent_and_late_results_do_not_reopen_or_create_files(self) -> None:
        self._result("before close")
        handler = self._handler()
        close_tui_logging(self.path)
        close_tui_logging(self.path)
        self._result("after close")
        absent = Path(self.tmpdir.name) / "absent.log"
        self._result("unconfigured", path=absent)
        self.assertIsNone(handler.stream)
        self.assertEqual([entry.findtext("name") for entry in self._results()], ["before close"])
        self.assertFalse(absent.exists())


class TestTUILoggingRuntime(unittest.TestCase):
    def setUp(self) -> None:
        # Watcher failures should fail the test instead of terminating the runner.
        fatal = patch.object(TUI, "_fatal", side_effect=AssertionError("unexpected watcher failure"))
        self.fatal_mock = fatal.start()
        self.addCleanup(fatal.stop)

    def _tui(self, function: CodeFunction, release: threading.Event) -> TUI:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tui = TUI(Runtime([function], client_factories={}), log_path=Path(temporary.name) / "tui.log")
        self.addCleanup(self._cleanup_run, tui, release)
        return tui

    def _cleanup_run(self, tui: TUI, release: threading.Event) -> None:
        release.set()
        for run in tui._runs:
            run.cancel_event.set()
            if run.node.thread is not None:
                run.node.thread.join(timeout=5)
            run.watcher_stop.set()
            if run.watcher_thread is not None:
                run.watcher_thread.join(timeout=5)
        close_tui_logging(tui.log_path)

    def _launch(self, tui: TUI, name: str) -> _RunRecord:
        tui._open_launch_form(0)
        assert tui._form_state is not None
        tui._form_state.fields[0].value = name
        count = len(tui._runs)
        tui._submit_form()
        self.assertEqual(len(tui._runs), count + 1)
        return tui._runs[-1]

    def _complete(self, run: _RunRecord) -> None:
        self.assertTrue(run.node.done.wait(timeout=5), "runtime node did not complete")
        if run.node.thread is not None:
            run.node.thread.join(timeout=5)
            self.assertFalse(run.node.thread.is_alive())
        assert run.watcher_thread is not None
        run.watcher_thread.join(timeout=5)
        self.assertFalse(run.watcher_thread.is_alive(), "TUI watcher did not finish")
        self.fatal_mock.assert_not_called()

    def test_real_watchers_log_each_root_once_and_exclude_child_results(self) -> None:
        release = threading.Event()
        child = CodeFunction(name="child", desc="child", args=[], callable=lambda ctx: "child-only result")

        def parent(ctx: RunContext) -> str:
            if not release.wait(timeout=5):
                raise TimeoutError("test did not release root")
            self.assertEqual(ctx.invoke(child, {}).result(), "child-only result")
            return "root output <ok> & complete"

        function = CodeFunction(name="parent", desc="parent", args=[], callable=parent, uses=[child])
        tui = self._tui(function, release)
        callback = Mock(side_effect=RuntimeError("billing callback failed"))
        tui.register_terminal_callback(callback)
        runs = [self._launch(tui, f"root {index}") for index in range(2)]
        self.assertEqual(_read_results(tui.log_path), [])
        release.set()
        for run in runs:
            self._complete(run)
        self.assertEqual(_read_results(tui.log_path), [])

        tui.pump_events()
        self.assertEqual(len(_read_results(tui.log_path)), len(runs))
        tui.pump_events()
        tui.on_session_stop()

        results = _read_results(tui.log_path)
        self.assertEqual(len(results), len(runs))
        self.assertEqual({entry.findtext("name") for entry in results}, {run.name for run in runs})
        self.assertTrue(all(entry.findtext("content") == "root output <ok> & complete" for entry in results))
        self.assertNotIn("child-only result", tui.log_path.read_text(encoding="utf-8"))
        self.assertEqual(callback.call_count, len(runs))

    def test_run_logs_unpumped_completion_and_closes_handler_on_normal_or_failed_render(self) -> None:
        for render_fails in (False, True):
            with self.subTest(render_fails=render_fails):
                release = threading.Event()

                def work(ctx: RunContext) -> str:
                    if not release.wait(timeout=5):
                        raise TimeoutError("test did not release root")
                    return "completed during render"

                function = CodeFunction(name="work", desc="work", args=[], callable=work)
                tui = self._tui(function, release)
                run = self._launch(tui, "shutdown snapshot")
                handler = next(
                    handler for handler in logging.getLogger("netflux").handlers
                    if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == tui.log_path
                )
                original_start = tui.on_session_start

                def start(*, interactive: bool) -> None:
                    original_start(interactive=interactive)
                    tui._should_exit = False

                def render(frame: str) -> None:
                    release.set()
                    self._complete(run)
                    self.assertFalse(tui._event_queue.empty())
                    self.assertEqual(_read_results(tui.log_path), [])
                    tui._should_exit = True
                    if render_fails:
                        raise ValueError("render failed")

                with patch("sys.stdin.isatty", return_value=False), patch(
                    "sys.stdout.isatty", return_value=False
                ), patch.object(tui, "on_session_start", side_effect=start), patch(
                    "netflux.tui._driver.ui_driver", side_effect=render
                ):
                    if render_fails:
                        with self.assertRaisesRegex(ValueError, "render failed"):
                            tui.run()
                    else:
                        tui.run()

                self.assertEqual(
                    [entry.findtext("content") for entry in _read_results(tui.log_path)],
                    ["completed during render"],
                )
                self.assertIsNone(handler.stream)
                self.assertNotIn(handler, logging.getLogger("netflux").handlers)
