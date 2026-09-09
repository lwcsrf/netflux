from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
import io
import logging
from pathlib import Path
import re
import tempfile
import threading
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from ...tui._logging import close_tui_logging, configure_tui_logging, log_tui_result


class _RecordingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class TestTUILoggingHandlerAudit(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.logger = logging.getLogger("netflux")
        self.previous_level = self.logger.level
        self.addCleanup(self.logger.setLevel, self.previous_level)
        self.logger.setLevel(logging.DEBUG)
        self.path = configure_tui_logging(Path(self.tmpdir.name) / "session.log")
        self.addCleanup(close_tui_logging)

    def _result(self, name: str = "session", content: str = "successful result", *, path=None) -> None:
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
        return [
            ET.fromstring(fragment)
            for fragment in re.findall(r"<session_result>.*?</session_result>", self._text(path), re.DOTALL)
        ]

    def _handler(self) -> logging.FileHandler:
        return next(
            handler for handler in self.logger.handlers
            if handler.name == "netflux.tui.file" and isinstance(handler, logging.FileHandler)
        )

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

        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            self._result()

        self.assertEqual(len(self._results()), 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), "")
        self.assertTrue(all(not recorder.records for recorder in recorders))
        self.assertEqual(unrelated_path.read_text(encoding="utf-8"), "")

    def test_result_does_not_change_ordinary_levels_or_exception_tracebacks(self) -> None:
        child = logging.getLogger("netflux.handler_audit")
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

    def test_result_bypass_is_independent_of_logger_and_global_thresholds(self) -> None:
        child = logging.getLogger("netflux.tui.result")
        old_disabled, old_level = child.disabled, child.level
        old_disable = logging.root.manager.disable
        self.addCleanup(setattr, child, "disabled", old_disabled)
        self.addCleanup(child.setLevel, old_level)
        self.addCleanup(logging.disable, old_disable)
        self.logger.setLevel(logging.CRITICAL)
        child.setLevel(logging.CRITICAL)
        child.disabled = True
        logging.disable(logging.CRITICAL)

        self._result()

        self.assertEqual(len(self._results()), 1)
        self.assertTrue(child.disabled)
        self.assertEqual(child.level, logging.CRITICAL)
        self.assertEqual(self.logger.level, logging.CRITICAL)
        self.assertEqual(logging.root.manager.disable, logging.CRITICAL)

    def test_shared_handler_lock_prevents_interleaving_with_error_threads(self) -> None:
        workers, writes = 8, 30
        start = threading.Barrier(workers)
        child = logging.getLogger("netflux.handler_audit.concurrent")
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
                    self._result(marker, "\n".join([f"{marker} <payload>&"] * 20))

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
            self.assertEqual(entry.findtext("content"), "\n".join([f"{name} <payload>&"] * 20))
        for worker in range(1, workers, 2):
            for item in range(writes):
                self.assertEqual(text.count(f"worker-{worker}-item-{item} error\n"), 1)

    def test_result_is_flushed_before_return_and_uses_existing_formatter(self) -> None:
        handler = self._handler()
        handler.setFormatter(logging.Formatter("CUSTOM %(levelname)s %(name)s: %(message)s"))
        self._result()
        self.assertTrue(self._text().startswith("CUSTOM INFO netflux.tui.result: \n<session_result>"))
        self.assertEqual(len(self._results()), 1)
        self.assertFalse(handler.stream.closed)

    def test_custom_handler_filters_remain_honored(self) -> None:
        class SkipOne(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                return "skip this session" not in record.getMessage()

        handler = self._handler()
        custom_filter = SkipOne()
        handler.addFilter(custom_filter)
        self._result("skip this session")
        self._result("keep this session")
        self.assertEqual([result.findtext("name") for result in self._results()], ["keep this session"])

    def test_reconfiguration_closes_previous_file_without_cross_routing_results(self) -> None:
        first_handler = self._handler()
        self._result("first")
        second_path = configure_tui_logging(Path(self.tmpdir.name) / "second.log")
        self.assertIsNone(first_handler.stream)
        self.assertNotIn(first_handler, self.logger.handlers)

        self._result("stale first", path=self.path)
        self._result("second", path=second_path)
        self.assertEqual([entry.findtext("name") for entry in self._results()], ["first"])
        self.assertEqual([entry.findtext("name") for entry in self._results(second_path)], ["second"])
        close_tui_logging(self.path)
        self._result("second remains open", path=second_path)
        self.assertEqual(len(self._results(second_path)), 2)

    def test_reconfiguration_same_path_appends_without_duplicate_handlers(self) -> None:
        for index in range(5):
            configure_tui_logging(self.path)
            self._result(str(index))
        self.assertEqual([entry.findtext("name") for entry in self._results()], list(map(str, range(5))))
        self.assertEqual(sum(handler.name == "netflux.tui.file" for handler in self.logger.handlers), 1)

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

    def test_path_normalization_targets_configured_file(self) -> None:
        alias = self.path.parent / "unused-directory" / ".." / self.path.name
        self._result("normalized", path=alias)
        self.assertEqual([entry.findtext("name") for entry in self._results()], ["normalized"])

    def test_file_write_error_uses_standard_logging_error_handling(self) -> None:
        handler = self._handler()
        with patch.object(handler.stream, "write", side_effect=OSError("simulated disk full")), patch.object(
            handler, "handleError"
        ) as handle_error:
            self._result("disk full")
        handle_error.assert_called_once()
        self.assertEqual(handle_error.call_args.args[0].name, "netflux.tui.result")
        self._result("recovered")
        self.assertEqual([entry.findtext("name") for entry in self._results()], ["recovered"])
