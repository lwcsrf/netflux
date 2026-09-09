from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import logging
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

from ...core import CancellationException, NodeState, NodeView
from ...runtime import Runtime
from ...tui._logging import close_tui_logging
from ...tui.tui import TUI, _RunUpdateEvent
from .test_tui_controllers import _make_code_function, _make_output_view, _make_view


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
        text = self.tui.log_path.read_text(encoding="utf-8")
        return [
            ET.fromstring(fragment)
            for fragment in re.findall(r"<session_result>.*?</session_result>", text, re.DOTALL)
        ]

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

    def test_queued_success_logs_root_once_across_callback_and_shutdown(self) -> None:
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

        callback = Mock()
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

    def test_only_session_results_bypass_error_logging_threshold(self) -> None:
        logger = logging.getLogger("netflux.test_tui_result_logging")
        previous_level = logger.level
        self.addCleanup(logger.setLevel, previous_level)
        logger.setLevel(logging.DEBUG)
        for level in (logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL):
            logger.log(level, "ordinary-%s-marker", logging.getLevelName(level))

        self._launch(_make_output_view(self.fn, output="successful result", update_seqnum=1))

        text = self.tui.log_path.read_text(encoding="utf-8")
        for level in ("DEBUG", "INFO", "WARNING"):
            self.assertNotIn(f"ordinary-{level}-marker", text)
        for level in ("ERROR", "CRITICAL"):
            self.assertIn(f"ordinary-{level}-marker", text)
        self.assertEqual(len(self._results()), 1)
        self.assertEqual(self._results()[0].findtext("content"), "successful result")
