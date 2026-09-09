from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

from ...core import NodeState, NodeView
from ...runtime import Runtime
from ...tui._contracts import TerminalSize
from ...tui._logging import close_tui_logging
from ...tui.tui import TUI, _RunUpdateEvent
from .test_tui_controllers import _make_code_function, _make_output_view, _make_view


class _UnprintableResult:
    def __str__(self) -> str:
        raise ValueError("result cannot be stringified")


class TestTUILoggingLifecycleAudit(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.fn = _make_code_function("root")
        self.runtime = Runtime([self.fn], client_factories={})
        self.tui = TUI(self.runtime, log_path=Path(temporary.name) / "tui.log")
        self.addCleanup(close_tui_logging, self.tui.log_path)

    def _launch(self, view: NodeView, name: str) -> None:
        self.tui._open_launch_form(0)
        assert self.tui._form_state is not None
        self.tui._form_state.fields[0].value = name
        with patch.object(self.runtime, "invoke", return_value=SimpleNamespace(id=view.id)), patch.object(
            self.runtime, "get_view", return_value=view
        ), patch("netflux.tui.tui.threading.Thread.start"):
            self.tui._submit_form()

    def _results(self) -> list[ET.Element]:
        text = self.tui.log_path.read_text(encoding="utf-8")
        return [
            ET.fromstring(fragment)
            for fragment in re.findall(r"<session_result>.*?</session_result>", text, re.DOTALL)
        ]

    def test_log_write_failure_does_not_abort_immediate_launch_or_billing(self) -> None:
        callback = Mock()
        self.tui.register_terminal_callback(callback)
        terminal = _make_output_view(self.fn, output="answer", update_seqnum=1)

        with patch("netflux.tui.tui.log_tui_result", side_effect=OSError("log unavailable")) as write_result:
            self._launch(terminal, "immediate")
            self.tui.register_terminal_callback(callback)
            with patch.object(self.runtime, "get_view", return_value=terminal):
                self.tui.on_session_stop()

        write_result.assert_called_once()
        self.assertEqual(self.tui.log_path.read_text(encoding="utf-8").count("TUI could not log the result"), 1)
        self.assertIsNone(self.tui._form_state)
        self.assertEqual(self.tui._selected_run, 0)
        self.assertTrue(self.tui._runs[0].terminal_browse_applied)
        callback.assert_called_once_with({})

    def test_unprintable_hidden_result_preserves_event_drain_and_callbacks(self) -> None:
        callback = Mock()
        self.tui.register_terminal_callback(callback)
        first = _make_view(self.fn, state=NodeState.Running, update_seqnum=1)
        second = _make_view(self.fn, state=NodeState.Running, update_seqnum=2)
        self._launch(first, "unprintable")
        self._launch(second, "healthy")
        first_done = replace(first, state=NodeState.Success, outputs=_UnprintableResult(), ended_at=3.0)
        second_done = replace(second, state=NodeState.Success, outputs="healthy output", ended_at=4.0)
        self.tui._event_queue.put(_RunUpdateEvent(0, first_done))
        self.tui._event_queue.put(_RunUpdateEvent(1, second_done))

        self.assertTrue(self.tui.pump_events())

        self.assertTrue(self.tui._event_queue.empty())
        self.assertEqual(callback.call_count, 2)
        self.assertTrue(self.tui._runs[0].auto_unread)
        self.assertTrue(self.tui._runs[1].terminal_browse_pending)
        self.assertIn("healthy", [entry.findtext("name") for entry in self._results()])

    def test_stop_continues_to_other_runs_when_one_result_cannot_be_logged(self) -> None:
        callback = Mock()
        self.tui.register_terminal_callback(callback)
        first = _make_view(self.fn, state=NodeState.Running, update_seqnum=1)
        second = _make_view(self.fn, state=NodeState.Running, update_seqnum=2)
        self._launch(first, "first")
        self._launch(second, "second")
        terminal_views = {
            1: replace(first, state=NodeState.Success, outputs="first output", ended_at=3.0),
            2: replace(second, state=NodeState.Success, outputs="second output", ended_at=4.0),
        }
        from ...tui.tui import log_tui_result

        def write_result(*args: object, **kwargs: object) -> None:
            if kwargs["name"] == "first":
                raise OSError("first result cannot be written")
            log_tui_result(*args, **kwargs)

        with patch.object(self.runtime, "get_view", side_effect=terminal_views.__getitem__), patch(
            "netflux.tui.tui.log_tui_result", side_effect=write_result
        ):
            self.tui.on_session_stop()

        self.assertEqual(callback.call_count, 2)
        self.assertTrue(all(run.watcher_stop.is_set() for run in self.tui._runs))
        self.assertEqual(self.tui._runs[1].latest_view, terminal_views[2])
        self.assertEqual([entry.findtext("name") for entry in self._results()], ["second"])

    def test_log_failure_does_not_prevent_graceful_cancel_exit(self) -> None:
        callback = Mock()
        self.tui.register_terminal_callback(callback)
        running = _make_view(self.fn, state=NodeState.Running, update_seqnum=1)
        self._launch(running, "canceled")
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

    def test_result_and_callback_are_once_only_under_callback_reentry(self) -> None:
        running = _make_view(self.fn, state=NodeState.Running, update_seqnum=1)
        self._launch(running, "reentrant")
        terminal = replace(running, state=NodeState.Success, outputs="answer", ended_at=3.0)

        def reenter(_bill: object) -> None:
            self.tui.register_terminal_callback(callback)
            self.tui._event_queue.put(_RunUpdateEvent(0, terminal))
            self.tui.pump_events()
            self.tui.on_session_stop()

        callback = Mock(side_effect=reenter)
        self.tui.register_terminal_callback(callback)
        self.tui._event_queue.put(_RunUpdateEvent(0, terminal))
        with patch.object(self.runtime, "get_view", return_value=terminal):
            self.tui.pump_events()
            self.tui.on_session_stop()

        callback.assert_called_once_with({})
        self.assertEqual(len(self._results()), 1)
