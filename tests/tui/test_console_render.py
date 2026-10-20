from __future__ import annotations

import ctypes
import re
import subprocess
import unittest
from dataclasses import replace
from threading import Event
from unittest.mock import Mock, patch

from rich.cells import cell_len

from ...core import (
    AgentFunction,
    CodeFunction,
    Continuation,
    ModelTextPart,
    NodeState,
    NodeView,
    RetriableException,
    RunContext,
    TokenBill,
    TokenUsage,
    TranscriptPosition,
    ThinkingBlockPart,
)
from ...providers import Provider
from ...tui import ConsoleRender
from ...tui.console import _clipboard_copy_failure_message, _copy_text_to_clipboard


_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _make_code_function(name: str) -> CodeFunction:
    def _callable(ctx: RunContext) -> str:
        del ctx
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


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _event_view(**changes) -> NodeView:
    return replace(NodeView(
        id=1, fn=_make_agent_function("root"), inputs={},
        state=NodeState.Running, outputs=None, exception=None,
        children=(), usage=None, transcript=(), started_at=100.0,
        ended_at=None, update_seqnum=1,
    ), **changes)


class TestConsoleRender(unittest.TestCase):
    def test_retry_headers_fit_narrow_panes_with_a_visible_ellipsis(self) -> None:
        message = "retry 漢字😀 " * 20
        child = _event_view(id=2, intermediate_events=(
            RetriableException(TranscriptPosition(0), ValueError(message), attempt=0),
        ))
        renderer = ConsoleRender(follow=False)
        renderer.render_body(width=40, height=10, view=_event_view(children=(child,)), tick=0)
        row = next(i for i, info in enumerate(renderer._line_infos) if info.key == "event:2:0")
        line = _strip_ansi(renderer._lines[row])
        self.assertLessEqual(cell_len(line), 40)
        self.assertTrue(line.endswith("…"))
        self.assertEqual(renderer._line_infos[row].copy_text, message)

    def test_events_split_text_preserve_copy_selection_and_final_result(self) -> None:
        events = (
            RetriableException(
                TranscriptPosition(0, 6), ValueError("first line\nsecond line"),
                attempt=0, backoff_seconds=5, backoff_started_at=100,
            ),
            Continuation(TranscriptPosition(0, 6)),
        )
        view = _event_view(
            transcript=(ModelTextPart(text="alpha "),),
            intermediate_events=events,
        )
        renderer = ConsoleRender(follow=False)
        with patch("netflux.tui.console.time.time", return_value=103):
            renderer.render_body(width=100, height=30, view=view, tick=0)
        retry_row = next(i for i, info in enumerate(renderer._line_infos) if info.key == "event:1:0")
        renderer._set_cursor(retry_row)
        renderer.toggle_expanded()

        final = replace(
            view, transcript=(ModelTextPart(text="alpha beta"),),
            state=NodeState.Success, outputs="duplicate output", ended_at=107,
        )
        renderer.render_body(width=100, height=30, view=final, tick=0)
        self.assertEqual(renderer._line_infos[renderer._cursor].key, "event:1:0")
        self.assertFalse(renderer._collapse_overrides["event:1:0"])
        keys = [info.key for info in renderer._line_infos if info.key]
        self.assertEqual(keys, [
            "n:1", "tp:1:0:model", "event:1:0", "event:1:0:backoff",
            "event:1:1", "tp:1:0:model:6",
        ])
        self.assertIn("ValueError | attempt 0 | first line", _strip_ansi("\n".join(renderer._lines)))
        self.assertIn("second line", _strip_ansi("\n".join(renderer._lines)))
        self.assertTrue(renderer.focus_terminal_result())
        with patch("netflux.tui.console._render_markdown_lines") as markdown:
            renderer.render_body(width=100, height=30, tick=0)
        markdown.assert_not_called()
        rendered = _strip_ansi("\n".join(renderer._lines))
        self.assertEqual(rendered.count("alpha "), 1)
        self.assertEqual(rendered.count("beta"), 1)
        self.assertNotIn("duplicate output", rendered)
        self.assertFalse(renderer._collapse_overrides["tp:1:0:model"])
        self.assertFalse(renderer._collapse_overrides["tp:1:0:model:6"])
        for row, info in enumerate(renderer._line_infos):
            if "tp:1:0:model" in info.anchors:
                renderer._set_cursor(row)
                with patch("netflux.tui.console._copy_text_to_clipboard", return_value=True) as copy:
                    self.assertTrue(renderer.copy_selected_text())
                copy.assert_called_once_with("alpha beta")

    def test_backoff_clock_and_cancellation_are_derived_from_snapshot(self) -> None:
        event = RetriableException(
            TranscriptPosition(0), ValueError("retry"), attempt=0,
            backoff_seconds=5, backoff_started_at=103,
        )
        events = (
            replace(event, backoff_seconds=2, backoff_started_at=100),
            replace(event, attempt=1, backoff_seconds=None, backoff_started_at=None),
            replace(event, attempt=2),
        )
        cases = (
            (NodeState.Running, None, 106, "│  ⠋ ▸ ⏱️ backoff 3.0s / 5.0s"),
            (NodeState.Running, None, 108, "│    ▸ ⏱️ backoff 5.0s / 5.0s"),
            (NodeState.Canceled, 107, 200, "│    ▸ ⏱️ backoff canceled (5.0s)"),
            (NodeState.Canceled, 109, 106, "│    ▸ ⏱️ backoff 5.0s / 5.0s"),
        )
        for state, ended_at, now, expected in cases:
            with self.subTest(state=state, ended_at=ended_at):
                view = _event_view(intermediate_events=events, state=state, ended_at=ended_at)
                renderer = ConsoleRender(follow=False)
                with patch("netflux.tui.console.time.time", return_value=now):
                    renderer.render_body(width=100, height=20, view=view, tick=0)
                    rows = {
                        info.key: _strip_ansi(line)
                        for info, line in zip(renderer._line_infos, renderer._lines) if info.key
                    }
                    self.assertEqual(rows["event:1:0:backoff"], "│    ▸ ⏱️ backoff 2.0s / 2.0s")
                    self.assertNotIn("event:1:1:backoff", rows)
                    row = next(i for i, info in enumerate(renderer._line_infos) if info.key == "event:1:2:backoff")
                    self.assertEqual(expected, _strip_ansi(renderer._lines[row]))
                    self.assertEqual(renderer._line_infos[row].toggle_col, 5)
                    line_count = len(renderer._lines)
                    renderer._set_cursor(row)
                    renderer.toggle_expanded()
                    renderer.render_body(width=100, height=20, tick=1)
                    expanded = expected.replace("▸", "▾").replace("⠋", "⠙")
                    self.assertEqual(_strip_ansi(renderer._lines[row]), expanded)
                    self.assertEqual(len(renderer._lines), line_count)

    def test_exception_jumps_follow_visible_agent_errors_and_skip_cancellation(self) -> None:
        retry = RetriableException(TranscriptPosition(0), ValueError("retry"), attempt=0)
        failed = _event_view(id=2, state=NodeState.Error, exception=ValueError("failure"))
        canceled = _event_view(
            id=3, state=NodeState.Canceled, exception=RuntimeError("canceled"),
            intermediate_events=(retry,), ended_at=103,
        )
        code = replace(failed, id=4, fn=_make_code_function("code"))
        hidden = replace(failed, id=5)
        collapsed = replace(failed, id=6, children=(hidden,), intermediate_events=(retry,))
        root = _event_view(children=(failed, canceled, code, collapsed))
        renderer = ConsoleRender(follow=True)
        renderer._collapse_overrides["n:6"] = True
        renderer.render_body(width=100, height=3, view=root, tick=0)
        self.assertEqual(
            [info.key for info in renderer._line_infos if info.is_exception_target],
            ["ae:2", "event:3:0", "n:6"],
        )
        self.assertTrue(renderer.right_pane_context().can_jump_exceptions)
        renderer._set_cursor(0, disable_follow=False)
        for action, keys in (
            ("next_exception", ("ae:2", "event:3:0", "n:6", "n:6")),
            ("prev_exception", ("event:3:0", "ae:2", "ae:2")),
        ):
            for key in keys:
                renderer.apply_action(action)
                renderer.render_body(width=100, height=3, tick=0)
                self.assertEqual(renderer._line_infos[renderer._cursor].key, key)
                self.assertTrue(0 <= renderer._cursor - renderer._scroll_offset < 3)
        self.assertFalse(renderer._follow_mode)
        renderer.expand_all_nodes()
        renderer.render_body(width=100, height=3, tick=0)
        self.assertEqual(
            [info.key for info in renderer._line_infos if info.is_exception_target],
            ["ae:2", "event:3:0", "event:6:0", "ae:5", "ae:6"],
        )

    def test_continuation_between_thinking_blocks_is_not_coalesced_away(self) -> None:
        view = _event_view(
            transcript=(
                ThinkingBlockPart(content="", signature=""),
                ThinkingBlockPart(content="reasoning", signature=""),
            ),
            intermediate_events=(Continuation(TranscriptPosition(1)),),
        )
        renderer = ConsoleRender()
        renderer.render_body(width=100, height=20, view=view, tick=0)
        keys = [info.key for info in renderer._line_infos]
        self.assertEqual(keys, ["n:1", "tp:1:0:thinking", "event:1:0", "tp:1:1:thinking"])
        continuation_line = renderer._lines[2]
        self.assertEqual(continuation_line, renderer._lines[1].replace("➰ thinking", "↪ continuation"))
        self.assertTrue(renderer._line_infos[2].expandable)
        self.assertIsNone(renderer._line_infos[2].copy_text)
        renderer._set_cursor(2)
        renderer.toggle_expanded()
        renderer.render_body(width=100, height=20, tick=0)
        self.assertEqual(renderer._lines[2], continuation_line.replace("▸", "▾"))
        self.assertEqual([info.key for info in renderer._line_infos], keys)

    def test_toggle_node_details_collapses_their_section_or_owning_node(self) -> None:
        target = NodeView(
            id=2,
            fn=_make_agent_function("target"),
            inputs={"first": "one", "last": "two"},
            state=NodeState.Success,
            outputs=None,
            exception=None,
            children=(),
            usage=None,
            transcript=(),
            started_at=0.0,
            ended_at=0.0,
            update_seqnum=1,
        )
        cases = (
            (NodeState.Success, TokenUsage(output_tokens_total=7), None, "n:2", (0,)),
            (NodeState.Error, None, ValueError("first line\nlast line"), "ae:2", (0, 1, 2)),
            (NodeState.Canceled, None, RuntimeError("first line\nlast line"), "ae:2", (0, 1, 2)),
        )
        for state, usage, exception, anchor, offsets in cases:
            for nested in (False, True):
                for offset in offsets:
                    with self.subTest(state=state, nested=nested, offset=offset):
                        view = replace(target, state=state, usage=usage, exception=exception)
                        if nested:
                            view = replace(
                                target,
                                id=1,
                                fn=_make_agent_function("parent"),
                                children=(replace(target, id=3), view),
                            )
                        renderer = ConsoleRender(follow=False)
                        renderer.render_body(width=80, height=30, view=view, tick=0)
                        rows = [
                            idx for idx, info in enumerate(renderer._line_infos)
                            if anchor in info.anchors
                        ]
                        if exception is not None:
                            self.assertTrue(_strip_ansi(renderer._lines[rows[0]]).lstrip("│ ").startswith("▾ "))
                        renderer._set_cursor(rows[offset])

                        renderer.toggle_expanded()

                        collapsed_key = anchor if exception is not None else "n:2"
                        self.assertEqual(renderer._collapse_overrides, {collapsed_key: True})
                        renderer.render_body(width=80, height=30, tick=0)
                        self.assertEqual(renderer._line_infos[renderer._cursor].key, collapsed_key)
                        keys = {info.key for info in renderer._line_infos}
                        self.assertEqual("aa:2:last" in keys, exception is not None)
                        if nested:
                            self.assertTrue({"n:1", "n:3", "aa:1:last", "aa:3:last"} <= keys)

    def test_toggle_expanded_content_collapses_its_section(self) -> None:
        view = NodeView(
            id=1,
            fn=_make_agent_function("root"),
            inputs={"first": "argument one\nargument two", "last": "other"},
            state=NodeState.Success,
            outputs=None,
            exception=None,
            children=(),
            usage=None,
            transcript=(ModelTextPart(text="transcript one\n\ntranscript two"),),
            started_at=0.0,
            ended_at=0.0,
            update_seqnum=1,
        )
        for key in ("aa:1:first", "tp:1:0:model"):
            with self.subTest(section=key):
                renderer = ConsoleRender(follow=False)
                renderer._collapse_overrides[key] = False
                renderer.render_body(width=80, height=20, view=view, tick=0)
                content_rows = [
                    idx for idx, info in enumerate(renderer._line_infos)
                    if info.key is None and key in info.anchors
                ]
                self.assertGreaterEqual(len(content_rows), 2)
                renderer._set_cursor(content_rows[-1])

                renderer.toggle_expanded()

                self.assertEqual(renderer._collapse_overrides, {key: True})
                renderer.render_body(width=80, height=20, tick=0)
                self.assertEqual(renderer._line_infos[renderer._cursor].key, key)
                self.assertFalse(any(
                    info.key is None and key in info.anchors
                    for info in renderer._line_infos
                ))

    def test_toggle_empty_agent_header_collapses_parent_not_previous_sibling(self) -> None:
        leaf = NodeView(
            id=4,
            fn=_make_agent_function("leaf"),
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
        )
        sibling = replace(leaf, id=3, inputs={"argument": "value"})
        parent = replace(leaf, id=2, children=(sibling, leaf))
        root = replace(leaf, id=1, children=(parent,))
        for action in ("keyboard", "mouse"):
            with self.subTest(action=action):
                renderer = ConsoleRender(follow=False)
                renderer.render_body(width=80, height=20, view=root, tick=0)
                row = next(
                    idx for idx, info in enumerate(renderer._line_infos)
                    if info.key == "n:4"
                )
                self.assertFalse(renderer._line_infos[row].expandable)
                renderer._set_cursor(row)

                if action == "mouse":
                    renderer.handle_mouse_event(x=0, y=row)
                else:
                    renderer.toggle_expanded()

                self.assertEqual(renderer._collapse_overrides, {"n:2": True})
                renderer.render_body(width=80, height=20, tick=0)
                self.assertEqual(renderer._line_infos[renderer._cursor].key, "n:2")
                self.assertEqual(
                    [info.key for info in renderer._line_infos], ["n:1", "n:2"]
                )

    def test_toggle_cancellation_footer_preserves_fallback_and_stops_following(self) -> None:
        cancel_event = Event()
        cancel_event.set()
        view = NodeView(
            id=1,
            fn=_make_agent_function("root"),
            inputs={"query": "first line\nlast line"},
            state=NodeState.Running,
            outputs=None,
            exception=None,
            children=(),
            usage=None,
            transcript=(),
            started_at=0.0,
            ended_at=None,
            update_seqnum=1,
        )
        for footer_offset in (1, 2):
            for expanded in (False, True):
                with self.subTest(footer_offset=footer_offset, expanded=expanded):
                    renderer = ConsoleRender(cancel_event=cancel_event, follow=True)
                    if expanded:
                        renderer._collapse_overrides["aa:1:query"] = False
                    renderer.render_body(width=80, height=20, view=view, tick=0)
                    self.assertIn("Cancelation pending", _strip_ansi(renderer._lines[-1]))
                    renderer._set_cursor(len(renderer._lines) - footer_offset, disable_follow=False)
                    self.assertTrue(renderer._follow_mode)

                    renderer.toggle_expanded()

                    self.assertEqual(renderer._collapse_overrides, {"aa:1:query": True})
                    self.assertFalse(renderer._follow_mode)
                    renderer.render_body(width=80, height=20, tick=0)
                    self.assertEqual(renderer._line_infos[renderer._cursor].key, "aa:1:query")

    def test_total_token_bill_uses_compact_k_suffixes(self) -> None:
        rendered = ConsoleRender._format_total_token_bill(
            {
                Provider.Gemini: TokenBill(
                    input_tokens_cache_read=91234,
                    input_tokens_regular=131499,
                    output_tokens_total=6499,
                )
            }
        )

        self.assertEqual(rendered, "g4a[CR:91k Reg:131k Out:6.5k]")

    def test_cache_write_uses_thousand_rounding(self) -> None:
        rendered = ConsoleRender._format_token_bill_fields(
            TokenBill(input_tokens_cache_write=1501)
        )

        self.assertEqual(rendered, "CW:2k")

    def test_copy_selected_result_uses_raw_root_result_text(self) -> None:
        fn = _make_code_function("root")
        view = NodeView(
            id=1,
            fn=fn,
            inputs={},
            state=NodeState.Success,
            outputs="# Summary\n\n- first item",
            exception=None,
            children=(),
            usage=None,
            transcript=(),
            started_at=0.0,
            ended_at=0.0,
            update_seqnum=1,
        )
        renderer = ConsoleRender()
        renderer.render_body(width=80, height=10, view=view, tick=0)
        self.assertTrue(renderer.focus_terminal_result())
        renderer.render_body(width=80, height=10, tick=0)

        with patch("netflux.tui.console._copy_text_to_clipboard", return_value=True) as copy_mock:
            self.assertTrue(renderer.copy_selected_text())

        copy_mock.assert_called_once_with("# Summary\n\n- first item")

    def test_copy_text_to_clipboard_uses_win32_unicode_path(self) -> None:
        text = "à ù • │ emoji 😀🧪 CJK 漢字 tail"
        expected = ctypes.create_unicode_buffer(text)
        destination = ctypes.create_string_buffer(ctypes.sizeof(expected))
        kernel32, user32 = Mock(), Mock()
        kernel32.GlobalAlloc.return_value = 1
        kernel32.GlobalLock.return_value = ctypes.addressof(destination)
        user32.OpenClipboard.return_value = True
        user32.EmptyClipboard.return_value = True
        user32.SetClipboardData.return_value = 1
        with patch("netflux.tui.console.sys.platform", "win32"), patch(
            "ctypes.WinDLL", side_effect=[kernel32, user32], create=True,
        ), patch("netflux.tui.console.subprocess.run") as run_mock:
            self.assertTrue(_copy_text_to_clipboard(text))

        kernel32.GlobalAlloc.assert_called_once_with(0x0042, ctypes.sizeof(expected))
        self.assertEqual(destination.raw, bytes(expected))
        user32.SetClipboardData.assert_called_once_with(13, 1)
        kernel32.GlobalFree.assert_not_called()
        run_mock.assert_not_called()

    def test_copy_text_to_clipboard_prefers_wl_copy_on_linux(self) -> None:
        def _which(name: str) -> str | None:
            mapping = {
                "wl-copy": "/usr/bin/wl-copy",
                "xclip": "/usr/bin/xclip",
                "xsel": "/usr/bin/xsel",
            }
            return mapping.get(name)

        with patch("netflux.tui.console.sys.platform", "linux"), patch(
            "netflux.tui.console.shutil.which",
            side_effect=_which,
        ), patch("netflux.tui.console.subprocess.run") as run_mock:
            run_mock.return_value = None
            self.assertTrue(_copy_text_to_clipboard("hello"))

        run_mock.assert_called_once()
        self.assertEqual(run_mock.call_args.args[0], ["wl-copy"])
        self.assertEqual(run_mock.call_args.kwargs["stdout"], subprocess.DEVNULL)
        self.assertEqual(run_mock.call_args.kwargs["stderr"], subprocess.DEVNULL)
        self.assertNotIn("capture_output", run_mock.call_args.kwargs)

    def test_copy_text_to_clipboard_falls_back_to_xclip_on_linux(self) -> None:
        def _which(name: str) -> str | None:
            mapping = {
                "wl-copy": None,
                "xclip": "/usr/bin/xclip",
                "xsel": "/usr/bin/xsel",
            }
            return mapping.get(name)

        with patch("netflux.tui.console.sys.platform", "linux"), patch(
            "netflux.tui.console.shutil.which",
            side_effect=_which,
        ), patch("netflux.tui.console.subprocess.run") as run_mock:
            run_mock.return_value = None
            self.assertTrue(_copy_text_to_clipboard("hello"))

        run_mock.assert_called_once()
        self.assertEqual(run_mock.call_args.args[0], ["xclip", "-selection", "clipboard"])
        self.assertEqual(run_mock.call_args.kwargs["stdout"], subprocess.DEVNULL)
        self.assertEqual(run_mock.call_args.kwargs["stderr"], subprocess.DEVNULL)
        self.assertNotIn("capture_output", run_mock.call_args.kwargs)

    def test_linux_clipboard_failure_message_mentions_install_when_no_backend(self) -> None:
        with patch("netflux.tui.console.sys.platform", "linux"), patch(
            "netflux.tui.console.shutil.which",
            return_value=None,
        ), patch("netflux.tui.console.subprocess.run") as run_mock:
            self.assertFalse(_copy_text_to_clipboard("hello"))
            self.assertEqual(
                _clipboard_copy_failure_message(),
                "Clipboard unavailable. Install wl-copy, xclip, or xsel.",
            )

        run_mock.assert_not_called()

    def test_focus_terminal_result_expands_root_result_and_renders_markdown(self) -> None:
        fn = _make_code_function("root")
        view = NodeView(
            id=1,
            fn=fn,
            inputs={},
            state=NodeState.Success,
            outputs="# Summary\n\n- first item\n- second item",
            exception=None,
            children=(),
            usage=None,
            transcript=(),
            started_at=0.0,
            ended_at=0.0,
            update_seqnum=1,
        )
        renderer = ConsoleRender(follow=False)

        renderer.render_body(width=80, height=10, view=view, tick=0)
        self.assertTrue(renderer.focus_terminal_result())
        rendered = _strip_ansi(renderer.render_body(width=80, height=10, tick=0))

        self.assertIn("Summary", rendered)
        self.assertIn("• first item", rendered)
        self.assertIn("• second item", rendered)
        self.assertNotIn("- first item", rendered)
        self.assertNotIn("# Summary", rendered)

    def test_agent_transcript_model_result_is_copyable_when_outputs_missing(self) -> None:
        fn = _make_agent_function("root")
        agent_view = NodeView(
            id=1,
            fn=fn,
            inputs={},
            state=NodeState.Success,
            outputs=None,
            exception=None,
            children=(),
            usage=None,
            transcript=(ModelTextPart(text="final transcript result"),),
            started_at=0.0,
            ended_at=0.0,
            update_seqnum=1,
        )
        renderer = ConsoleRender()
        renderer.render_body(width=80, height=10, view=agent_view, tick=0)
        self.assertTrue(renderer.focus_terminal_result())
        renderer.render_body(width=80, height=10, tick=0)

        with patch("netflux.tui.console._copy_text_to_clipboard", return_value=True) as copy_mock:
            self.assertTrue(renderer.copy_selected_text())

        copy_mock.assert_called_once_with("final transcript result")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
