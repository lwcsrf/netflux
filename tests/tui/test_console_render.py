from __future__ import annotations

import re
import subprocess
import unittest
from unittest.mock import patch

from ...core import (
    AgentFunction,
    CodeFunction,
    ModelTextPart,
    NodeState,
    NodeView,
    RunContext,
    TokenBill,
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


class TestConsoleRender(unittest.TestCase):
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

        self.assertEqual(rendered, "g31pp[CR:91k Reg:131k Out:6.5k]")

    def test_cache_write_uses_thousand_rounding(self) -> None:
        rendered = ConsoleRender._format_token_bill_fields(
            TokenBill(input_tokens_cache_write=1501)
        )

        self.assertEqual(rendered, "CW:2k")

    def test_copy_terminal_result_uses_raw_root_result_text(self) -> None:
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
        renderer.assign_view(view)

        with patch("netflux.tui.console._copy_text_to_clipboard", return_value=True) as copy_mock:
            self.assertTrue(renderer.copy_terminal_result())

        copy_mock.assert_called_once_with("# Summary\n\n- first item")

    def test_copy_text_to_clipboard_uses_win32_unicode_path(self) -> None:
        with patch("netflux.tui.console.sys.platform", "win32"), patch(
            "netflux.tui.console._copy_text_to_clipboard_windows",
            return_value=True,
        ) as win_copy, patch("netflux.tui.console.subprocess.run") as run_mock:
            self.assertTrue(_copy_text_to_clipboard("à ù • │ emoji 😀 CJK 漢字"))

        win_copy.assert_called_once_with("à ù • │ emoji 😀 CJK 漢字")
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
        renderer.assign_view(agent_view)

        with patch("netflux.tui.console._copy_text_to_clipboard", return_value=True) as copy_mock:
            self.assertTrue(renderer.copy_terminal_result())

        copy_mock.assert_called_once_with("final transcript result")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
