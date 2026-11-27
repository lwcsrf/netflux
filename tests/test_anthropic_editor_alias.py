import tempfile
import unittest
from contextlib import nullcontext
from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path
from typing import Sequence, cast
from unittest.mock import MagicMock, Mock

if find_spec("anthropic") is None:
    raise unittest.SkipTest("Anthropic SDK is optional")

import anthropic
from anthropic.types import (
    Message, RawMessageStartEvent, TextBlockParam, ToolResultBlockParam, ToolUseBlockParam,
)
from anthropic.types.message_create_params import MessageCreateParamsBase

from ..core import AgentFunction, AgentNode, CodeFunction, Function, ToolResultPart, ToolUsePart
from ..func_lib.text_editor_func import TextEditor
from ..providers import ModelNames, Provider
from ..runtime import Runtime


NATIVE_EDITOR = "str_replace_based_edit_tool"


def _tool_use(name: str, call_id: str, **args: object) -> ToolUseBlockParam:
    return ToolUseBlockParam(type="tool_use", id=call_id, name=name, input=args)


def _run_tool_turn(
    uses: Sequence[Function],
    calls: Sequence[ToolUseBlockParam],
) -> tuple[AgentNode, list[MessageCreateParamsBase]]:
    fn = AgentFunction(
        name="agent", desc="", args=[], uses=uses,
        system_prompt="system", user_prompt_template="prompt",
    )
    client = Mock(spec=anthropic.Anthropic)
    streams: list[nullcontext[MagicMock]] = []
    for content, stop_reason in [
        (calls, "tool_use"),
        ([{"type": "text", "text": "done"}], "end_turn"),
    ]:
        message = Message.model_validate({
            "id": "msg", "model": ModelNames[Provider.Anthropic],
            "role": "assistant", "type": "message",
            "content": deepcopy(content), "stop_reason": stop_reason, "input_transformations": [],
            "usage": {"input_tokens": 1, "output_tokens": 1},
        })
        stream = MagicMock()
        stream.__iter__.return_value = [RawMessageStartEvent(type="message_start", message=message)]
        stream.get_final_message.return_value = message
        streams.append(nullcontext(stream))
    client.messages.stream.side_effect = streams
    runtime = Runtime([fn], client_factories={Provider.Anthropic: lambda: client})
    node = runtime.invoke(None, fn, {})
    assert isinstance(node, AgentNode)
    node.result()
    requests: list[MessageCreateParamsBase] = [
        call.kwargs for call in client.messages.stream.call_args_list
    ]
    return node, requests


class TestAnthropicEditorAlias(unittest.TestCase):
    def test_alias_executes_locally_and_replays_original_tool_uses(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir, "source.txt")
            target = Path(tmpdir, "target.txt")
            source.write_bytes(b"one\ntwo\n")
            target.write_text("before\n")
            calls = [
                _tool_use(NATIVE_EDITOR, "view-id", command="view", path=str(source), view_range=[2, -1]),
                _tool_use(NATIVE_EDITOR, "insert-id", command="insert", path=str(target),
                          insert_line=1, insert_text="after\n"),
            ]
            editor = TextEditor()
            node, requests = _run_tool_turn([editor], calls)
            self.assertEqual(target.read_text(), "before\nafter\n")

        self.assertEqual([tool.get("name") for tool in requests[0].get("tools", [])], ["text_editor"])
        assistant, user = list(requests[1]["messages"])[-2:]
        self.assertEqual(assistant, {"role": "assistant", "content": calls})
        results = cast(list[ToolResultBlockParam], user["content"])
        self.assertEqual([r["tool_use_id"] for r in results], ["view-id", "insert-id"])
        self.assertEqual([r.get("is_error") for r in results], [False, False])
        view_content = cast(list[TextBlockParam], results[0].get("content"))
        self.assertEqual(view_content[0]["text"], "2|two\n")
        uses = [p for p in node.transcript if isinstance(p, ToolUsePart)]
        self.assertEqual([(p.tool_name, p.tool_use_id, p.args) for p in uses],
                         [(NATIVE_EDITOR, c["id"], c["input"]) for c in calls])
        outputs = [p for p in node.transcript if isinstance(p, ToolResultPart)]
        self.assertEqual([(p.tool_name, p.tool_use_id) for p in outputs],
                         [(NATIVE_EDITOR, c["id"]) for c in calls])
        for child, call in zip(node.children, calls, strict=True):
            self.assertIs(child.fn, editor)
            self.assertEqual(child.tool_use_id, call["id"])

    def test_registered_name_wins_and_an_unregistered_editor_stays_unavailable(self) -> None:
        custom = CodeFunction(name=NATIVE_EDITOR, desc="", args=[], callable=lambda ctx: "custom result")
        lookalike = CodeFunction(name="text_editor", desc="", args=[], callable=lambda ctx: "lookalike")
        for uses, expected_error in [([TextEditor(), custom], False), ([lookalike], True), ([], True)]:
            with self.subTest(uses=[f.name for f in uses]):
                node, _ = _run_tool_turn(uses, [_tool_use(NATIVE_EDITOR, "call-id")])
                result, = [p for p in node.transcript if isinstance(p, ToolResultPart)]
                self.assertEqual(result.is_error, expected_error)
                if expected_error:
                    self.assertIn("unknown tool", result.outputs)
                    self.assertEqual(node.children, [])
                else:
                    self.assertEqual(result.outputs, "custom result")

    def test_explicit_editor_arguments_take_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir, "source.txt")
            target = Path(tmpdir, "target.txt")
            source.write_bytes(b"one\ntwo\nthree\n")
            target.write_bytes(b"before\n")
            calls = [
                _tool_use(NATIVE_EDITOR, "mixed-range", command="view", path=str(source),
                          view_end_line=3, view_range=[2, 2]),
                _tool_use(NATIVE_EDITOR, "bad-range", command="view", path=str(source),
                          view_start_line=3, view_range="bad"),
                _tool_use(NATIVE_EDITOR, "whole-file", command="view", path=str(source)),
                _tool_use(NATIVE_EDITOR, "mixed-insert", command="insert", path=str(target),
                          insert_line=1, new_str="explicit\n", insert_text="inferred\n"),
            ]
            node, requests = _run_tool_turn([TextEditor()], calls)
            self.assertEqual(target.read_text(), "before\nexplicit\n")

        self.assertEqual(list(requests[1]["messages"])[-2]["content"], calls)
        results = [part for part in node.transcript if isinstance(part, ToolResultPart)]
        self.assertEqual([part.outputs for part in results], [
            "2|two\n3|three\n", "3|three\n", "1|one\n2|two\n3|three\n", "Insert successful.",
        ])

    def test_unusable_native_arguments_report_advertised_editor_fields(self) -> None:
        calls = [
            _tool_use(NATIVE_EDITOR, "bad-range", command="view", path="file", view_range="bad"),
            _tool_use(NATIVE_EDITOR, "no-text", command="insert", path="file", insert_line=0),
            _tool_use(NATIVE_EDITOR, "bad-text", command="insert", path="file", insert_line=0,
                      insert_text=[]),
        ]
        node, _ = _run_tool_turn([TextEditor()], calls)
        results = [part for part in node.transcript if isinstance(part, ToolResultPart)]
        self.assertEqual([(part.tool_use_id, part.is_error) for part in results],
                         [(call["id"], True) for call in calls])
        self.assertIn("view_start_line", results[0].outputs)
        self.assertIn("view_end_line", results[0].outputs)
        self.assertNotIn("view_range", results[0].outputs)
        for result in results[1:]:
            self.assertEqual(result.outputs, "TextEditorException: Missing argument: new_str")
