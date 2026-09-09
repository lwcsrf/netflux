"""Exercise provider behavioral contracts through the real SDK without networking."""

import json
from unittest.mock import Mock

import pytest

openai = pytest.importorskip("openai", minversion="3.8.0")
import httpx2

from ..core import (
    AgentFunction, CodeFunction, FunctionArg, ModelProviderException,
    ModelStatusPart, ModelTextPart, ToolResultPart, ToolUsePart,
)
from ..func_lib import raise_exception, status_update
from ..func_lib.bash_func import BashCommandTimeoutException
from ..providers import Provider
from ..runtime import Runtime
from .test_openai_provider import _call, _message, _reasoning, _response, _tool


def _body(*output):
    body = _response().to_dict(mode="json", use_api_names=True)
    # Preserve omitted fields exactly, so the SDK really sees absent phase values.
    body["output"] = list(output)
    body["usage"]["output_tokens_details"]["reasoning_tokens"] = (
        2 if any(item["type"] == "reasoning" for item in output) else 0
    )
    return body


@pytest.fixture
def run_wire_agent():
    def run(responses, *, uses=(), registered=()):
        responses = iter(responses)
        requests = []

        def handle(request):
            requests.append(json.loads(request.content))
            return httpx2.Response(200, json=next(responses))

        client = openai.OpenAI(
            api_key="test-key", base_url="https://openai.invalid/v1",
            http_client=httpx2.Client(
                transport=httpx2.MockTransport(handle), trust_env=False,
            ),
            _strict_response_validation=True,
        )
        fn = AgentFunction(
            name="agent", desc="Behavior audit", args=[], system_prompt="System",
            user_prompt_template="Task", uses=uses, default_model=Provider.OpenAI,
        )
        runtime = Runtime([fn, *registered], client_factories={Provider.OpenAI: lambda: client})
        node = runtime.invoke(None, fn, {}, max_agent_levels=1)
        assert node.done.wait(5), "OpenAI behavior test did not finish"
        node.thread.join(5)
        assert not node.thread.is_alive()
        assert client.is_closed()
        return node, requests

    return run


@pytest.mark.parametrize("enabled", [False, True], ids=["empty-tools", "direct-uses"])
def test_serialized_request_exposes_only_author_direct_uses(run_wire_agent, enabled):
    hidden_called = Mock(return_value="hidden")
    hidden = CodeFunction(name="hidden", desc="Transitive only", args=[], callable=lambda ctx: hidden_called())
    delegate = AgentFunction(
        name="delegate", desc="Nested agent", args=[], system_prompt="", user_prompt_template="",
        uses=[hidden], default_model=Provider.OpenAI,
    )
    # An author may use a familiar tool name; it must still be a local function.
    bash = CodeFunction(name="bash", desc="Author bash", args=[], callable=lambda ctx: "local")
    uses = [bash, delegate] if enabled else []
    node, requests = run_wire_agent(
        [_body(_call(name="hidden")), _body(_message("Done"))], uses=uses,
        registered=[delegate, hidden, status_update, raise_exception],
    )

    assert node.result() == "Done"
    hidden_called.assert_not_called()
    assert not node.children
    assert len(requests) == 2
    request = requests[0]
    assert "tools" in request
    assert [tool["name"] for tool in request["tools"]] == [fn.name for fn in uses]
    assert set(node.func_map) == {fn.name for fn in uses}
    for tool in request["tools"]:
        assert tool["type"] == "function"
        assert tool["allowed_callers"] == ["direct"]
        assert tool["async"] is False
        assert tool["defer_loading"] is False
    assert request["tool_choice"] == "auto"
    assert request["parallel_tool_calls"] is True
    assert not ({"mcp_servers", "prompt", "conversation", "previous_response_id"} & request.keys())
    assert requests[1]["tools"] == request["tools"]
    result = requests[1]["input"][-1]["output"]
    assert len(result) == 1 and result[0]["type"] == "input_text"
    assert result[0]["text"].startswith("<outcome>exception</outcome>\n<result_content>\n")
    assert "Invoking unknown tool: 'hidden'" in result[0]["text"]
    assert result[0]["text"].endswith("\n</result_content>")


@pytest.mark.parametrize("phase_field", ["null", "omitted"])
@pytest.mark.parametrize("shape", ["text", "calls", "text-calls", "calls-text", "text-calls-text"])
def test_unphased_text_and_calls_follow_full_continuation_contract(run_wire_agent, phase_field, shape):
    def message(text):
        item = _message(text, phase=None)
        if phase_field == "omitted":
            del item["phase"]
        return item

    calls = [_call("first"), _call("second")]
    first_output = []
    if shape == "text":
        first_output.append(message("  Done \n"))
    else:
        if shape.startswith("text-"):
            first_output.append(message("Before calls"))
        first_output.extend(calls)
        if shape.endswith("-text"):
            first_output.append(message("After calls"))
    responses = [_body(*first_output)]
    if shape != "text":
        responses.append(_body(message("  Done \n")))
    called = Mock(return_value="tool result")
    node, requests = run_wire_agent(responses, uses=[_tool(lambda ctx: called())])

    assert node.result() == "Done"
    assert isinstance(node.transcript[-1], ModelTextPart)
    assert node.transcript[-1].text == node.result()
    if shape == "text":
        called.assert_not_called()
        assert len(requests) == 1
        assert not node.children
        return

    assert called.call_count == len(node.children) == 2
    assert len(requests) == 2
    replay = requests[1]["input"]
    assert replay[:1] == requests[0]["input"]
    assert replay[1:1 + len(first_output)] == first_output
    results = replay[1 + len(first_output):]
    assert [item["type"] for item in results] == ["function_call_output"] * 2
    assert [item["call_id"] for item in results] == ["first", "second"]
    assert [item["output"] for item in results] == [[{"type": "input_text", "text": (
        "<outcome>success</outcome>\n<result_content>\ntool result\n</result_content>"
    )}]] * 2
    texts = [part.text for part in node.transcript if isinstance(part, ModelTextPart)]
    expected = [item["content"][0]["text"] for item in first_output if item["type"] == "message"]
    assert texts == expected + ["Done"]


@pytest.mark.parametrize("output", [
    [],
    [_message(" \t\n", phase=None)],
    [_reasoning()],
    [_message("Still working", phase="commentary")],
    [_message("Earlier text", phase=None), _reasoning()],
], ids=["empty", "whitespace", "reasoning-only", "commentary-only", "text-before-reasoning"])
def test_no_calls_without_terminal_answer_is_a_provider_error(run_wire_agent, output):
    # OpenAI intentionally requires a nonblank trailing answer. Earlier text,
    # commentary, and reasoning do not become a successful empty/fallback result.
    called = Mock(return_value="must not execute")
    node, requests = run_wire_agent([
        _body(*output),
    ], uses=[_tool(lambda ctx: called())])

    with pytest.raises(ModelProviderException, match="without function calls or non-empty final assistant text"):
        node.result()
    called.assert_not_called()
    assert not node.children
    assert len(requests) == 1
    assert not any(isinstance(part, (ToolUsePart, ToolResultPart)) for part in node.transcript)


@pytest.mark.parametrize("kind", ["disabled", "invalid-arguments", "custom-same-name"])
def test_non_special_status_calls_remain_ordinary_tool_results(run_wire_agent, kind):
    args = {"msg": "Working"}
    uses = []
    if kind == "invalid-arguments":
        uses = [status_update]
        args = {"msg": 123}
    elif kind == "custom-same-name":
        uses = [CodeFunction(
            name="status_update", desc="Author function", args=[FunctionArg("msg", str)],
            callable=lambda ctx, *, msg: f"custom: {msg}",
        )]
    node, requests = run_wire_agent([
        _body(_call(name="status_update", arguments=args)), _body(_message("Done")),
    ], uses=uses)

    assert node.result() == "Done"
    assert not any(isinstance(part, ModelStatusPart) for part in node.transcript)
    uses_parts = [part for part in node.transcript if isinstance(part, ToolUsePart)]
    results = [part for part in node.transcript if isinstance(part, ToolResultPart)]
    assert len(uses_parts) == len(results) == 1
    assert results[0].tool_use_id == uses_parts[0].tool_use_id
    is_error = kind != "custom-same-name"
    assert results[0].is_error is is_error
    wire_result = requests[1]["input"][-1]
    assert wire_result["call_id"] == "call_original"
    outcome = "exception" if is_error else "success"
    assert wire_result["output"] == [{
        "type": "input_text",
        "text": f"<outcome>{outcome}</outcome>\n<result_content>\n{results[0].outputs}\n</result_content>",
    }]
    if is_error:
        assert not node.children
    else:
        assert results[0].outputs == "custom: Working"
        assert node.children[0].tool_use_id == uses_parts[0].tool_use_id


def test_status_update_keeps_native_result_and_child_correlation(run_wire_agent):
    call = _call(name="status_update", arguments={"msg": "Working"})
    node, requests = run_wire_agent([
        _body(call), _body(_message("Done")),
    ], uses=[status_update])

    assert node.result() == "Done"
    assert not any(isinstance(part, (ToolUsePart, ToolResultPart)) for part in node.transcript)
    view = node.watch()
    statuses = [part for part in view.transcript if isinstance(part, ModelStatusPart)]
    assert len(statuses) == len(view.children) == 1
    status = statuses[0]
    child = view.children[0]
    assert status.text == child.inputs["msg"] == "Working"
    assert status.tool_use_id == child.tool_use_id
    assert status.tool_use_id != call["call_id"]
    assert child.fn is status_update
    assert child.outputs == "ok"
    assert requests[1]["input"][1] == call
    wire_result = requests[1]["input"][-1]
    assert wire_result["call_id"] == call["call_id"]
    assert wire_result["output"] == [{"type": "input_text", "text": (
        "<outcome>success</outcome>\n<result_content>\nok\n</result_content>"
    )}]


@pytest.mark.parametrize("error_cls", [None, ValueError, BashCommandTimeoutException])
def test_tool_output_preserves_content_and_marks_outcome(run_wire_agent, error_cls):
    text = '"quoted"\nC:\\tmp\\image.png\nUnicode: café 雪'
    if error_cls is BashCommandTimeoutException:
        text = f"Command timed out.\n\n--- partial output ---\n{text}"

    def tool(ctx):
        if error_cls is not None:
            raise error_cls(text)
        return text

    node, requests = run_wire_agent([
        _body(_call()), _body(_message("Done")),
    ], uses=[_tool(tool)])

    assert node.result() == "Done"
    content = f"{error_cls.__name__}: {text}" if error_cls is not None else text
    outcome = "exception" if error_cls is not None else "success"
    expected = f"<outcome>{outcome}</outcome>\n<result_content>\n{content}\n</result_content>"
    assert requests[1]["input"][-1]["output"] == [{"type": "input_text", "text": expected}]
    result = next(part for part in node.transcript if isinstance(part, ToolResultPart))
    assert result.outputs == content
    assert result.is_error is (error_cls is not None)


@pytest.mark.parametrize("unsupported", [
    {"type": "shell_call", "id": "shell", "call_id": "shell-call", "status": "completed",
     "action": {"commands": ["echo test"]}},
    {"type": "web_search_call", "id": "search", "status": "completed",
     "action": {"type": "search", "query": "test"}},
    {"type": "mcp_call", "id": "mcp", "name": "tool", "server_label": "unconfigured",
     "arguments": "{}", "status": "completed"},
    {"type": "mcp_list_tools", "id": "mcp-list", "server_label": "unconfigured",
     "tools": [{"name": "injected", "input_schema": {"type": "object"}}]},
    {"type": "additional_tools", "id": "additional", "role": "assistant",
     "tools": [{"type": "function", "name": "injected", "parameters": {"type": "object"}}]},
    {"type": "tool_search_call", "id": "tool-search", "execution": "client",
     "status": "completed", "arguments": {"query": "more tools"}},
], ids=lambda item: item["type"])
def test_unsupported_provider_tools_reject_entire_response_before_dispatch(run_wire_agent, unsupported):
    called = Mock(return_value="must not execute")
    node, requests = run_wire_agent([
        _body(_call(), unsupported),
    ], uses=[_tool(lambda ctx: called())])

    with pytest.raises(ModelProviderException, match="Unsupported OpenAI response output item"):
        node.result()
    called.assert_not_called()
    assert not node.children
    assert len(requests) == 1
    assert node.history == requests[0]["input"]
    assert [tool["name"] for tool in node.tools] == ["tool"]
