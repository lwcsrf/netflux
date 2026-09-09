import copy
import json
from threading import Event
from unittest.mock import Mock

import pytest

openai = pytest.importorskip("openai", minversion="3.8.0")
from openai.types.responses import Response

from ..core import (
    AgentException,
    AgentFunction,
    CancellationException,
    CodeFunction,
    FunctionArg,
    ModelProviderException,
    ModelStatusPart,
    ModelTextPart,
    NodeState,
    ThinkingBlockPart,
    ToolResultPart,
    ToolUsePart,
    UserTextPart,
)
from ..func_lib import raise_exception, status_update
from ..providers import ModelNames, Provider
from ..providers.openai import OaiAgentNode
from ..runtime import Runtime


def _message(*texts, phase="final_answer"):
    return {
        "type": "message", "id": "msg_" + texts[0], "role": "assistant",
        "status": "completed", "phase": phase,
        "content": [
            {"type": "output_text", "text": text, "annotations": []}
            for text in texts
        ],
    }


def _reasoning(name="reasoning"):
    return {
        "type": "reasoning", "id": name, "status": "completed",
        "summary": [], "encrypted_content": "encrypted_" + name,
    }


def _call(call_id="call_original", arguments=None, name="tool"):
    return {
        "type": "function_call", "id": "fc_" + call_id,
        "call_id": call_id, "name": name, "status": "completed",
        "arguments": json.dumps(arguments or {}),
    }


def _response(*output, status="completed"):
    return Response.model_validate({
        "id": "resp_test", "created_at": 0, "object": "response",
        "model": "gpt-6-astra", "output": list(output), "status": status,
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        "truncation": "disabled",
        "reasoning": {"mode": "standard", "effort": "max", "context": "all_turns"},
        "prompt_cache_options": {"mode": "implicit", "ttl": "30m"},
        "incomplete_details": {"reason": "max_output_tokens"} if status == "incomplete" else None,
        "usage": {
            "input_tokens": 20,
            "input_tokens_details": {"cached_tokens": 5, "cache_write_tokens": 3},
            "output_tokens": 8,
            "output_tokens_details": {
                "reasoning_tokens": 2 if any(item["type"] == "reasoning" for item in output) else 0,
            },
            "total_tokens": 28,
        },
    })


@pytest.fixture
def start_agent():
    def start(responses, *, tool=None, tools=(), cancel_event=None, cancel_during_response=False):
        requests = []
        pending_responses = iter(responses)
        client = Mock(spec=openai.OpenAI)
        client.responses = Mock()

        def create(**kwargs):
            # The provider appends to its replay list after each request.
            requests.append(copy.deepcopy(kwargs))
            response = next(pending_responses)
            if cancel_during_response:
                cancel_event.set()
            return response

        client.responses.create.side_effect = create
        fn = AgentFunction(
            name="agent", desc="Test agent", args=[], system_prompt="System instructions",
            user_prompt_template="Task", uses=[tool] if tool else list(tools),
            default_model=Provider.OpenAI,
        )
        runtime = Runtime([fn], client_factories={Provider.OpenAI: lambda: client})
        node = runtime.invoke(None, fn, {}, cancel_event=cancel_event, max_agent_levels=1)
        assert node.done.wait(5), "Mocked OpenAI agent did not finish"
        node.thread.join(5)
        assert not node.thread.is_alive()
        client.close.assert_called_once_with()
        if node.state == NodeState.Success:
            assert isinstance(node.transcript[-1], ModelTextPart)
            assert node.result() == node.transcript[-1].text
        return node, requests

    return start


def _tool(callable=None, args=()):
    return CodeFunction(
        name="tool", desc="Test tool", args=list(args),
        callable=callable or (lambda ctx: "tool result"),
    )


def test_request_uses_invocation_system_prompt(start_agent):
    node, requests = start_agent([_response(_message("Done"))])
    assert node.result() == "Done"
    assert requests[0]["instructions"].startswith("System instructions\n\nWARNING:")
    assert "Agent level 1 of 1" in requests[0]["instructions"]


def test_optional_enum_null_reaches_framework_callable(start_agent):
    received = []
    enum = {"read", "write"}

    def record(ctx, *, mode=None, required_mode):
        received.append({"mode": mode, "required_mode": required_mode})

    tool = _tool(
        record,
        [FunctionArg("mode", str, optional=True, enum=enum),
         FunctionArg("required_mode", str, enum=enum)],
    )
    args = {"mode": None, "required_mode": "read"}
    node, requests = start_agent([
        _response(_call(arguments=args)), _response(_message("Done")),
    ], tool=tool)

    assert node.result() == "Done"
    assert received == [args]
    schema = requests[0]["tools"][0]["parameters"]
    assert requests[0]["tools"][0]["strict"] is True
    assert requests[0]["tools"][0]["allowed_callers"] == ["direct"]
    assert requests[0]["tools"][0]["async"] is False
    assert requests[0]["tools"][0]["defer_loading"] is False
    assert schema["required"] == ["mode", "required_mode"]
    assert schema["properties"]["mode"]["type"] == ["string", "null"]
    assert schema["properties"]["mode"]["enum"] == ["read", "write", None]
    assert schema["properties"]["required_mode"]["enum"] == ["read", "write"]
    assert enum == {"read", "write"}


def test_commentary_and_tool_items_keep_native_transcript_and_replay_order(start_agent):
    first = _response(
        _message("Checking", phase="commentary"), _reasoning("first"), _call(),
        _message("Waiting", phase="commentary"),
    )
    second = _response(
        _message("Calculating", phase="commentary"), _reasoning("second"),
        _message("The answer", "is"), _message("42"),
    )
    node, requests = start_agent([first, second], tool=_tool())

    assert node.result() == "The answer\nis\n42"
    assert [type(part) for part in node.transcript] == [
        UserTextPart, ModelTextPart, ThinkingBlockPart, ToolUsePart, ModelTextPart,
        ToolResultPart, ModelTextPart, ThinkingBlockPart,
        ModelTextPart,
    ]
    assert [part.text for part in node.transcript if isinstance(part, ModelTextPart)] == [
        "Checking", "Waiting", "Calculating", "The answer\nis\n42",
    ]
    assert requests[1]["input"][1:5] == [
        item.to_dict(mode="json", use_api_names=True, exclude_unset=True)
        for item in first.output
    ]
    result_item = requests[1]["input"][5]
    assert result_item["type"] == "function_call_output"
    assert result_item["call_id"] == "call_original"
    assert node.children[0].tool_use_id == f"openai-{node.id}-1-tool"
    assert node.transcript[3].tool_use_id == node.children[0].tool_use_id
    assert node.transcript[5].tool_use_id == node.children[0].tool_use_id
    assert node.history[6:] == [
        item.to_dict(mode="json", use_api_names=True, exclude_unset=True)
        for item in second.output
    ]


@pytest.mark.parametrize("phase", [None, "final_answer"])
def test_result_contains_only_trailing_assistant_blocks(start_agent, phase):
    node, _ = start_agent([_response(
        _message("Earlier text", phase=phase), _reasoning(),
        _message("Final", "answer", phase=phase), _message("only", phase=phase),
    )])
    assert node.result() == "Final\nanswer\nonly"
    assert node.transcript[1].text == "Earlier text"


def test_commentary_alone_cannot_be_function_result(start_agent):
    node, _ = start_agent([_response(_message("Still working", phase="commentary"))])
    with pytest.raises(ModelProviderException):
        node.result()


def test_explicit_final_phase_takes_precedence_over_unphased_text(start_agent):
    node, _ = start_agent([_response(
        _message("Unphased preamble", phase=None),
        _message("Final answer"), _message("continued"),
    )])
    assert node.result() == "Final answer\ncontinued"
    assert node.transcript[1].text == "Unphased preamble"


def test_adjacent_text_coalesces_within_phase_boundaries(start_agent):
    first = _response(
        _message("Checking", "inputs", phase="commentary"),
        _message("Continuing", phase="commentary"),
        _message("Preamble", "details", phase=None),
        _call(),
    )
    node, requests = start_agent([
        first,
        _response(
            _message("Finished", "checking", phase="commentary"),
            _message("Unphased preamble", phase=None),
            _message("Final", "answer"), _message("continued"),
        ),
    ], tool=_tool())

    assert [part.text for part in node.transcript if isinstance(part, ModelTextPart)] == [
        "Checking\ninputs\nContinuing", "Preamble\ndetails",
        "Finished\nchecking", "Unphased preamble", "Final\nanswer\ncontinued",
    ]
    assert node.result() == "Final\nanswer\ncontinued"
    assert requests[1]["input"][1:-1] == [
        item.to_dict(mode="json", use_api_names=True, exclude_unset=True)
        for item in first.output
    ]


@pytest.mark.parametrize("phase", [None, "final_answer"])
def test_final_text_normalization_is_shared_with_transcript(start_agent, phase):
    node, _ = start_agent([_response(
        _message(" \t", "  First  ", "", phase=phase),
        _message("\n", " Second \n", phase=phase),
    )])

    assert node.result() == "First  \n Second"
    assert node.transcript == [UserTextPart(text="Task"), ModelTextPart(text=node.result())]


def test_interleaved_unphased_text_stays_separate_from_final_answer(start_agent):
    response = _response(
        _message("Preamble", phase=None), _message("Final"),
        _message("Aside", phase=None), _message("answer"),
        _message("Suffix", phase=None),
    )
    node, _ = start_agent([response])

    assert node.result() == "Final\nanswer"
    assert [part.text for part in node.transcript if isinstance(part, ModelTextPart)] == [
        "Preamble\nAside\nSuffix", "Final\nanswer",
    ]
    assert node.history[1:] == [
        item.to_dict(mode="json", use_api_names=True, exclude_unset=True)
        for item in response.output
    ]


def test_commentary_resets_final_answer_selection(start_agent):
    node, _ = start_agent([_response(
        _message("Earlier answer"), _message("Checking", phase="commentary"),
        _message("Actual", "answer"),
    )])

    assert node.result() == "Actual\nanswer"
    assert [part.text for part in node.transcript if isinstance(part, ModelTextPart)] == [
        "Earlier answer", "Checking", "Actual\nanswer",
    ]


def test_coalescing_does_not_mutate_published_snapshots(start_agent, monkeypatch):
    snapshots = []
    original = Runtime.post_transcript_update

    def record(runtime, node):
        original(runtime, node)
        snapshots.append(runtime._node_observables[node.id].view)

    monkeypatch.setattr(Runtime, "post_transcript_update", record)
    node, _ = start_agent([_response(
        _message("Checking", "inputs", phase="commentary"), _message("Done"),
    )])

    assert snapshots[0].transcript[-1].text == "Checking"
    assert snapshots[1].transcript[-1].text == "Checking\ninputs"
    assert node.transcript[1].text == "Checking\ninputs"
    assert snapshots[0].transcript[-1] is not node.transcript[1]


@pytest.mark.parametrize("separate_message", [False, True])
def test_terminal_refusal_is_preserved_when_answer_projection_is_deferred(start_agent, separate_message):
    messages = [_message("Partial answer")]
    if separate_message:
        messages.append(_message(""))
    refusal = messages[-1]
    refusal["content"].append({"type": "refusal", "refusal": "Cannot continue"})
    node, _ = start_agent([_response(*messages)])

    with pytest.raises(ModelProviderException, match="Cannot continue"):
        node.result()
    assert node.transcript[-1].text == "Partial answer\nCannot continue"


def test_blank_messages_do_not_merge_distinct_phases(start_agent):
    node, _ = start_agent([
        _response(
            _message("Checking", phase="commentary"),
            _message(" \n"), _message("Draft"), _call(),
        ),
        _response(_message("Done")),
    ], tool=_tool())

    assert [part.text for part in node.transcript if isinstance(part, ModelTextPart)] == [
        "Checking", "Draft", "Done",
    ]


@pytest.mark.parametrize("cancel", [False, True])
def test_refusal_is_retained_and_fails_before_tools_even_when_cancelled(start_agent, cancel):
    refusal_text = "I cannot complete this request."
    refusal = _message("Refusal")
    refusal["id"] = "msg_refusal"
    refusal["content"] = [{"type": "refusal", "refusal": refusal_text}]
    called = Mock(return_value="tool result")
    node, requests = start_agent(
        [_response(_reasoning(), _call(), refusal),
         _response(_message("Unexpected continuation"))],
        tool=_tool(lambda ctx: called()),
        cancel_event=Event(), cancel_during_response=cancel,
    )

    with pytest.raises(ModelProviderException) as caught:
        node.result()
    failure = caught.value
    assert type(failure) is ModelProviderException
    assert failure.inner_exception is None
    assert refusal_text in str(failure)
    assert "msg_refusal" in str(failure)
    assert [part.text for part in node.transcript if isinstance(part, ModelTextPart)] == [refusal_text]
    called.assert_not_called()
    assert node.children == []
    assert len(requests) == 1
    usage = node.token_usage
    assert (usage.input_tokens_regular, usage.input_tokens_cache_read, usage.input_tokens_cache_write) == (12, 5, 3)
    assert (usage.output_tokens_total, usage.output_tokens_reasoning, usage.output_tokens_text) == (8, 2, 6)
    assert (usage.input_tokens_total, usage.context_window_in, usage.context_window_out) == (20, 20, 8)


def test_completed_result_wins_cancellation_during_request_and_retains_usage(start_agent):
    node, requests = start_agent(
        [_response(_reasoning(), _message("Done"))],
        cancel_event=Event(), cancel_during_response=True,
    )
    assert node.result() == "Done"
    assert len(requests) == 1
    usage = node.token_usage
    assert (usage.input_tokens_regular, usage.input_tokens_cache_read, usage.input_tokens_cache_write) == (12, 5, 3)
    assert (usage.output_tokens_total, usage.output_tokens_reasoning, usage.output_tokens_text) == (8, 2, 6)


@pytest.mark.parametrize("cancel", [False, True])
def test_incomplete_response_preserves_error_and_token_usage(start_agent, cancel):
    node, _ = start_agent(
        [_response(status="incomplete")],
        cancel_event=Event(), cancel_during_response=cancel,
    )
    with pytest.raises(ModelProviderException, match="max_output_tokens"):
        node.result()
    assert node.token_usage.input_tokens_total == 20
    assert node.token_usage.output_tokens_total == 8


def test_cancellation_after_response_prevents_new_tool_invocations(start_agent):
    called = Mock(return_value="tool result")
    node, requests = start_agent(
        [_response(_call())], tool=_tool(lambda ctx: called()),
        cancel_event=Event(), cancel_during_response=True,
    )
    with pytest.raises(CancellationException):
        node.result()
    called.assert_not_called()
    assert node.children == []
    assert len(requests) == 1
    assert node.token_usage.input_tokens_total == 20
    assert isinstance(node.transcript[-1], ToolUsePart)


@pytest.mark.parametrize("same_batch", [False, True])
@pytest.mark.parametrize("tool_error", [False, True])
def test_repeated_native_call_ids_use_distinct_framework_ids(
    start_agent, monkeypatch, same_batch, tool_error,
):
    # These synthetic responses exercise the local adapter's ID separation;
    # they make no assertion about the server emitting or accepting repeated IDs.
    first_call = _call(arguments={"value": 1})
    second_call = _call(arguments={"value": 2})
    second_call["id"] = "fc_second"
    responses = (
        [_response(first_call, second_call)] if same_batch
        else [_response(first_call), _response(second_call)]
    ) + [_response(_message("Done"))]

    def tool(ctx, *, value):
        if tool_error and value == 2:
            raise ValueError("second call failed")
        return str(value)

    # Keep an ID regression from reaching Runtime's process-fatal duplicate
    # child check while still exercising real child creation and view mapping.
    invoke = OaiAgentNode.invoke_tool_function
    invoked_ids = []

    def guarded_invoke(self, name, args, tool_use_id):
        assert tool_use_id not in invoked_ids
        invoked_ids.append(tool_use_id)
        return invoke(self, name, args, tool_use_id)

    monkeypatch.setattr(OaiAgentNode, "invoke_tool_function", guarded_invoke)
    node, requests = start_agent(
        responses, tool=_tool(tool, [FunctionArg("value", int)]),
    )
    assert node.result() == "Done"
    expected_ids = [f"openai-{node.id}-{counter}-tool" for counter in (1, 2)]
    assert invoked_ids == expected_ids
    assert [child.tool_use_id for child in node.children] == expected_ids
    assert len(requests) == (2 if same_batch else 3)

    replay = requests[-1]["input"]
    assert [item for item in replay if item["type"] == "function_call"] == [
        first_call, second_call,
    ]
    outputs = [item for item in replay if item["type"] == "function_call_output"]
    assert [item["call_id"] for item in outputs] == ["call_original", "call_original"]
    assert [item["output"] for item in outputs] == [
        [{"type": "input_text", "text": "<outcome>success</outcome>\n<result_content>\n1\n</result_content>"}],
        [{"type": "input_text", "text": (
            "<outcome>exception</outcome>\n<result_content>\n"
            "ValueError: second call failed\n</result_content>" if tool_error else
            "<outcome>success</outcome>\n<result_content>\n2\n</result_content>"
        )}],
    ]

    view = node.watch()
    assert len(view.transcript_child_map) == 4
    for child_view, tool_use_id in zip(view.children, expected_ids):
        parts = [part for part in view.transcript
                 if isinstance(part, (ToolUsePart, ToolResultPart))
                 and part.tool_use_id == tool_use_id]
        assert [type(part) for part in parts] == [ToolUsePart, ToolResultPart]
        assert all(view.transcript_child_map[id(part)] is child_view for part in parts)
        assert parts[1].is_error is (tool_error and child_view.inputs["value"] == 2)


def test_call_ids_can_repeat_in_independent_agent_nodes(start_agent):
    for _ in range(2):
        node, _ = start_agent([
            _response(_call()), _response(_message("Done")),
        ], tool=_tool())
        assert node.result() == "Done"
        assert node.children[0].tool_use_id == f"openai-{node.id}-1-tool"


@pytest.mark.parametrize("field,value", [
    ("caller", {"type": "program", "caller_id": "code"}),
    ("async", True),
    ("namespace", "unexpected"),
])
def test_unsupported_function_protocol_never_dispatches_tools(start_agent, field, value):
    payload = _response(_call()).to_dict()
    payload["output"][0][field] = value
    response = Response.model_validate(payload)
    called = Mock()
    node, requests = start_agent([response], tool=_tool(lambda ctx: called()))

    with pytest.raises(ModelProviderException, match=field):
        node.result()
    called.assert_not_called()
    assert not node.children
    assert len(requests) == 1


@pytest.mark.parametrize("arguments", ["{", "null", "[]", '"text"'])
def test_malformed_arguments_reject_entire_tool_batch(start_agent, arguments):
    invalid = _call("invalid")
    invalid["arguments"] = arguments
    called = Mock()
    node, requests = start_agent([
        _response(_call(), invalid),
    ], tool=_tool(lambda ctx: called()))

    with pytest.raises(ModelProviderException, match="arguments"):
        node.result()
    called.assert_not_called()
    assert not node.children
    assert len(requests) == 1
    assert node.history == requests[0]["input"]


@pytest.mark.parametrize("failure", ["unknown_tool", "bad_arguments", "tool_error"])
def test_tool_errors_return_to_model_and_other_calls_complete(start_agent, failure):
    invoked = []

    def tool(ctx, *, value):
        invoked.append(value)
        if value == 1:
            raise ValueError("Tool failed")
        return "Succeeded"

    bad_call = _call("bad", arguments={"value": 1})
    if failure == "unknown_tool":
        bad_call["name"] = "missing"
    elif failure == "bad_arguments":
        bad_call["arguments"] = "{}"
    node, requests = start_agent([
        _response(bad_call, _call("good", arguments={"value": 2})),
        _response(_message("Recovered")),
    ], tool=_tool(tool, [FunctionArg("value", int)]))

    assert node.result() == "Recovered"
    assert 2 in invoked
    assert len(requests) == 2
    outputs = [item["output"] for item in requests[1]["input"]
               if item["type"] == "function_call_output"]
    results = [part for part in node.transcript if isinstance(part, ToolResultPart)]
    assert outputs[0] == [{"type": "input_text", "text": (
        "<outcome>exception</outcome>\n"
        f"<result_content>\n{results[0].outputs}\n</result_content>"
    )}]
    assert outputs[1] == [{"type": "input_text", "text": (
        "<outcome>success</outcome>\n<result_content>\nSucceeded\n</result_content>"
    )}]
    assert [part.is_error for part in results] == [True, False]


def test_tool_batch_starts_all_children_before_waiting(start_agent):
    second_started = Event()

    def tool(ctx, *, value):
        if value == 1:
            assert second_started.wait(2), "Second tool was not started in parallel"
        else:
            second_started.set()
        return value

    node, requests = start_agent([
        _response(_call("first", {"value": 1}), _call("second", {"value": 2})),
        _response(_message("Done")),
    ], tool=_tool(tool, [FunctionArg("value", int)]))

    assert node.result() == "Done"
    outputs = [item["output"] for item in requests[1]["input"]
               if item["type"] == "function_call_output"]
    assert outputs == [
        [{"type": "input_text", "text": "<outcome>success</outcome>\n<result_content>\n1\n</result_content>"}],
        [{"type": "input_text", "text": "<outcome>success</outcome>\n<result_content>\n2\n</result_content>"}],
    ]


@pytest.mark.parametrize("cancel", [False, True])
def test_declared_agent_exception_joins_siblings_and_wins_cancellation(start_agent, cancel):
    cancel_event = Event()
    sibling_finished = Event()

    def sibling(ctx):
        if cancel:
            cancel_event.set()
        sibling_finished.set()
        return "Finished sibling"

    node, requests = start_agent([_response(
        _call("raise", {"msg": "Cannot finish"}, name="raise_exception"),
        _call("sibling"),
    )], tools=[raise_exception, _tool(sibling)], cancel_event=cancel_event)

    with pytest.raises(AgentException, match="Cannot finish"):
        node.result()
    assert sibling_finished.is_set()
    assert all(child.done.is_set() for child in node.children)
    assert isinstance(node.transcript[-1], ToolResultPart)
    assert node.transcript[-1].outputs == "Finished sibling"
    assert len(requests) == 1


def test_status_updates_use_framework_status_parts_and_native_tool_outputs(start_agent):
    node, requests = start_agent([
        _response(_call(arguments={"msg": "Working"}, name="status_update")),
        _response(_message("Done")),
    ], tool=status_update)

    assert node.result() == "Done"
    assert isinstance(node.transcript[1], ModelStatusPart)
    assert node.transcript[1].text == "Working"
    assert not any(isinstance(part, ToolResultPart) for part in node.transcript)
    assert requests[1]["input"][-1]["output"] == [{"type": "input_text", "text": (
        "<outcome>success</outcome>\n<result_content>\nok\n</result_content>"
    )}]


def test_configured_model_and_default_cache_ttl_are_not_restricted(start_agent, monkeypatch):
    monkeypatch.setitem(ModelNames, Provider.OpenAI, "another-recent-model")
    node, requests = start_agent([_response(_message("Done"))])
    assert node.result() == "Done"
    assert requests[0]["model"] == "another-recent-model"
    assert requests[0]["prompt_cache_options"] == {"mode": "implicit"}


@pytest.mark.parametrize("failure", ["incomplete_message", "missing_usage", "missing_reasoning"])
def test_partial_final_text_is_retained_on_later_validation_failure(start_agent, failure):
    payload = _response(_message("Partial answer")).to_dict()
    if failure == "incomplete_message":
        message = _message("Incomplete later text")
        message["status"] = "incomplete"
        payload["output"].append(message)
    elif failure == "missing_usage":
        payload["usage"] = None
    else:
        payload["usage"]["output_tokens_details"]["reasoning_tokens"] = 2
    response = Response.model_validate(payload)
    node, requests = start_agent([response])

    with pytest.raises(ModelProviderException) as caught:
        node.result()
    assert caught.value.inner_exception is None
    assert [part.text for part in node.transcript if isinstance(part, ModelTextPart)] == ["Partial answer"]
    assert node.history == requests[0]["input"]
    assert len(requests) == 1
