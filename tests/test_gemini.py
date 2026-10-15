"""Manual continuation preserves completed turns and retries only the current hop."""

import copy
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

genai = pytest.importorskip("google.genai")
from google.genai import errors, types

from ..core import (
    AgentFunction, CodeFunction, RunContext, ThinkingBlockPart,
    ToolResultPart, ToolUsePart, UserTextPart,
)
from ..providers import Provider
from ..providers import gemini


def response(parts=None, *, reason="STOP", token=None, usage=None):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(
            content=types.Content(role="model", parts=parts) if parts is not None else None,
            finish_reason=reason,
            continuation_token=token,
        )],
        usage_metadata=types.GenerateContentResponseUsageMetadata(**(
            usage or {"prompt_token_count": 10, "candidates_token_count": 1}
        )),
    )


@pytest.fixture
def make_agent(monkeypatch):
    monkeypatch.setattr(gemini.time, "sleep", lambda _: None)

    def make(responses, *, uses=(), cancel_event=None):
        client = Mock(spec=genai.Client)
        runtime = Mock()
        ctx = RunContext(runtime=runtime, node=None)
        fn = AgentFunction(
            name="test_agent", desc="", args=[], system_prompt="system",
            user_prompt_template="prompt", uses=list(uses), default_model=Provider.Gemini,
        )
        node = gemini.GeminiAgentNode(
            ctx, 1, fn, {}, None, cancel_event, lambda: client,
        )
        ctx.node = node
        requests, before_requests, updates = [], [], []
        replies = iter(responses)

        def generate(**kwargs):
            requests.append(copy.deepcopy(kwargs))
            before_requests.append((copy.deepcopy(node._history), list(node.transcript)))
            reply = next(replies)
            if isinstance(reply, Exception):
                raise reply
            return reply

        client.models.generate_content.side_effect = generate
        runtime.post_transcript_update.side_effect = lambda _: updates.append(
            (list(node.transcript), copy.copy(node.token_usage))
        )
        return SimpleNamespace(
            node=node, client=client, runtime=runtime, requests=requests,
            before_requests=before_requests, updates=updates,
        )

    return make


def test_buffers_content_and_preserves_parts_and_whitespace(make_agent):
    parts = [
        types.Part(text="hello"),
        types.Part(thought_signature=b"signature"),
        types.Part(text=" \n"),
        types.Part(text="world"),
    ]
    multi = make_agent([
        response(parts[:1], reason="CONTINUATION", token=b"first"),
        response(parts[1:3], reason="CONTINUATION", token=b"second"),
        response(parts[3:]),
    ])
    multi.node.run()
    single = make_agent([response(parts)])
    single.node.run()

    assert multi.node._history == single.node._history
    assert multi.node._history[-1].parts == parts
    assert multi.node.transcript == single.node.transcript
    multi.runtime.post_success.assert_called_once_with(multi.node, "hello \nworld")
    assert sum(isinstance(p, ThinkingBlockPart) for p in multi.node.transcript) == 1
    for history, transcript in multi.before_requests:
        assert len(history) == 1
        assert transcript == [UserTextPart(text="prompt")]
    assert [r["config"].continuation_token for r in multi.requests] == [None, b"first", b"second"]
    assert all(r["config"].automatic_continuation is False for r in multi.requests)
    assert all(r["config"].automatic_function_calling.disable for r in multi.requests)
    assert all(r["config"].max_output_tokens == gemini.MAX_TOKENS for r in multi.requests)


@pytest.mark.parametrize("reason", ["MALFORMED_FUNCTION_CALL", "MALFORMED_RESPONSE"])
def test_retries_latest_hop_without_committing_or_duplicating_parts(make_agent, reason):
    agent = make_agent([
        response([types.Part(text="first ")], reason="CONTINUATION", token=b"checkpoint"),
        errors.ServerError(503, {"error": {"message": "retry this hop"}}),
        response([types.Part(text="discard this")], reason=reason,
                 token=b"rejected-checkpoint"),
        response([types.Part(text="last")]),
    ])
    agent.node.run()

    assert len(agent.requests) == 4
    assert agent.requests[1] == agent.requests[2] == agent.requests[3]
    assert agent.requests[1]["contents"] == agent.requests[0]["contents"]
    assert agent.requests[1]["config"].continuation_token == b"checkpoint"
    assert all(len(history) == 1 for history, _ in agent.before_requests)
    assert [p.text for p in agent.node._history[-1].parts] == ["first ", "last"]
    agent.runtime.post_success.assert_called_once_with(agent.node, "first last")
    assert agent.node.token_usage.input_tokens_total == 20


@pytest.mark.parametrize("content", [
    None, types.Content(role="model"), types.Content(role="model", parts=[]),
], ids=["no-content", "no-parts", "empty-parts"])
def test_buffered_tool_waits_through_retry_and_empty_stop_then_resets(make_agent, content):
    tool = CodeFunction(name="lookup", desc="", args=[], callable=lambda ctx: "found")
    parts = [
        types.Part(thought_signature=b"signature"),
        types.Part(function_call=types.FunctionCall(name="lookup", args={}, id="call-1")),
    ]
    incomplete = response(reason=None)
    incomplete.candidates[0].content = content
    agent = make_agent([
        response(parts, reason="CONTINUATION", token=b"checkpoint"),
        incomplete,  # Buffered calls must not make a malformed hop complete.
        response(token=b"stale-stop-token"),
        response([types.Part(text="done")]),
    ], uses=[tool])
    child = Mock()
    child.result.return_value = "found"
    agent.node.invoke_tool_function = Mock(return_value=child)
    original_generate = agent.client.models.generate_content.side_effect

    def generate(**kwargs):
        # Calls are deferred until the empty STOP completes the buffered generation.
        assert agent.node.invoke_tool_function.call_count == (1 if len(agent.requests) == 3 else 0)
        return original_generate(**kwargs)

    agent.client.models.generate_content.side_effect = generate
    agent.node.run()

    agent.node.invoke_tool_function.assert_called_once_with("lookup", {}, "gemini-1-1-lookup")
    assert [r["config"].continuation_token for r in agent.requests] == [
        None, b"checkpoint", b"checkpoint", None,
    ]
    assert agent.requests[1] == agent.requests[2]
    assert len(agent.before_requests[1][0]) == 1
    assert agent.before_requests[1][1] == [UserTextPart(text="prompt")]
    next_history = agent.requests[3]["contents"]
    assert [c.role for c in next_history] == ["user", "model", "user"]
    assert next_history[1].parts == parts
    assert next_history[2].parts[0].function_response.id == "call-1"
    assert next_history[2].parts[0].function_response.response == {"output": "found"}
    assert agent.node._history[-1].parts == [types.Part(text="done")]
    assert sum(isinstance(p, ToolUsePart) for p in agent.node.transcript) == 1
    assert sum(isinstance(p, ToolResultPart) for p in agent.node.transcript) == 1
    agent.runtime.post_success.assert_called_once_with(agent.node, "done")


def test_missing_finish_reason_with_tool_call_preserves_legacy_compatibility(make_agent):
    tool = CodeFunction(name="lookup", desc="", args=[], callable=lambda ctx: "found")
    agent = make_agent([
        response([types.Part(function_call=types.FunctionCall(name="lookup", args={}))],
                 reason=None),
        response([types.Part(text="done")]),
    ], uses=[tool])
    child = Mock()
    child.result.return_value = "found"
    agent.node.invoke_tool_function = Mock(return_value=child)

    agent.node.run()

    agent.node.invoke_tool_function.assert_called_once_with("lookup", {}, "gemini-1-1-lookup")
    assert len(agent.requests) == 2
    assert agent.requests[1]["contents"][-1].parts[0].function_response.response == {"output": "found"}
    agent.runtime.post_success.assert_called_once_with(agent.node, "done")


def test_no_candidates_preserves_prompt_feedback_without_retrying(make_agent):
    agent = make_agent([types.GenerateContentResponse(
        prompt_feedback=types.GenerateContentResponsePromptFeedback(
            block_reason="SAFETY", block_reason_message="Prompt blocked by safety policy.",
        ),
    )])

    with pytest.raises(RuntimeError, match="Gemini returned no candidates") as exc:
        agent.node.run()

    assert "SAFETY" in str(exc.value)
    assert "Prompt blocked by safety policy." in str(exc.value)
    assert len(agent.requests) == 1
    assert len(agent.node._history) == 1
    agent.runtime.post_success.assert_not_called()
    agent.client.close.assert_called_once()


def test_malformed_response_retries_are_bounded(make_agent):
    agent = make_agent([
        response([types.Part(text="discard this")], reason="MALFORMED_RESPONSE"),
    ] * 8)

    with pytest.raises(RuntimeError, match="MALFORMED_RESPONSE"):
        agent.node.run()

    assert len(agent.requests) == 8
    assert all(request == agent.requests[0] for request in agent.requests)
    assert len(agent.node._history) == 1
    assert agent.node.transcript == [UserTextPart(text="prompt")]
    agent.runtime.post_success.assert_not_called()
    agent.client.close.assert_called_once()


@pytest.mark.parametrize("parts", [None, []])
def test_single_hop_empty_stop_returns_empty_success(make_agent, parts):
    agent = make_agent([response(parts)])
    agent.node.run()

    assert len(agent.requests) == 1
    agent.runtime.post_success.assert_called_once_with(agent.node, "")


def test_usage_sums_each_hop_and_context_reports_latest_hop(make_agent):
    agent = make_agent([
        response([], reason="CONTINUATION", token=b"checkpoint", usage={
            "prompt_token_count": 100, "tool_use_prompt_token_count": 2,
            "cached_content_token_count": 40, "thoughts_token_count": 5,
            "candidates_token_count": 3,
        }),
        response([types.Part(text="done")], usage={
            "prompt_token_count": 110, "tool_use_prompt_token_count": 4,
            "cached_content_token_count": 70, "thoughts_token_count": 7,
            "candidates_token_count": 2,
        }),
    ])
    agent.node.run()

    usage = agent.node.token_usage
    assert usage.input_tokens_cache_read == 110
    assert usage.input_tokens_regular == 106
    assert usage.input_tokens_total == 216
    assert usage.output_tokens_reasoning == 12
    assert usage.output_tokens_text == 5
    assert usage.output_tokens_total == 17
    assert (usage.context_window_in, usage.context_window_out) == (114, 9)
    transcript, first_usage = agent.updates[0]
    assert transcript == [UserTextPart(text="prompt")]
    assert (first_usage.context_window_in, first_usage.context_window_out) == (102, 8)


@pytest.mark.parametrize("parts", [None, []])
def test_token_without_finish_reason_can_continue_without_content(make_agent, parts):
    agent = make_agent([
        response(parts, reason=None, token=b"checkpoint"),
        response([types.Part(text="done")]),
    ])
    agent.node.run()

    assert len(agent.requests) == 2
    assert agent.requests[1]["config"].continuation_token == b"checkpoint"
    agent.runtime.post_success.assert_called_once_with(agent.node, "done")


@pytest.mark.parametrize("reason,token", [
    ("CONTINUATION", None), ("MAX_TOKENS", None), ("MAX_TOKENS", b"budget-token"),
])
def test_invalid_or_exhausted_hop_does_not_commit_or_execute(make_agent, reason, token):
    tool = CodeFunction(name="lookup", desc="", args=[], callable=lambda ctx: "found")
    agent = make_agent([
        response([types.Part(text="unfinished")], reason="CONTINUATION", token=b"checkpoint"),
        response([types.Part(function_call=types.FunctionCall(name="lookup", args={}))],
                 reason=reason, token=token),
    ], uses=[tool])
    agent.node.invoke_tool_function = Mock()

    with pytest.raises((AssertionError, RuntimeError)):
        agent.node.run()

    assert len(agent.requests) == 2
    assert len(agent.node._history) == 1
    assert agent.node.transcript == [UserTextPart(text="prompt")]
    agent.node.invoke_tool_function.assert_not_called()
    agent.runtime.post_success.assert_not_called()
    agent.client.close.assert_called_once()


def test_cancellation_between_hops_keeps_buffer_uncommitted(make_agent):
    cancel_event = Event()
    agent = make_agent([
        response([types.Part(text="unfinished")], reason="CONTINUATION", token=b"checkpoint"),
    ], cancel_event=cancel_event)
    agent.runtime.post_transcript_update.side_effect = lambda _: cancel_event.set()

    agent.node.run()

    assert len(agent.requests) == 1
    assert len(agent.node._history) == 1
    assert agent.node.transcript == [UserTextPart(text="prompt")]
    agent.runtime.post_cancel.assert_called_once_with(agent.node, None)
    agent.runtime.post_success.assert_not_called()
