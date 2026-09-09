"""Agent-declared failures belong to the agent that produced them."""

from contextlib import nullcontext
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ..core import AgentException, AgentFunction, CodeFunction, FunctionArg, ModelTextPart, ToolResultPart
from ..func_lib import raise_exception
from ..func_lib.raise_exception import RaiseException
from ..providers import Provider
from ..runtime import Runtime


@pytest.fixture(params=[Provider.OpenAI, Provider.Anthropic, Provider.Gemini], ids=lambda p: p.value)
def harness(request):
    provider = request.param
    module, minimum = {
        Provider.OpenAI: ("openai", "3.8.0"),
        Provider.Anthropic: ("anthropic", "1.3.0"),
        Provider.Gemini: ("google.genai", "2.22.0"),
    }[provider]
    sdk = pytest.importorskip(module, minversion=minimum)

    def response(turn):
        # A string is a final answer; a list is one batch of (name, arguments).
        calls = [] if isinstance(turn, str) else turn
        if provider is Provider.OpenAI:
            from openai.types.responses import Response

            output = [
                {"type": "function_call", "id": f"fc_{index}", "call_id": f"call_{index}",
                 "name": name, "arguments": json.dumps(args), "status": "completed"}
                for index, (name, args) in enumerate(calls)
            ] if calls else [{
                "type": "message", "id": "msg_test", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": turn, "annotations": []}],
            }]
            return Response.model_validate({
                "id": "resp_test", "created_at": 0, "object": "response", "model": "test",
                "output": output, "status": "completed", "parallel_tool_calls": True,
                "tool_choice": "auto", "tools": [], "truncation": "disabled",
                "reasoning": {"mode": "standard", "effort": "max", "context": "all_turns"},
                "prompt_cache_options": {"mode": "implicit", "ttl": "30m"},
                "usage": {
                    "input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
                    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            })
        if provider is Provider.Anthropic:
            from anthropic.types import Message

            return Message.model_validate({
                "id": "msg_test", "type": "message", "role": "assistant", "model": "test",
                "stop_reason": "tool_use" if calls else "end_turn", "stop_sequence": None,
                "content": [
                    {"type": "tool_use", "id": f"call_{index}", "name": name, "input": args}
                    for index, (name, args) in enumerate(calls)
                ] if calls else [{"type": "text", "text": turn}],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            })
        from google.genai import types

        parts = [types.Part(function_call=types.FunctionCall(name=name, args=args))
                 for name, args in calls] if calls else [types.Part(text=turn)]
        return types.GenerateContentResponse(
            candidates=[types.Candidate(
                content=types.Content(role="model", parts=parts), finish_reason=types.FinishReason.STOP,
            )],
            usage_metadata=types.GenerateContentResponseUsageMetadata(
                prompt_token_count=1, candidates_token_count=1, total_token_count=2,
            ),
        )

    def run(root, turns):
        pending = {name: iter(sequence) for name, sequence in turns.items()}
        requests = []
        clients = []

        def factory():
            client_type = {Provider.OpenAI: "OpenAI", Provider.Anthropic: "Anthropic",
                           Provider.Gemini: "Client"}[provider]
            client = Mock(spec=getattr(sdk, client_type))
            clients.append(client)

            def create(**kwargs):
                prompt = (kwargs["instructions"] if provider is Provider.OpenAI
                          else kwargs["system"] if provider is Provider.Anthropic
                          else kwargs["config"].system_instruction)
                name = prompt.split("\n", 1)[0]
                requests.append(name)
                result = response(next(pending[name]))
                if provider is Provider.Anthropic:
                    stream = Mock()
                    stream.get_final_message.return_value = result
                    return nullcontext(stream)
                return result

            if provider is Provider.OpenAI:
                client.responses = Mock()
                client.responses.create.side_effect = create
            elif provider is Provider.Anthropic:
                client.messages = Mock()
                client.messages.stream.side_effect = create
            else:
                client.models = Mock()
                client.models.generate_content.side_effect = create
            return client

        runtime = Runtime([root], client_factories={provider: factory})
        node = runtime.invoke(None, root, {}, max_agent_levels=2)
        assert node.done.wait(5), "Mocked provider did not finish"
        node.thread.join(5)
        assert not node.thread.is_alive()
        for client in clients:
            client.close.assert_called_once_with()
        return node, requests

    def agent(name, uses):
        return AgentFunction(
            name=name, desc=name, args=[], system_prompt=name,
            user_prompt_template="Task", uses=uses, default_model=provider,
        )

    return SimpleNamespace(agent=agent, run=run)


def _assert_recovered(node, requests):
    assert node.result() == "Recovered"
    assert isinstance(node.transcript[-1], ModelTextPart)
    assert node.transcript[-1].text == node.result()
    results = [part for part in node.transcript if isinstance(part, ToolResultPart)]
    assert len(results) == 1
    assert results[0].is_error
    assert results[0].outputs.startswith("AgentException: Agent 'nested'")
    assert requests == ["outer", "nested", "outer"]


def test_named_raise_exception_does_not_short_circuit_a_nested_failure(harness):
    nested = harness.agent("nested", [])

    def delegate_or_fail(ctx, *, msg):
        if ctx.node.parent.fn.name == "outer":
            return ctx.invoke(nested, {}).result()
        # Produce the same exception (including its owner) as the built-in.
        return raise_exception.callable(ctx, msg=msg)

    shared = CodeFunction(
        name="raise_exception", desc="Delegate or report failure", args=[FunctionArg("msg", str)],
        callable=delegate_or_fail, uses=[nested],
    )
    # Runtime requires globally unique names. A shared function in this legal
    # recursive graph can forward a nested failure without duplicating its name.
    nested.uses_funcs.append(shared)
    outer = harness.agent("outer", [shared])
    node, requests = harness.run(outer, {
        "outer": [[("raise_exception", {"msg": "Delegate"})], "Recovered"],
        "nested": [[("raise_exception", {"msg": "Nested failure"})]],
    })

    nested_node = node.children[0].children[0]
    assert isinstance(nested_node.exception, AgentException)
    assert nested_node.exception.node_id == nested_node.id != node.id
    _assert_recovered(node, requests)


@pytest.mark.parametrize("implementation", ["singleton", "fresh_instance", "author_function"])
def test_direct_raise_exception_still_short_circuits(harness, implementation):
    failure = raise_exception
    if implementation == "fresh_instance":
        failure = RaiseException()
    elif implementation == "author_function":
        failure = CodeFunction(
            name="raise_exception", desc="Author-defined failure", args=[FunctionArg("msg", str)],
            callable=lambda ctx, *, msg: raise_exception.callable(ctx, msg=msg),
        )
    outer = harness.agent("outer", [failure])
    node, requests = harness.run(outer, {"outer": [[("raise_exception", {"msg": "Cannot finish"})]]})

    with pytest.raises(AgentException, match="Cannot finish") as caught:
        node.result()
    assert caught.value is node.children[0].exception
    assert caught.value.node_id == node.id
    assert caught.value.agent_name == "outer"
    assert requests == ["outer"]


@pytest.mark.parametrize("wrapped", [False, True], ids=["agent_child", "code_wrapper"])
def test_nested_raise_exception_returns_to_the_parent_as_a_tool_error(harness, wrapped):
    nested = harness.agent("nested", [raise_exception])
    target = CodeFunction(
        name="wrapper", desc="Invoke nested agent", args=[], uses=[nested],
        callable=lambda ctx: ctx.invoke(nested, {}).result(),
    ) if wrapped else nested
    outer = harness.agent("outer", [target])
    node, requests = harness.run(outer, {
        "outer": [[(target.name, {})], "Recovered"],
        "nested": [[("raise_exception", {"msg": "Nested failure"})]],
    })

    _assert_recovered(node, requests)
