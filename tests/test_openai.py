from unittest.mock import Mock, patch

import pytest

openai = pytest.importorskip("openai", minversion="3.54.0")

from openai.types.responses import Response, ResponseError, ResponseFunctionToolCall
from openai.types.responses.response import InputTransformation, PromptCacheOptions
from openai.types.shared import Reasoning

from ..core import AgentFunction, ModelProviderException, RunContext
from ..providers import ModelNames, Provider
from ..providers.openai import OaiAgentNode


@pytest.fixture
def node():
    fn = AgentFunction(
        name="continuity", desc="", args=[], system_prompt="system",
        user_prompt_template="prompt", uses=[], default_model=Provider.OpenAI,
    )
    runtime = Mock()
    runtime.max_agent_levels.return_value = 8
    with openai.OpenAI(api_key="test", base_url="https://example.invalid/v1") as client:
        yield OaiAgentNode(
            RunContext(runtime=runtime, node=None), 1, fn, {},
            parent=None, cancel_event=None, client_factory=lambda: client,
        )


def make_response(**overrides):
    fields = dict(
        id="resp_continuity", status="completed", model=ModelNames[Provider.OpenAI],
        output=[], usage=None, truncation="disabled",
        reasoning=Reasoning(mode="standard", effort="max", context="all_turns"),
        prompt_cache_options=PromptCacheOptions(mode="implicit", ttl="30m"),
    )
    fields.update(overrides)
    return Response.model_construct(**fields)


@pytest.mark.parametrize("fields", [
    {}, {"input_transformations": None}, {"input_transformations": []},
], ids=["omitted", "null", "empty"])
def test_accepts_response_without_reported_input_transformations(node, fields):
    node.assert_response_contract(make_response(**fields))


@pytest.mark.parametrize("reason", [
    "organization_mismatch", "end_user_mismatch", "model_mismatch",
    "prefix_mismatch", "turn_boundary",
])
def test_rejects_reported_reasoning_removal_with_context(node, reason):
    response = make_response(input_transformations=[
        InputTransformation(id="rs_removed", type="reasoning", action="removed", reason=reason),
        InputTransformation(id="rs_also_removed", type="reasoning", action="removed", reason="prefix_mismatch"),
    ])

    with pytest.raises(ModelProviderException) as raised:
        node.assert_response_contract(response)

    error = raised.value
    assert error.node_id == node.id
    assert error.agent_name == node.agent_fn.name
    for detail in (response.id, "rs_removed", "rs_also_removed", reason):
        assert detail in str(error)


@pytest.mark.parametrize("status", ["completed", "failed"])
def test_reported_removal_stops_loop_without_retry_or_tool_dispatch(node, status):
    response = make_response(
        status=status,
        error=ResponseError(code="server_error", message="retryable failure") if status == "failed" else None,
        input_transformations=[
            InputTransformation(id="rs_removed", type="reasoning", action="removed", reason="prefix_mismatch"),
        ],
        output=[ResponseFunctionToolCall.model_construct(
            id="fc_pending", call_id="call_pending", type="function_call",
            name="should_not_run", arguments="{}", status="completed",
        )],
    )
    raw_response = Mock()
    raw_response.parse.return_value = response
    history_before = list(node.history)
    transcript_before = list(node.transcript)

    with patch.object(node.client.responses.with_raw_response, "create", return_value=raw_response) as create, \
            patch.object(node, "retry_delay", side_effect=AssertionError("Must not retry reasoning loss")) as retry, \
            patch.object(node, "consume_response_output") as consume, \
            patch.object(node, "execute_tool_batch") as execute:
        with pytest.raises(ModelProviderException, match="rs_removed"):
            node.run_agent_loop()

    create.assert_called_once()
    retry.assert_not_called()
    consume.assert_not_called()
    execute.assert_not_called()
    assert node.history == history_before
    assert node.transcript == transcript_before
    assert node.intermediate_events == []
