from unittest.mock import Mock, patch

import httpx2
import pytest

openai = pytest.importorskip("openai")

from ..core import AgentFunction
from ..providers import ModelNames, Provider
from ..providers.openai import OaiAgentNode, OaiResponseError
from ..runtime import Runtime


@pytest.fixture
def response_node():
    body = {
        "id": "resp_continuity",
        "created_at": 0,
        "model": ModelNames[Provider.OpenAI],
        "object": "response",
        "status": "completed",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "truncation": "disabled",
        "reasoning": {"mode": "standard", "effort": "max", "context": "all_turns"},
        "prompt_cache_options": {"mode": "implicit", "ttl": "30m"},
        "usage": {
            "input_tokens": 0,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 0,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 0,
        },
    }
    handler = Mock(side_effect=lambda request: httpx2.Response(200, json=body))
    client = openai.OpenAI(
        api_key="test-key",
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
    )
    fn = AgentFunction(
        name="continuity", desc="", args=[], system_prompt="system",
        user_prompt_template="prompt", default_model=Provider.OpenAI,
    )
    runtime = Runtime([fn], client_factories={Provider.OpenAI: lambda: client})
    with patch.object(OaiAgentNode, "start"):
        node = runtime.invoke(None, fn, {})
    try:
        yield node, body, handler
    finally:
        node.close_client()


@pytest.mark.parametrize("diagnostic", ["omitted", None, []])
def test_absent_or_empty_input_transformations_are_accepted(response_node, diagnostic):
    node, body, handler = response_node
    if diagnostic != "omitted":
        body["input_transformations"] = diagnostic

    response = node.create_response_with_retry()

    assert response.id == "resp_continuity"
    handler.assert_called_once()


def test_reported_input_transformations_stop_before_output_or_tools(response_node):
    node, body, handler = response_node
    body["input_transformations"] = [
        {"id": "rs_prefix", "type": "reasoning", "action": "removed", "reason": "prefix_mismatch"},
        {"id": "rs_turn", "type": "reasoning", "action": "removed", "reason": "turn_boundary"},
    ]
    original_history = list(node.history)

    with patch.object(node, "consume_response_output") as consume, \
            patch.object(node, "execute_tool_batch") as execute:
        with pytest.raises(OaiResponseError) as error:
            node.run_agent_loop()

    for detail in ("resp_continuity", "rs_prefix", "prefix_mismatch", "rs_turn", "turn_boundary"):
        assert detail in str(error.value)
    handler.assert_called_once()
    consume.assert_not_called()
    execute.assert_not_called()
    assert node.history == original_history
