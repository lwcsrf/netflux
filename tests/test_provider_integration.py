import json
from pathlib import Path
import subprocess
import sys

import httpx2
import pytest

from ..core import AgentFunction
from ..demos import client_factory
from ..providers import ModelNames, Provider
from ..runtime import Runtime


def test_provider_and_factory_imports_do_not_require_provider_sdks():
    # A fresh interpreter catches eager imports even when another test has
    # already imported all installed SDKs into this process.
    code = """
import importlib.abc
import sys

class BlockProviderSDKs(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'anthropic', 'google', 'openai', 'xai_sdk'}:
            raise AssertionError(f'Unexpected provider SDK import: {fullname}')

sys.meta_path.insert(0, BlockProviderSDKs())
from netflux.demos.client_factory import CLIENT_FACTORIES
from netflux.providers import Provider
assert Provider.OpenAI in CLIENT_FACTORIES
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_openai_factory_and_runtime_complete_a_real_sdk_request(monkeypatch, tmp_path):
    openai = pytest.importorskip("openai", minversion="3.8.0")
    from ..providers.openai import OaiAgentNode

    (tmp_path / "openai.key").write_text("  test-openai-key\n", encoding="utf-8")
    monkeypatch.setattr(client_factory, "DEMO_DIR", tmp_path)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.test/v1")
    requests = []

    def respond(request):
        requests.append(request)
        return httpx2.Response(200, json={
            "id": "resp_integration", "created_at": 0, "object": "response",
            "model": ModelNames[Provider.OpenAI], "status": "completed",
            "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
            "truncation": "disabled",
            "reasoning": {"mode": "standard", "effort": "max", "context": "all_turns"},
            "prompt_cache_options": {"mode": "implicit", "ttl": "30m"},
            "output": [{
                "id": "msg_integration", "type": "message", "role": "assistant",
                "status": "completed", "phase": "final_answer",
                "content": [{"type": "output_text", "text": "Done", "annotations": []}],
            }],
            "usage": {
                "input_tokens": 4,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens": 1, "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 5,
            },
        })

    real_http_client = openai.DefaultHttpx2Client
    clients = []

    def create_http_client(**kwargs):
        client = real_http_client(
            transport=httpx2.MockTransport(respond), trust_env=False, **kwargs,
        )
        clients.append(client)
        return client

    monkeypatch.setattr(openai, "DefaultHttpx2Client", create_http_client)
    fn = AgentFunction(
        name="agent", desc="Test OpenAI integration", args=[],
        system_prompt="Complete the task", user_prompt_template="Task", uses=[],
        default_model=Provider.OpenAI,
    )
    runtime = Runtime([fn], client_factories=client_factory.CLIENT_FACTORIES)
    node = runtime.invoke(None, fn, {})
    assert node.done.wait(5), "Mocked OpenAI agent did not finish"
    node.thread.join(5)
    assert not node.thread.is_alive()
    assert isinstance(node, OaiAgentNode)
    assert node.result() == "Done"
    assert runtime.get_view(node.id).provider is Provider.OpenAI
    assert len(requests) == 1
    assert requests[0].url.path == "/v1/responses"
    assert requests[0].headers["authorization"] == "Bearer test-openai-key"
    assert json.loads(requests[0].content)["model"] == ModelNames[Provider.OpenAI]
    assert len(clients) == 1
    assert clients[0].is_closed
    assert clients[0].timeout == httpx2.Timeout(connect=10, read=900, write=120, pool=10)


def test_openai_factory_disables_sdk_retries(monkeypatch, tmp_path):
    openai = pytest.importorskip("openai", minversion="3.8.0")
    (tmp_path / "openai.key").write_text("test-openai-key", encoding="utf-8")
    monkeypatch.setattr(client_factory, "DEMO_DIR", tmp_path)

    with client_factory.openai_client_factory() as client:
        assert isinstance(client, openai.OpenAI)
        assert client.max_retries == 0


@pytest.mark.parametrize("contents", [None, " \n"])
def test_openai_factory_requires_a_nonempty_key(monkeypatch, tmp_path, contents):
    pytest.importorskip("openai", minversion="3.8.0")
    monkeypatch.setattr(client_factory, "DEMO_DIR", tmp_path)
    if contents is not None:
        (tmp_path / "openai.key").write_text(contents, encoding="utf-8")
    error = FileNotFoundError if contents is None else RuntimeError
    with pytest.raises(error, match="openai.key"):
        client_factory.openai_client_factory()
