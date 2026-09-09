"""Image tool results through the real Responses SDK and Runtime, without networking."""

import base64
import json
from threading import Event

import pytest

openai = pytest.importorskip("openai", minversion="3.8.0")
import httpx2
from openai.types.responses import Response

Image = pytest.importorskip("PIL.Image")

from ..core import (
    AgentFunction, AgentNode, CodeFunction, ModelStatusPart, ToolResultPart,
)
from ..func_lib import status_update
from ..func_lib.view_image import ImageResult, view_image
from ..providers import Provider
from ..providers import openai as provider_mod
from ..runtime import Runtime


def _call(call_id, name, **args):
    return {
        "type": "function_call", "id": "fc_" + call_id, "call_id": call_id,
        "name": name, "status": "completed", "arguments": json.dumps(args),
    }


def _message(text, phase="final_answer"):
    return {
        "type": "message", "id": "msg_" + text, "role": "assistant",
        "status": "completed", "phase": phase,
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _reasoning(name):
    return {
        "type": "reasoning", "id": name, "status": "completed",
        "summary": [], "encrypted_content": "encrypted_" + name,
    }


def _response(*output):
    return Response.model_validate({
        "id": "resp_image", "created_at": 0, "object": "response",
        "model": "gpt-6-astra", "output": list(output), "status": "completed",
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        "truncation": "disabled",
        "reasoning": {"mode": "standard", "effort": "max", "context": "all_turns"},
        "prompt_cache_options": {"mode": "implicit", "ttl": "30m"},
        "usage": {
            "input_tokens": 20,
            "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
            "output_tokens": 8,
            "output_tokens_details": {
                "reasoning_tokens": 2 if any(item["type"] == "reasoning" for item in output) else 0,
            },
            "total_tokens": 28,
        },
    }).to_dict(mode="json", use_api_names=True)


@pytest.fixture
def run_images(monkeypatch):
    monkeypatch.setattr(provider_mod.time, "sleep", lambda _: None)

    def run(outcomes, uses):
        outcomes = iter(outcomes)
        requests, clients = [], []

        def handle(request):
            requests.append(json.loads(request.content))
            outcome = next(outcomes)
            if callable(outcome):
                outcome = outcome()
            if isinstance(outcome, Exception):
                raise outcome
            return httpx2.Response(200, json=outcome)

        def factory():
            client = openai.OpenAI(
                api_key="test-key", base_url="https://openai.invalid/v1",
                http_client=httpx2.Client(transport=httpx2.MockTransport(handle), trust_env=False),
                _strict_response_validation=True,
            )
            clients.append(client)
            return client

        fn = AgentFunction(
            name="agent", desc="Image test", args=[], system_prompt="System",
            user_prompt_template="View images", uses=uses, default_model=Provider.OpenAI,
        )
        runtime = Runtime([fn], client_factories={Provider.OpenAI: factory})
        node = runtime.invoke(None, fn, {}, max_agent_levels=1)
        assert node.done.wait(10), "Mocked image agent did not finish"
        node.thread.join(5)
        assert not node.thread.is_alive()
        assert all(client.is_closed() for client in clients)
        return node, requests, clients

    return run


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (3, 2), "red").save(path)
    return path


def test_image_batch_serializes_nested_outputs_in_call_order(
    run_images, image_path, tmp_path, monkeypatch,
):
    def fail_stringification(self):
        raise AssertionError("Provider must extract ImageResult.status directly")

    monkeypatch.setattr(ImageResult, "__str__", fail_stringification)
    second_path = tmp_path / "second.png"
    Image.new("RGB", (2, 3), "blue").save(second_path)
    second_finished = Event()
    completion_order = []
    read_image = view_image.callable

    def read(ctx, *, path):
        if path == str(image_path):
            assert second_finished.wait(2), "Image children did not start in parallel"
        result = read_image(ctx, path=path)
        completion_order.append(path)
        if path == str(second_path):
            second_finished.set()
        return result

    monkeypatch.setattr(view_image, "callable", read)
    tool = CodeFunction(name="tool", desc="Ordinary", args=[], callable=lambda ctx: "text")
    calls = [
        _call("first-image", "view_image", path=str(image_path)),
        _call("ordinary", "tool"),
        _call("second-image", "view_image", path=str(second_path)),
        _call("status", "status_update", msg="Viewing images"),
        _call("missing-image", "view_image", path=str(tmp_path / "missing.png")),
    ]
    node, requests, _ = run_images([
        _response(*calls), _response(_message("Done")),
    ], [view_image, tool, status_update])

    assert node.result() == "Done"
    assert completion_order == [str(second_path), str(image_path)]
    history = requests[1]["input"]
    outputs = [item for item in history if item["type"] == "function_call_output"]
    assert [item["call_id"] for item in outputs] == [call["call_id"] for call in calls]
    assert sum(item.get("role") == "user" for item in history) == 1
    transcript_results = [part for part in node.transcript if isinstance(part, ToolResultPart)]
    assert len(transcript_results) == 4
    for index, path in [(0, image_path), (2, second_path)]:
        result = node.children[index].result()
        assert isinstance(result, ImageResult)
        parts = outputs[index]["output"]
        assert isinstance(parts, list)
        assert [part["type"] for part in parts] == ["input_text", "input_image"]
        assert parts[0]["text"] == f"<outcome>success</outcome>\n<result_content>\n{result.status}\n</result_content>"
        assert parts[1]["detail"] == "auto"
        prefix, encoded = parts[1]["image_url"].split(",", 1)
        assert prefix == "data:image/png;base64"
        assert base64.b64decode(encoded) == result.data == path.read_bytes()
        assert encoded not in str(node.transcript)
        assert transcript_results[index].outputs is result
        assert not transcript_results[index].is_error
        assert transcript_results[index].tool_use_id == node.children[index].tool_use_id
        assert transcript_results[index].tool_use_id != outputs[index]["call_id"]
    assert outputs[1]["output"] == [{"type": "input_text", "text": (
        "<outcome>success</outcome>\n<result_content>\ntext\n</result_content>"
    )}]
    assert outputs[3]["output"] == [{"type": "input_text", "text": (
        "<outcome>success</outcome>\n<result_content>\nok\n</result_content>"
    )}]
    assert outputs[4]["output"] == [{
        "type": "input_text", "text": (
            "<outcome>exception</outcome>\n"
            f"<result_content>\n{transcript_results[-1].outputs}\n</result_content>"
        ),
    }]
    assert "FileNotFoundError" in transcript_results[-1].outputs
    assert transcript_results[-1].is_error
    with pytest.raises(FileNotFoundError):
        node.children[-1].result()
    assert len([part for part in node.transcript if isinstance(part, ModelStatusPart)]) == 1


def test_image_history_survives_connection_retry_and_later_tool_cycle(run_images, image_path):
    first = _response(
        _reasoning("first"), _message("Viewing", "commentary"),
        _call("image-call", "view_image", path=str(image_path)),
    )
    second = _response(_reasoning("second"), _call("followup-call", "followup"))
    tool_calls = []

    def disconnect():
        image_path.unlink()
        return httpx2.ConnectError("Simulated disconnect")

    def followup(ctx):
        assert not image_path.exists()
        tool_calls.append("followup")
        return "continued"

    tool = CodeFunction(name="followup", desc="Continue", args=[], callable=followup)
    node, requests, clients = run_images([
        first, disconnect, second, _response(_reasoning("third"), _message("Done")),
    ], [view_image, tool])

    assert node.result() == "Done"
    assert len(clients) == 2
    assert len(node.children) == 2
    assert tool_calls == ["followup"]
    assert requests[1] == requests[2]
    previous_input = requests[2]["input"]
    assert requests[3]["input"][:len(previous_input)] == previous_input
    assert previous_input[1:4] == first["output"]
    assert requests[3]["input"][len(previous_input):-1] == second["output"]
    assert len([item for item in node.history if item["type"] == "function_call_output"]) == 2
    for request in requests:
        assert request["store"] is False
        assert request["reasoning"]["context"] == "all_turns"
        assert sum(item.get("role") == "user" for item in request["input"]) == 1


@pytest.mark.parametrize("fmt, mime", [("JPEG", "image/jpeg"), ("WEBP", "image/webp")])
def test_native_formats_keep_their_bytes_and_status(run_images, tmp_path, fmt, mime):
    path = tmp_path / "image"
    Image.new("RGB", (3, 2), "red").save(path, format=fmt)
    node, requests, _ = run_images([
        _response(_call("image", "view_image", path=str(path))), _response(_message("Done")),
    ], [view_image])

    assert node.result() == "Done"
    result = node.children[0].result()
    status, image = requests[1]["input"][-1]["output"]
    prefix, encoded = image["image_url"].split(",", 1)
    assert prefix == f"data:{mime};base64"
    assert base64.b64decode(encoded) == result.data == path.read_bytes()
    assert status["text"] == f"<outcome>success</outcome>\n<result_content>\n{result.status}\n</result_content>"


@pytest.mark.parametrize("name", ["view_image", "custom_image"])
def test_custom_image_result_preserves_identity_and_status(run_images, image_path, monkeypatch, name):
    def fail_stringification(self):
        raise AssertionError("Provider must extract ImageResult.status directly")

    monkeypatch.setattr(ImageResult, "__str__", fail_stringification)
    snapshot = view_image.callable(None, path=str(image_path))
    custom = CodeFunction(name=name, desc="Author image", args=[], callable=lambda ctx: snapshot)
    node, requests, _ = run_images([
        _response(_call("custom", name)), _response(_message("Done")),
    ], [custom])

    assert node.result() == "Done"
    assert requests[1]["input"][-1]["output"] == [
        {"type": "input_text", "text": (
            f"<outcome>success</outcome>\n<result_content>\n{snapshot.status}\n</result_content>"
        )},
        {"type": "input_image", "detail": "auto", "image_url": (
            f"data:{snapshot.mime_type};base64,{snapshot.base64_data}"
        )},
    ]
    transcript_result = next(part for part in node.transcript if isinstance(part, ToolResultPart))
    assert transcript_result.outputs is snapshot and not transcript_result.is_error


def test_unrelated_function_named_view_image_remains_textual(run_images):
    custom = CodeFunction(
        name="view_image", desc="Author function", args=[], callable=lambda ctx: "author output",
    )
    node, requests, _ = run_images([
        _response(_call("custom", "view_image")), _response(_message("Done")),
    ], [custom])

    assert node.result() == "Done"
    assert requests[1]["input"][-1]["output"] == [{"type": "input_text", "text": (
        "<outcome>success</outcome>\n<result_content>\nauthor output\n</result_content>"
    )}]


def test_plain_builtin_result_remains_text(run_images, monkeypatch):
    monkeypatch.setattr(view_image, "callable", lambda ctx, *, path: "plain output")
    node, requests, _ = run_images([
        _response(_call("image", "view_image", path="unused")), _response(_message("Done")),
    ], [view_image])

    assert node.result() == "Done"
    assert requests[1]["input"][-1]["output"] == [{"type": "input_text", "text": (
        "<outcome>success</outcome>\n<result_content>\nplain output\n</result_content>"
    )}]
    transcript_result = next(part for part in node.transcript if isinstance(part, ToolResultPart))
    assert transcript_result.outputs == "plain output" and not transcript_result.is_error


def test_falsey_invocation_exception_is_a_text_error(run_images, monkeypatch):
    class FalseyError(Exception):
        def __bool__(self):
            return False

    def fail_invoke(self, tool_name, tool_args, tool_use_id):
        raise FalseyError("cannot invoke")

    monkeypatch.setattr(AgentNode, "invoke_tool_function", fail_invoke)
    node, requests, _ = run_images([
        _response(_call("failed", "view_image", path="unused")), _response(_message("Recovered")),
    ], [view_image])

    assert node.result() == "Recovered"
    assert not node.children
    result = requests[1]["input"][-1]
    assert result["call_id"] == "failed"
    assert result["output"] == [{"type": "input_text", "text": (
        "<outcome>exception</outcome>\n<result_content>\nFalseyError: cannot invoke\n</result_content>"
    )}]
    transcript_result = next(part for part in node.transcript if isinstance(part, ToolResultPart))
    assert transcript_result.is_error and transcript_result.outputs == "FalseyError: cannot invoke"
