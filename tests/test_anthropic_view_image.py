"""Check nested image results through the Anthropic SDK's real SSE transport."""

import base64
import json

import pytest

anthropic = pytest.importorskip("anthropic", minversion="1.3.0")
import httpx2

Image = pytest.importorskip("PIL.Image")

from ..core import (
    AgentFunction, AgentNode, CodeFunction, FunctionArg,
    ModelStatusPart, ToolResultPart, ToolUsePart,
)
from ..func_lib import status_update, view_image
from ..func_lib.view_image import ImageResult
from ..providers import Provider
from ..runtime import Runtime


def _call(id, name, **args):
    return {"type": "tool_use", "id": id, "name": name, "input": args}


def _stream(blocks):
    events = [{"type": "message_start", "message": {
        "id": "msg_test", "type": "message", "role": "assistant", "model": "test",
        "content": [], "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 0},
    }}]
    for index, block in enumerate(blocks):
        events.extend([
            {"type": "content_block_start", "index": index, "content_block": block},
            {"type": "content_block_stop", "index": index},
        ])
    events.extend([
        {"type": "message_delta", "delta": {
            "stop_reason": "tool_use" if any(b["type"] == "tool_use" for b in blocks) else "end_turn",
            "stop_sequence": None,
        }, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    ])
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


@pytest.fixture
def run_agent():
    def run(turns, uses):
        pending = iter(turns)
        requests = []

        def handle(request):
            requests.append(json.loads(request.content))
            return httpx2.Response(
                200, headers={"content-type": "text/event-stream"},
                text=_stream(next(pending)),
            )

        with anthropic.Anthropic(
            api_key="test-key", base_url="https://anthropic.invalid",
            http_client=httpx2.Client(transport=httpx2.MockTransport(handle), trust_env=False),
            _strict_response_validation=True,
        ) as client:
            agent = AgentFunction(
                name="agent", desc="Inspect images", args=[], system_prompt="System",
                user_prompt_template="Task", uses=uses, default_model=Provider.Anthropic,
            )
            runtime = Runtime([agent], client_factories={Provider.Anthropic: lambda: client})
            node = runtime.invoke(None, agent, {}, max_agent_levels=1)
            assert node.done.wait(5), "Anthropic image test did not finish"
            node.thread.join(5)
            assert not node.thread.is_alive()
        return node, requests

    return run


def test_images_remain_nested_in_order_with_thinking_and_cache_replay(run_agent, tmp_path, monkeypatch):
    def fail_stringification(self):
        raise AssertionError("Provider must extract ImageResult.status directly")

    monkeypatch.setattr(ImageResult, "__str__", fail_stringification)
    paths = [tmp_path / "first.png", tmp_path / "second.png"]
    for path, color in zip(paths, ["red", "blue"]):
        Image.new("RGB", (2, 3), color).save(path)
    snapshots = [path.read_bytes() for path in paths]
    original = view_image.callable

    def load_and_remove(ctx, *, path):
        result = original(ctx, path=path)
        # Delivery must use the successful snapshot, even after its path disappears.
        from pathlib import Path
        Path(path).unlink()
        return result

    monkeypatch.setattr(view_image, "callable", load_and_remove)
    ordinary = CodeFunction(name="ordinary", desc="Plain result", args=[], callable=lambda ctx: "ok")
    thinking = {"type": "thinking", "thinking": "Inspect both images.", "signature": "signed-thinking"}
    redacted = {"type": "redacted_thinking", "data": "encrypted-thinking"}
    first_turn = [
        thinking, redacted,
        _call("first", "view_image", path=str(paths[0])),
        _call("plain", "ordinary"),
        _call("status", "status_update", msg="Inspecting images"),
        _call("second", "view_image", path=str(paths[1])),
    ]
    node, requests = run_agent([
        first_turn, [_call("later", "ordinary")], [{"type": "text", "text": "Done"}],
    ], [view_image, ordinary, status_update])

    assert node.result() == "Done"
    assert len(requests) == 3
    replay = requests[1]["messages"]
    assert [message["role"] for message in replay] == ["user", "assistant", "user"]
    assert replay[1]["content"] == first_turn
    results = replay[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["first", "plain", "status", "second"]
    assert all(r["type"] == "tool_result" and r["is_error"] is False for r in results)
    for index, expected in zip([0, 3], snapshots):
        summary, image = results[index]["content"]
        assert image["type"] == "image"
        assert image["source"]["type"] == "base64"
        assert image["source"]["media_type"] == "image/png"
        assert base64.b64decode(image["source"]["data"]) == expected
        assert summary == {"type": "text", "text": node.children[index].result().status}
    assert [r.get("cache_control") for r in results] == [None, None, None, {"type": "ephemeral"}]
    assert all("cache_control" not in r for r in node._history[2]["content"])
    assert requests[2]["messages"][1:3] == node._history[1:3]
    assert requests[2]["messages"][2]["content"][-1]["content"] == results[-1]["content"]
    assert len([part for part in node.transcript if isinstance(part, ModelStatusPart)]) == 1
    transcript_results = [part for part in node.transcript if isinstance(part, ToolResultPart)]
    assert [part.tool_use_id for part in transcript_results] == ["first", "plain", "second", "later"]
    assert len([part for part in node.transcript if isinstance(part, ToolUsePart)]) == 4
    for index, child_index in [(0, 0), (2, 3)]:
        assert transcript_results[index].outputs is node.children[child_index].result()
        assert not transcript_results[index].is_error
    assert transcript_results[1].outputs == transcript_results[3].outputs == "ok"
    assert all(base64.b64encode(data).decode() not in repr(node.transcript) for data in snapshots)
    assert all(isinstance(child.result(), ImageResult) for child in node.children if child.fn is view_image)


@pytest.mark.parametrize("invalid_args", [False, True], ids=["missing-file", "missing-argument"])
def test_image_errors_are_matching_text_results(run_agent, tmp_path, invalid_args):
    args = {} if invalid_args else {"path": str(tmp_path / "missing.png")}
    node, requests = run_agent([
        [_call("failed", "view_image", **args)], [{"type": "text", "text": "Recovered"}],
    ], [view_image])

    assert node.result() == "Recovered"
    result = requests[1]["messages"][-1]["content"][0]
    assert result["tool_use_id"] == "failed" and result["is_error"] is True
    assert [part["type"] for part in result["content"]] == ["text"]
    assert result["content"][0]["text"]
    transcript_result = next(part for part in node.transcript if isinstance(part, ToolResultPart))
    assert transcript_result.is_error
    assert transcript_result.outputs == result["content"][0]["text"]
    if not invalid_args:
        with pytest.raises(FileNotFoundError):
            node.children[0].result()


@pytest.mark.parametrize("fmt, mime", [("JPEG", "image/jpeg"), ("WEBP", "image/webp")])
def test_native_formats_keep_their_bytes_and_status(run_agent, tmp_path, fmt, mime):
    path = tmp_path / "image"
    Image.new("RGB", (3, 2), "red").save(path, format=fmt)
    node, requests = run_agent([
        [_call("image", "view_image", path=str(path))], [{"type": "text", "text": "Done"}],
    ], [view_image])

    assert node.result() == "Done"
    result = node.children[0].result()
    status, image = requests[1]["messages"][-1]["content"][0]["content"]
    assert image["source"]["media_type"] == mime
    assert base64.b64decode(image["source"]["data"]) == result.data == path.read_bytes()
    assert status["text"] == result.status


def test_unrelated_function_named_view_image_remains_text_only(run_agent):
    custom = CodeFunction(
        name="view_image", desc="Author tool", args=[FunctionArg("path", str)],
        callable=lambda ctx, *, path: "author output",
    )
    node, requests = run_agent([
        [_call("custom", "view_image", path="unused")], [{"type": "text", "text": "Done"}],
    ], [custom])

    assert node.result() == "Done"
    assert requests[1]["messages"][-1]["content"][0]["content"] == [
        {"type": "text", "text": "author output"},
    ]


def test_plain_builtin_result_remains_text(run_agent, monkeypatch):
    monkeypatch.setattr(view_image, "callable", lambda ctx, *, path: "plain output")
    node, requests = run_agent([
        [_call("plain", "view_image", path="unused")], [{"type": "text", "text": "Done"}],
    ], [view_image])

    assert node.result() == "Done"
    result = requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] is False
    assert result["content"] == [{"type": "text", "text": "plain output"}]
    transcript_result = next(part for part in node.transcript if isinstance(part, ToolResultPart))
    assert transcript_result.outputs == "plain output" and not transcript_result.is_error


@pytest.mark.parametrize("name", ["view_image", "custom_image"])
def test_custom_image_result_preserves_identity_and_status(run_agent, tmp_path, monkeypatch, name):
    def fail_stringification(self):
        raise AssertionError("Provider must extract ImageResult.status directly")

    monkeypatch.setattr(ImageResult, "__str__", fail_stringification)
    path = tmp_path / "custom.png"
    Image.new("RGB", (2, 3), "red").save(path)
    snapshot = view_image.callable(None, path=str(path))
    custom = CodeFunction(name=name, desc="Author image", args=[], callable=lambda ctx: snapshot)
    node, requests = run_agent([
        [_call("custom", name)], [{"type": "text", "text": "Done"}],
    ], [custom])

    assert node.result() == "Done"
    result = requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] is False
    assert result["content"] == [
        {"type": "text", "text": snapshot.status},
        {"type": "image", "source": {
            "type": "base64", "media_type": snapshot.mime_type, "data": snapshot.base64_data,
        }},
    ]
    transcript_result = next(part for part in node.transcript if isinstance(part, ToolResultPart))
    assert transcript_result.outputs is snapshot and not transcript_result.is_error


def test_falsey_invocation_exception_is_a_text_error(run_agent, monkeypatch):
    class FalseyError(Exception):
        def __bool__(self):
            return False

    def fail_invoke(self, tool_name, tool_args, tool_use_id):
        raise FalseyError("cannot invoke")

    monkeypatch.setattr(AgentNode, "invoke_tool_function", fail_invoke)
    node, requests = run_agent([
        [_call("failed", "view_image", path="unused")], [{"type": "text", "text": "Recovered"}],
    ], [view_image])

    assert node.result() == "Recovered"
    assert not node.children
    result = requests[1]["messages"][-1]["content"][0]
    assert result["tool_use_id"] == "failed" and result["is_error"] is True
    assert result["content"] == [{"type": "text", "text": "FalseyError: cannot invoke"}]
    transcript_result = next(part for part in node.transcript if isinstance(part, ToolResultPart))
    assert transcript_result.is_error and transcript_result.outputs == "FalseyError: cannot invoke"
