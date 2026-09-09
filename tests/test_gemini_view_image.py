"""Exercise tool text and nested images through Gemini's real HTTP serializer."""

import base64
import json

import httpx2
import pytest
from PIL import Image

genai = pytest.importorskip("google.genai", minversion="2.22.0")
from google.genai import types

from ..core import (
    AgentFunction, AgentNode, CodeFunction, ModelProviderException,
    ModelStatusPart, ThinkingBlockPart, ToolResultPart, ToolUsePart,
)
from ..func_lib import status_update
from ..func_lib.view_image import ImageResult, view_image
from ..providers import Provider
from ..runtime import Runtime


def _response(calls=None, *, signature=b"reasoning-signature"):
    parts = [types.Part(function_call=types.FunctionCall(id=id, name=name, args=args))
             for id, name, args in calls] if calls else [types.Part(text="Done")]
    if calls:
        parts[0].thought_signature = signature
    return types.GenerateContentResponse(
        candidates=[types.Candidate(
            content=types.Content(role="model", parts=parts),
            finish_reason=types.FinishReason.STOP,
        )],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=1, candidates_token_count=1, total_token_count=2,
        ),
    )


@pytest.fixture
def run_agent():
    clients = []

    def run(uses, responses):
        pending = iter(responses)
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            return httpx2.Response(
                200, json=next(pending).model_dump(mode="json", by_alias=True, exclude_none=True),
            )

        def factory():
            client = genai.Client(api_key="test-key", http_options=types.HttpOptions(
                base_url="https://gemini.test",
                httpx_client=httpx2.Client(transport=httpx2.MockTransport(respond), trust_env=False),
            ))
            clients.append(client)
            return client

        agent = AgentFunction(
            name="agent", desc="Inspect images", args=[], system_prompt="Inspect images",
            user_prompt_template="Task", uses=uses, default_model=Provider.Gemini,
        )
        runtime = Runtime([agent], client_factories={Provider.Gemini: factory})
        node = runtime.invoke(None, agent, {})
        assert node.done.wait(5), "Mocked Gemini agent did not finish"
        node.thread.join(5)
        assert not node.thread.is_alive()
        return node, requests

    yield run
    for client in clients:
        client.close()


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (3, 2), "green").save(path)
    return path


def test_converted_image_uses_prepared_bytes_and_mime(run_agent, tmp_path):
    from io import BytesIO

    path = tmp_path / "image.tiff"
    Image.new("RGBA", (12, 8), (255, 0, 0, 128)).save(path)
    source_data = path.read_bytes()
    node, requests = run_agent([view_image], [
        _response([("converted", "view_image", {"path": str(path)})]), _response(),
    ])

    assert node.result() == "Done"
    result = requests[1]["contents"][-1]["parts"][0]["functionResponse"]
    inline = result["parts"][0]["inline_data"]
    assert inline["mime_type"] == "image/png"
    data = base64.urlsafe_b64decode(inline["data"])
    with Image.open(BytesIO(data)) as image:
        image.load()
        assert image.format == "PNG" and image.size == (12, 8)
        assert image.getpixel((0, 0)) == (255, 0, 0, 128)
    assert node.children[0].result().data == data
    assert path.read_bytes() == source_data


@pytest.mark.parametrize("fmt, mime", [("JPEG", "image/jpeg"), ("WEBP", "image/webp")])
def test_native_formats_keep_their_bytes_and_status(run_agent, tmp_path, fmt, mime):
    path = tmp_path / "image"
    Image.new("RGB", (3, 2), "red").save(path, format=fmt)
    node, requests = run_agent([view_image], [
        _response([("image", "view_image", {"path": str(path)})]), _response(),
    ])

    assert node.result() == "Done"
    result = node.children[0].result()
    response = requests[1]["contents"][-1]["parts"][0]["functionResponse"]
    image = response["parts"][0]["inline_data"]
    assert image["mime_type"] == mime
    assert base64.urlsafe_b64decode(image["data"]) == result.data == path.read_bytes()
    assert response["response"] == {"output": result.status}


def test_nested_images_replay_with_signatures_and_mixed_tool_results(run_agent, image_path, monkeypatch):
    def fail_stringification(self):
        raise AssertionError("Provider must extract ImageResult.status directly")

    monkeypatch.setattr(ImageResult, "__str__", fail_stringification)
    plain = CodeFunction(name="plain", desc="Plain result", args=[], callable=lambda ctx: "plain output")
    first = _response([
        ("native-status", "status_update", {"msg": "Inspecting images"}),
        ("native-image", "view_image", {"path": str(image_path)}),
        ("native-plain", "plain", {}),
        ("native-missing", "view_image", {"path": str(image_path.with_name("missing.png"))}),
    ])
    second = _response([
        ("native-next", "view_image", {"path": str(image_path)}),
    ], signature=b"next-signature")
    node, requests = run_agent([view_image, status_update, plain], [first, second, _response()])

    assert node.result() == "Done"
    assert len(requests) == 3
    history = requests[-1]["contents"]
    assert [message["role"] for message in history] == ["user", "model", "user", "model", "user"]
    assert history[:3] == requests[1]["contents"]
    assert history[1] == first.candidates[0].content.model_dump(mode="json", by_alias=True, exclude_none=True)
    assert history[3] == second.candidates[0].content.model_dump(mode="json", by_alias=True, exclude_none=True)
    for message in (history[2], history[4]):
        assert all(set(part) == {"functionResponse"} for part in message["parts"])

    responses = [part["functionResponse"] for part in history[2]["parts"]]
    assert [response["id"] for response in responses] == [
        "native-status", "native-image", "native-plain", "native-missing",
    ]
    assert responses[0]["response"] == {"output": "ok"}
    assert responses[2]["response"] == {"output": "plain output"}
    assert set(responses[3]["response"]) == {"error"}
    assert "FileNotFoundError" in responses[3]["response"]["error"]
    for index in (0, 2, 3):
        assert "parts" not in responses[index]

    image = node.children[1].result()
    assert isinstance(image, ImageResult)
    assert responses[1]["response"] == {"output": image.status}
    # Verify the actual SDK wire shape, then decode through its native types.
    wire_data = responses[1]["parts"][0]["inline_data"]["data"]
    assert responses[1]["parts"] == [{
        "inline_data": {"mime_type": "image/png", "data": wire_data},
    }]
    # ProtoJSON accepts both base64 alphabets. The SDK uses URL-safe encoding,
    # while ImageResult stores standard base64; this fixture exercises that gap.
    assert base64.b64encode(image.data) != base64.urlsafe_b64encode(image.data)
    assert base64.urlsafe_b64decode(wire_data) == image.data
    encoded = types.FunctionResponse.model_validate(responses[1])
    canonical = encoded.model_dump(mode="json", by_alias=True, exclude_none=True)
    canonical_data = canonical["parts"][0]["inlineData"]["data"]
    assert canonical["parts"] == [{"inlineData": {"mimeType": "image/png", "data": canonical_data}}]
    assert base64.urlsafe_b64decode(canonical_data) == image.data
    assert encoded.parts[0].inline_data.data == image_path.read_bytes()
    assert history[4]["parts"][0]["functionResponse"]["parts"] == responses[1]["parts"]
    assert history[4]["parts"][0]["functionResponse"]["id"] == "native-next"

    uses = [part for part in node.transcript if isinstance(part, ToolUsePart)]
    results = [part for part in node.transcript if isinstance(part, ToolResultPart)]
    assert [part.tool_use_id for part in results] == [part.tool_use_id for part in uses]
    assert all(part.tool_use_id.startswith("gemini-") for part in results)
    assert [part.is_error for part in results] == [False, False, True, False]
    assert results[0].outputs is image
    assert results[-1].outputs is node.children[-1].result()
    assert results[2].outputs == responses[3]["response"]["error"]
    with pytest.raises(FileNotFoundError):
        node.children[3].result()
    assert image.base64_data not in repr(node.transcript)
    assert len([part for part in node.transcript if isinstance(part, ModelStatusPart)]) == 1
    assert [part.signature for part in node.transcript if isinstance(part, ThinkingBlockPart)] == [
        base64.b64encode(b"reasoning-signature").decode(),
        base64.b64encode(b"next-signature").decode(),
    ]


def test_replay_keeps_snapshot_after_source_disappears(run_agent, image_path, monkeypatch):
    original = view_image.callable

    def load_and_remove(ctx, *, path):
        result = original(ctx, path=path)
        image_path.unlink()
        return result

    monkeypatch.setattr(view_image, "callable", load_and_remove)
    plain = CodeFunction(name="plain", desc="Continue", args=[], callable=lambda ctx: "ok")
    node, requests = run_agent([view_image, plain], [
        _response([("image", "view_image", {"path": str(image_path)})]),
        _response([("next", "plain", {})]), _response(),
    ])

    assert node.result() == "Done"
    assert not image_path.exists()
    history = requests[1]["contents"]
    assert requests[2]["contents"][:len(history)] == history
    response = history[-1]["parts"][0]["functionResponse"]
    assert base64.urlsafe_b64decode(response["parts"][0]["inline_data"]["data"]) == node.children[0].result().data


@pytest.mark.parametrize("native_id", [None, "native-call"])
@pytest.mark.parametrize("name", ["view_image", "custom_image"])
def test_custom_image_result_preserves_identity_and_status(run_agent, image_path, monkeypatch, native_id, name):
    def fail_stringification(self):
        raise AssertionError("Provider must extract ImageResult.status directly")

    monkeypatch.setattr(ImageResult, "__str__", fail_stringification)
    image = view_image.callable(None, path=str(image_path))
    custom = CodeFunction(
        name=name, desc="Application tool", args=[], callable=lambda ctx: image,
    )
    node, requests = run_agent([custom], [_response([(native_id, name, {})]), _response()])

    assert node.result() == "Done"
    batch = requests[1]["contents"][-1]
    assert batch["role"] == "user"
    result = batch["parts"][0]["functionResponse"]
    assert result.get("id") == native_id
    assert result["name"] == name
    assert result["response"] == {"output": image.status}
    assert result["parts"] == [{"inline_data": {
        "mime_type": image.mime_type, "data": base64.urlsafe_b64encode(image.data).decode(),
    }}]
    transcript_result = next(part for part in node.transcript if isinstance(part, ToolResultPart))
    assert transcript_result.outputs is image and not transcript_result.is_error


def test_invalid_image_arguments_return_a_normal_error(run_agent):
    node, requests = run_agent([view_image], [_response([("bad-args", "view_image", {})]), _response()])

    assert node.result() == "Done"
    result = requests[1]["contents"][-1]["parts"][0]["functionResponse"]
    assert result["id"] == "bad-args"
    assert set(result["response"]) == {"error"}
    assert "parts" not in result
    assert not node.children
    transcript_result = next(part for part in node.transcript if isinstance(part, ToolResultPart))
    assert transcript_result.is_error
    assert transcript_result.outputs == result["response"]["error"]


@pytest.mark.parametrize("tool_result", [
    None,
    "",
    'First "quoted" line\nSecond\tline with \\path and \u96ea\n{"already": "JSON text"}',
])
def test_tool_text_survives_http_json_without_extra_serialization(run_agent, tool_result):
    plain = CodeFunction(name="plain", desc="Plain result", args=[], callable=lambda ctx: tool_result)
    node, requests = run_agent([plain], [_response([("plain-call", "plain", {})]), _response()])

    assert node.result() == "Done"
    result = requests[1]["contents"][-1]["parts"][0]["functionResponse"]
    expected_text = "" if tool_result is None else tool_result
    assert result["response"] == {"output": expected_text}
    assert "parts" not in result
    transcript_result = next(part for part in node.transcript if isinstance(part, ToolResultPart))
    assert transcript_result.outputs == expected_text
    assert not transcript_result.is_error


def test_image_preparation_fault_is_not_a_tool_failure(run_agent, image_path, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("SDK image serialization failed")

    monkeypatch.setattr(types.FunctionResponsePart, "from_bytes", fail)

    node, requests = run_agent([view_image], [
        _response([("native-image", "view_image", {"path": str(image_path)})]),
    ])

    with pytest.raises(ModelProviderException) as caught:
        node.result()
    assert len(requests) == 1
    image = node.children[0].result()
    assert isinstance(image, ImageResult)
    assert node.children[0].exception is None
    transcript_result = next(part for part in node.transcript if isinstance(part, ToolResultPart))
    assert transcript_result.outputs is image
    assert not [part for part in node.transcript if isinstance(part, ToolResultPart) and part.is_error]
    assert isinstance(caught.value.inner_exception, RuntimeError)


def test_plain_builtin_result_remains_text(run_agent, monkeypatch):
    monkeypatch.setattr(view_image, "callable", lambda ctx, *, path: "plain output")
    node, requests = run_agent([view_image], [
        _response([("image", "view_image", {"path": "unused"})]), _response(),
    ])

    assert node.result() == "Done"
    result = requests[1]["contents"][-1]["parts"][0]["functionResponse"]
    assert result["response"] == {"output": "plain output"}
    assert "parts" not in result
    transcript_result = next(part for part in node.transcript if isinstance(part, ToolResultPart))
    assert transcript_result.outputs == "plain output" and not transcript_result.is_error


def test_falsey_invocation_exception_is_a_text_error(run_agent, monkeypatch):
    class FalseyError(Exception):
        def __bool__(self):
            return False

    def fail_invoke(self, tool_name, tool_args, tool_use_id):
        raise FalseyError("cannot invoke")

    monkeypatch.setattr(AgentNode, "invoke_tool_function", fail_invoke)
    node, requests = run_agent([view_image], [
        _response([("failed", "view_image", {"path": "unused"})]), _response(),
    ])

    assert node.result() == "Done"
    assert not node.children
    result = requests[1]["contents"][-1]["parts"][0]["functionResponse"]
    assert result["id"] == "failed"
    assert result["response"] == {"error": "FalseyError: cannot invoke"}
    assert "parts" not in result
    transcript_result = next(part for part in node.transcript if isinstance(part, ToolResultPart))
    assert transcript_result.is_error and transcript_result.outputs == "FalseyError: cannot invoke"
