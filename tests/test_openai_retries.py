"""Exercise provider retries through the real SDK and Runtime, without networking."""

import json
import ssl
from datetime import datetime, timezone
from email.utils import format_datetime
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

openai = pytest.importorskip("openai", minversion="3.8.0")
import httpx2
from openai._constants import MAX_RETRY_AFTER_DELAY

from ..core import AgentFunction, CancellationException, ModelProviderException, NodeState, ToolUsePart
from ..providers import Provider
from ..providers import openai as provider_mod
from ..runtime import Runtime
from .test_openai_provider import _call, _message, _response, _tool


def _ok(*output, status="completed", code=None):
    body = _response(*output, status=status).to_dict(mode="json", use_api_names=True)
    if code is not None:
        body["error"] = {"code": code, "message": "Response failed"}
    return httpx2.Response(200, json=body)


def _error(status=503, *, code="server_error", error_type="server_error", headers=None):
    return httpx2.Response(status, headers=headers, json={
        "error": {"message": "Request failed", "code": code, "type": error_type},
    })


@pytest.fixture
def run_agent(monkeypatch):
    delays = []
    monkeypatch.setattr(provider_mod.time, "sleep", delays.append)
    monkeypatch.setattr(provider_mod.random, "uniform", lambda *args: 0.0)

    def run(outcomes, *, tool=None, cancel_event=None, close_error=False,
            factory_error=None, strict_validation=False):
        pending = iter(outcomes)
        requests, clients, transports = [], [], []

        def handle(request):
            requests.append(json.loads(request.content))
            outcome = next(pending)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        class Transport(httpx2.MockTransport):
            close_count = 0

            def close(self):
                self.close_count += 1
                super().close()
                if close_error:
                    raise RuntimeError("Cleanup failed")

        def factory():
            if clients and factory_error is not None:
                raise factory_error
            transport = Transport(handle)
            # A user-supplied client may enable SDK retries. The provider must
            # override them to retain its own attempt and cancellation budget.
            client = openai.OpenAI(
                api_key="test-key", base_url="https://openai.invalid/v1",
                max_retries=5, http_client=httpx2.Client(transport=transport, trust_env=False),
                _strict_response_validation=strict_validation,
            )
            transports.append(transport)
            clients.append(client)
            return client

        fn = AgentFunction(
            name="agent", desc="Retry test", args=[], system_prompt="System",
            user_prompt_template="Task", uses=[tool] if tool else [],
            default_model=Provider.OpenAI,
        )
        runtime = Runtime([fn], client_factories={Provider.OpenAI: factory})
        node = runtime.invoke(None, fn, {}, cancel_event=cancel_event, max_agent_levels=1)
        assert node.done.wait(5), "OpenAI test did not finish"
        node.thread.join(5)
        assert not node.thread.is_alive()
        assert all(client.is_closed() for client in clients)
        assert [transport.close_count for transport in transports] == [1] * len(transports)
        return SimpleNamespace(node=node, requests=requests, clients=clients, delays=delays)

    return run


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 504])
def test_transient_http_errors_retry_without_rebuilding_client(run_agent, status):
    result = run_agent([_error(status), _ok(_message("Done"))])
    assert result.node.result() == "Done"
    assert len(result.requests) == 2
    assert result.requests[0] == result.requests[1]
    assert len(result.clients) == 1
    assert result.delays == [3.0]
    assert result.node.token_usage.input_tokens_total == 20


@pytest.mark.parametrize("status", [400, 401, 403, 404, 405, 413, 418, 422])
def test_terminal_http_errors_are_not_retried(run_agent, status):
    result = run_agent([_error(status), _ok(_message("Unexpected"))])
    with pytest.raises(ModelProviderException) as caught:
        result.node.result()
    assert isinstance(caught.value.inner_exception, openai.APIStatusError)
    assert caught.value.inner_exception.status_code == status
    assert len(result.requests) == len(result.clients) == 1
    assert result.delays == []


@pytest.mark.parametrize("field", ["code", "error_type"])
@pytest.mark.parametrize("code", ["insufficient_quota", "future_error"])
def test_rate_limit_retry_classification_depends_on_http_status(run_agent, field, code):
    result = run_agent([
        _error(429, **{field: code}), _ok(_message("Done")),
    ])
    assert result.node.result() == "Done"
    assert len(result.requests) == 2
    assert result.delays == [3.0]


@pytest.mark.parametrize("status,override,retries", [
    (503, "false", False), (429, "false", False),
    (503, "true", True), (418, "true", False), (400, "true", False),
    (401, "true", False), (403, "true", False),
])
def test_server_retry_override_can_only_veto_the_retry_whitelist(run_agent, status, override, retries):
    result = run_agent([
        _error(status, headers={"x-should-retry": override}), _ok(_message("Done")),
    ])
    if retries:
        assert result.node.result() == "Done"
    else:
        with pytest.raises(ModelProviderException):
            result.node.result()
    assert len(result.requests) == 1 + retries
    assert len(result.delays) == int(retries)


@pytest.mark.parametrize("status,code,error_cls", [
    (503, "future_error", openai.InternalServerError),
    (429, "insufficient_quota", openai.RateLimitError),
])
def test_http_attempt_budget_disables_sdk_retries_and_preserves_details(run_agent, status, code, error_cls):
    result = run_agent([_error(status, code=code) for _ in range(9)])
    with pytest.raises(ModelProviderException) as caught:
        result.node.result()
    assert isinstance(caught.value.inner_exception, error_cls)
    assert caught.value.inner_exception.code == code
    assert code in str(caught.value)
    assert len(result.requests) == 8
    assert len(result.clients) == 1
    assert result.delays == [3.0, 6.0, 12.0, 24.0, 30.0, 30.0, 30.0]


@pytest.mark.parametrize("error_cls", [
    httpx2.ConnectError, httpx2.ReadError, httpx2.WriteError,
    httpx2.RemoteProtocolError, httpx2.ConnectTimeout, httpx2.ReadTimeout,
    httpx2.WriteTimeout, httpx2.PoolTimeout,
])
def test_transport_errors_retry_with_a_fresh_client(run_agent, error_cls):
    result = run_agent([error_cls("Transport failed"), _ok(_message("Done"))])
    assert result.node.result() == "Done"
    assert len(result.requests) == len(result.clients) == 2
    assert result.requests[0] == result.requests[1]
    assert result.delays == [3.0]


def test_connection_attempt_budget_does_not_create_an_unused_client(run_agent):
    result = run_agent([httpx2.ConnectError("Disconnected") for _ in range(9)])
    with pytest.raises(ModelProviderException) as caught:
        result.node.result()
    assert isinstance(caught.value.inner_exception, openai.APIConnectionError)
    assert len(result.requests) == len(result.clients) == 8
    assert len(result.delays) == 7


@pytest.mark.parametrize("error_cls", [
    ValueError, httpx2.LocalProtocolError, ssl.SSLCertVerificationError,
])
def test_sdk_connection_errors_retry_without_classifying_their_causes(run_agent, error_cls):
    result = run_agent([error_cls("Connection failed"), _ok(_message("Done"))])
    assert result.node.result() == "Done"
    assert len(result.requests) == len(result.clients) == 2
    assert result.delays == [3.0]


def test_exhausted_connection_retries_preserve_the_original_cause_chain(run_agent):
    error = httpx2.ConnectError("TLS failed")
    error.__cause__ = ssl.SSLCertVerificationError("Certificate verify failed")
    result = run_agent([error] * 8)
    with pytest.raises(ModelProviderException) as caught:
        result.node.result()
    assert isinstance(caught.value.inner_exception, openai.APIConnectionError)
    assert caught.value.inner_exception.__cause__ is error
    assert isinstance(error.__cause__, ssl.SSLCertVerificationError)
    assert len(result.requests) == len(result.clients) == 8
    assert len(result.delays) == 7


@pytest.mark.parametrize("code", ["server_error", "rate_limit_exceeded", "vector_store_timeout"])
def test_failed_responses_retry_without_replaying_or_running_partial_tools(run_agent, code):
    called = Mock(return_value="tool result")
    result = run_agent([
        _ok(_call("discarded"), status="failed", code=code),
        _ok(_call("accepted")),
        _error(),
        _ok(_message("Done")),
    ], tool=_tool(lambda ctx: called()))
    assert result.node.result() == "Done"
    called.assert_called_once_with()
    assert len(result.node.children) == 1
    assert len(result.requests) == 4
    assert result.requests[0] == result.requests[1]
    assert result.requests[2] == result.requests[3]
    assert [item["type"] for item in result.requests[2]["input"]] == [
        "message", "function_call", "function_call_output",
    ]
    assert result.requests[2]["input"][1]["call_id"] == "accepted"
    assert len([part for part in result.node.transcript if isinstance(part, ToolUsePart)]) == 1
    usage = result.node.token_usage
    assert (usage.input_tokens_total, usage.output_tokens_total) == (60, 24)
    assert (usage.input_tokens_regular, usage.input_tokens_cache_read, usage.input_tokens_cache_write) == (36, 15, 9)
    assert (usage.context_window_in, usage.context_window_out) == (20, 8)


def test_failed_response_budget_preserves_usage_from_all_attempts(run_agent):
    result = run_agent([_ok(status="failed", code="server_error") for _ in range(9)])
    with pytest.raises(ModelProviderException, match="server_error"):
        result.node.result()
    assert len(result.requests) == 8
    assert len(result.delays) == 7
    assert len(result.clients) == 1
    assert (result.node.token_usage.input_tokens_total, result.node.token_usage.output_tokens_total) == (160, 64)


def test_http_transport_and_failed_response_errors_share_one_attempt_budget(run_agent):
    outcomes = [
        _error(), httpx2.ReadTimeout("Timed out"),
        _ok(status="failed", code="server_error"),
    ]
    result = run_agent(outcomes * 3)
    with pytest.raises(ModelProviderException) as caught:
        result.node.result()
    assert isinstance(caught.value.inner_exception, openai.APITimeoutError)
    assert len(result.requests) == 8
    assert len(result.delays) == 7
    assert len(result.clients) == 3
    assert (result.node.token_usage.input_tokens_total, result.node.token_usage.output_tokens_total) == (40, 16)


@pytest.mark.parametrize("status,code", [
    ("failed", "invalid_prompt"), ("failed", "bio_policy"),
    ("failed", None),
    ("incomplete", None), ("completed", "server_error"),
    ("cancelled", "server_error"),
])
def test_terminal_response_outcomes_are_not_retried(run_agent, status, code):
    result = run_agent([_ok(status=status, code=code), _ok(_message("Unexpected"))])
    with pytest.raises(ModelProviderException) as caught:
        result.node.result()
    assert type(caught.value) is ModelProviderException
    assert caught.value.inner_exception is None
    if status != "completed":
        assert "resp_test" in caught.value.message
        assert f"status={status!r}" in caught.value.message
    if code is not None:
        assert code in caught.value.message
    assert len(result.requests) == 1
    assert result.delays == []
    assert (result.node.token_usage.input_tokens_total, result.node.token_usage.output_tokens_total) == (20, 8)


@pytest.mark.parametrize("invalid_json", [False, True])
@pytest.mark.parametrize("strict_validation", [False, True])
@pytest.mark.parametrize("reconnect", [False, True])
def test_sdk_response_parsing_errors_are_terminal(run_agent, invalid_json, strict_validation, reconnect):
    response = (
        httpx2.Response(200, content="{broken", headers={"content-type": "application/json"})
        if invalid_json else httpx2.Response(200, json={"object": "response"})
    )
    outcomes = [httpx2.ConnectError("Disconnected")] if reconnect else []
    result = run_agent(outcomes + [response, _ok(_message("Unexpected"))],
                       strict_validation=strict_validation)
    with pytest.raises(ModelProviderException) as caught:
        result.node.result()
    expected = json.JSONDecodeError if invalid_json else openai.APIResponseValidationError
    error = caught.value.inner_exception
    assert error is not None and isinstance(error, expected)
    diagnostic = error if invalid_json else error.__cause__
    assert diagnostic is not None
    assert str(diagnostic) in str(caught.value)
    assert len(result.requests) == len(result.clients) == 1 + reconnect
    assert result.delays == ([3.0] if reconnect else [])


def test_sdk_validation_error_without_cause_preserves_its_message(run_agent):
    response = httpx2.Response(200, request=httpx2.Request("POST", "https://openai.invalid/v1/responses"))
    error = openai.APIResponseValidationError(response, {"object": "response"}, message="Missing response output")
    result = run_agent([error, _ok(_message("Unexpected"))])
    with pytest.raises(ModelProviderException, match="Missing response output") as caught:
        result.node.result()
    assert caught.value.inner_exception is error
    assert error.__cause__ is None
    assert str(error) in caught.value.message
    assert result.node.state == NodeState.Error
    assert len(result.requests) == len(result.clients) == 1
    assert result.delays == []


@pytest.mark.parametrize("headers,expected", [
    ({"retry-after": "2.5"}, 2.5),
    ({"retry-after-ms": "1500"}, 1.5),
    ({"retry-after-ms": "1500", "retry-after": "50"}, 1.5),
    ({"retry-after": str(MAX_RETRY_AFTER_DELAY)}, MAX_RETRY_AFTER_DELAY),
])
def test_server_retry_delay_is_respected(run_agent, headers, expected):
    result = run_agent([_error(429, headers=headers), _ok(_message("Done"))])
    assert result.node.result() == "Done"
    assert result.delays == [expected]


def test_http_date_retry_after_is_respected(run_agent, monkeypatch):
    now = 2_000_000_000
    monkeypatch.setattr(provider_mod.time, "time", lambda: now)
    retry_at = format_datetime(datetime.fromtimestamp(now + 37, timezone.utc), usegmt=True)
    result = run_agent([_error(429, headers={"retry-after": retry_at}), _ok(_message("Done"))])
    assert result.node.result() == "Done"
    assert result.delays == [37.0]


@pytest.mark.parametrize("header", ["retry-after", "retry-after-ms"])
@pytest.mark.parametrize("value", ["invalid", "nan", "inf", "-1", "0"])
def test_invalid_server_delay_uses_normal_backoff(run_agent, header, value):
    result = run_agent([_error(429, headers={header: value}), _ok(_message("Done"))])
    assert result.node.result() == "Done"
    assert result.delays == [3.0]


@pytest.mark.parametrize("headers", [
    {"retry-after": str(MAX_RETRY_AFTER_DELAY + 0.1)},
    {"retry-after-ms": str(MAX_RETRY_AFTER_DELAY * 1000 + 1)},
])
def test_excessive_server_delay_fails_instead_of_retrying_early(run_agent, headers):
    result = run_agent([_error(429, headers=headers), _ok(_message("Unexpected"))])
    with pytest.raises(ModelProviderException) as caught:
        result.node.result()
    assert isinstance(caught.value.inner_exception, openai.RateLimitError)
    assert len(result.requests) == 1
    assert result.delays == []


@pytest.mark.parametrize("connection_error", [False, True])
@pytest.mark.parametrize("wait_result", [False, True])
def test_cancellation_during_retry_wait_prevents_request_and_rebuild(run_agent, connection_error, wait_result):
    class CancelDuringWait(Event):
        def wait(self, timeout=None):
            self.set()
            return wait_result

    result = run_agent([
        httpx2.ConnectError("Disconnected") if connection_error else _error(),
        _ok(_message("Unexpected")),
    ], cancel_event=CancelDuringWait())
    with pytest.raises(CancellationException):
        result.node.result()
    assert len(result.requests) == len(result.clients) == 1


def test_failed_client_rebuild_preserves_the_factory_exception(run_agent):
    factory_error = ValueError("Invalid replacement client configuration")
    result = run_agent([httpx2.ConnectError("Disconnected")], factory_error=factory_error)
    with pytest.raises(ModelProviderException) as caught:
        result.node.result()
    assert caught.value.inner_exception is factory_error
    assert len(result.requests) == len(result.clients) == 1


@pytest.mark.parametrize("path,value", [
    (("output", 1, "call_id"), ["wrong"]),
    (("output", 1, "call_id"), 42),
    (("output", 1, "call_id"), True),
    (("output", 1, "name"), ["tool"]),
    (("output", 1, "caller"), {"type": "future"}),
    (("output", 1, "caller"), {"type": "code_interpreter", "tool_id": "code"}),
    (("output", 1, "async"), "future"),
    (("output", 1, "async"), 2),
    (("output", 2, "phase"), "future_phase"),
    (("output", 2, "role"), "user"),
    (("output", 2, "type"), "future_message"),
    (("output", 2, "content", 0, "type"), "future_content"),
    (("error",), {"code": "future_error", "message": "Response failed"}),
])
def test_malformed_response_fails_sdk_validation_before_dispatch(run_agent, path, value):
    payload = _response(_call("valid"), _call("invalid"), _message("Untrusted output")).to_dict()
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    called = Mock()
    result = run_agent([httpx2.Response(200, json=payload)], tool=_tool(lambda ctx: called()))

    with pytest.raises(ModelProviderException) as caught:
        result.node.result()
    error = caught.value
    validation = error.inner_exception
    assert validation is not None and isinstance(validation, openai.APIResponseValidationError)
    assert validation.body == payload
    assert validation.__cause__ is not None
    assert str(validation.__cause__) in error.message
    assert error.message in str(error)
    assert str(path[-1]) in error.message
    invalid_value = next(iter(value.values())) if isinstance(value, dict) else value
    assert repr(invalid_value) in error.message
    assert error.provider is type(result.node)
    assert (error.agent_name, error.node_id) == ("agent", result.node.id)
    assert result.node.state == NodeState.Error
    called.assert_not_called()
    assert result.node.children == []
    assert result.node.history == result.requests[0]["input"]
    assert len(result.requests) == 1
    assert result.delays == []


@pytest.mark.parametrize("field", ["call_id", "name"])
def test_empty_function_identity_prevents_entire_batch_dispatch(run_agent, field):
    payload = _response(_call("valid"), _call("invalid")).to_dict()
    payload["output"][1][field] = ""
    called = Mock()
    result = run_agent(
        [httpx2.Response(200, json=payload)], tool=_tool(lambda ctx: called()),
    )

    with pytest.raises(ModelProviderException, match=f"invalid function.*{field}") as caught:
        result.node.result()
    assert caught.value.inner_exception is None
    called.assert_not_called()
    assert result.node.children == []
    assert result.node.history == result.requests[0]["input"]
    assert len(result.requests) == 1


@pytest.mark.parametrize("details,field", [
    ("input_tokens_details", "cached_tokens"),
    ("input_tokens_details", "cache_write_tokens"),
    ("output_tokens_details", "reasoning_tokens"),
])
def test_negative_usage_breakdowns_fail_without_corrupting_totals(run_agent, details, field):
    payload = _response(_message("Done")).to_dict()
    payload["usage"][details][field] = -1
    result = run_agent([httpx2.Response(200, json=payload)])

    with pytest.raises(ModelProviderException, match="negative token counts") as caught:
        result.node.result()
    assert f"{field}=-1" in caught.value.message
    assert caught.value.inner_exception is None
    assert result.node.token_usage.input_tokens_total == 0
    assert result.node.token_usage.output_tokens_total == 0
    assert len(result.requests) == 1


@pytest.mark.parametrize("success", [False, True])
def test_cleanup_errors_do_not_replace_the_terminal_outcome(run_agent, success):
    result = run_agent([_ok(_message("Done")) if success else _error(401)], close_error=True)
    if success:
        assert result.node.result() == "Done"
    else:
        with pytest.raises(ModelProviderException) as caught:
            result.node.result()
        assert isinstance(caught.value.inner_exception, openai.AuthenticationError)
