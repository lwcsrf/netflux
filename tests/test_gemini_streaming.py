import unittest
from unittest.mock import Mock, patch

import h2.exceptions
from h2.connection import ConnectionInputs, H2ConnectionStateMachine
import httpcore2
import httpx2

from ..core import AgentFunction, CodeFunction, Continuation, ModelProviderException, RetriableException, RunContext
from ..providers import Provider
from ..runtime import Runtime

try:
    import google.genai as genai
except ModuleNotFoundError as exc:
    if exc.name not in ("google", "google.genai"):
        raise
    raise unittest.SkipTest("Gemini provider tests require google-genai") from exc

from google.genai import errors, types
from requests.exceptions import ChunkedEncodingError

from ..providers import gemini


def chunk(*parts, finish=None, token=None, counts=None):
    candidate = types.Candidate(
        content=types.Content(role="model", parts=list(parts)) if parts else None,
        finish_reason=finish,
        continuation_token=token,
    )
    usage = None
    if counts is not None:
        usage = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=counts[0], candidates_token_count=counts[1],
        )
    return types.GenerateContentResponse(candidates=[candidate], usage_metadata=usage)


class TestGeminiStreaming(unittest.TestCase):
    def run_streams(self, streams, tools=()):
        streams = iter(streams)
        self.requests = []

        def stream(*, model, contents, config):
            self.assertIs(config.automatic_continuation, False)
            self.requests.append((config.continuation_token, list(contents)))
            for item in next(streams):
                if isinstance(item, Exception):
                    raise item
                yield item

        def factory():
            client = Mock(spec=genai.Client)
            client.models.generate_content_stream.side_effect = stream
            client.models.generate_content.side_effect = AssertionError("Expected streaming request")
            return client

        agent = AgentFunction(
            name="agent", desc="", args=[], system_prompt="system", user_prompt_template="prompt",
            uses=list(tools), default_model=Provider.Gemini,
        )
        self.client_factory = Mock(side_effect=factory)
        runtime = Runtime([agent], client_factories={Provider.Gemini: self.client_factory})
        with patch.object(gemini, "time"):
            node = runtime.invoke(None, agent, {})
            self.assertEqual(node.result(), "done")
        return node

    def test_checkpoint_retry_multihop_and_tool_replay(self):
        def tool(ctx: RunContext) -> str:
            return "ok"

        tool_fn = CodeFunction(name="lookup", desc="", args=[], callable=tool)
        call = types.Part(
            function_call=types.FunctionCall(name="lookup", args={}, id="call-1"),
            thought_signature=b"call-signature",
        )
        metadata = types.Part(text="original", part_metadata={"source": "model"})
        signed_empty = types.Part(text="", thought_signature=b"text-signature")
        timeout = httpx2.ReadTimeout("stream interrupted")
        node = self.run_streams([
            [
                chunk(types.Part(text="first")),
                chunk(types.Part(text=" hop\n"), finish=types.FinishReason.CONTINUATION,
                      token=b"next-hop"),
                chunk(counts=(3, 2)),
            ],
            [
                chunk(types.Part(text="A"), token=b"first-checkpoint", counts=(5, 1)),
                chunk(types.Part(text=" B"), token=b"last-checkpoint", counts=(5, 2)),
                chunk(call, counts=(5, 99)),
                timeout,
            ],
            [
                chunk(types.Part(text=" C"), metadata, signed_empty, counts=(8, 1)),
                chunk(call, finish=types.FinishReason.STOP, counts=(8, 4)),
            ],
            [chunk(types.Part(text="done"), finish=types.FinishReason.STOP, counts=(12, 1))],
        ], tools=[tool_fn])

        self.assertEqual([token for token, _ in self.requests],
                         [None, b"next-hop", b"last-checkpoint", None])
        original = [types.Content(role="user", parts=[types.Part(text="prompt")])]
        for _, history in self.requests[:3]:
            self.assertEqual(history, original)
        _, replay = self.requests[3]
        self.assertEqual(replay[1].parts, [
            types.Part(text="first hop\n"), types.Part(text="A B C"), metadata, signed_empty, call,
        ])
        self.assertEqual(replay[2].parts[0].function_response.id, "call-1")
        self.assertEqual(len(node.children), 1)
        continuation, retry = node.intermediate_events
        self.assertIsInstance(continuation, Continuation)
        self.assertIsInstance(retry, RetriableException)
        self.assertIs(retry.exception, timeout)
        self.assertEqual(retry.attempt, 0)
        self.assertEqual(node.token_usage.input_tokens_total, 3 + 5 + 8 + 12)
        self.assertEqual(node.token_usage.output_tokens_total, 2 + 2 + 4 + 1)

    def test_prompt_feedback_before_disconnect_stops_without_retry(self):
        feedback = types.GenerateContentResponsePromptFeedback(block_reason=types.BlockedReason.SAFETY)
        blocked = types.GenerateContentResponse(prompt_feedback=feedback)
        with self.assertRaisesRegex(ModelProviderException, "SAFETY"):
            self.run_streams([
                [blocked, httpx2.ReadTimeout("stream disconnected")],
                [chunk(types.Part(text="done"), finish=types.FinishReason.STOP, counts=(1, 1))],
            ])
        self.assertEqual(len(self.requests), 1)

    def test_consecutive_interruptions_only_count_new_checkpoints(self):
        node = self.run_streams([
            [chunk(types.Part(text="d"), token=b"first", counts=(2, 1))],
            [chunk(types.Part(text="discard"), counts=(30, 10))],
            [chunk(types.Part(text="o"), token=b"second", counts=(3, 1)),
             chunk(types.Part(text="discard"), counts=(3, 10))],
            [chunk(types.Part(text="ne"), finish=types.FinishReason.STOP, counts=(4, 2))],
        ])
        self.assertEqual([token for token, _ in self.requests], [None, b"first", b"first", b"second"])
        self.assertEqual([event.attempt for event in node.intermediate_events], [0, 1, 2])
        self.assertEqual(node.token_usage.input_tokens_total, 2 + 3 + 4)
        self.assertEqual(node.token_usage.output_tokens_total, 1 + 1 + 2)

    def test_chunked_encoding_error_resumes_from_checkpoint(self):
        disconnect = ChunkedEncodingError("Response ended prematurely")
        node = self.run_streams([
            [
                chunk(types.Part(text="do"), token=b"checkpoint", counts=(2, 1)),
                chunk(types.Part(text="discard")),
                disconnect,
            ],
            [chunk(types.Part(text="ne"), finish=types.FinishReason.STOP, counts=(3, 1))],
        ])
        self.assertEqual([token for token, _ in self.requests], [None, b"checkpoint"])
        retry, = node.intermediate_events
        self.assertIsInstance(retry, RetriableException)
        self.assertIs(retry.exception, disconnect)

    def protocol_errors_after_goaway(self, next_input):
        connection = H2ConnectionStateMachine()
        connection.process_input(ConnectionInputs.RECV_GOAWAY)
        with self.assertRaises(h2.exceptions.ProtocolError) as caught:
            connection.process_input(next_input)
        body_error = caught.exception

        # Reproduce the header-read exception chain from HTTPX2's transport.
        core_error = httpcore2.LocalProtocolError(body_error)
        core_error.__context__ = body_error
        header_error = httpx2.LocalProtocolError("wrapped HTTP/2 error")
        header_error.__cause__ = core_error
        return body_error, header_error

    def test_goaway_rebuilds_client_and_resumes_from_checkpoint(self):
        body_error, header_error = self.protocol_errors_after_goaway(ConnectionInputs.RECV_DATA)
        body_error.args = ("changed h2 message",)  # Only the typed transition identifies this failure.
        node = self.run_streams([
            [chunk(types.Part(text="do"), token=b"checkpoint", counts=(2, 1)),
             chunk(types.Part(text="discard")), body_error],
            [header_error],
            [chunk(types.Part(text="ne"), finish=types.FinishReason.STOP, counts=(3, 1))],
        ])
        self.assertEqual(self.client_factory.call_count, 3)
        self.assertEqual([token for token, _ in self.requests], [None, b"checkpoint", b"checkpoint"])
        self.assertEqual([event.exception for event in node.intermediate_events], [body_error, header_error])

    def test_goaway_exact_message_without_exception_chain(self):
        message = "Invalid input ConnectionInputs.RECV_DATA in state ConnectionState.CLOSED"
        matches = gemini.GeminiAgentNode._is_http2_goaway_cause
        for error_type in (h2.exceptions.ProtocolError, httpx2.LocalProtocolError):
            with self.subTest(error_type=error_type):
                self.assertTrue(matches(error_type(message)))
                self.assertFalse(matches(error_type(f"Other error: {message}")))
        self.assertFalse(matches(RuntimeError(message)))

    def test_unrelated_h2_and_local_protocol_errors_are_terminal(self):
        for error in self.protocol_errors_after_goaway(ConnectionInputs.RECV_HEADERS):
            with self.subTest(error_type=type(error)):
                with self.assertRaises(ModelProviderException) as caught:
                    self.run_streams([[error]])
                self.assertIs(caught.exception.inner_exception, error)
                self.assertEqual(self.client_factory.call_count, 1)
                self.assertEqual(len(self.requests), 1)

    def test_upstream_499_retries_from_latest_checkpoint(self):
        cancelled = errors.ClientError(499, {"error": {"status": "CANCELLED"}})
        node = self.run_streams([
            [chunk(types.Part(text="do"), token=b"checkpoint", counts=(2, 1)), cancelled],
            [chunk(types.Part(text="ne"), finish=types.FinishReason.STOP, counts=(3, 1))],
        ])
        self.assertEqual([token for token, _ in self.requests], [None, b"checkpoint"])
        retry, = node.intermediate_events
        self.assertIs(retry.exception, cancelled)

    def test_unfinished_tool_call_without_checkpoint_is_retried(self):
        call = types.Part(function_call=types.FunctionCall(name="uncommitted", args={}))
        usage = types.GenerateContentResponseUsageMetadata(prompt_token_count=5, candidates_token_count=2)
        node = self.run_streams([
            [chunk(finish=types.FinishReason.STOP), httpx2.ReadTimeout("stream interrupted after STOP")],
            [chunk(call, counts=(2, 1))],
            [
                chunk(types.Part(text="done")),
                chunk(finish=types.FinishReason.STOP),
                types.GenerateContentResponse(usage_metadata=usage),
            ],
        ])
        self.assertEqual([token for token, _ in self.requests], [None, None, None])
        self.assertEqual(self.requests[0][1], self.requests[2][1])
        self.assertEqual(node.token_usage.output_tokens_total, 2)
        self.assertEqual([event.attempt for event in node.intermediate_events], [0, 1])
