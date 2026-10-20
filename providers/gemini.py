from typing import Any, Callable, Dict, List, Optional, Union, cast
from types import MappingProxyType
import copy
import base64
import time
import random
from threading import Event
import httpx2
from overrides import override

from ..core import (
    Node, RunContext, Function, CodeFunction, AgentNode, AgentException,
    UserTextPart, ModelTextPart, ModelStatusPart, ThinkingBlockPart, ToolUsePart, ToolResultPart,
    TokenUsage,
)
from ..func_lib import ImageResult, raise_exception, status_update
from . import ModelNames, Provider

import google.genai as genai
from google.genai import types
from google.genai import errors as genai_errors

"""
## Misc Research Notes.

* Strict structured outputs (if needed, e.g. if framework supports a ReturnStructured concept
  in the future)
    * `config={"response_mime_type": "application/json", "response_schema": list[Recipe]}`
    * Refer to: https://ai.google.dev/gemini-api/docs/structured-output
    * Validation: Post-validate all structured outputs with jsonschema or Pydantic;
      don't assume perfection. (Google notes validators aren't applied

* Thinking with Interleaved Function Calls:
    * Refer to: https://ai.google.dev/gemini-api/docs/function-calling?example=meeting#thinking
        * Preserving the thought signatures in the user request callbacks. This works very
          similarly to Anthropic and other model providers. The idea is that they are the
          decrypted reasoning tokens, which the model can use to restore the continuous CoT
          across tool calls, while remaining stateless on the server side.
    * The manual loop (Automatic Function Calling disabled):
        1. Send generateContent with your tool declarations and prompt.
        2. If the response contains functionCall(s), execute those functions in your app.
        3. Build Part.from_function_response(name=..., response=...) for each call.
        4. Send another generateContent with:
            - the same configs and tools
            - all the session messages thus far, and:
            - append the previous response model content (containing the last “thinking”
              thought signatures and the last function call(s)),
        5. Repeat until the model returns a non-thinking text answer and no follow-up function
           calls.
    * Interleaved reasoning with tools — Gemini 3.1 Pro behaves like Opus 4.1 (strong evidence
      for both models).
        * After a reasoning phase, the model can emit one or more functionCalls; once tool
          results are returned, it resumes with new reasoning before deciding whether to call
          more tools or produce text. Empirically you see thought-signature parts preceding
          the calls, then, after tool responses, new thought-signatures and another round of
          calls/text. This repeats across many cycles—matching Opus 4.1's interleaved
          “think → tool → think → …” pattern.
    * Full reasoning continuity across tool cycles — same continuity model as Opus 4.1 (strong
      evidence for both models).
        * As long as you replay the entire prior model response (including thought-signatures)
          plus your function responses, the next turn continues a single, coherent chain of
          reasoning starting from the last user text prompt onward. Empirically, we ran
          experiments whose final outputs were dependent on early internal thought state from
          many tool cycles ago, and these do prove that the context window includes thought
          parts from many tool cycles ago - just like with Opus 4.1 when you fully replay
          history.

* No particular tool specially used in training for bash or text editor like opus.
    - Re-use the `Bash` and `TextEditor` functions inspired by Anthropic's spec, but you need
      to provide the full tool specs unlike Anthropic.

* Multi-hop continuations; starting with Gemini 4 Argon:
    * Long generations can now be forced to span multiple request cycles, called "hops",
      indicated by stop_reason and continuation_token (encapsulates the multi-hop state thus far).
    * A continuation token preserves generation state over multi-hops;
      the next hop resumes from original input + continuation token.
    * The outputs of multi-hop combine into one model response = one thinking-tools agent cycle.
    * Manual continuation gives callers control over retry policy and backoff, so
      retries can be scoped to the current hop.
    * Ref: `https://ai.google.dev/gemini-api/docs/generate-content/long-decode`.
"""

MAX_TOKENS = 131_072
THINKING_CFG = types.ThinkingConfig(
    thinking_level=types.ThinkingLevel.HIGH,
    # Disable Thought Summaries always: they are not useful, and we don't want to
    # accidentally allow them to be included as past thinking content ever.
    include_thoughts=False,
)
# Prevent agent loop runaway. Max model request hops, excluding retries.
MAX_STEPS = 768

class GeminiAgentNode(AgentNode):
    """
    AgentNode impl for Gemini using `google-genai` SDK typed objects exclusively.

    - History is List[types.Content]; Parts are types.Part (text, thought, thought_signature, function_call).
    - Parallel function execution; result aggregation.
    - Thought summaries are never stored; signatures are preserved in history and also recorded into Transcript as ThinkingBlockPart.
    - Final assistant content is appended to history even on the last turn.
    - No caching (handled by Gemini service transparently).
    """
    def __init__(
        self,
        ctx: RunContext,
        id: int,
        fn: Function,
        inputs: Dict[str, Any],
        parent: Optional['Node'],
        cancel_event: Optional[Event],
        client_factory: Callable[[], Any],
        tool_use_id: Optional[str] = None,
    ):
        super().__init__(ctx, id, fn, inputs, parent, cancel_event, client_factory, tool_use_id)
        client: genai.Client = cast(genai.Client, client_factory())
        assert isinstance(client, genai.Client), \
            "GeminiAgentNode expected client_factory to return `google.genai.Client`"
        self.client: genai.Client = client
        self.model = ModelNames[Provider.Gemini]
        self._history: List[types.Content] = []   # Typed conversation history we replay every turn
        self._tool_call_counter = 0
        self._tools: List[types.Tool] = self._build_tool_params()
        self._token_usage = TokenUsage()

        # Seed initial user message.
        # Substitute inputs into the templated user prompt.
        user_text = self.build_user_text()
        self.transcript.append(UserTextPart(text=user_text))
        self._history.append(
            types.Content(role="user", parts=[types.Part.from_text(text=user_text)])
        )

    @property
    @override
    def token_usage(self) -> TokenUsage:
        return self._token_usage

    @property
    @override
    def provider(self) -> Provider:
        return Provider.Gemini

    def append_model_text(self, text: str) -> None:
        """Coalesce adjacent nonblank text."""
        if not text or len(text) == 0:
            return
        if self.transcript and isinstance(self.transcript[-1], ModelTextPart):
            text = self.transcript[-1].text + text
            # Published snapshots (like NodeView) use immutable frozen Parts list
            # but reference shared Part objects, so always replace rather than mutate.
            self.transcript[-1] = ModelTextPart(text=text)
        else:
            self.transcript.append(ModelTextPart(text=text))
        self.ctx.post_transcript_update()

    def _new_tool_use_id(self, tool_name: str) -> str:
        self._tool_call_counter += 1
        return f"gemini-{self.id}-{self._tool_call_counter}-{tool_name}"

    def is_valid_status_update(self, fc: types.FunctionCall) -> bool:
        if not fc.name or self.func_map.get(fc.name) is not status_update:
            return False
        try:
            status_update.validate_coerce_args(fc.args or {})
        except ValueError:
            return False
        return True

    def _close_client(self) -> None:
        if self.client is None:
            return
        try:
            self.client.close()
        except Exception:
            pass
        finally:
            self.client = None  # type: ignore[assignment]

    def run(self) -> None:
        try:
            self.run_agent_loop()
        finally:
            self._close_client()

    def run_agent_loop(self) -> None:
        config = types.GenerateContentConfig(
            system_instruction=self.system_prompt(),
            tools=self._tools,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.AUTO
                ),
            ),
            # We should only use manual function call loop, never auto function calling (incompatible
            # with our transcripts, event posting, exceptions model, etc).
            # Refer to: https://googleapis.github.io/python-genai/#function-calling
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            # We control retries and scope the retry down to the latest continuation only.
            automatic_continuation=False,
            thinking_config=THINKING_CFG,
            max_output_tokens=MAX_TOKENS,
        )
        pending_parts: list[types.Part] = []   # To accumulate all parts in multi-hop cycle.
        continuation_token: Optional[bytes] = None

        # Agent loop.
        # Prior to Argon, every thinking-tools agent cycle was one iteration of this loop.
        # Since Argon, each iteration is one "hop" within a thinking-tools agent cycle.
        for _ in range(MAX_STEPS):
            if self.is_cancel_requested():
                self.ctx.post_cancel()
                self._close_client()
                return

            hop_config: types.GenerateContentConfig = config.model_copy(
                update={"continuation_token": continuation_token}
            )
            resp: types.GenerateContentResponse
            candidate: types.Candidate
            max_attempts = 8
            base_delay = 3  # seconds
            attempt = 1
            # Retry logic.
            while True:
                force_retry: bool = False
                try:
                    resp = self.client.models.generate_content(
                        model=self.model,
                        contents=cast(types.ContentListUnionDict, self._history),
                        config=hop_config,
                    )

                    # Blocked prompts can legitimately return no candidates (different than block during generation).
                    # Not likely retriable, so raise non-retriable. Very rare in practice.
                    if not resp.candidates:
                        raise RuntimeError(
                            f"Gemini returned no candidates. Prompt feedback: {resp.prompt_feedback!r}")
                    candidate = resp.candidates[0]

                    # Todo: Use SDK enum once it's added.
                    if candidate.finish_reason == "MISSING_THOUGHT_SIGNATURE":
                        raise AssertionError(
                            f"Gemini reported {candidate.finish_reason}; "
                            "our reasoning-continuity protocol assumptions need investigation. "
                            f"Details: {candidate.finish_message!r}.")

                    if candidate.finish_reason == types.FinishReason.TOO_MANY_TOOL_CALLS:
                        raise RuntimeError(f"Unexpected finish_reason: {candidate.finish_reason}.")

                    # False positive for safety block or SDK proactively detecting malformed
                    # function call.
                    if candidate.finish_reason and candidate.finish_reason in (
                        types.FinishReason.SAFETY,
                        types.FinishReason.RECITATION,
                        types.FinishReason.LANGUAGE,
                        types.FinishReason.BLOCKLIST,
                        types.FinishReason.PROHIBITED_CONTENT,
                        types.FinishReason.SPII,
                        types.FinishReason.MALFORMED_FUNCTION_CALL,
                        types.FinishReason.UNEXPECTED_TOOL_CALL,
                        "MALFORMED_RESPONSE",  # Not yet exposed by the SDK enum.
                    ):
                        # Force retry below. Burn one attempt.
                        force_retry = True
                        raise RuntimeError(f"Candidate finish_reason is: {candidate.finish_reason}.")

                    # Workaround for an empirical issue solved by retry.
                    if ((not candidate.content or not candidate.content.parts)
                            and not candidate.finish_reason
                            and not candidate.continuation_token):
                        # Force retry below. Burn one attempt.
                        force_retry = True
                        raise RuntimeError(f"Response has no content and no finish_reason and no continuation.")

                    break
                except (
                    genai_errors.APIError,
                    genai_errors.UnknownApiResponseError,
                    httpx2.HTTPStatusError,
                    httpx2.TransportError,
                    RuntimeError,
                ) as e:
                    # Retry on rate limits, 5xx responses, and connection/transport issues, or forced from above.
                    is_retriable: bool = force_retry
                    is_connection: bool = False

                    if isinstance(e, httpx2.TransportError) and not isinstance(e, httpx2.ProtocolError):
                        is_retriable = True
                        is_connection = True
                    if isinstance(e, httpx2.RemoteProtocolError):
                        is_retriable = True
                        is_connection = True
                    if isinstance(e, httpx2.HTTPStatusError):
                        status_code = e.response.status_code
                        if status_code in (408, 409, 429) or status_code >= 500:
                            is_retriable = True
                    if isinstance(e, genai_errors.APIError):
                        if e.code in (408, 409, 429) or e.code >= 500:
                            is_retriable = True
                    if isinstance(e, genai_errors.UnknownApiResponseError):
                        is_retriable = True

                    if not is_retriable or attempt >= max_attempts:
                        self._close_client()
                        raise

                    delay = base_delay * (2 ** (attempt - 1))
                    delay = min(delay, 30)
                    # Add small jitter to prevent thundering herd.
                    delay += random.uniform(0, delay * 0.1)

                    self.record_retriable_exception(e, attempt=attempt - 1, backoff_seconds=delay)

                    if self.is_cancel_requested():
                        self.ctx.post_cancel()
                        self._close_client()
                        return

                    # Sleep unless/until canceled.
                    if self.cancel_event:
                        if self.cancel_event.wait(delay):
                            self.ctx.post_cancel()
                            self._close_client()
                            return
                    else:
                        time.sleep(delay)
                    self.ctx.post_update()

                    # Rebuild client on transport errors to reset broken sessions/sockets.
                    if is_connection:
                        self._close_client()
                        self.client = self.client_factory()

                    attempt += 1
                    continue
            
            if self.is_cancel_requested():
                self.ctx.post_cancel()
                self._close_client()
                return

            # Incremental token accounting.
            assert resp.usage_metadata is not None, "Gemini response missing usage metadata"
            self._accumulate_usage(resp.usage_metadata)

            # Validate each hop before adding its parts to the pending generation.
            if candidate.content:
                assert candidate.content.role in (None, "model"), (
                    f"Unexpected Gemini response role: {candidate.content.role}")
                self._check_thoughts_sanity(candidate.content)
            hop_parts: types.Parts = (candidate.content.parts or []) if candidate.content else []

            if candidate.finish_reason == types.FinishReason.CONTINUATION or (
                candidate.finish_reason is None and candidate.continuation_token
            ):
                if not candidate.continuation_token:
                    raise RuntimeError("Gemini returned CONTINUATION without a continuation token.")
                pending_parts.extend(hop_parts)
                continuation_token = candidate.continuation_token
                
                # Publish the continuation and usage while both transcripts remain buffered.
                self.record_continuation()
                continue

            if candidate.finish_reason == types.FinishReason.MAX_TOKENS:
                raise RuntimeError(f"Gemini exhausted max_output_tokens ({MAX_TOKENS}).")

            assert candidate.finish_reason == types.FinishReason.STOP or (
                candidate.finish_reason is None and any(p.function_call for p in hop_parts)
            ), (
                "Expected a completed Gemini generation. "
                f"Got: {candidate.finish_reason} {candidate.finish_message}")

            # Multi-hop agent cycle is ready to proceed after this final hop.
            pending_parts.extend(hop_parts)
            parts: types.Parts = pending_parts
            pending_parts = []
            continuation_token = None
            if not parts:
                self.ctx.post_success(self._final_text())
                self._close_client()
                return

            self._history.append(types.Content(role="model", parts=parts))

            part: types.Part
            calls: List[types.FunctionCall] = []
            text_chunks: List[str] = []
            for part in parts:
                thought_sig: Optional[bytes] = part.thought_signature
                if thought_sig:
                    # Thought signatures are always recorded in transcript as ThinkingBlockPart
                    # without the "though" text for Gemini (since it's empty/hidden by policy right now).
                    if isinstance(thought_sig, (bytes, bytearray)):
                        sig_b64 = base64.b64encode(thought_sig).decode("utf-8")
                    else:
                        sig_b64 = str(thought_sig)
                    self.transcript.append(ThinkingBlockPart(content="", signature=sig_b64))
                    self.ctx.post_transcript_update()
                    # Ensure our understanding of the protocol is correct that function calls come last.
                    assert not calls, "Gemini thought_signature parts should precede function_call parts."

                func_call: Optional[types.FunctionCall] = part.function_call
                if func_call:
                    calls.append(func_call)
                    # Append to transcript later in the loop below.

                text: Optional[str] = part.text
                if text:
                    text_chunks.append(text)
                    # Ensure our understanding of the protocol is correct that function calls come last.
                    assert not calls, "Gemini text parts should precede function_call parts."

            for text in text_chunks:
                self.append_model_text(text)

            # No function calls → finalize with assistant text.
            if not calls:
                assert candidate.finish_reason == types.FinishReason.STOP, (
                    "Expected finish_reason=STOP when no function calls")
                self.ctx.post_success(self._final_text())
                self._close_client()
                return

            # At this point, there are function calls to process. First, sanity check:
            assert candidate.finish_reason in (None, types.FinishReason.STOP), (
                f"Expected finish_reason STOP or None when function calls are present. Got: {candidate.finish_reason}")

            # Make sure we check for cancellation right before commencing possibly
            # lengthy sub-tasks.
            if self.is_cancel_requested():
                self.ctx.post_cancel()
                self._close_client()
                return

            # Execute requested tools in parallel and aggregate all function responses.
            # Even if some tool invocations fail early, continue processing others.
            result_parts: list[types.Part] = []
            children: List[Optional[Node]] = []                  # Index to match `calls` 1:1.
            invoke_exceptions: List[Optional[Exception]] = []    # Index to match `calls` 1:1.
            tool_use_ids: List[str] = []
            for fc in calls:
                assert fc.name, "Gemini returned a function call with a missing or empty name."
                name: str = fc.name
                tool_args: Dict[str, Any] = fc.args or {}

                # `fc.id` lacks uniqueness across sessions, so netflux transcripts will use `tool_use_id`
                # for call/response as the Runtime-unique key, while gemini response will use `fc.id`.
                tool_use_id = self._new_tool_use_id(name)
                tool_use_ids.append(tool_use_id)

                args_ro = MappingProxyType(copy.deepcopy(tool_args))
                if self.is_valid_status_update(fc):
                    self.transcript.append(
                        ModelStatusPart(text=args_ro["msg"], tool_use_id=tool_use_id)
                    )
                else:
                    self.transcript.append(
                        ToolUsePart(tool_use_id=tool_use_id, tool_name=name, args=args_ro)
                    )
                self.ctx.post_transcript_update()

                try:
                    children.append(self.invoke_tool_function(name, tool_args, tool_use_id))
                    invoke_exceptions.append(None)
                except Exception as ex:
                    children.append(None)
                    invoke_exceptions.append(ex)

            pending_agent_ex: Optional[AgentException] = None

            # WaitAll + transcribe results.
            for fc, child, invoke_ex, tool_use_id in zip(calls, children, invoke_exceptions, tool_use_ids):
                assert fc.name, "Gemini returned a function call with a missing or empty name."
                result: Any
                out_text: str
                is_error: bool

                # Did the invocation itself fail-fast? (e.g. bad args)
                if invoke_ex is not None:
                    result = invoke_ex
                    out_text = AgentNode.stringify_exception(invoke_ex)
                    is_error = True

                # Check if the child Node is success / error.
                else:
                    assert child
                    try:
                        # This will re-raise any exception that happened inside the tool function.
                        result = child.result()
                        if result is None:
                            out_text = ""
                        elif isinstance(result, ImageResult):
                            out_text = result.status
                        else:
                            out_text = str(result)
                        is_error = False
                    except Exception as ex:
                        if (
                            isinstance(ex, AgentException)
                            and ex.node_id == self.id
                            and isinstance(child.fn, CodeFunction)
                            and (child.fn is raise_exception or child.fn.name == "raise_exception")
                        ):
                            # This agent decided to raise an exception. Keep processing the rest of the batch
                            # per spec before propagating the exception outside the loop.
                            pending_agent_ex = ex
                            continue
                        result = ex
                        out_text = AgentNode.stringify_exception(ex)
                        is_error = True

                # Transcript result in common framework types.
                if not self.is_valid_status_update(fc):
                    self.transcript.append(
                        ToolResultPart(
                            tool_use_id=tool_use_id,
                            tool_name=fc.name,
                            outputs=result if isinstance(result, ImageResult) else out_text,
                            is_error=is_error,
                        )
                    )
                    self.ctx.post_transcript_update()

                # Nest media beside the text response; SDK faults stay outside the tool handler.
                image_parts: Optional[list[types.FunctionResponsePart]] = None
                if isinstance(result, ImageResult):
                    image_parts = [types.FunctionResponsePart.from_bytes(
                        data=result.data, mime_type=result.mime_type,
                    )]
                result_parts.append(types.Part(
                    function_response=types.FunctionResponse(
                        id=fc.id,
                        name=fc.name,
                        response={"error" if is_error else "output": out_text},
                        parts=image_parts,
                    )
                ))

            # Now that we finished collecting + transcribing children, it's a good time to react
            # to agent wanting to raise exception, or cancellation request, in that
            # order of priority.
            if pending_agent_ex:
                self.ctx.post_exception(pending_agent_ex)
                self._close_client()
                return
            if self.is_cancel_requested():
                self.ctx.post_cancel()
                self._close_client()
                return

            assert len(result_parts) == len(calls)
            
            # Per protocol: next user message contains only function results.
            # Aggregated function results to single Content message.
            self._history.append(types.Content(role="user", parts=result_parts))

        self._close_client()
        raise RuntimeError(f"Gemini agent loop exceeded MAX_STEPS ({MAX_STEPS} model hops) "
                           "without producing a final response.")

    def _accumulate_usage(self, usage: types.GenerateContentResponseUsageMetadata):
        """For updating TokenUsage after each SDK response in the agent loop."""
        assert usage.prompt_token_count is not None, "Gemini response missing prompt token count"

        cache_read = usage.cached_content_token_count or 0
        prompt_tokens = usage.prompt_token_count
        tool_prompt_tokens = usage.tool_use_prompt_token_count or 0
        reasoning_tokens = usage.thoughts_token_count or 0
        text_tokens = usage.candidates_token_count or 0

        token_usage = self._token_usage
        token_usage.input_tokens_cache_read += cache_read
        token_usage.input_tokens_regular += (prompt_tokens + tool_prompt_tokens) - cache_read
        token_usage.input_tokens_total += prompt_tokens + tool_prompt_tokens
        token_usage.output_tokens_reasoning = (token_usage.output_tokens_reasoning or 0) + reasoning_tokens
        token_usage.output_tokens_text = (token_usage.output_tokens_text or 0) + text_tokens
        token_usage.output_tokens_total += reasoning_tokens + text_tokens
        token_usage.context_window_in = prompt_tokens + tool_prompt_tokens
        token_usage.context_window_out = reasoning_tokens + text_tokens

    def _build_tool_params(self) -> list[types.Tool]:
        decls = [self._make_function_declaration(t) for t in self.agent_fn.uses]
        return [types.Tool(function_declarations=decls)] if decls else []

    def _make_function_declaration(self, fn: Function) -> types.FunctionDeclaration:
        params_props: Dict[str, types.Schema] = {}
        for arg in fn.args:
            enum: Union[List[str], None] = None
            if arg.argtype is str and arg.enum is not None:
                enum = sorted(list(arg.enum))

            arg_schema = types.Schema(
                type=self._gemini_type_for_arg(arg.argtype),
                description=arg.desc,
                enum=enum,
            )
            params_props[arg.name] = arg_schema

        params_required = [arg.name for arg in fn.args if not arg.optional]

        return types.FunctionDeclaration(
            name=fn.name,
            description=fn.desc,
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties=params_props,
                required=params_required,
            ),
        )

    def _check_thoughts_sanity(self, content: types.Content):
        # Ensure empty `thought` text.
        # For Gemini, thoughts are currently hidden. Only thought signatures are used for replay.
        # It would be very ambiguous if we somehow replay partial thoughts, or api behavior changes.
        # This is a sanity check to eliminate any such uncertainty.
        if not content:
            return
        parts = content.parts or []
        for p in parts:
            if p.thought is None:
                continue
            if isinstance(p.thought, str):
                assert p.thought.strip() == "", "Gemini thought text is supposed to be empty."
            if isinstance(p.thought, bool):
                if p.thought:
                    assert p.text is None or p.text.strip() == "", "Gemini thought text is supposed to be empty."

    def _final_text(self) -> str:
        """Return the final answer exactly as recorded after the last tool."""
        for part in reversed(self.transcript):
            if isinstance(part, ModelTextPart):
                return part.text
            if isinstance(part, (ToolUsePart, ToolResultPart, ModelStatusPart)):
                break

        # Gemini can successfully stop without text. Record that empty result
        # explicitly so even these successes have a matching final text part.
        self.transcript.append(ModelTextPart(text=""))
        self.ctx.post_transcript_update()
        return ""

    @staticmethod
    def _gemini_type_for_arg(py_t: type) -> types.Type:
        if py_t is str:   return types.Type.STRING
        if py_t is int:   return types.Type.INTEGER
        if py_t is float: return types.Type.NUMBER
        if py_t is bool:  return types.Type.BOOLEAN
        return types.Type.STRING
