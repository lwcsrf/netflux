from typing import Any, Callable, Dict, List, Optional, Union
from types import MappingProxyType
import copy
import base64
import time
import random
from multiprocessing.synchronize import Event
import httpx
from overrides import override

from ..core import (
    Node, RunContext, Function, AgentNode, AgentException,
    UserTextPart, ModelTextPart, ThinkingBlockPart, ToolUsePart, ToolResultPart,
    TokenUsage,
)
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
            - append your functionResponse parts (role can be `tool` or `user`).
        5. Repeat until the model returns a non-thinking text answer and no follow-up function
           calls.
    * Interleaved reasoning with tools — Gemini 2.5 Pro behaves like Opus 4.1 (strong evidence
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
"""

MAX_TOKENS = 64000
THINKING_CFG = types.ThinkingConfig(
    thinking_level=types.ThinkingLevel.HIGH,
    # Disable Thought Summaries always: they are not useful, and we don't want to
    # accidentally allow them to be included as past thinking content ever.
    include_thoughts=False,
)
# Prevent agent loop runaway. Max tool call + response cycles before giving up.
MAX_STEPS = 64

class GeminiAgentNode(AgentNode):
    """
    AgentNode impl for Gemini using `google-genai` SDK typed objects exclusively.

    - History is List[types.Content]; Parts are types.Part (text, thought, thought_signature, function_call).
    - Parallel tool execution; aggregate results into a single role="tool" message per cycle including parallel tool calls.
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
    ):
        super().__init__(ctx, id, fn, inputs, parent, cancel_event, client_factory)
        client: Any = client_factory()
        if not isinstance(client, genai.Client):
            raise TypeError(
                "GeminiAgentNode expected client_factory to return google.genai.Client"
            )
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

    def _new_tool_use_id(self, tool_name: str) -> str:
        self._tool_call_counter += 1
        return f"gemini-{self.id}-{self._tool_call_counter}-{tool_name}"

    def run(self) -> None:
        config = types.GenerateContentConfig(
            system_instruction=self.agent_fn.system_prompt,
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
            thinking_config=THINKING_CFG,
            max_output_tokens=MAX_TOKENS,
        )

        # Agent loop.
        for _ in range(MAX_STEPS):
            if self.is_cancel_requested():
                self.ctx.post_cancel()
                self.client.close()
                return

            # One thinking-tool turn.
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
                        contents=self._history,
                        config=config,
                    )

                    # Should never happen in practice. If it does, will not be retried below.
                    if not resp.candidates:
                        # Will be converted by wrapper to ModelProviderException with more context.
                        raise RuntimeError("Gemini returned no candidates.")
                    candidate = resp.candidates[0]

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
                    ):
                        # Force retry below. Burn one attempt.
                        force_retry = True
                        raise RuntimeError(f"Candidate finish_reason is: {candidate.finish_reason}.")

                    # Workaround for an empirical issue solved by retry (observed Oct 2025).
                    if not candidate.content and not candidate.finish_reason:
                        # Force retry below. Burn one attempt.
                        force_retry = True
                        raise RuntimeError(f"Response has no content and no finish_reason.")

                    break
                except (
                    genai_errors.APIError,
                    genai_errors.UnknownApiResponseError,
                    httpx.HTTPStatusError,
                    httpx.TransportError,
                    RuntimeError,
                ) as e:
                    # Retry on rate limits, 5xx responses, and connection/transport issues, or forced from above.
                    is_retriable: bool = force_retry
                    is_connection: bool = False

                    if isinstance(e, httpx.TransportError) and not isinstance(e, httpx.ProtocolError):
                        is_retriable = True
                        is_connection = True
                    if isinstance(e, httpx.RemoteProtocolError):
                        is_retriable = True
                        is_connection = True
                    if isinstance(e, httpx.HTTPStatusError):
                        status_code = e.response.status_code
                        if status_code in (408, 409, 429) or status_code >= 500:
                            is_retriable = True
                    if isinstance(e, genai_errors.APIError):
                        if e.code in (408, 409, 429) or e.code >= 500:
                            is_retriable = True
                    if isinstance(e, genai_errors.UnknownApiResponseError):
                        is_retriable = True

                    if not is_retriable or attempt >= max_attempts:
                        raise

                    if self.is_cancel_requested():
                        self.ctx.post_cancel()
                        self.client.close()
                        return

                    delay = base_delay * (2 ** (attempt - 1))
                    delay = min(delay, 30)
                    # Add small jitter to prevent thundering herd.
                    delay += random.uniform(0, delay * 0.1)

                    # Sleep unless/until canceled.
                    if self.cancel_event:
                        if self.cancel_event.wait(delay):
                            self.ctx.post_cancel()
                            self.client.close()
                            return
                    else:
                        time.sleep(delay)

                    # Rebuild client on transport errors to reset broken sessions/sockets.
                    if is_connection:
                        self.client.close()
                        self.client = self.client_factory()

                    attempt += 1
                    continue
            
            if self.is_cancel_requested():
                self.ctx.post_cancel()
                self.client.close()
                return

            # Incremental token accounting.
            assert resp.usage_metadata is not None, "Gemini response missing usage metadata"
            self._accumulate_usage(resp.usage_metadata)

            # Sanity checks.
            if candidate.content:
                assert candidate.content.role == "model", (
                    f"Last response role must be 'model'. Got: {candidate.content.role}") 
            else:
                assert candidate.finish_reason == types.FinishReason.STOP, (
                    f"finish_reason != STOP and no content is present. "
                    f"Got: {candidate.finish_reason} {candidate.finish_message}")
                # No new content and we are done.
                # Finalize with whatever is in the transcript.
                self.ctx.post_success(self._final_text())
                self.client.close()
                return
            
            # Thoughts are supposed to be empty (hidden) or we want to know of API change.
            self._check_thoughts_sanity(candidate.content)           

            # Always append sanitized model content (keeps history complete) for replay.
            self._history.append(candidate.content)

            part: types.Part
            calls: List[types.FunctionCall] = []
            for part in candidate.content.parts:  # pyright: ignore[reportOptionalIterable]
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
                    self.transcript.append(ModelTextPart(text=text))
                    self.ctx.post_transcript_update()
                    # Ensure our understanding of the protocol is correct that function calls come last.
                    assert not calls, "Gemini text parts should precede function_call parts."

            # No function calls → finalize with assistant text.
            if not calls:
                assert candidate.finish_reason == types.FinishReason.STOP, (
                    "Expected finish_reason=STOP when no function calls")
                self.ctx.post_success(self._final_text())
                self.client.close()
                return

            # At this point, there are function calls to process. First, sanity check:
            assert candidate.finish_reason in (None, types.FinishReason.STOP), (
                f"Expected finish_reason STOP or None when function calls are present. Got: {candidate.finish_reason}")

            # Make sure we check for cancellation right before commencing possibly
            # lengthy sub-tasks.
            if self.is_cancel_requested():
                self.ctx.post_cancel()
                self.client.close()
                return

            # Execute requested tools in parallel and aggregate all function responses.
            # Even if some tool invocations fail early, continue processing others.
            result_parts: list[types.Part] = []
            children: List[Optional[Node]] = []                  # Index to match `calls` 1:1.
            invoke_exceptions: List[Optional[Exception]] = []    # Index to match `calls` 1:1.
            tool_use_ids: List[str] = []
            for fc in calls:
                assert fc.name
                name: str = fc.name
                tool_args: Dict[str, Any] = fc.args or {}
                tool_use_id = fc.id or self._new_tool_use_id(name)
                tool_use_ids.append(tool_use_id)

                args_ro = MappingProxyType(copy.deepcopy(tool_args))
                self.transcript.append(
                    ToolUsePart(tool_use_id=tool_use_id, tool_name=name, args=args_ro)
                )
                self.ctx.post_transcript_update()

                try:
                    children.append(self.invoke_tool_function(name, tool_args))
                    invoke_exceptions.append(None)
                except Exception as ex:
                    children.append(None)
                    invoke_exceptions.append(ex)

            pending_agent_ex: Optional[AgentException] = None

            # WaitAll + transcribe results.
            for fc, child, invoke_ex, tool_use_id in zip(calls, children, invoke_exceptions, tool_use_ids):
                assert fc.name
                response: dict[str, Any] = {}  # for gemini `FunctionResponse.response` field.
                out_text: str
                is_error: bool

                # Did the invocation itself fail-fast? (e.g. bad args)
                if invoke_ex:
                    out_text = AgentNode.stringify_exception(invoke_ex)
                    is_error = True
                    response["error"] = out_text

                # Check if the child Node is success / error.
                else:
                    assert child
                    try:
                        # This will re-raise any exception that happened inside the tool function.
                        result: Any = child.result()
                        out_text = "" if result is None else str(result)
                        is_error = False
                        response["output"] = out_text
                    except AgentException as ex:
                        # Agent decided to raise an exception. Keep processing the rest of the batch
                        # per spec before propagating the exception outside the loop.
                        pending_agent_ex = ex
                        continue
                    except Exception as ex:
                        out_text = AgentNode.stringify_exception(ex)
                        is_error = True
                        response["error"] = out_text

                # Transcript result in common framework types.
                self.transcript.append(
                    ToolResultPart(
                        tool_use_id=tool_use_id,
                        tool_name=fc.name,
                        outputs=out_text,
                        is_error=is_error,
                    )
                )
                self.ctx.post_transcript_update()

                # Transcript result in gemini sdk types.
                result_parts.append(types.Part(
                    function_response=types.FunctionResponse(
                        id=tool_use_id,
                        name=fc.name,
                        response=response,
                    )
                ))

            # Now that we finished collecting + transcribing children, it's a good time to react
            # to agent wanting to raise exception, or cancellation request, in that
            # order of priority.
            if pending_agent_ex:
                self.ctx.post_exception(pending_agent_ex)
                self.client.close()
                return
            if self.is_cancel_requested():
                self.ctx.post_cancel()
                self.client.close()
                return
            
            # Per protocol: next user message contains only function results.
            # Aggregated function results to single Content message.
            self._history.append(types.Content(role="tool", parts=result_parts))

        raise RuntimeError(f"Gemini agent loop exceeded MAX_STEPS ({MAX_STEPS}) "
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
        """
        Extract final text from transcript: concatenate all ModelTextPart text
        that comes after the last function call found (ToolResultPart).
        """
        last_func_idx = -1
        for i, part in enumerate(self.transcript):
            if isinstance(part, ToolResultPart):
                last_func_idx = i

        final_text_chunks: List[str] = []
        for i in range(last_func_idx + 1, len(self.transcript)):
            part = self.transcript[i]
            if isinstance(part, ModelTextPart):
                if part.text.strip():
                    final_text_chunks.append(part.text)
       
        return "\n".join(final_text_chunks)

    @staticmethod
    def _gemini_type_for_arg(py_t: type) -> types.Type:
        if py_t is str:   return types.Type.STRING
        if py_t is int:   return types.Type.INTEGER
        if py_t is float: return types.Type.NUMBER
        if py_t is bool:  return types.Type.BOOLEAN
        return types.Type.STRING
