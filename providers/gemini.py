from typing import Any, Callable, Dict, List, Optional, Union
import base64

from ..core import (
    Node, RunContext, Function, AgentNode, AgentException,
    UserTextPart, ModelTextPart, ThinkingBlockPart, ToolUsePart, ToolResultPart,
    TokenUsage,
)
from . import ModelNames, Provider

import google.genai as genai
from google.genai import types
from overrides import override

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

# Max tool call + response cycles before giving up.
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
        client_factory: Callable[[], Any],
    ):
        super().__init__(ctx, id, fn, inputs, parent, client_factory)
        client = client_factory()
        if not isinstance(client, genai.Client):
            raise TypeError(
                "GeminiAgentNode expected client_factory to return google.genai.Client"
            )
        self.client = client
        self._tool_call_counter = 0
        self._token_usage = TokenUsage()

    @property
    @override
    def token_usage(self) -> TokenUsage:
        return self._token_usage

    @staticmethod
    def _gemini_type_enum(py_t: type) -> types.Type:
        if py_t is str:   return types.Type.STRING
        if py_t is int:   return types.Type.INTEGER
        if py_t is float: return types.Type.NUMBER
        if py_t is bool:  return types.Type.BOOLEAN
        return types.Type.STRING

    def _make_function_declaration(self, fn: Function) -> types.FunctionDeclaration:
        params_props: Dict[str, types.Schema] = {}
        for arg in fn.args:
            enum: Union[List[str], None] = None
            if arg.argtype is str and arg.enum is not None:
                enum = sorted(list(arg.enum))

            arg_schema = types.Schema(
                type=self._gemini_type_enum(arg.argtype),
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

    def _build_gemini_tools(self) -> list[types.Tool]:
        decls = [self._make_function_declaration(t) for t in self.agent_fn.uses]
        return [types.Tool(function_declarations=decls)] if decls else []

    def _append_thought_signatures(self, candidate: types.Candidate):
        """Append all thought signatures in the candidate as ThinkingBlockPart
        in the framework transcript (not the SDK replay transcript)."""
        if not candidate.content or not candidate.content.parts:
            return
        for part in candidate.content.parts:
            sig = part.thought_signature  # bytes | None
            if sig is None:
                continue
            if isinstance(sig, (bytes, bytearray)):
                sig_b64 = base64.b64encode(sig).decode("utf-8")
            else:
                sig_b64 = str(sig)
            self.transcript.append(ThinkingBlockPart(content="", signature=sig_b64))

    def _check_sanity(self, content: types.Content):
        # Ensure empty `thought` text.
        # For Gemini, thoughts are currently hidden. Only thought signatures are used for replay.
        # It would be very ambiguous if we somehow replay partial thoughts, or api behavior changes.
        # This is a sanity check to eliminate any such uncertainty.
        parts = content.parts or []
        for p in parts:
            if p.thought is None:
                continue
            if isinstance(p.thought, str):
                assert p.thought.strip() == "", "Gemini thought text is supposed to be empty."
            if isinstance(p.thought, bool):
                if p.thought:
                    assert p.text is None or p.text.strip() == "", "Gemini thought text is supposed to be empty."

    def _collect_function_calls(self, candidate: types.Candidate) -> list[types.FunctionCall]:
        if not candidate.content:
            return []
        parts = candidate.content.parts or []
        return [p.function_call for p in parts if p.function_call is not None]

    def _extract_text(self, candidate: types.Candidate) -> str:
        if not candidate.content:
            return ""
        parts = candidate.content.parts or []
        chunks = [p.text for p in parts if p.text is not None]
        return "\n".join([t for t in chunks if isinstance(t, str) and t.strip()]).strip()

    def _new_tool_use_id(self, tool_name: str) -> str:
        self._tool_call_counter += 1
        return f"gemini-{self.id}-{self._tool_call_counter}-{tool_name}"

    def run(self) -> None:
        tools = self._build_gemini_tools()
        config = types.GenerateContentConfig(
            system_instruction=self.agent_fn.system_prompt or "",
            tools=tools,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode=types.FunctionCallingConfigMode.AUTO
                )
            ),
            # We should only use manual function call loop, never auto function calling (incompatible
            # with our transcripts, event posting, exceptions model, etc).
            # Refer to: https://googleapis.github.io/python-genai/#function-calling
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            thinking_config=types.ThinkingConfig(
                thinking_budget=32768,
                # Disable Thought Summaries always: they are not useful, and we don't want to
                # accidentally allow them to be included as past thinking content ever.
                include_thoughts=False,
            ),
            max_output_tokens=64000,
        )

        # Substitute inputs into the templated user prompt.
        user_text = self.build_user_text()
        self.transcript.append(UserTextPart(text=user_text))
        contents: types.ContentListUnionDict = [
            types.Content(role="user", parts=[types.Part.from_text(text=user_text)])
        ]

        for _ in range(MAX_STEPS):
            # TODO: Add retry/backoff for transient Google Generative AI client errors.
            resp = self.client.models.generate_content(
                model=ModelNames[Provider.Gemini],
                contents=contents,
                config=config,
            )
            assert resp.usage_metadata is not None
            self._accumulate_usage(resp.usage_metadata)
            if not resp.candidates:
                raise RuntimeError("Gemini returned no candidates.")
            candidate = resp.candidates[0]

            # Record thought signatures (never summaries) in framework-type transcript.
            self._append_thought_signatures(candidate)

            # Always append sanitized model content (keeps history complete) for replay.
            assert candidate.content
            self._check_sanity(candidate.content)
            contents.append(candidate.content)

            # Gather function calls requested.
            calls: List[types.FunctionCall] = self._collect_function_calls(candidate)

            # No function calls → finalize with assistant text.
            if not calls:
                final_text: str = self._extract_text(candidate)
                self.transcript.append(ModelTextPart(text=final_text))
                self.ctx.post_success(final_text)
                return

            # Execute requested tools in parallel and
            # aggregate all function responses into one tool message.
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

                self.transcript.append(
                    ToolUsePart(tool_use_id=tool_use_id, tool_name=name, args=tool_args)
                )

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

                if invoke_ex:
                    out_text = AgentNode.stringify_exception(invoke_ex)
                    is_error = True
                    response["error"] = out_text
                else:
                    assert child
                    try:
                        # This will re-raise any exception that happened inside the tool function.
                        result: Any = child.result()
                        out_text = "" if result is None else str(result)
                        is_error = False
                        response["output"] = out_text
                    except AgentException as ex:
                        # Special case where agent decided to RaiseException. Record and
                        # finish processing the rest of the batch before surfacing.
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

                # Transcript result in gemini sdk types.
                result_parts.append(types.Part(
                    function_response=types.FunctionResponse(
                        id=tool_use_id,
                        name=fc.name,
                        response=response,
                    )
                ))

            # To single aggregated tool results message.
            if pending_agent_ex:
                self.ctx.post_exception(pending_agent_ex)
                return
            contents.append(types.Content(role="tool", parts=result_parts))

        raise RuntimeError("Gemini tool loop exceeded MAX_STEPS without producing a final answer.")

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
