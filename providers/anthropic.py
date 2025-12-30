from types import SimpleNamespace, MappingProxyType
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Union, cast
import copy
from multiprocessing.synchronize import Event
import time
import random
from overrides import override

from ..core import (
    Node, RunContext, Function, AgentNode, AgentException, ModelProviderException,
    UserTextPart, ModelTextPart, ThinkingBlockPart, ToolUsePart, ToolResultPart,
    TokenUsage,
)
from . import ModelNames, Provider


import anthropic
import httpx
from anthropic.types import (
    Message, MessageParam,
    Usage,
    TextBlock, TextBlockParam,
    ThinkingBlock, ThinkingBlockParam,
    RedactedThinkingBlock, RedactedThinkingBlockParam,
    ToolUseBlock, ToolUseBlockParam,
    ToolResultBlockParam,
    ToolParam, ToolUnionParam,
    CacheControlEphemeralParam,
    ToolChoiceAutoParam, ThinkingConfigEnabledParam,

)
from anthropic.types.tool_param import InputSchemaTyped

"""
## Misc Research Notes (applicable to Claude 4 models)

"Tool" = Anthropic LLM invoking our framework concept of `Function`.

* Extended Thinking Interleaved with Tool Use:
    * This is the only mode really compatible with our agentic framework since we want
      long-running agentic tasks with continuous reasoning.
    * In this mode, Claude is allowed to emit new thinking blocks after tool results, potentially
      leading to another tool_use in the next assistant turn—something it won't do in the
      non-interleaved mode. E.g.:
        * thinking → tool_use(s) → (user sends tool_result(s)) → thinking → text
        * thinking → tool_use(s) → (user sends tool_result(s)) → thinking → tool_use(s) →
          (user sends tool_result(s)) → thinking → text
        * etc
    * Ref: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#example-passing-thinking-blocks-with-tool-results
    * Ref: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#tool-use-with-interleaved-thinking
    * To simplify caching and api usage, always request with the full history of thinking blocks,
      tool use request, tool use response, etc, and for the prompt caching policy we select for
      the agent, always put the `cache_control` on every request latest msg.
    * While tool results appear as user messages in the API structure, they're part of a
      continuous reasoning flow. Preserving thinking blocks maintains this conceptual flow across
      multiple API calls.
    * With interleaved thinking, Claude can:
        * Reason about the results of a tool call before deciding what to do next
        * Chain multiple tool calls with reasoning steps in between
        * Make more nuanced decisions based on intermediate results
    * When streaming with tool use, we should wait to receive the full tool use inputs before
      invoking tool, since our framework is not real-time and does not support streaming
      arguments.
        * Ref: https://docs.anthropic.com/en/docs/build-with-claude/streaming#streaming-request-with-tool-use
    * Our empirical experiments:
        * Conversation shape: One initial user text prompt starts the session. After that, every
          "user" request is just a `tool_result` only (no user `text` type part), or multiple
          `tool_result` if the assistant requested parallel tool use. And every non-final
          `assistant` response contains reasoning (or redacted-reasoning) block and signature,
          [optional `text` block seen sometimes], `tool_use` block (parallel or single). Final
          `assistant` response contains reasoning (or redacted-reasoning) block and signature,
          followed by final `text` block.
        * Signatures will be included from the assistant on reasoning or redacted-reasoning
          blocks. These must be included when sending the conversation history back in user
          requests.
        * Model will decrypt redacted reasoning blocks when they are sent back (with signatures).
          It is only the user that cannot see them.
        * Our replay policy: on every user request (tool use follow-ups), you always replay the
          full conversation history (all elements) since the initial user text prompt, in exact
          sequence sent and received, unmodified.
        * Continuous reasoning: When this is done properly (Never alter, reorder, trim, or
          re-wrap any assistant block. You only append new tool_result blocks to transcript and
          then send the request), and the tool-reasoning interleaving beta header is enabled,
          and tool choice "auto" is used, you will attain **full fluent context since the last
          user text prompt**: the model's follow-up reasoning step has direct access to the
          entire chain of prior reasoning and actions in its context window. Thus, you get 
          **reasoning continuity**: the model will produce new thinking that references earlier
          thinking you replayed. From the model's perspective, this behaves like **one
          continuous, ever-expanding assistant turn across many tool cycles incorporating
          continuous reasoning**.
        * In experiments, we observed direct evidence of the assistant referring to an early
          thinking block dozens of tool-cycles apart in another thinking block toward the end.

* Tools that Claude was optimized to use during training:
    * text editor
        * Ref: https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/text-editor-tool
        * Specified without reference implementation.
        * We implemented very robustly in `func_lib/text_editor.py`.
    * bash tool
        * Ref: https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/bash-tool
        * Specified without reference implementation.   
        * We implemented in `func_lib/bash.py`.

* Strict structured outputs
    * should be achieved via tools (their arg schema is the output schema)
"""

# Enables: Extended Thinking Interleaved with Tool Use.
# Since we want agentic task completion end to end, we must always add the
# header on each of our requests.
INTERLEAVED_BETA = {"anthropic-beta": "interleaved-thinking-2025-05-14"}
# "With interleaved thinking, the budget_tokens can exceed the max_tokens parameter,
# as it represents the total budget across all thinking blocks within one assistant turn."
MAX_TOKENS = 64_000
THINKING_CFG = ThinkingConfigEnabledParam(type="enabled", budget_tokens=80_000)
# 5-minute TTL prompt cache watermark on the latest user request msg (initial + after tool_result).
CACHE_TTL = "5m"
# Prevent agent loop runaway. Max tool call + response cycles before giving up.
MAX_STEPS = 256


class AnthropicAgentNode(AgentNode):
    """
    Anthropic agent driver using strongly-typed SDK.

    - Messages: history kept as List[MessageParam] with *Param content blocks.
    - Replay: response Blocks -> corresponding *Param blocks.
    - Tool calls: executed in parallel here (impl detail), results batched into
      a single user message session continuation request.
    - Cache watermark: applied just-in-time to the latest user message before each request.
    """

    def __init__(
        self,
        ctx: RunContext,
        id: int,
        fn: Function,
        inputs: Dict[str, Any],
        parent: Optional[Node],
        cancel_event: Optional[Event],
        client_factory: Callable[[], Any],
    ):
        super().__init__(ctx, id, fn, inputs, parent, cancel_event, client_factory)
        client: Any = client_factory()
        if not isinstance(client, anthropic.Anthropic):
            raise TypeError(
                "AnthropicAgentNode expected client_factory to return anthropic.Anthropic"
            )
        self.client: anthropic.Anthropic = client
        self.model = ModelNames[Provider.Anthropic]
        self._history: List[MessageParam] = []   # Typed conversation history we replay every turn
        self._tools: List[ToolUnionParam] = self._build_tool_params()
        self._token_usage = TokenUsage()

        # Seed initial user message (cache watermark will be added pre-send).
        # Substitute inputs into the templated user prompt.
        user_text = self.build_user_text()
        self.transcript.append(UserTextPart(text=user_text))
        self._history.append(
            cast(MessageParam, {"role": "user", "content": [TextBlockParam(text=user_text, type="text")]})
        )

    @property
    @override
    def token_usage(self) -> TokenUsage:
        return self._token_usage

    def run(self) -> None:
        # Agent loop.
        for _ in range(MAX_STEPS):
            if self.is_cancel_requested():
                self.ctx.post_cancel()
                self.client.close()
                return

            # Apply watermark to *latest* user message only (just-in-time).
            # The longest prefix cache partial hit wins. We get partial credit for
            # a cached prefix and the rest are new tokens which we incrementally pay to cache.
            msgs: List[MessageParam] = self._messages_with_latest_cache_ttl(self._history, CACHE_TTL)

            # One thinking-tool turn.
            resp: Message
            max_attempts = 8
            base_delay = 3  # seconds
            attempt = 1
            # Retry logic.
            while True:
                try:
                    with self.client.messages.stream(
                        model=self.model,
                        system=self.agent_fn.system_prompt,
                        messages=msgs,
                        tools=self._tools,
                        tool_choice=ToolChoiceAutoParam(type="auto"),
                        max_tokens=MAX_TOKENS,
                        thinking=THINKING_CFG,
                        extra_headers=INTERLEAVED_BETA,
                    ) as stream:
                        resp = stream.get_final_message()
                        break

                except (
                    anthropic.APIConnectionError,
                    anthropic.RateLimitError,
                    anthropic.APIStatusError,
                    httpx.TransportError,
                    httpx.HTTPStatusError,
                ) as e:
                    is_retriable: bool = False
                    is_connection: bool = False

                    # Retry on known transient conditions: connection issues, rate limits, or 5xx responses.
                    if isinstance(e, anthropic.RateLimitError):
                        is_retriable = True
                    if isinstance(e, anthropic.APIStatusError):
                        # WORKAROUND: some HTTP-200 with late overloaded SSE are not `class OverloadedError(APIStatusError)`
                        # but the SDK is still flagging them as APIStatusError.
                        if e.status_code in (408, 409, 429) or e.status_code >= 500 or "overloaded" in e.message.lower():
                            is_retriable = True
                    if isinstance(e, httpx.HTTPStatusError):
                        status_code = e.response.status_code
                        if status_code in (408, 409, 429) or status_code >= 500:
                            is_retriable = True
                    if isinstance(e, httpx.TransportError) and not isinstance(e, httpx.ProtocolError):
                        is_retriable = True
                        is_connection = True
                    if isinstance(e, (anthropic.APIConnectionError, httpx.RemoteProtocolError)):
                        is_retriable = True
                        is_connection = True
                    
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

                    # Sleep unless canceled.
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
            self._accumulate_usage(resp.usage)

            # Map response ContentBlocks -> *Param blocks for strict replay,
            # while also projecting into our framework transcript.
            assistant_params: List[Union[
                TextBlockParam, ToolUseBlockParam, ThinkingBlockParam, RedactedThinkingBlockParam
            ]] = []
            tool_uses: List[ToolUseBlock] = []
            final_text_chunks: List[str] = []

            for blk in resp.content:
                if isinstance(blk, ThinkingBlock):
                    # Record in our own framework-type msg for transcript.
                    self.transcript.append(
                        ThinkingBlockPart(content=blk.thinking, signature=blk.signature, redacted=False)
                    )
                    self.ctx.post_transcript_update()
                    # Add sdk-type msg for session replay.
                    assistant_params.append(
                        ThinkingBlockParam(signature=blk.signature, thinking=blk.thinking, type=blk.type)
                    )

                elif isinstance(blk, RedactedThinkingBlock):
                    self.transcript.append(
                        ThinkingBlockPart(content=blk.data, signature="", redacted=True)
                    )
                    self.ctx.post_transcript_update()
                    assistant_params.append(
                        RedactedThinkingBlockParam(data=blk.data, type=blk.type)
                    )

                elif isinstance(blk, ToolUseBlock):
                    args = cast(Dict[str, Any], blk.input or {})
                    # Make args mapping immutable for transcript snapshot
                    args_ro = MappingProxyType(copy.deepcopy(args))
                    self.transcript.append(
                        ToolUsePart(tool_use_id=blk.id, tool_name=blk.name, args=args_ro)
                    )
                    self.ctx.post_transcript_update()
                    tool_uses.append(blk)
                    assistant_params.append(
                        ToolUseBlockParam(id=blk.id, name=blk.name, input=args, type="tool_use")
                    )

                elif isinstance(blk, TextBlock):
                    if blk.text and blk.text.strip():
                        final_text_chunks.append(blk.text)
                        # Non-final interleaved text should also be replayed
                        assistant_params.append(TextBlockParam(text=blk.text, type="text"))

            # Append assistant turn to history for strict session replay.
            self._history.append(
                MessageParam(role="assistant", content=assistant_params)
            )

            # If no tool uses -> finalize with the accumulated text.
            if not tool_uses:
                # Assert expectation that the protocol is the way we think:
                # model is finishing the turn with final text completion.
                if resp.stop_reason != "end_turn":
                    raise ModelProviderException(
                        message=f"Expected stop_reason 'end_turn' for final text, got "
                                f"'{resp.stop_reason!r}'; debug protocol adherence.",
                        provider=type(self),
                        agent_name=self.agent_fn.name,
                        node_id=self.id,
                    )
                
                final_text = "\n".join(t for t in final_text_chunks if t).strip()
                self.transcript.append(ModelTextPart(text=final_text))
                self.ctx.post_transcript_update()
                self.ctx.post_success(final_text)
                self.client.close()
                return

            # Make sure we check for cancellation right before commencing possibly
            # lengthy sub-tasks.
            if self.is_cancel_requested():
                self.ctx.post_cancel()
                self.client.close()
                return

            # Assert expectation: model requested tool use in this turn.
            if resp.stop_reason != "tool_use":
                raise ModelProviderException(
                    message=f"Expected stop_reason 'tool_use' before tool execution, got "
                            f"'{resp.stop_reason!r}'; debug protocol adherence.",
                    provider=type(self),
                    agent_name=self.agent_fn.name,
                    node_id=self.id,
                )

            # Execute requested tools in parallel and aggregate tool_result blocks.
            # Even if some tool invocations fail early, continue processing others.
            result_blocks: List[ToolResultBlockParam] = []
            children: List[Optional[Node]] = []                 # Index to match `tool_uses` 1:1.
            invoke_exceptions: List[Optional[Exception]] = []   # Index to match `tool_uses` 1:1.
            for tu in tool_uses:
                args: Dict[str, Any] = cast(Dict[str, Any], tu.input or {})

                try:
                    children.append(self.invoke_tool_function(tu.name, args))
                    invoke_exceptions.append(None)
                except Exception as ex:
                    children.append(None)
                    invoke_exceptions.append(ex)

            pending_agent_ex: Optional[AgentException] = None

            # WaitAll + transcribe results.
            for tu, child, invoke_ex in zip(tool_uses, children, invoke_exceptions):
                out_text: str
                is_error: bool

                # Did the invocation itself fail-fast? (e.g. bad args)
                if invoke_ex:
                    out_text = AgentNode.stringify_exception(invoke_ex)
                    is_error = True

                # Check if the child Node is success / error.
                else:
                    assert child
                    try:
                        # This will re-raise any exception that happened inside the tool function.
                        result: Any = child.result()
                        out_text = "" if result is None else str(result)
                        is_error = False
                    except AgentException as ex:
                        # Agent decided to raise an exception. Keep processing the rest of the batch
                        # per spec before propagating the exception outside the loop.
                        pending_agent_ex = ex
                        continue
                    except Exception as ex:
                        out_text = AgentNode.stringify_exception(ex)
                        is_error = True

                self.transcript.append(
                    ToolResultPart(
                        tool_use_id=tu.id,
                        tool_name=tu.name,
                        outputs=out_text,
                        is_error=is_error,
                    )
                )
                self.ctx.post_transcript_update()

                result_blocks.append(
                    ToolResultBlockParam(
                        tool_use_id=tu.id,
                        type="tool_result",
                        content=[TextBlockParam(text=out_text, type="text")],
                        is_error=is_error,
                    )
                )

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
            
            # Per protocol: next user message contains only tool_result blocks
            self._history.append(cast(MessageParam, {"role": "user", "content": result_blocks}))

        raise RuntimeError(f"Anthropic agent loop exceeded MAX_STEPS ({MAX_STEPS}) "
                           "without producing a final response.")

    def _accumulate_usage(self, usage: Usage) -> None:
        cache_read = usage.cache_read_input_tokens or 0
        cache_write = usage.cache_creation_input_tokens or 0
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens

        token_usage = self._token_usage
        token_usage.input_tokens_cache_read += cache_read
        token_usage.input_tokens_cache_write = (token_usage.input_tokens_cache_write or 0) + cache_write
        token_usage.input_tokens_regular += input_tokens
        token_usage.input_tokens_total += input_tokens + cache_write + cache_read
        token_usage.output_tokens_total += output_tokens
        token_usage.context_window_in = input_tokens + cache_write + cache_read
        token_usage.context_window_out = output_tokens

    def _build_tool_params(self) -> List[ToolUnionParam]:
        funcs: Sequence[Function] = self.agent_fn.uses
        tools: List[ToolUnionParam] = []
        for f in funcs:
            # Build args spec.
            props: Dict[str, Dict[str, Any]] = {}
            required: List[str] = []
            for arg in f.args:
                arg_schema: Dict[str, Any] = {
                    "type": self.json_type_for_arg(arg.argtype),
                    "description": arg.desc,
                }
                if arg.argtype is str and arg.enum is not None:
                    arg_schema["enum"] = sorted(list(arg.enum))
                props[arg.name] = arg_schema

                if not arg.optional:
                    required.append(arg.name)

            input_schema = cast(InputSchemaTyped, {
                "type": "object",
                "properties": props,
                "required": required,
            })

            tools.append(
                ToolParam(name=f.name, description=f.desc, input_schema=input_schema)
            )

        return tools

    @staticmethod
    def _messages_with_latest_cache_ttl(msgs: List[MessageParam], ttl: Literal['5m', '1h']) -> List[MessageParam]:
        """
        Deep-copy and attach CacheControlEphemeralParam(ttl) only to the last element of the
        latest user message's (text or function response) content list.
        Leave earlier content blocks untouched.
        """
        out: List[MessageParam] = copy.deepcopy(msgs)

        # Find last user message
        idx = None
        for i in range(len(out) - 1, -1, -1):
            if out[i].get("role") == "user":
                idx = i
                break
        if idx is None:
            return out

        cc: CacheControlEphemeralParam
        if ttl == '5m':
            # Omit the `ttl` argument (default) because older clients may not support it.
            cc = CacheControlEphemeralParam(type="ephemeral")
        else:
            cc = CacheControlEphemeralParam(type="ephemeral", ttl=ttl)
        orig_blocks = cast(List[Any], out[idx]["content"])
        assert orig_blocks, "User message content is empty"
        new_blocks: List[Any] = []

        to_obj = lambda x: x if not isinstance(x, dict) else SimpleNamespace(**x)

        last_block_idx = len(orig_blocks) - 1
        for i, blk in enumerate(orig_blocks):
            if i != last_block_idx:
                new_blocks.append(blk)
                continue

            b = to_obj(blk)
            if getattr(b, "type", None) == "text":
                new_blocks.append(TextBlockParam(text=b.text, type="text", cache_control=cc))
            elif getattr(b, "type", None) == "tool_result":
                new_blocks.append(
                    ToolResultBlockParam(
                        tool_use_id=b.tool_use_id,
                        type="tool_result",
                        content=b.content,
                        is_error=b.is_error,
                        cache_control=cc,
                    )
                )
            else:
                # Unknown block type; leave untouched (no cache control).
                new_blocks.append(blk)

        out[idx] = cast(MessageParam, {"role": "user", "content": new_blocks})
        return out

    @staticmethod
    def json_type_for_arg(py_t: type) -> str:
        if py_t is str:   return "string"
        if py_t is int:   return "integer"
        if py_t is float: return "number"
        if py_t is bool:  return "boolean"
        return "string"  # fallback
