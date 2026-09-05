from types import MappingProxyType
from typing import Any, Callable, Dict, List, Optional, Sequence, cast
import copy
import json
import random
import time
from threading import Event
from overrides import override

from ..core import (
    Node, RunContext, Function, AgentNode, AgentException, ModelProviderException,
    UserTextPart, ModelTextPart, ThinkingBlockPart, ToolUsePart, ToolResultPart,
    TokenUsage,
)
from . import ModelNames, Provider

import httpx
import openai
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessageParam


"""
## GitHub Copilot provider

Drives a model served through the GitHub Copilot API surface
(`https://api.githubcopilot.com`), which is OpenAI *chat/completions* compatible.
This lets us consume the *same* GitHub Copilot subscription entitlement used by
the VS Code chat experience, with billing routed through GitHub Copilot rather
than a direct Anthropic account.

Notable differences vs. the Anthropic provider:
* Protocol is OpenAI-style `messages` + `tool_calls`, not Anthropic content blocks.
* The Copilot proxy does not expose Anthropic's extended-thinking / 1M-context
  beta knobs through this OpenAI-compatible surface. We request the strongest
  configuration the surface accepts (high `max_tokens`, the Claude Opus model id)
  and degrade gracefully. If the proxy returns provider-native `reasoning`
  content, we capture it into the transcript as a thinking block.
* Auth uses a short-lived Copilot bearer token minted from a GitHub OAuth token.
  Token refresh is handled by the client factory: on 401 we rebuild the client,
  which mints a fresh bearer.
"""

MAX_TOKENS = 64_000
# Prevent agent loop runaway. Max tool call + response cycles before giving up.
MAX_STEPS = 256
# Chat-completion models (incl. Claude via the Copilot surface) sometimes emit a
# bare narration turn ("Let me validate that next.") with NO tool call and a
# "stop" finish, expecting a chat partner to say "continue". In an autonomous
# loop that would end the agent prematurely. When that happens we nudge the model
# to either call the next tool or produce its final answer, up to this many
# CONSECUTIVE times (reset whenever the model actually calls a tool).
MAX_CONTINUE_NUDGES = 3
CONTINUE_NUDGE_TEXT = (
    "You ended your turn without calling a tool. If your task is not yet complete, "
    "call the appropriate tool NOW (do not narrate what you will do — actually call "
    "it). If your task IS complete, reply with your final answer only."
)


class CopilotAgentNode(AgentNode):
    """
    GitHub Copilot agent driver (OpenAI chat/completions compatible).

    - Messages: history kept as List[ChatCompletionMessageParam] (plain dicts).
    - Replay: full history is resent every turn (assistant tool_calls + tool results).
    - Tool calls: executed in parallel here; results appended as role="tool" messages.
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
        tool_use_id: Optional[str] = None,
    ):
        super().__init__(ctx, id, fn, inputs, parent, cancel_event, client_factory, tool_use_id)
        client: Any = client_factory()
        if not isinstance(client, OpenAI):
            raise TypeError(
                "CopilotAgentNode expected client_factory to return openai.OpenAI"
            )
        self.client: OpenAI = client
        self.model = ModelNames[Provider.Copilot]
        self._history: List[ChatCompletionMessageParam] = []
        self._tools: List[Dict[str, Any]] = self._build_tool_params()
        self._token_usage = TokenUsage()
        # Count consecutive bare-text (no tool call) turns to bound auto-continue.
        self._consecutive_nudges = 0

        # Seed system + initial user message.
        self._history.append(
            cast(ChatCompletionMessageParam,
                 {"role": "system", "content": self.agent_fn.system_prompt})
        )
        user_text = self.build_user_text()
        self.transcript.append(UserTextPart(text=user_text))
        self._history.append(
            cast(ChatCompletionMessageParam, {"role": "user", "content": user_text})
        )

    @property
    @override
    def token_usage(self) -> TokenUsage:
        return self._token_usage

    @property
    @override
    def provider(self) -> Provider:
        return Provider.Copilot

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
        for _ in range(MAX_STEPS):
            if self.is_cancel_requested():
                self.ctx.post_cancel()
                self._close_client()
                return

            resp: ChatCompletion
            max_attempts = 8
            base_delay = 3  # seconds
            attempt = 1
            while True:
                try:
                    resp = self.client.chat.completions.create(
                        model=self.model,
                        messages=self._history,
                        tools=cast(Any, self._tools) if self._tools else openai.NOT_GIVEN,
                        tool_choice="auto" if self._tools else openai.NOT_GIVEN,
                        max_tokens=MAX_TOKENS,
                    )
                    break

                except (
                    openai.APIConnectionError,
                    openai.RateLimitError,
                    openai.APIStatusError,
                    openai.AuthenticationError,
                    httpx.TransportError,
                    httpx.HTTPStatusError,
                ) as e:
                    is_retriable = False
                    is_connection = False

                    if isinstance(e, openai.RateLimitError):
                        is_retriable = True
                    if isinstance(e, openai.AuthenticationError):
                        # Bearer likely expired; rebuilding the client mints a fresh one.
                        is_retriable = True
                        is_connection = True
                    if isinstance(e, openai.APIStatusError):
                        code = e.status_code
                        if code in (408, 409, 429) or code >= 500:
                            is_retriable = True
                    if isinstance(e, httpx.HTTPStatusError):
                        code = e.response.status_code
                        if code in (408, 409, 429) or code >= 500:
                            is_retriable = True
                    if isinstance(e, httpx.TransportError) and not isinstance(e, httpx.ProtocolError):
                        is_retriable = True
                        is_connection = True
                    if isinstance(e, (openai.APIConnectionError, httpx.RemoteProtocolError)):
                        is_retriable = True
                        is_connection = True

                    if not is_retriable or attempt >= max_attempts:
                        self._close_client()
                        raise

                    if self.is_cancel_requested():
                        self.ctx.post_cancel()
                        self._close_client()
                        return

                    delay = base_delay * (2 ** (attempt - 1))
                    delay = min(delay, 30)
                    delay += random.uniform(0, delay * 0.1)

                    if self.cancel_event:
                        if self.cancel_event.wait(delay):
                            self.ctx.post_cancel()
                            self._close_client()
                            return
                    else:
                        time.sleep(delay)

                    if is_connection:
                        self._close_client()
                        self.client = self.client_factory()

                    attempt += 1
                    continue

            if self.is_cancel_requested():
                self.ctx.post_cancel()
                self._close_client()
                return

            if resp.usage is not None:
                self._accumulate_usage(resp.usage)

            if not resp.choices:
                self._close_client()
                raise ModelProviderException(
                    message="Copilot response contained no choices.",
                    provider=type(self),
                    agent_name=self.agent_fn.name,
                    node_id=self.id,
                )

            choice = resp.choices[0]
            msg = choice.message

            # Capture provider-native reasoning if the proxy surfaces it.
            reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
            if reasoning:
                self.transcript.append(
                    ThinkingBlockPart(content=str(reasoning), signature="", redacted=False)
                )
                self.ctx.post_transcript_update()

            tool_calls = list(msg.tool_calls or [])

            # Build the assistant message for replay.
            assistant_msg: Dict[str, Any] = {"role": "assistant"}
            assistant_msg["content"] = msg.content or None
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments or "{}",
                        },
                    }
                    for tc in tool_calls
                ]
            self._history.append(cast(ChatCompletionMessageParam, assistant_msg))

            # Transcribe any tool-use requests.
            for tc in tool_calls:
                parsed_args = self._parse_tool_args(tc.function.arguments)
                args_ro = MappingProxyType(copy.deepcopy(parsed_args))
                self.transcript.append(
                    ToolUsePart(tool_use_id=tc.id, tool_name=tc.function.name, args=args_ro)
                )
                self.ctx.post_transcript_update()

            # No tool calls -> either nudge to continue, or finalize.
            if not tool_calls:
                final_text = (msg.content or "").strip()
                # If the model stopped mid-task (bare narration, no tool call),
                # nudge it to continue up to a bounded number of times before
                # accepting the text as the final answer.
                if self._tools and self._consecutive_nudges < MAX_CONTINUE_NUDGES:
                    self._consecutive_nudges += 1
                    self._history.append(
                        cast(ChatCompletionMessageParam,
                             {"role": "user", "content": CONTINUE_NUDGE_TEXT})
                    )
                    continue

                self.transcript.append(ModelTextPart(text=final_text))
                self.ctx.post_transcript_update()
                self.ctx.post_success(final_text)
                self._close_client()
                return

            # The model called at least one tool; reset the bare-text nudge budget.
            self._consecutive_nudges = 0

            if self.is_cancel_requested():
                self.ctx.post_cancel()
                self._close_client()
                return

            # Execute requested tools (in parallel via invoke) and gather results.
            children: List[Optional[Node]] = []
            invoke_exceptions: List[Optional[Exception]] = []
            parsed_args_list: List[Dict[str, Any]] = []
            for tc in tool_calls:
                parsed_args = self._parse_tool_args(tc.function.arguments)
                parsed_args_list.append(parsed_args)
                try:
                    children.append(self.invoke_tool_function(tc.function.name, parsed_args, tc.id))
                    invoke_exceptions.append(None)
                except Exception as ex:
                    children.append(None)
                    invoke_exceptions.append(ex)

            pending_agent_ex: Optional[AgentException] = None

            for tc, child, invoke_ex in zip(tool_calls, children, invoke_exceptions):
                out_text: str
                is_error: bool

                if invoke_ex:
                    out_text = AgentNode.stringify_exception(invoke_ex)
                    is_error = True
                else:
                    assert child
                    try:
                        result: Any = child.result()
                        out_text = "" if result is None else str(result)
                        is_error = False
                    except AgentException as ex:
                        pending_agent_ex = ex
                        continue
                    except Exception as ex:
                        out_text = AgentNode.stringify_exception(ex)
                        is_error = True

                self.transcript.append(
                    ToolResultPart(
                        tool_use_id=tc.id,
                        tool_name=tc.function.name,
                        outputs=out_text,
                        is_error=is_error,
                    )
                )
                self.ctx.post_transcript_update()

                self._history.append(
                    cast(ChatCompletionMessageParam, {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": out_text,
                    })
                )

            if pending_agent_ex:
                self.ctx.post_exception(pending_agent_ex)
                self._close_client()
                return
            if self.is_cancel_requested():
                self.ctx.post_cancel()
                self._close_client()
                return

        self._close_client()
        raise RuntimeError(f"Copilot agent loop exceeded MAX_STEPS ({MAX_STEPS}) "
                           "without producing a final response.")

    @staticmethod
    def _parse_tool_args(raw: Optional[str]) -> Dict[str, Any]:
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _accumulate_usage(self, usage: Any) -> None:
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        cached = 0
        details = getattr(usage, "prompt_tokens_details", None)
        if details is not None:
            cached = getattr(details, "cached_tokens", 0) or 0
        regular = prompt_tokens - cached

        tu = self._token_usage
        tu.input_tokens_cache_read += cached
        tu.input_tokens_regular += regular
        tu.input_tokens_total += prompt_tokens
        tu.output_tokens_total += completion_tokens
        tu.context_window_in = prompt_tokens
        tu.context_window_out = completion_tokens

    def _build_tool_params(self) -> List[Dict[str, Any]]:
        funcs: Sequence[Function] = self.agent_fn.uses
        tools: List[Dict[str, Any]] = []
        for f in funcs:
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

            tools.append({
                "type": "function",
                "function": {
                    "name": f.name,
                    "description": f.desc,
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": required,
                    },
                },
            })
        return tools

    @staticmethod
    def json_type_for_arg(py_t: type) -> str:
        if py_t is str:   return "string"
        if py_t is int:   return "integer"
        if py_t is float: return "number"
        if py_t is bool:  return "boolean"
        return "string"
