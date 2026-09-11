import copy
import json
import math
import random
import time
from dataclasses import dataclass
from threading import Event
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast
from overrides import override
from typing_extensions import Final, Literal, Required, TypedDict

import openai
from openai._constants import MAX_RETRY_AFTER_DELAY
from openai.types.responses import (
    EasyInputMessageParam, FunctionToolParam, Response, ResponseError,
    ResponseFunctionToolCall, ResponseInputItemParam, ResponseInputParam, ResponseInputTextParam,
    ResponseOutputMessage, ResponseOutputRefusal, ResponseOutputText, ResponseReasoningItem,
    ResponseUsage, ToolChoiceOptions,
)
from openai.types.responses.response_create_params import PromptCacheOptions
from openai.types.responses.response_function_tool_call import CallerDirect
from openai.types.responses.response_input_item_param import FunctionCallOutput
from openai.types.shared_params import FunctionParameters, Reasoning

from ..core import (
    AgentException, AgentNode, AllowedArgTypeUnion, CodeFunction, Function,
    ModelStatusPart, ModelProviderException, ModelTextPart, Node, RunContext,
    ThinkingBlockPart, TokenUsage, ToolResultPart, ToolUsePart, UserTextPart,
)
from ..func_lib import raise_exception, status_update
from . import ModelNames, Provider


MAX_TOKENS: Final[int] = 128_000
MAX_STEPS: Final[int] = 768

# SDK request types are TypedDicts; protocol values (including ResponseStatus)
# are Literal strings, not enums. Constructors provide static type checking.
# Keep all compatible reasoning available throughout the agent run.
REASONING_CONFIG: Final[Reasoning] = Reasoning(
    mode="standard", effort="max", context="all_turns",
)

# Implicit caching fits the append-only tool loop.
PROMPT_CACHE_OPTIONS: Final[PromptCacheOptions] = PromptCacheOptions(
    mode="implicit",
    # ttl="30m",  # SDK-documented default.
)

TOOL_CHOICE: Final[ToolChoiceOptions] = "auto"
MAX_ATTEMPTS: Final[int] = 8
BASE_RETRY_DELAY_SECONDS: Final[float] = 3.0
MAX_RETRY_DELAY_SECONDS: Final[float] = 30.0
TRANSIENT_RESPONSE_ERRORS: Final = frozenset({
    "server_error", "rate_limit_exceeded", "vector_store_timeout",
})


# OpenAI and Pydantic expose JSON Schema as a generic dict; these TypedDicts
# check the strict object/property subset we generate before the SDK cast,
# so that it's like using strong types.
JsonScalarType = Literal["string", "integer", "number", "boolean"]
JsonTypeName = Literal["string", "integer", "number", "boolean", "null"]
JsonTypeSpec = Union[JsonScalarType, List[JsonTypeName]]


class PropertySchema(TypedDict, total=False):
    type: Required[JsonTypeSpec]
    description: Required[str]
    enum: List[Optional[str]]


class ObjectParametersSchema(TypedDict):
    type: Literal["object"]
    properties: Dict[str, PropertySchema]
    required: List[str]
    additionalProperties: Literal[False]


@dataclass(frozen=True)
class PendingToolCall:
    """An SDK call plus parsed arguments and netflux's independent child ID."""

    item: ResponseFunctionToolCall
    args: Dict[str, Any]
    tool_use_id: str


@dataclass(frozen=True)
class ModelTextUpdate:
    """Text staged for publication after response validation, with phase boundaries."""

    text: str
    merge: bool


TranscriptUpdate = Union[ThinkingBlockPart, ToolUsePart, ModelStatusPart, ModelTextUpdate]


class OaiAgentNode(AgentNode):
    """AgentNode implementation using OpenAI's modern Responses API.

    Replays every native output item in order with ``store=False`` and truncation
    disabled. Encrypted reasoning and standard/max/all_turns configuration are
    required; tools must be synchronous, direct, and non-namespaced. Unexpected
    protocols fail before tool dispatch. ``transcript`` holds the normalized
    netflux view; ``history`` retains the complete OpenAI conversation.
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
    ) -> None:
        super().__init__(
            ctx,
            id,
            fn,
            inputs,
            parent,
            cancel_event,
            client_factory,
            tool_use_id,
        )

        self.model: str = ModelNames[Provider.OpenAI]

        self.history: ResponseInputParam = []
        self.tool_call_counter = 0
        self.tools = [self.make_function_tool(fn) for fn in self.agent_fn.uses]
        self.usage = TokenUsage()

        user_text = self.build_user_text()
        self.transcript.append(UserTextPart(text=user_text))

        self.history.append(EasyInputMessageParam(
            type="message", role="user",
            content=[ResponseInputTextParam(type="input_text", text=user_text)],
        ))

        # Create network resources only after all pure provider setup succeeds.
        self.client: openai.OpenAI = self.new_client()
        self.client_closed = False

    @property
    @override
    def token_usage(self) -> TokenUsage:
        return self.usage

    @property
    @override
    def provider(self) -> Provider:
        return Provider.OpenAI

    def append_model_text(self, text: str, *, merge: bool = True) -> None:
        """Coalesce adjacent nonblank text unless a provider phase boundary intervenes."""
        if not text.strip():
            return
        if merge and self.transcript and isinstance(self.transcript[-1], ModelTextPart):
            text = self.transcript[-1].text + "\n" + text
            # Published snapshots share frozen parts, so replace rather than mutate.
            self.transcript[-1] = ModelTextPart(text=text)
        else:
            self.transcript.append(ModelTextPart(text=text))
        self.ctx.post_transcript_update()

    def new_tool_use_id(self, tool_name: str) -> str:
        self.tool_call_counter += 1
        return f"openai-{self.id}-{self.tool_call_counter}-{tool_name}"

    def is_valid_status_update(self, tool_name: str, args: Dict[str, Any]) -> bool:
        if self.func_map.get(tool_name) is not status_update:
            return False
        try:
            status_update.validate_coerce_args(args)
        except ValueError:
            return False
        return True

    def new_client(self) -> openai.OpenAI:
        client: Any = self.client_factory()
        if not isinstance(client, openai.OpenAI):
            raise TypeError(
                "OaiAgentNode expected client_factory to return openai.OpenAI"
            )
        # Each node owns its factory client. Disable SDK retries while retaining
        # its configuration (with_options() drops strict response validation).
        client.max_retries = 0
        return client

    def close_client(self) -> None:
        if self.client_closed:
            return
        try:
            self.client.close()
        except Exception:
            # Provider cleanup must not mask the real terminal result/failure.
            pass
        finally:
            self.client_closed = True

    def run(self) -> None:
        try:
            self.run_agent_loop()
        except ModelProviderException as ex:
            # Already contextualized; do not wrap it again in AgentNode.run_wrapper.
            self.ctx.post_exception(ex)
        finally:
            self.close_client()

    def run_agent_loop(self) -> None:
        for _ in range(MAX_STEPS):
            response = self.create_response_with_retry()
            if response is None:
                # Cancellation was already posted; exceptions propagate instead.
                return

            pending_calls, final_text = self.consume_response_output(response)

            if not pending_calls:
                if not final_text:
                    raise self.provider_error(
                        "OpenAI completed without function calls or non-empty "
                        "final assistant text."
                    )
                self.ctx.post_success(final_text)
                return

            if self.is_cancel_requested():
                self.ctx.post_cancel()
                return

            function_outputs = self.execute_tool_batch(pending_calls)
            if function_outputs is None:
                # execute_tool_batch already posted an AgentException or cancellation.
                return

            # Append tool results after the complete preceding response.output.
            self.history.extend(function_outputs)

        raise self.provider_error(
            f"OpenAI agent loop exceeded MAX_STEPS ({MAX_STEPS}) "
            "without producing a final response."
        )

    def create_response_with_retry(self) -> Optional[Response]:
        attempt = 1
        while True:
            if self.is_cancel_requested():
                self.ctx.post_cancel()
                return None
            try:
                response = self.client.responses.create(
                    model=self.model,
                    instructions=self.system_prompt(),
                    input=self.history,
                    tools=self.tools,
                    tool_choice=TOOL_CHOICE,
                    parallel_tool_calls=True,
                    reasoning=REASONING_CONFIG,
                    prompt_cache_options=PROMPT_CACHE_OPTIONS,
                    store=False,
                    truncation="disabled",
                    max_output_tokens=MAX_TOKENS,
                    background=False,
                    stream=False,
                    # Stateless responses include encrypted reasoning by default.
                )
            except (openai.APIConnectionError, openai.APIStatusError) as ex:
                delay = self.retry_delay(ex, attempt)
                if delay is None or attempt == MAX_ATTEMPTS:
                    raise
                should_rebuild_client = isinstance(ex, openai.APIConnectionError)
            else:
                # Failed/incomplete responses also consume tokens. Never replay
                # their partial output or dispatch their tools when retrying.
                if response.usage is not None:
                    self.accumulate_usage(response.usage)
                delay = (
                    self.retry_delay(response.error, attempt)
                    if response.status == "failed" and response.error is not None else None
                )
                if delay is None or attempt == MAX_ATTEMPTS:
                    self.assert_response_contract(response)
                    return response
                should_rebuild_client = False

            if self.is_cancel_requested():
                self.ctx.post_cancel()
                return None

            if self.cancel_event is not None:
                if self.cancel_event.wait(delay):
                    self.ctx.post_cancel()
                    return None
            else:
                time.sleep(delay)

            if self.is_cancel_requested():
                self.ctx.post_cancel()
                return None
            if should_rebuild_client:
                self.close_client()
                self.client = self.new_client()
                self.client_closed = False
            attempt += 1

    def retry_delay(
        self,
        error: Union[openai.APIConnectionError, openai.APIStatusError, ResponseError],
        attempt: int,
    ) -> Optional[float]:
        """Retry only whitelisted SDK transport, HTTP, and response failures."""
        is_retriable = (
            isinstance(error, openai.APIConnectionError)  # Includes APITimeoutError.
            or isinstance(error, openai.APIStatusError)
            and (error.status_code in (408, 409, 429) or error.status_code >= 500)
            or isinstance(error, ResponseError) and error.code in TRANSIENT_RESPONSE_ERRORS
        )
        if not is_retriable:
            return None

        if isinstance(error, openai.APIStatusError):
            headers = error.response.headers
            if headers.get("x-should-retry") == "false":
                return None
            # Reuse the SDK parser for milliseconds, seconds and HTTP dates.
            retry_after = self.client._parse_retry_after_header(headers)
            if retry_after is not None and math.isfinite(retry_after) and retry_after > 0:
                return retry_after if retry_after <= MAX_RETRY_AFTER_DELAY else None

        delay = min(BASE_RETRY_DELAY_SECONDS * 2 ** (attempt - 1), MAX_RETRY_DELAY_SECONDS)
        return delay + random.uniform(0.0, delay * 0.1)

    def assert_response_contract(self, response: Response) -> None:
        if response.status != "completed":
            raise self.provider_error(
                f"OpenAI response {response.id!r} did not complete: "
                f"status={response.status!r}, "
                f"incomplete_details={response.incomplete_details!r}, "
                f"error={response.error!r}."
            )

        if response.error is not None:
            raise self.provider_error(
                f"OpenAI returned an error on a completed response: {response.error!r}."
            )

        if response.incomplete_details is not None:
            raise self.provider_error(
                "OpenAI returned incomplete_details on a completed response: "
                f"{response.incomplete_details!r}."
            )

        if response.previous_response_id is not None:
            raise self.provider_error(
                "OpenAI unexpectedly associated the response with "
                f"previous_response_id={response.previous_response_id!r}; "
                "this provider is manual-replay only."
            )

        if response.conversation is not None:
            raise self.provider_error(
                "OpenAI unexpectedly associated the response with a server-side "
                f"conversation: {response.conversation!r}."
            )

        if response.truncation != "disabled":
            raise self.provider_error(
                "OpenAI did not honor fail-closed truncation: "
                f"expected 'disabled', got {response.truncation!r}."
            )

        reasoning = response.reasoning
        if reasoning is None:
            raise self.provider_error(
                "OpenAI omitted the effective reasoning configuration."
            )

        actual_reasoning = (
            reasoning.mode,
            reasoning.effort,
            reasoning.context,
        )
        expected_reasoning = ("standard", "max", "all_turns")
        if actual_reasoning != expected_reasoning:
            raise self.provider_error(
                "OpenAI changed the effective reasoning configuration: "
                f"expected={expected_reasoning!r}, actual={actual_reasoning!r}."
            )

        cache_options = response.prompt_cache_options
        if cache_options is None:
            raise self.provider_error(
                "OpenAI omitted the effective prompt-cache configuration."
            )
        if cache_options.mode != "implicit":
            raise self.provider_error(
                "OpenAI changed the effective prompt-cache configuration: "
                f"expected mode='implicit', actual={cache_options.mode!r}."
            )

    def consume_response_output(
        self,
        response: Response,
    ) -> Tuple[List[PendingToolCall], str]:
        """Validate the full response before dispatching tools."""
        updates: List[TranscriptUpdate] = []
        final_text_chunks: List[str] = []
        try:
            replay_items, pending_calls, final_text = self.prepare_response_output(
                response, updates, final_text_chunks,
            )
        except Exception:
            # Retain partial text, including refusals, without invoking any tools
            # or adding an incomplete response to native replay history.
            if final_text_chunks:
                updates.append(ModelTextUpdate("\n".join(final_text_chunks), merge=False))
            self.publish_transcript_updates(updates)
            raise

        # Preserve the complete response output contiguously and in server order.
        self.history.extend(replay_items)

        self.publish_transcript_updates(updates)

        # Emit the selected terminal answer once, after all other projected text.
        # This keeps commentary/unphased preambles separate and guarantees that
        # the final ModelTextPart is exactly the result, even when unphased text
        # follows an explicit answer. Native history retains exact server order.
        self.append_model_text(final_text, merge=False)
        return pending_calls, final_text

    def publish_transcript_updates(
        self,
        updates: Sequence[TranscriptUpdate],
    ) -> None:
        for update in updates:
            if isinstance(update, ModelTextUpdate):
                self.append_model_text(update.text, merge=update.merge)
            else:
                self.transcript.append(update)
                self.ctx.post_transcript_update()

    def prepare_response_output(
        self,
        response: Response,
        updates: List[TranscriptUpdate],
        final_text_chunks: List[str],
    ) -> Tuple[List[ResponseInputItemParam], List[PendingToolCall], str]:
        """Stage exact replay items and transcript updates without dispatching.

        The temporary replay list is completed before mutating ``self.history``.
        Thus a newly introduced or malformed output kind fails the entire response
        rather than leaving a partially appended provider transcript.
        """

        replay_items: List[ResponseInputItemParam] = []
        pending_calls: List[PendingToolCall] = []
        trailing_messages: List[Tuple[int, ResponseOutputMessage]] = []
        reasoning_item_count = 0

        # Only the trailing answer messages in a terminal response can supply the
        # result. Reasoning, function calls, and commentary delimit that sequence;
        # explicit final_answer messages take precedence over unphased messages.
        # https://developers.openai.com/api/docs/guides/reasoning#phase-parameter
        if not any(isinstance(item, ResponseFunctionToolCall) for item in response.output):
            for index in range(len(response.output) - 1, -1, -1):
                item = response.output[index]
                if not isinstance(item, ResponseOutputMessage) or item.phase == "commentary":
                    break
                trailing_messages.append((index, item))
        final_phase = (
            "final_answer"
            if any(item.phase == "final_answer" for _, item in trailing_messages)
            else None
        )
        final_message_indices = {
            index for index, item in trailing_messages if item.phase == final_phase
        }
        previous_message: Optional[ResponseOutputMessage] = None

        for index, item in enumerate(response.output):
            if not isinstance(item, ResponseOutputMessage):
                previous_message = None

            if isinstance(item, ResponseReasoningItem):
                reasoning_item_count += 1
                encrypted_content = self.validate_reasoning_item(item)

                # OpenAI reasoning remains opaque. encrypted_content is the
                # provider-native continuation signature retained in history.
                updates.append(ThinkingBlockPart(
                    content="", signature=encrypted_content, redacted=True,
                ))
            elif isinstance(item, ResponseFunctionToolCall):
                self.validate_function_call(item)
                args = self.parse_function_arguments(item)

                # Match Gemini: generate netflux IDs independently of provider
                # IDs. Keep OpenAI's original call_id in replay and API outputs;
                # use tool_use_id for transcript/child correlation in netflux.
                tool_use_id = self.new_tool_use_id(item.name)
                if self.is_valid_status_update(item.name, args):
                    updates.append(ModelStatusPart(
                        text=args["msg"], tool_use_id=tool_use_id,
                    ))
                else:
                    updates.append(ToolUsePart(
                        tool_use_id=tool_use_id, tool_name=item.name,
                        args=MappingProxyType(copy.deepcopy(args)),
                    ))
                pending_calls.append(PendingToolCall(
                    item=item, args=args, tool_use_id=tool_use_id,
                ))

            elif isinstance(item, ResponseOutputMessage):
                self.validate_output_message(item)

                is_final = index in final_message_indices
                has_text = self.prepare_output_message(
                    item, updates,
                    merge=previous_message is not None and previous_message.phase == item.phase,
                    deferred_chunks=final_text_chunks if is_final else None,
                )
                if not is_final and has_text:
                    previous_message = item

            else:
                # Never silently omit output from protocols we have not enabled.
                raise self.provider_error(
                    "Unsupported OpenAI response output item: "
                    f"{type(item).__name__}: {item!r}."
                )

            # Use SDK aliases (notably async_) and retain every native field for
            # replay, including encrypted reasoning, message phases and call IDs.
            replay_items.append(cast(
                ResponseInputItemParam,
                item.to_dict(mode="json", use_api_names=True, exclude_unset=True),
            ))

        usage = response.usage
        if usage is None:
            raise self.provider_error(
                "OpenAI completed a response without usage information."
            )
        reasoning_tokens = usage.output_tokens_details.reasoning_tokens
        if reasoning_tokens > 0 and reasoning_item_count == 0:
            raise self.provider_error(
                "OpenAI billed reasoning tokens but returned no reasoning item to "
                "replay; reasoning continuity cannot be guaranteed."
            )

        return replay_items, pending_calls, "\n".join(final_text_chunks).strip()

    def validate_reasoning_item(self, item: ResponseReasoningItem) -> str:
        if item.status not in (None, "completed"):
            raise self.provider_error(
                "OpenAI returned a non-completed reasoning item: "
                f"id={item.id!r}, status={item.status!r}."
            )

        encrypted_content = item.encrypted_content
        if not encrypted_content:
            raise self.provider_error(
                "OpenAI returned a reasoning item without encrypted_content while "
                "store=False; replaying it would silently lose reasoning continuity."
            )

        # No summary was requested, and raw reasoning is intentionally opaque.
        # Fail loudly if either representation unexpectedly appears so it cannot be
        # confused with the encrypted continuation state.
        if item.summary:
            raise self.provider_error(
                "OpenAI returned a reasoning summary although no summary was requested."
            )
        if item.content:
            raise self.provider_error(
                "OpenAI unexpectedly returned raw reasoning text content."
            )

        return encrypted_content

    def validate_function_call(
        self,
        item: ResponseFunctionToolCall,
    ) -> None:
        if item.status not in (None, "completed"):
            raise self.provider_error(
                "OpenAI returned a non-completed function call: "
                f"call_id={item.call_id!r}, status={item.status!r}."
            )

        if not isinstance(item.call_id, str) or not item.call_id:
            raise self.provider_error(
                f"OpenAI returned an invalid function call_id: {item.call_id!r}."
            )

        if not isinstance(item.name, str) or not item.name:
            raise self.provider_error(
                f"OpenAI returned an invalid function name {item.name!r} "
                f"for call_id={item.call_id!r}."
            )

        if item.async_ is not None and item.async_ is not False:
            raise self.provider_error(
                "OpenAI emitted an unsupported async function-call value: "
                f"{item.async_!r}."
            )

        if item.caller is not None and (
            not isinstance(item.caller, CallerDirect) or item.caller.type != "direct"
        ):
            raise self.provider_error(
                "OpenAI emitted a non-direct function caller although this "
                "provider did not opt into programmatic tool calling: "
                f"{item.caller!r}."
            )

        if item.namespace is not None:
            raise self.provider_error(
                "OpenAI unexpectedly emitted a namespaced function call: "
                f"namespace={item.namespace!r}, name={item.name!r}."
            )

    def parse_function_arguments(
        self,
        item: ResponseFunctionToolCall,
    ) -> Dict[str, Any]:
        try:
            parsed: Any = json.loads(item.arguments)
        except json.JSONDecodeError as ex:
            raise self.provider_error(
                "OpenAI returned malformed JSON function arguments for "
                f"{item.name!r}: {item.arguments!r}."
            ) from ex

        if not isinstance(parsed, dict):
            raise self.provider_error(
                "OpenAI function arguments must decode to a JSON object; "
                f"tool={item.name!r}, decoded_type={type(parsed).__name__}."
            )

        return parsed

    def validate_output_message(self, item: ResponseOutputMessage) -> None:
        # SDK response parsing is permissive; validate the protocol values before
        # interpreting or replaying assistant messages.
        if item.type != "message" or item.role != "assistant":
            raise self.provider_error(
                "Unsupported OpenAI assistant message: "
                f"id={item.id!r}, type={item.type!r}, role={item.role!r}."
            )

        if item.status != "completed":
            raise self.provider_error(
                "OpenAI returned a non-completed assistant message: "
                f"id={item.id!r}, status={item.status!r}."
            )

        if item.phase not in (None, "commentary", "final_answer"):
            raise self.provider_error(
                "Unsupported OpenAI assistant message phase: "
                f"id={item.id!r}, phase={item.phase!r}."
            )

    def prepare_output_message(
        self, item: ResponseOutputMessage, updates: List[TranscriptUpdate], *, merge: bool,
        deferred_chunks: Optional[List[str]] = None,
    ) -> bool:
        has_text = False

        for part in item.content:
            if isinstance(part, ResponseOutputText) and part.type == "output_text":
                text = part.text
            elif isinstance(part, ResponseOutputRefusal) and part.type == "refusal":
                text = part.refusal
            else:
                raise self.provider_error(
                    "Unsupported OpenAI assistant content part: "
                    f"{type(part).__name__}: {part!r}."
                )

            if text and text.strip():
                has_text = True
                if deferred_chunks is None:
                    updates.append(ModelTextUpdate(text, merge=merge))
                    merge = True
                else:
                    deferred_chunks.append(text)

            if isinstance(part, ResponseOutputRefusal):
                # Preserve the explanation in the transcript and the terminal exception.
                raise self.provider_error(
                    f"OpenAI refusal in assistant message {item.id!r}: {text}"
                )

        return has_text

    def execute_tool_batch(
        self,
        pending_calls: Sequence[PendingToolCall],
    ) -> Optional[List[FunctionCallOutput]]:
        """Start all requested tools, wait for all, and return ordered outputs.

        Child Nodes begin running when invoked, so invoking the complete batch before
        waiting preserves the parallel behavior of the Anthropic and Gemini drivers.
        Ordinary invocation/tool failures are returned to the model. When this
        agent calls raise_exception, finish waiting for the rest of the batch,
        then post its declared failure through the Runtime.
        """

        children: List[Optional[Node]] = []
        invoke_exceptions: List[Optional[Exception]] = []

        for pending in pending_calls:
            try:
                children.append(self.invoke_tool_function(
                    pending.item.name,
                    pending.args,
                    pending.tool_use_id,
                ))
                invoke_exceptions.append(None)
            except Exception as ex:
                children.append(None)
                invoke_exceptions.append(ex)

        pending_agent_ex: Optional[AgentException] = None
        outputs: List[FunctionCallOutput] = []

        for pending, child, invoke_ex in zip(
            pending_calls,
            children,
            invoke_exceptions,
        ):
            out_text: str
            is_error: bool

            if invoke_ex is not None:
                out_text = AgentNode.stringify_exception(invoke_ex)
                is_error = True
            else:
                if child is None:
                    raise self.provider_error(
                        "OpenAI tool batch lost a child Node without an invocation "
                        "exception."
                    )
                try:
                    result: Any = child.result()
                    out_text = "" if result is None else str(result)
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
                    out_text = AgentNode.stringify_exception(ex)
                    is_error = True

            if not self.is_valid_status_update(pending.item.name, pending.args):
                self.transcript.append(ToolResultPart(
                    tool_use_id=pending.tool_use_id, tool_name=pending.item.name,
                    outputs=out_text, is_error=is_error,
                ))
                self.ctx.post_transcript_update()

            wire_output = json.dumps(
                {"error" if is_error else "output": out_text},
                ensure_ascii=False, separators=(",", ":"),
            )

            outputs.append(FunctionCallOutput(
                type="function_call_output", call_id=pending.item.call_id, output=wire_output,
            ))

        # React only after the full batch has been joined/transcribed.
        if pending_agent_ex is not None:
            self.ctx.post_exception(pending_agent_ex)
            return None

        if self.is_cancel_requested():
            self.ctx.post_cancel()
            return None

        return outputs

    def accumulate_usage(self, usage: ResponseUsage) -> None:
        cached_input = usage.input_tokens_details.cached_tokens
        cache_write_input = usage.input_tokens_details.cache_write_tokens
        reasoning_output = usage.output_tokens_details.reasoning_tokens
        if min(cached_input, cache_write_input, reasoning_output) < 0:
            raise self.provider_error(f"OpenAI returned negative token counts: {usage!r}.")
        regular_input = usage.input_tokens - cached_input - cache_write_input

        if regular_input < 0:
            raise self.provider_error(
                "OpenAI returned inconsistent input-token accounting: "
                f"input_tokens={usage.input_tokens}, "
                f"cached_tokens={cached_input}, "
                f"cache_write_tokens={cache_write_input}."
            )

        non_reasoning_output = usage.output_tokens - reasoning_output
        if non_reasoning_output < 0:
            raise self.provider_error(
                "OpenAI returned inconsistent output-token accounting: "
                f"output_tokens={usage.output_tokens}, "
                f"reasoning_tokens={reasoning_output}."
            )

        expected_total = usage.input_tokens + usage.output_tokens
        if usage.total_tokens != expected_total:
            raise self.provider_error(
                "OpenAI returned inconsistent total-token accounting: "
                f"total_tokens={usage.total_tokens}, expected={expected_total}."
            )

        token_usage = self.usage
        token_usage.input_tokens_cache_read += cached_input
        token_usage.input_tokens_cache_write = (
            token_usage.input_tokens_cache_write or 0
        ) + cache_write_input
        token_usage.input_tokens_regular += regular_input
        token_usage.input_tokens_total += usage.input_tokens

        token_usage.output_tokens_reasoning = (
            token_usage.output_tokens_reasoning or 0
        ) + reasoning_output
        token_usage.output_tokens_text = (
            token_usage.output_tokens_text or 0
        ) + non_reasoning_output
        token_usage.output_tokens_total += usage.output_tokens

        token_usage.context_window_in = usage.input_tokens
        token_usage.context_window_out = usage.output_tokens

    def make_function_tool(self, fn: Function) -> FunctionToolParam:
        properties: Dict[str, PropertySchema] = {}

        for arg in fn.args:
            json_type = self.json_type_for_arg(arg.argtype)
            property_schema = PropertySchema(
                type=[json_type, "null"] if arg.optional else json_type, description=arg.desc,
            )
            if arg.argtype is str and arg.enum is not None:
                enum_values: List[Optional[str]] = list(sorted(arg.enum))
                if arg.optional:
                    # Netflux already accepts None for optional args. Strict
                    # schemas must allow null in both the type and the enum.
                    enum_values.append(None)
                property_schema["enum"] = enum_values

            properties[arg.name] = property_schema

        # In OpenAI strict mode every property is listed in required. Netflux
        # optional args remain optional semantically by being nullable.
        schema = ObjectParametersSchema(
            type="object", properties=properties,
            required=[arg.name for arg in fn.args], additionalProperties=False,
        )

        tool = FunctionToolParam(
            type="function", name=fn.name, description=fn.desc,
            parameters=cast(FunctionParameters, schema), strict=True,
            allowed_callers=["direct"], defer_loading=False,
        )
        tool["async"] = False  # Reserved Python keyword in the SDK's TypedDict.
        return tool

    @staticmethod
    def json_type_for_arg(py_type: AllowedArgTypeUnion) -> JsonScalarType:
        if py_type is str:
            return "string"
        if py_type is int:
            return "integer"
        if py_type is float:
            return "number"
        if py_type is bool:
            return "boolean"
        raise TypeError(f"Unsupported FunctionArg type for OpenAI: {py_type!r}")

    def provider_error(self, message: str) -> ModelProviderException:
        return ModelProviderException(
            message=message, provider=type(self),
            agent_name=self.agent_fn.name, node_id=self.id,
        )
