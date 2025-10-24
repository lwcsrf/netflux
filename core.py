from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Type, Union, get_args, Mapping
import inspect
from threading import Thread
import multiprocessing as mp
from multiprocessing import Lock
from multiprocessing.synchronize import Event
from overrides import override

AllowedArgTypeUnion = Union[Type[str], Type[int], Type[float], Type[bool]]
AllowedArgTypeTuple: tuple[type, ...] = tuple(
    inner for t in get_args(AllowedArgTypeUnion) for inner in get_args(t)
)

from .providers import Provider

class CancellationException(Exception):
    """Raised when a node cooperatively acknowledges a cancellation request."""

    def __init__(self, message: str = "Operation was canceled (no reason provided)"):
        super().__init__(message)

class NodeState(Enum):
    Waiting = "Waiting"
    Running = "Running"
    Success = "Success"
    Error = "Error"
    Canceled = "Canceled"

TerminalNodeStates = (NodeState.Success, NodeState.Error, NodeState.Canceled)

class SessionScope(Enum):
    # Refers to the lifetime of the entire top-level invocation tree.
    TopLevel = "TopLevel"
    # Refers to the lifetime of the direct parent Node (if any).
    Parent = "Parent"
    # Refers to the lifetime of the current Node itself.
    Self = "Self"

@dataclass
class TokenUsage:
    # Input tokens that were part of a cache hit
    input_tokens_cache_read: int = 0
    # Input tokens that are not part of any cache hit and whose intermediate internal forward state will be cached for
    # follow-up requests, with a special billing rate for the cache write tokens.
    input_tokens_cache_write: Optional[int] = None
    # Input tokens that were not part of a cache hit. Some model providers (e.g. Gemini) make this a free cache write
    # always, and others (e.g. Anthropic) do not (instead will have a non-null `input_tokens_cache_write`).
    input_tokens_regular: int = 0
    # The sum of the above 3, which are mutually exclusive.
    input_tokens_total: int = 0

    # only the output tokens that were part of reasoning ("thinking"), applicable only if the output breakdown is
    # known.
    output_tokens_reasoning: Optional[int] = None
    # only the output tokens that were part of non-reasoning assistant text or tool calls (sometimes text generated for
    # a tool call argument is a very significant output deliberately), applicable only if the output breakdown is
    # known.
    output_tokens_text: Optional[int] = None
    # the sum of all output tokens (including tool call, reasoning, non-reasoning tokens), which are mutually exclusive.
    output_tokens_total: int = 0

    # Context window token totals for the last completed request-response cycle.
    # These values are overwritten on each cycle to reflect the latest context window size.
    context_window_in: int = 0
    context_window_out: int = 0

class SessionBag:
    """Thread-safe namespace/key object registry for a Node scope."""
    def __init__(self) -> None:
        self._lock = Lock()
        self._values: Dict[str, Dict[str, Any]] = {}

    def get_or_put(self, namespace: str, key: str, factory: Callable[[], Any]) -> Any:
        with self._lock:
            ns = self._values.setdefault(namespace, {})
            if key in ns:
                return ns[key]
            value = factory()
            ns[key] = value
            return value

@dataclass(frozen=True)
class FunctionArg:
    name: str
    argtype: AllowedArgTypeUnion
    desc: str = ""
    optional: bool = False
    enum: Optional[Set[str]] = None

    def __post_init__(self) -> None:
        """Spec-time validation of declared arg type and (optional) enum constraint."""
        if self.argtype not in AllowedArgTypeTuple:
            raise ValueError(
                f"Arg '{self.name}' has unsupported type {self.argtype!r}; "
                f"only {AllowedArgTypeTuple} are allowed."
            )

        if self.enum is not None:
            if self.argtype is not str:
                raise ValueError(
                    f"Arg '{self.name}': enum constraint is only supported for str args."
                )

            # Ensure non-empty, strings only, unique; store as immutable tuple
            if not isinstance(self.enum, (set, frozenset)) or len(self.enum) == 0:
                raise ValueError(f"Arg '{self.name}': enum must be a set of string literals.")
            if any(not isinstance(v, str) for v in self.enum):
                raise ValueError(f"Arg '{self.name}': enum values must be strings.")

    def validate_value(self, value: Any) -> None:
        # Allow None literal only when optional arg.
        if value is None:
            if self.optional:
                return
            raise ValueError(f"Arg '{self.name}' is required and cannot be None")

        # bool is a subclass of int; enforce exact match semantics
        if self.argtype is bool:
            if type(value) is not bool:
                raise ValueError(
                    f"Arg '{self.name}' expects bool, got {type(value).__name__}"
                )
            return

        if self.argtype is int and type(value) is bool:
            raise ValueError(f"Arg '{self.name}' expects int, got bool")

        if self.argtype is float and type(value) is bool:
            raise ValueError(f"Arg '{self.name}' expects float, got bool")

        if not isinstance(value, self.argtype):
            raise ValueError(
                f"Arg '{self.name}' expects {self.argtype.__name__}, got {type(value).__name__}"
            )

        # Enum constraint (strings only). Give a model useful feedback for retries.
        if self.argtype is str and self.enum is not None:
            if value not in self.enum:
                allowed = ", ".join(sorted(self.enum))
                raise ValueError(
                    f"Arg '{self.name}' must be one of: {allowed}; got '{value}'"
                )

class Function(ABC):
    def __init__(
        self,
        name: str,
        desc: str,
        args: Sequence[FunctionArg],
    ) -> None:
        self.name = name
        self.desc = desc
        self.args: List[FunctionArg] = list(args)

    def validate_coerce_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate whether an invocation with the given `args` is valid for this Function.
        """
        expect = {a.name: a for a in self.args}
        unknown = set(args.keys()) - set(expect.keys())
        if unknown:
            raise ValueError(
                f"Unknown arg(s) for {self.name}: {sorted(unknown)}; expected {sorted(expect)}"
            )

        # Only required (non-optional) args must be present.
        missing = [name for name, spec in expect.items() if name not in args and not spec.optional]
        if missing:
            raise ValueError(
                f"Missing required arg(s) for {self.name}: {missing}"
            )

        # Validate provided values (optional args may be omitted; if explicitly given as None,
        # `validate_value()` will accept only when optional=True).
        # Here we can also add light coercion for minor mistakes where it makes sense.
        coerced: Dict[str, Any] = dict(args)  # copy
        for k, v in list(args.items()):
            spec = expect[k]
            coerced[k] = v

            # Tolerate boolean-like strings.
            if spec.argtype is bool and isinstance(v, str):
                s = v.strip().lower()
                if s == "true" or s == "false":
                    coerced[k] = (s == "true")

            expect[k].validate_value(coerced[k])
        return coerced

    @property
    @abstractmethod
    def uses(self) -> List['Function']:
        raise NotImplementedError()

    def is_agent(self) -> bool:
        return isinstance(self, AgentFunction)

    def is_code(self) -> bool:
        return isinstance(self, CodeFunction)

class AgentFunction(Function):
    def __init__(
        self,
        *,
        name: str,
        desc: str,
        args: Sequence[FunctionArg],
        system_prompt: str,
        user_prompt_template: str,
        uses: Sequence[Function] = (),
        default_model: Provider = Provider.Anthropic,
    ) -> None:
        super().__init__(name, desc, args)
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        self.uses_funcs: List[Function] = list(uses)
        self.default_model = default_model

        # Check if any Functions have dup names.
        names = [t.name for t in self.uses_funcs]
        dups = [n for n in set(names) if names.count(n) > 1]
        if dups:
            raise ValueError(f"{name}: duplicate tool names not allowed: {sorted(dups)}")

    @property
    def uses(self) -> List[Function]:
        return list(self.uses_funcs)

class CodeFunction(Function):
    def __init__(
        self,
        *,
        name: str,
        desc: str,
        args: Sequence[FunctionArg],
        callable: Callable[..., Any],
        uses: Sequence[Function] = (),
    ) -> None:
        super().__init__(name, desc, args)
        self.callable = callable
        self._uses: List[Function] = list(uses)
        self._validate_callable_signature()

        # Check if any Functions have duplicate names.
        names = [t.name for t in self._uses]
        dups = [n for n in set(names) if names.count(n) > 1]
        if dups:
            raise ValueError(f"{self.name}: duplicate uses names not allowed: {sorted(dups)}")

    @property
    def uses(self) -> List[Function]:
        return list(self._uses)

    def _validate_callable_signature(self) -> None:
        sig = inspect.signature(self.callable)
        params = list(sig.parameters.values())
        if not params or params[0].kind not in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY):
            raise TypeError(f"{self.name}: first parameter must be a RunContext")

        # Forbid varargs/kwargs to keep the contract tight
        if any(p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD) for p in params):
            raise TypeError(f"{self.name}: *args/**kwargs are not allowed; declare explicit parameters matching FunctionArg[]: {self.args}")

        # Check contract:
        # - All arguments after RunContext must be KEYWORD_ONLY (use *, before them).
        # - Required args (optional=False): no default.
        # - Optional args (optional=True): default must be None.
        remaining: List[inspect.Parameter] = params[1:]

        non_kwonly = [p.name for p in remaining if p.kind is not inspect.Parameter.KEYWORD_ONLY]
        if non_kwonly:
            raise TypeError(
                f"{self.name}: all callable parameters after RunContext must be keyword-only; "
                f"violations: {non_kwonly}"
            )

        expected = [a.name for a in self.args]
        actual = [p.name for p in remaining if p.kind is inspect.Parameter.KEYWORD_ONLY]
        if actual != expected:
            raise TypeError(f"{self.name}: callable signature args {actual} do not match spec {expected}")

        # Default value rules per optionality
        by_name: Dict[str, inspect.Parameter] = {p.name: p for p in remaining}
        for a in self.args:
            p = by_name[a.name]
            if a.optional:
                # Optional must have default=None
                if p.default is inspect._empty:
                    raise TypeError(
                        f"{self.name}: optional arg '{a.name}' must have default=None"
                    )
                if p.default is not None:
                    raise TypeError(
                        f"{self.name}: optional arg '{a.name}' default must be None, got {p.default!r}"
                    )
            else:
                # Required must not have a default
                if p.default is not inspect._empty:
                    raise TypeError(
                        f"{self.name}: required arg '{a.name}' must not have a default"
                    )

@dataclass(frozen=True)
class TranscriptPart:
    pass

@dataclass(frozen=True)
class UserTextPart(TranscriptPart):
    text: str

@dataclass(frozen=True)
class ModelTextPart(TranscriptPart):
    text: str

@dataclass(frozen=True)
class ToolUsePart(TranscriptPart):
    tool_use_id: str
    tool_name: str
    args: Mapping[str, Any]

@dataclass(frozen=True)
class ToolResultPart(TranscriptPart):
    tool_use_id: str
    tool_name: str
    outputs: Any
    is_error: bool

@dataclass(frozen=True)
class ThinkingBlockPart(TranscriptPart):
    content: str
    signature: str
    redacted: bool = False

@dataclass
class RunContext:
    runtime: 'Runtime'      # type: ignore
    node: Optional['Node']  # None outside of any top-level task.
    object_bags: Dict[SessionScope, SessionBag] = field(default_factory=dict)
    cancel_event: Optional[Event] = None

    def invoke(
        self,
        fn: Function,
        args: Dict[str, Any],
        provider: Optional[Provider] = None,
        cancel_event: Optional[Event] = None,
    ) -> 'Node':
        """
        Proxy to the Runtime to invoke a Function and create associated Node + edges.
        Returns the created `Node`.

        For AgentFunction calls only, optionally specify `provider` to override the default model.

        Provide `cancel_event` to enforce a particular cancellation scope for the child node.
        When omitted, the Runtime inherits the caller's CancelEvent (if any). It is correct
        practice to make sure a custom `cancel_event` has cancelation criteria that includes
        that of the caller (downward cancelation propagation).
        """
        return self.runtime.invoke(
            self.node,
            fn,
            args,
            provider=provider,
            cancel_event=cancel_event,
        )

    def post_status_update(self, state: 'NodeState') -> None:
        if self.node is None:
            raise RuntimeError("post_status_update may only be called from within a Node execution context")
        self.runtime.post_status_update(self.node, state)

    def post_success(self, outputs: Any) -> None:
        if self.node is None:
            raise RuntimeError("post_success may only be called from within a Node execution context")
        self.runtime.post_success(self.node, outputs)

    def post_exception(self, exception: Exception) -> None:
        if self.node is None:
            raise RuntimeError("post_exception may only be called from within a Node execution context")
        self.runtime.post_exception(self.node, exception)

    def post_cancel(self, exception: Optional[CancellationException] = None) -> None:
        """
        Inform the Runtime that the current Node is being cooperatively canceled.
        The `exception` may be provided to indicate the reason for cancellation,
        otherwise a generic `CancellationException` will be used.
        """
        if self.node is None:
            raise RuntimeError("post_cancel may only be called from within a Node execution context")
        self.runtime.post_cancel(self.node, exception)

    def post_transcript_update(self) -> None:

        if self.node is None:
            raise RuntimeError("post_transcript_update may only be called from within a Node execution context")
        self.runtime.post_transcript_update(self.node)

    def cancel_requested(self) -> bool:
        return bool(self.cancel_event and self.cancel_event.is_set())

    def _resolve_bag(self, scope: SessionScope) -> SessionBag:
        if self.node is None:
            raise RuntimeError("SessionBags are only available within a Node execution context")
        if not self.object_bags:
            raise RuntimeError("SessionBags have not been initialized for this RunContext")
        if scope is SessionScope.Parent and self.node.parent is None:
            raise NoParentSessionError("Current node has no parent.")

        try:
            return self.object_bags[scope]
        except KeyError as exc:
            raise RuntimeError(f"Session scope {scope.value} is not available") from exc

    def get_or_put(
        self,
        scope: SessionScope,
        namespace: str,
        key: str,
        factory: Callable[[], Any],
    ) -> Any:
        """Concurrency-safe get-or-else-create for an object in a session-scoped bag."""
        bag = self._resolve_bag(scope)
        return bag.get_or_put(namespace, key, factory)

@dataclass(frozen=True)
class NodeView:
    id: int
    fn: Function
    inputs: Dict[str, Any]
    state: NodeState
    outputs: Optional[Any]
    exception: Optional[Exception]
    children: tuple['NodeView', ...]
    usage: Optional[TokenUsage]
    transcript: tuple[TranscriptPart, ...]
    started_at: Optional[float]
    ended_at: Optional[float]  # Any terminal state.
    update_seqnum: int  # Seqnum when this NodeView was generated.

class Node(ABC):
    def __init__(
        self,
        ctx: RunContext,
        id: int,
        fn: Function,
        inputs: Dict[str, Any],
        parent: Optional['Node'],
        cancel_event: Optional[Event],
    ) -> None:
        self.ctx: RunContext = ctx
        self.id: int = id
        self.fn: Function = fn
        self.outputs: Optional[Any] = None
        self.exception: Optional[Exception] = None
        self.state: NodeState = NodeState.Waiting
        self.parent: Optional[Node] = parent
        self.children: List[Node] = []
        self.thread: Optional[Thread] = None
        self.done: Event = mp.Event()
        self.session_bag: SessionBag = SessionBag()
        self.cancel_event: Optional[Event] = cancel_event
        self.started_at: Optional[float] = None
        self.ended_at: Optional[float] = None

        assert isinstance(ctx, RunContext)
        assert isinstance(fn, Function)

        self.inputs: Dict[str, Any] = fn.validate_coerce_args(dict(inputs))

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.id} fn={self.fn.name} state={self.state.value}>"

    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = Thread(target=self.run_wrapper, name=f"netflux-node-{self.id}", daemon=True)
        self.thread.start()

    def run_wrapper(self) -> None:
        self.ctx.post_status_update(NodeState.Running)

        try:
            self.run()
        except CancellationException as ex:
            self.ctx.post_cancel(ex)
        except Exception as ex:
            self.ctx.post_exception(ex)
        assert self.state in TerminalNodeStates
        assert self.done.is_set()

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError

    @property
    def is_done(self) -> bool:
        return self.done.is_set()

    def wait(self):
        return self.done.wait()

    def result(self):
        self.wait()
        if self.state == NodeState.Error:
            assert self.exception
            raise self.exception
        if self.state == NodeState.Canceled:
            assert self.exception and isinstance(self.exception, CancellationException)
            raise self.exception
        return self.outputs

    def watch(self, as_of_seq: int = 0, *, timeout: Optional[float] = None) -> Optional[NodeView]:
        """Return the latest NodeView update (blocking until seq > view.update_seqnum).

        If ``timeout`` (seconds) elapses before a newer snapshot is available,
        returns ``None`` (mirroring the underlying condition wait semantics).
        """
        return self.ctx.runtime.watch(self, as_of_seq=as_of_seq, timeout=timeout)

    def is_cancel_requested(self) -> bool:
        if self.cancel_event is None:
            return False
        return self.cancel_event.is_set()

class CodeNode(Node):
    """
    The invoked CodeFunction is expected to expose a Python callable under the attribute
    `callable` that accepts `(run_ctx, *, arg0, arg1, etc)` and returns any
    string-serializable object.

    A per-invocation RunContext is expected to be created by the Runtime of the
    framework so the Callable can use it to invoke other Functions through the
    framework (thus creating child Nodes). The Callable may return an unstructured or
    structured output, but it must be string-serializable.

    A caller will typically call `node.result()` to wait for completion and get the
    output string. Alternatively, the Callable may raise an Exception, which will be
    propagated upon calling `node.result()`.
    """
    def __init__(
        self,
        ctx: RunContext,
        id: int,
        fn: Function,
        inputs: Dict[str, Any],
        parent: Optional[Node],
        cancel_event: Optional[Event],
    ):
        super().__init__(ctx, id, fn, inputs, parent, cancel_event)
        assert isinstance(self.fn, CodeFunction)

    def run(self) -> None:
        assert isinstance(self.fn, CodeFunction)
        func: CodeFunction = self.fn
        try:
            result = func.callable(self.ctx, **self.inputs)
            self.ctx.post_success(result)
        except CancellationException as e:
            self.ctx.post_cancel(e)
        except Exception as e:
            self.ctx.post_exception(e)

class AgentNode(Node):
    """
    Base class for LLM-backed agent invocations. Handles:
      - Templating the initial user prompt from the agent's template + inputs
      - Holding a normalized transcript (UserTextPart, ThinkingBlockPart, ToolUsePart, ToolResultPart, ModelTextPart)
      - Serializing tool invocations even if the model requests them in parallel

    Subclasses must implement `run()` to drive a provider-specific tool loop, and therein
    use the `RunContext` to submit children `Function` calls as a result of tool use.
    Before returning from `run()`, they should have set `self.outputs` or `self.exception`.
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
    ) -> None:
        super().__init__(ctx, id, fn, inputs, parent, cancel_event)
        assert isinstance(fn, AgentFunction), "AgentNode must wrap an AgentFunction"
        self.agent_fn: AgentFunction = fn
        self.transcript: List[TranscriptPart] = []

        if not callable(client_factory):
            raise TypeError("AgentNode requires a callable client_factory")
        self.client_factory: Callable[[], Any] = client_factory

        # For each `Function` the agent may invoke, map from string tool names
        # (as will be referred to by model tool call responses) to the `Function`.
        self.func_map: Dict[str, Function] = {t.name: t for t in self.agent_fn.uses}

    def get_transcript(self) -> List[TranscriptPart]:
        return list(self.transcript)  # shallow copy

    @property
    @abstractmethod
    def token_usage(self) -> TokenUsage:
        raise NotImplementedError

    @override
    def run_wrapper(self) -> None:
        self.ctx.post_status_update(NodeState.Running)
        try:
            self.run()
        except AgentException as e:
            # Safety: agents shouldn't bubble this far, but if they do, honor it.
            self.ctx.post_exception(e)
        except CancellationException as e:
            # Safety: agents shouldn't bubble this far, as they would have `post_cancel()`
            # already in their `run()`, but if they do, honor it.
            self.ctx.post_cancel(e)
        except Exception as ex:
            # Any other case must be a provider fault.
            mpe = ModelProviderException(
                message=f"The provider of the agent's model (the driver) faulted due to: "
                        f"{self.stringify_exception(ex)}",
                provider=type(self),
                agent_name=self.agent_fn.name,
                node_id=self.id,
                inner_exception=ex,
            )
            self.ctx.post_exception(mpe)
        assert self.state in TerminalNodeStates
        assert self.done.is_set()

    def build_user_text(self) -> str:
        # Templated user prompt injection.
        # This will raise on any invalid substitutions (todo: dedicated exception).
        return self.agent_fn.user_prompt_template.format(**self.inputs)

    def invoke_tool_function(
        self, tool_name: str, tool_args: Dict[str, Any],
    ) -> Node:
        if tool_name not in self.func_map:
            raise RuntimeError(
                f"Invoking unknown tool: '{tool_name}'. Tools available: {self.func_map.keys()}")
        fn: Function = self.func_map[tool_name]

        return self.ctx.invoke(fn, tool_args)

    @staticmethod
    def stringify_exception(ex: Exception) -> str:
        """Convert an exception to a single-line string with its type and message.
        Subtypes can use this to ensure adherence to the prescribed contract:
            "{type}: {string_rep}"
        E.g.:
            "FileNotFoundError: [Errno 2] No such file or directory: 'config.json'"
        """
        return f"{type(ex).__name__}: {ex.__str__()}"

class NoParentSessionError(Exception):
    pass

class AgentException(Exception):
    """Exception raised when an agent decides to invoke `RaiseException` by its own volition.

    This exception differentiates agent-initiated errors from infrastructure/service errors.
    Used when an agent explicitly decides to fault for any reason related to its task execution.
    Examples:
    - Tool use (`CodeFunction`) errors (e.g. file not found, invalid argument, etc).
    - Recurrent sub-agent (`AgentFunction`) errors make this agent unable to complete its task.
    - Missing context or preconditions that the agent deems necessary to complete its task.
    - Agent explicitly decides to abort its task for any reason.

    Attributes:
        message: The error detail produced by the agent
        agent_name: Name of the faulting agent
        node_id: Instance ID of the Node that faulted
    """

    def __init__(self, message: str, agent_name: str, node_id: int):
        super().__init__(message)
        self.message = message
        self.agent_name = agent_name
        self.node_id = node_id

    def __str__(self) -> str:
        return f"Agent '{self.agent_name}' (node {self.node_id}): {self.message}"

class ModelProviderException(Exception):
    """Exception raised when AgentNode implementations (`see: providers/`) fail for any reason.

    This exception is always unrelated to the agent's task and is never raised by the agent.
    Used for infrastructure, configuration, or service-level failures.

    Examples include:
    - Provider AgentNode malimplementation (not following protocol; not using SDK correctly)
    - Connection socket broken or can't open to remote service; timeouts or network failures.
    - Authentication / configuration issues
    - Provider overloaded, client rate-limited, load shedding or quota issues
    - Core framework bugs during ctx.invoke(..)
    - Other faults by the remote provider service

    Attributes:
        message: The error message
        provider: Name of the provider class that faulted
        agent_name: Name of agent being processed when provider faulted
        node_id: Instance ID of the Agent being processed
        inner_exception: The original exception (e.g. from SDK) that caused this failure
    """

    def __init__(self, message: str, provider: type, agent_name: str,
                 node_id: int, inner_exception: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.agent_name = agent_name
        self.node_id = node_id
        self.inner_exception = inner_exception

    def __str__(self) -> str:
        base_msg = f"'{self.provider.__name__}' failed processing agent '{self.agent_name}' (node {self.node_id}): {self.message}"
        if self.inner_exception:
            base_msg += f" (caused by: {AgentNode.stringify_exception(self.inner_exception)})"
        return base_msg
