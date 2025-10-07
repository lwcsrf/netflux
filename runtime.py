from dataclasses import dataclass
import copy
import logging
import time
from typing import Any, Callable, Deque, Dict, List, Mapping, Optional, Sequence, Union
import multiprocessing as mp
from multiprocessing import Lock
from multiprocessing.synchronize import Event, Condition
from collections import deque

from .core import (
    Node,
    NodeState,
    NodeView,
    RunContext,
    Function,
    CodeFunction,
    AgentFunction,
    CodeNode,
    AgentNode,
    SessionScope,
    SessionBag,
    CancellationException,
    TokenUsage,
)
from .providers import Provider, get_AgentNode_impl


@dataclass
class NodeObservable:
    cond: Condition
    touch_seqno: int  # last global seqno that modified this node
    view: NodeView    # immutable snapshot reflecting state at touch_seqno

class Runtime:
    def __init__(self,
        specs: Sequence[Function],
        *,
        client_factories: Mapping[Provider, Callable[[], Any]],
    ):
        self._functions: List[Function] = list(specs)  # shallow copy
        # Index functions by name; uniqueness enforced.
        self._fn_by_name: Dict[str, Function] = {}
        self._client_factories: Dict[Provider, Callable[[], Any]] = \
            self.validate_client_factories(client_factories)

        # Auto-register transitive Function dependencies via BFS over .uses
        queue: Deque[Function] = deque(specs)
        while queue:
            fn: Function = queue.popleft()
            if not isinstance(fn, Function):
                raise TypeError(f"Runtime spec includes non-Function: {fn!r}")

            if fn.name in self._fn_by_name:
                if self._fn_by_name[fn.name] is fn:
                    continue
                else:
                    raise ValueError(
                        f"Duplicate Function name '{fn.name}' found during registration."
                    )

            self._functions.append(fn)
            self._fn_by_name[fn.name] = fn
            for dep in fn.uses:
                queue.append(dep)

        self._lock = Lock()
        self._next_node_id: int = 0
        self._roots: List[Node] = []
        self._nodes_by_id: Dict[int, Node] = {}
        self._providers: Dict[Provider, type[AgentNode]] = {}
        self._node_observables: Dict[int, NodeObservable] = {}
        self._global_seqno: int = 0

    @staticmethod
    def validate_client_factories(
        factories: Mapping[Provider, Callable[[], Any]],
    ) -> Dict[Provider, Callable[[], Any]]:
        validated: Dict[Provider, Callable[[], Any]] = {}
        for provider, factory in factories.items():
            if not isinstance(provider, Provider):
                raise TypeError(
                    "client_factories keys must be Provider instances; "
                    f"got {provider!r}"
                )
            if not callable(factory):
                raise TypeError(
                    "client_factories values must be callables that create SDK clients"
                )
            validated[provider] = factory
        return validated

    def get_ctx(self) -> RunContext:
        """Return a RunContext not tied to any specific Node
        (suitable for top-level Function invokes by users)."""
        return RunContext(runtime=self, node=None)

    def list_toplevel_views(self) -> List[NodeView]:
        """Return a snapshot of the latest NodeViews for all top-level tasks."""
        with self._lock:
            # Build a consistent snapshot of all root views at this moment
            return [self._node_observables[root.id].view for root in self._roots]

    def get_view(self, node_id: int) -> NodeView:
        """Return the latest NodeView snapshot for the given node id."""
        with self._lock:
            if node_id not in self._nodes_by_id:
                raise KeyError(f"No node with id {node_id}")
            return self._node_observables[node_id].view

    def invoke(
        self,
        caller: Optional[Node],
        fn: Function,
        inputs: Dict[str, Any],
        provider: Optional[Provider] = None,
        cancel_event: Optional[Event] = None,
    ) -> Node:
        """
        Create and start a Node for `fn` with `inputs`, recording parent/child relationships.
        Returns the created Node.

        If `cancel_event` is provided, it defines the cancellation scope for the new
        child `Node` that will be created. Otherwise the caller's CancelEvent (if any)
        is inherited automatically. This enables cooperative cancellation chaining,
        where the caller being canceled also cancels its children, but children can have
        further customized cancellation scope (e.g. timeouts).
        """
        # Ensure the function is registered.
        reg_fn = self._fn_by_name.get(fn.name)
        if reg_fn is None:
            raise ValueError(f"Function '{fn.name}' is not registered with this Runtime.")
        if reg_fn is not fn:
            raise ValueError(
                f"Invoked function '{fn.name}' is not registered with this Runtime "
                f"even though it shares a name with another Function that is registered."
            )

        inputs = fn.validate_coerce_args(inputs)

        with self._lock:
            node_id = self._next_node_id
            self._next_node_id += 1

        # Determine cancellation scope inheritance (per docstring).
        if not cancel_event and caller:
            cancel_event = caller.cancel_event

        # Create a per-invocation RunContext; node will be injected post-construction
        ctx = RunContext(runtime=self, node=None, cancel_event=cancel_event)

        # Choose Node subtype
        node: Node
        if isinstance(fn, CodeFunction):
            if provider is not None:
                raise ValueError(f"Provider override is only valid for AgentFunction; invoking CodeFunction '{fn.name}'.")
            node = CodeNode(ctx, node_id, fn, inputs, caller, cancel_event)

        elif isinstance(fn, AgentFunction):
            provider = provider or fn.default_model
            if provider not in self._providers:
                self._providers[provider] = get_AgentNode_impl(provider)
            impl: type[AgentNode] = self._providers[provider]
            factory = self._client_factories.get(provider)
            if factory is None:
                raise ValueError(
                    f"No client factory registered for provider '{provider.value}'. "
                    "Update Runtime(client_factories=...) to include this provider."
                )
            node = impl(ctx, node_id, fn, inputs, caller, cancel_event, factory)
        else:
            raise TypeError(f"Unknown Function subtype: {type(fn).__name__}")

        # Back-reference the Node on its own RunContext
        ctx.node = node
        ctx.object_bags = self._build_session_bags(node)

        # Register global node mapping
        with self._lock:
            self._nodes_by_id[node_id] = node
            if caller is None:
                self._roots.append(node)
            else:
                caller.children.append(node)

            self._global_seqno += 1  # Every state change bumps it.
            self._publish_tree_update(node)

        node.start()
        return node

    def _build_session_bags(self, node: Node) -> Dict[SessionScope, SessionBag]:
        current: Node = node
        while current.parent is not None:
            current = current.parent
        top_level_bag: SessionBag = current.session_bag

        bags: Dict[SessionScope, SessionBag] = {
            SessionScope.TopLevel: top_level_bag,
            SessionScope.Self: node.session_bag,
        }
        if node.parent is not None:
            bags[SessionScope.Parent] = node.parent.session_bag
        return bags

    def _ensure_observable(self, node: Node) -> NodeObservable:
        """Should only be used during `_publish_tree_update()` while holding lock."""
        observable = self._node_observables.get(node.id)
        if observable is None:
            observable = NodeObservable(
                cond=mp.Condition(self._lock),
                touch_seqno=self._global_seqno,
                view=self._build_node_view(node),
            )
            self._node_observables[node.id] = observable
        return observable

    def _build_node_view(self, node: Node) -> NodeView:
        """Should only be used during `_publish_tree_update()` while holding lock."""
        child_views = tuple(
            self._ensure_observable(child).view for child in node.children
        )
        usage: Optional[TokenUsage] = None
        if isinstance(node, AgentNode):
            usage = copy.deepcopy(node.token_usage)

        return NodeView(
            id=node.id,
            fn=node.fn,
            inputs=node.inputs,        # Immutable in Node lifetime.
            state=node.state,
            outputs=node.outputs,      # Safe to share ref to immutable outputs (once created and set).
            exception=node.exception,  # Ditto.
            children=child_views,
            usage=usage,
            started_at=node.started_at,
            ended_at=node.ended_at,
            update_seqnum=self._global_seqno,
        )

    def _publish_tree_update(self, node: Node) -> None:
        seq = self._global_seqno
        current: Optional[Node] = node
        while current is not None:
            observable = self._ensure_observable(current)
            assert observable.touch_seqno <= seq
            if observable.touch_seqno < seq:
                observable.view = self._build_node_view(current)
                observable.touch_seqno = seq
                observable.cond.notify_all()
            current = current.parent

    def watch(
        self,
        node: Union[Node, int],
        as_of_seq: int = 0,
        *,
        timeout: Optional[float] = None,
    ) -> Optional[NodeView]:
        """Block until a newer snapshot is available and return it.

        If ``timeout`` (seconds) is provided and elapses before a newer snapshot
        becomes available (i.e., ``touch_seqno > as_of_seq``), return ``None``.
        Mirrors the underlying condition semantics (no exception on timeout).
        """
        node_id = self._resolve_node(node)
        with self._lock:
            observable = self._node_observables[node_id]

            # Fast path: already newer than caller's as_of_seq
            if observable.touch_seqno > as_of_seq:
                return observable.view

            # Compute absolute deadline if a timeout is provided
            deadline: Optional[float] = None
            if timeout is not None:
                # Treat non-positive timeout as an immediate poll (no wait)
                if timeout <= 0:
                    return None
                deadline = time.monotonic() + timeout

            while observable.touch_seqno <= as_of_seq:
                remaining: Optional[float] = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                notified = observable.cond.wait(timeout=remaining)
                # Even if notified is False (timeout), loop condition above
                # guards correctness; on timeout the remaining <= 0 branch
                # will return None on the next iteration.

            return observable.view

    def _resolve_node(self, node: Union[Node, int]) -> int:
        if isinstance(node, Node):
            return node.id
        return node

    def post_status_update(self, node: Node, state: NodeState) -> None:
        with self._lock:
            self._global_seqno += 1
            # Record start time on first transition to Running
            if state is NodeState.Running and not node.started_at:
                node.started_at = time.time()
            node.state = state
            self._publish_tree_update(node)

    def post_success(self, node: Node, outputs: Any) -> None:
        with self._lock:
            self._global_seqno += 1
            node.outputs = outputs
            node.state = NodeState.Success
            if not node.ended_at:
                node.ended_at = time.time()
            self._publish_tree_update(node)
            node.done.set()

    def post_exception(self, node: Node, exception: Exception) -> None:
        with self._lock:
            self._global_seqno += 1
            node.exception = exception
            node.state = NodeState.Error
            if not node.ended_at:
                node.ended_at = time.time()
            self._publish_tree_update(node)
            node.done.set()

        # Log immediately so there is trace of it even if consumer never collects .result()
        logging.error(f"Node {node.id} ({node.fn.name}) faulted: {exception}")

    def post_cancel(
        self,
        node: Node,
        exception: Optional[CancellationException] = None,
    ) -> None:
        """`exception` can be provided to explain the reason for cancellation
        otherwise the Node is assigned a no-reason CancellationException."""
        with self._lock:
            self._global_seqno += 1
            node.exception = exception or CancellationException()
            node.state = NodeState.Canceled
            if not node.ended_at:
                node.ended_at = time.time()
            self._publish_tree_update(node)
            node.done.set()
