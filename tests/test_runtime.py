import logging
import queue
import threading
import time
import unittest
from typing import Any, Iterable, List, Optional, Union
from unittest.mock import patch
import multiprocessing as mp
from multiprocessing.synchronize import Event

from ..runtime import Runtime, NodeObservable
from ..core import (
    AgentFunction,
    AgentNode,
    CodeFunction,
    CodeNode,
    Function,
    Node,
    NodeState,
    NodeView,
    RunContext,
    SessionScope,
    TokenUsage,
    CancellationException,
)
from ..providers import Provider


def _make_code_callable(result: Any = None, *, start_event: Optional[threading.Event] = None,
                        proceed_event: Optional[threading.Event] = None) -> Any:
    """Factory for deterministic CodeFunction callables used in tests."""

    def _callable(ctx: RunContext) -> Any:
        if start_event is not None:
            start_event.set()
        if proceed_event is not None:
            if not proceed_event.wait(timeout=5):
                raise TimeoutError("proceed_event was not set in time")
        return result

    return _callable


def _make_code_function(name: str, *, callable=None, uses: Optional[Iterable[Function]] = None) -> CodeFunction:
    if callable is None:
        callable = _make_code_callable()
    return CodeFunction(
        name=name,
        desc=f"code fn {name}",
        args=[],
        callable=callable,
        uses=list(uses or []),
    )


def _make_agent_function(name: str, *, uses: Optional[Iterable[Function]] = None) -> AgentFunction:
    return AgentFunction(
        name=name,
        desc=f"agent fn {name}",
        args=[],
        system_prompt="system",
        user_prompt_template="prompt",
        uses=list(uses or []),
        default_model=Provider.Anthropic,
    )


class DummyFunction(Function):
    def __init__(self, name: str = "dummy") -> None:
        super().__init__(name=name, desc=f"dummy {name}", args=[])

    @property
    def uses(self) -> List[Function]:
        return []


class DummyNode(Node):
    def __init__(
        self,
        ctx: RunContext,
        id: int,
        fn: Function,
        inputs: dict[str, Any],
        parent: Optional[Node],
        cancel_event=None,
    ) -> None:
        super().__init__(ctx, id, fn, inputs, parent, cancel_event)

    def run(self) -> None:  # pragma: no cover - not executed in these tests
        pass


def _register_dummy_node(runtime: Runtime, node: Node) -> None:
    """Register a manually constructed Node with the Runtime (mirrors invoke setup).
    
    IMPORTANT: If this node has children in node.children, they must already be
    registered (have observables) before calling this function, otherwise
    _build_node_view will fail.
    """
    ctx = node.ctx
    assert ctx.node is node
    if not ctx.object_bags:
        ctx.object_bags = runtime._build_session_bags(node)
    with runtime._lock:
        runtime._nodes_by_id[node.id] = node
        if node.parent is None:
            runtime._roots.append(node)
        runtime._global_seqno += 1
        # Create NodeObservable with initial view
        runtime._node_observables[node.id] = NodeObservable(
            cond=mp.Condition(runtime._lock),
            touch_seqno=runtime._global_seqno,
            view=runtime._build_node_view(node),
        )


class TestRuntimeClientFactories(unittest.TestCase):
    def test_validate_client_factories_type_checks_keys_and_values(self) -> None:
        with self.assertRaisesRegex(TypeError, "Provider"):
            Runtime.validate_client_factories({"bad": lambda: None})  # type: ignore

        with self.assertRaisesRegex(TypeError, "callables"):
            Runtime.validate_client_factories({Provider.Anthropic: object()})  # type: ignore


class TestRuntimeRegistration(unittest.TestCase):
    def test_rejects_duplicate_function_names_in_seeds(self) -> None:
        fn1 = _make_code_function("dup", callable=_make_code_callable())
        fn2 = _make_code_function("dup", callable=_make_code_callable())
        with self.assertRaisesRegex(ValueError, "Duplicate Function name 'dup'"):
            Runtime([fn1, fn2], client_factories={Provider.Anthropic: lambda: None})

    def test_rejects_duplicate_function_names_across_transitives(self) -> None:
        shared_name = "child"
        child_dep = _make_code_function(shared_name)
        parent = _make_code_function("parent", uses=[child_dep])
        conflicting = _make_code_function(shared_name)
        with self.assertRaisesRegex(ValueError, f"Duplicate Function name '{shared_name}'"):
            Runtime([parent, conflicting], client_factories={Provider.Anthropic: lambda: None})


class TestRuntimeInvocation(unittest.TestCase):
    def test_invoke_rejects_unregistered_function(self) -> None:
        runtime = Runtime([], client_factories={})
        code_fn = _make_code_function("standalone")
        with self.assertRaisesRegex(ValueError, "is not registered"):
            runtime.invoke(None, code_fn, {})

    def test_invoke_rejects_name_collision_with_different_instance(self) -> None:
        fn = _make_code_function("registered")
        runtime = Runtime([fn], client_factories={})
        impostor = _make_code_function("registered")
        with self.assertRaisesRegex(ValueError, "shares a name"):
            runtime.invoke(None, impostor, {})

    def test_invoke_disallows_provider_override_for_code_function(self) -> None:
        fn = _make_code_function("noop")
        runtime = Runtime([fn], client_factories={Provider.Anthropic: lambda: None})
        with self.assertRaisesRegex(ValueError, "Provider override is only valid for AgentFunction"):
            runtime.invoke(None, fn, {}, provider=Provider.Anthropic)

    def test_invoke_creates_and_starts_code_node(self) -> None:
        start_event = threading.Event()
        proceed_event = threading.Event()

        fn = _make_code_function(
            "controlled",
            callable=_make_code_callable("done", start_event=start_event, proceed_event=proceed_event),
        )
        runtime = Runtime([fn], client_factories={})
        node = runtime.invoke(None, fn, {})
        self.assertIsInstance(node, CodeNode)
        self.assertTrue(start_event.wait(timeout=1), "code callable did not start")
        self.assertEqual(node.state, NodeState.Running)
        self.assertIsNotNone(node.thread)
        assert node.thread  # silences pylance
        self.assertTrue(node.thread.is_alive())

        proceed_event.set()
        self.assertEqual(node.result(), "done")
        self.assertEqual(node.state, NodeState.Success)
        self.assertEqual(node.outputs, "done")
        assert node.thread
        node.thread.join(timeout=1)
        self.assertFalse(node.thread.is_alive())

    def test_invoke_creates_agent_node_with_provider_impl_and_factory(self) -> None:
        agent_fn = _make_agent_function("agent")
        factory_result = object()
        factory_called = threading.Event()

        def factory() -> object:
            factory_called.set()
            return factory_result

        class FakeAgentNode(AgentNode):
            last_client: Optional[object] = None

            def __init__(
                self,
                ctx: RunContext,
                id: int,
                fn: Function,
                inputs: dict[str, Any],
                parent: Optional[Node],
                cancel_event=None,
                client_factory=None,
            ) -> None:
                super().__init__(ctx, id, fn, inputs, parent, cancel_event, client_factory)
                type(self).last_client = None

            def run(self) -> None:
                type(self).last_client = self.client_factory()
                self.ctx.post_success("agent-output")

            @property
            def token_usage(self) -> TokenUsage:
                return TokenUsage()

        FakeAgentNode.last_client = None

        with patch("netflux.runtime.get_AgentNode_impl", return_value=FakeAgentNode):
            runtime = Runtime([agent_fn], client_factories={Provider.Anthropic: factory})
            node = runtime.invoke(None, agent_fn, {})
            self.assertIsInstance(node, FakeAgentNode)
            self.assertTrue(factory_called.wait(timeout=1))
            self.assertEqual(node.result(), "agent-output")
            self.assertIs(FakeAgentNode.last_client, factory_result)
            assert isinstance(node, AgentNode)  # silences pylance
            self.assertIs(node.agent_fn, agent_fn)

    def test_invoke_links_parent_child_relationship(self) -> None:
        child_result = "child-value"

        def child_callable(ctx: RunContext) -> str:
            return child_result

        child_fn = _make_code_function("child", callable=child_callable)
        captured_children: List[Node] = []

        def parent_callable(ctx: RunContext) -> str:
            child_node = ctx.invoke(child_fn, {})
            captured_children.append(child_node)
            res = child_node.result()
            assert isinstance(res, str)
            return res

        parent_fn = _make_code_function("parent", callable=parent_callable, uses=[child_fn])
        runtime = Runtime([parent_fn], client_factories={})
        parent_node = runtime.invoke(None, parent_fn, {})
        self.assertEqual(parent_node.result(), child_result)
        self.assertEqual(len(parent_node.children), 1)
        child_node = captured_children[0]
        self.assertIs(child_node.parent, parent_node)
        self.assertEqual(parent_node.children[0], child_node)

    def test_invoke_initializes_session_bags(self) -> None:
        child_nodes: List[Node] = []

        def child_callable(ctx: RunContext) -> str:
            return "child"

        child_fn = _make_code_function("child", callable=child_callable)

        def parent_callable(ctx: RunContext) -> str:
            node = ctx.invoke(child_fn, {})
            child_nodes.append(node)
            res = node.result()
            assert isinstance(res, str)
            return res

        parent_fn = _make_code_function("parent", callable=parent_callable, uses=[child_fn])
        runtime = Runtime([parent_fn], client_factories={})
        parent_node = runtime.invoke(None, parent_fn, {})
        parent_node.result()
        self.assertEqual(set(parent_node.ctx.object_bags.keys()), {SessionScope.TopLevel, SessionScope.Self})
        self.assertIs(parent_node.ctx.object_bags[SessionScope.TopLevel], parent_node.session_bag)
        self.assertIs(parent_node.ctx.object_bags[SessionScope.Self], parent_node.session_bag)

        child_node = child_nodes[0]
        child_bags = child_node.ctx.object_bags
        self.assertEqual(
            set(child_bags.keys()),
            {SessionScope.TopLevel, SessionScope.Parent, SessionScope.Self},
        )
        self.assertIs(child_bags[SessionScope.Self], child_node.session_bag)
        self.assertIs(child_bags[SessionScope.Parent], parent_node.session_bag)
        self.assertIs(child_bags[SessionScope.TopLevel], parent_node.session_bag)

    def test_invoke_propagates_cancel_event_to_children(self) -> None:
        cancel_event = mp.Event()
        observed: List[Optional[Event]] = []

        def child_callable(ctx: RunContext) -> str:
            observed.append(ctx.cancel_event)
            return "child"

        child_fn = _make_code_function("child", callable=child_callable)

        def parent_callable(ctx: RunContext) -> str:
            observed.append(ctx.cancel_event)
            node = ctx.invoke(child_fn, {})
            node.result()
            return "parent"

        parent_fn = _make_code_function("parent", callable=parent_callable, uses=[child_fn])
        runtime = Runtime([parent_fn], client_factories={})
        parent_node = runtime.invoke(None, parent_fn, {}, cancel_event=cancel_event)
        self.assertEqual(parent_node.result(), "parent")

        self.assertIs(parent_node.ctx.cancel_event, cancel_event)
        self.assertEqual(len(parent_node.children), 1)
        child_node = parent_node.children[0]
        self.assertIs(child_node.ctx.cancel_event, cancel_event)
        self.assertEqual(len(observed), 2)
        self.assertIs(observed[0], cancel_event)
        self.assertIs(observed[1], cancel_event)

    def test_invoke_allows_cancel_event_override(self) -> None:
        parent_event = mp.Event()
        override_event = mp.Event()
        observed_child: List[Optional[Event]] = []

        def child_callable(ctx: RunContext) -> str:
            observed_child.append(ctx.cancel_event)
            return "child"

        child_fn = _make_code_function("child", callable=child_callable)

        def parent_callable(ctx: RunContext) -> str:
            self.assertIs(ctx.cancel_event, parent_event)
            node = ctx.invoke(child_fn, {}, cancel_event=override_event)
            node.result()
            return "parent"

        parent_fn = _make_code_function("parent", callable=parent_callable, uses=[child_fn])
        runtime = Runtime([parent_fn], client_factories={})
        node = runtime.invoke(None, parent_fn, {}, cancel_event=parent_event)
        node.result()

        self.assertEqual(len(observed_child), 1)
        self.assertIs(observed_child[0], override_event)
        self.assertEqual(len(node.children), 1)
        child_node = node.children[0]
        self.assertIs(child_node.ctx.cancel_event, override_event)

    def test_cancel_event_triggers_cancellation(self) -> None:
        cancel_event = mp.Event()
        started = threading.Event()

        def blocking_callable(ctx: RunContext) -> str:
            started.set()
            while not ctx.cancel_requested():
                time.sleep(0.01)
            raise CancellationException()

        fn = _make_code_function("blocking", callable=blocking_callable)
        runtime = Runtime([fn], client_factories={})
        node = runtime.invoke(None, fn, {}, cancel_event=cancel_event)

        self.assertTrue(started.wait(timeout=1), "code callable did not start")
        cancel_event.set()

        with self.assertRaises(CancellationException):
            node.result()

        self.assertEqual(node.state, NodeState.Canceled)
        self.assertIsNotNone(node.exception)
        self.assertIsInstance(node.exception, CancellationException)


class TestRuntimeObservability(unittest.TestCase):
    def test_list_toplevel_views_returns_snapshots(self) -> None:
        fn = _make_code_function("view", callable=lambda ctx: "output")
        runtime = Runtime([fn], client_factories={})
        node = runtime.invoke(None, fn, {})
        node.result()
        views = runtime.list_toplevel_views()
        self.assertEqual(len(views), 1)
        view = views[0]
        self.assertIsInstance(view, NodeView)
        self.assertEqual(view.id, node.id)
        self.assertEqual(view.state, NodeState.Success)
        self.assertEqual(view.outputs, "output")
        self.assertEqual(view.children, ())

    def test_watch_blocks_until_newer_seq(self) -> None:
        start_event = threading.Event()
        proceed_event = threading.Event()
        fn = _make_code_function(
            "watch",
            callable=_make_code_callable("done", start_event=start_event, proceed_event=proceed_event),
        )
        runtime = Runtime([fn], client_factories={})
        node = runtime.invoke(None, fn, {})
        first_view = node.watch()
        self.assertIsNotNone(first_view)
        assert first_view is not None  # type narrowing
        self.assertEqual(first_view.state, NodeState.Running)

        results: queue.Queue[NodeView] = queue.Queue()
        watcher_started = threading.Event()

        def watcher() -> None:
            watcher_started.set()
            view = runtime.watch(node, as_of_seq=first_view.update_seqnum)
            self.assertIsNotNone(view)
            assert view is not None  # type narrowing
            results.put(view)

        thread = threading.Thread(target=watcher, daemon=True)
        thread.start()
        self.assertTrue(watcher_started.wait(timeout=1))
        self.assertTrue(results.empty())

        proceed_event.set()
        node.result()
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        updated_view = results.get(timeout=1)
        self.assertGreater(updated_view.update_seqnum, first_view.update_seqnum)
        self.assertEqual(updated_view.state, NodeState.Success)


class TestRuntimeStateTransitions(unittest.TestCase):
    def _make_runtime_with_dummy_node(self, *, node_id: int = 1) -> tuple[Runtime, Node]:
        runtime = Runtime([], client_factories={})
        dummy_fn = DummyFunction(name=f"fn{node_id}")
        ctx = RunContext(runtime=runtime, node=None)
        node = DummyNode(ctx=ctx, id=node_id, fn=dummy_fn, inputs={}, parent=None)
        ctx.node = node
        _register_dummy_node(runtime, node)
        return runtime, node

    def test_post_status_update_mutates_state_and_notifies(self) -> None:
        runtime, node = self._make_runtime_with_dummy_node()
        initial_view = runtime.get_view(node.id)

        results: queue.Queue[NodeView] = queue.Queue()
        watcher_started = threading.Event()

        def watcher() -> None:
            watcher_started.set()
            view = runtime.watch(node.id, as_of_seq=initial_view.update_seqnum)
            self.assertIsNotNone(view)
            assert view is not None  # type narrowing
            results.put(view)

        thread = threading.Thread(target=watcher, daemon=True)
        thread.start()
        self.assertTrue(watcher_started.wait(timeout=1))
        self.assertTrue(results.empty())

        runtime.post_status_update(node, NodeState.Running)
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        updated = results.get(timeout=1)
        self.assertEqual(node.state, NodeState.Running)
        self.assertEqual(updated.state, NodeState.Running)
        self.assertGreater(updated.update_seqnum, initial_view.update_seqnum)

    def test_post_success_sets_outputs_and_marks_done(self) -> None:
        runtime, node = self._make_runtime_with_dummy_node(node_id=2)
        payload = {"ok": True}
        runtime.post_success(node, payload)
        self.assertEqual(node.state, NodeState.Success)
        self.assertIs(node.outputs, payload)
        self.assertTrue(node.done.is_set())
        view = runtime.get_view(node.id)
        self.assertEqual(view.state, NodeState.Success)
        self.assertIs(view.outputs, payload)

    def test_post_exception_sets_exception_and_logs(self) -> None:
        runtime, node = self._make_runtime_with_dummy_node(node_id=3)
        exc = RuntimeError("boom")
        with self.assertLogs(level=logging.ERROR) as captured:
            runtime.post_exception(node, exc)
        self.assertEqual(node.state, NodeState.Error)
        self.assertIs(node.exception, exc)
        self.assertTrue(node.done.is_set())
        view = runtime.get_view(node.id)
        self.assertEqual(view.state, NodeState.Error)
        self.assertIs(view.exception, exc)
        self.assertTrue(any("boom" in msg for msg in captured.output))

    def test_post_cancel_sets_canceled_state(self) -> None:
        runtime, node = self._make_runtime_with_dummy_node(node_id=4)
        runtime.post_cancel(node)

        self.assertEqual(node.state, NodeState.Canceled)
        self.assertIsNotNone(node.exception)
        self.assertIsInstance(node.exception, CancellationException)
        self.assertTrue(node.done.is_set())

        view = runtime.get_view(node.id)
        self.assertEqual(view.state, NodeState.Canceled)
        self.assertIsNotNone(view.exception)
        self.assertIsInstance(view.exception, CancellationException)

    def test_publish_viewtree_update_refreshes_ancestors(self) -> None:
        runtime = Runtime([], client_factories={})
        parent_ctx = RunContext(runtime=runtime, node=None)
        child_ctx = RunContext(runtime=runtime, node=None)
        parent = DummyNode(ctx=parent_ctx, id=10, fn=DummyFunction("parent"), inputs={}, parent=None)
        child = DummyNode(ctx=child_ctx, id=11, fn=DummyFunction("child"), inputs={}, parent=parent)
        parent_ctx.node = parent
        child_ctx.node = child
        # Register child first (so parent can reference it in its view)
        _register_dummy_node(runtime, child)
        parent.children.append(child)
        child.parent = parent
        _register_dummy_node(runtime, parent)

        parent_view_before = runtime.get_view(parent.id)
        child_view_before = runtime.get_view(child.id)

        with runtime._lock:
            runtime._global_seqno += 1
            child.state = NodeState.Running
            runtime._publish_viewtree_update(child)

        parent_view_after = runtime.get_view(parent.id)
        self.assertGreater(parent_view_after.update_seqnum, parent_view_before.update_seqnum)
        self.assertEqual(len(parent_view_after.children), 1)
        child_in_parent = parent_view_after.children[0]
        self.assertIsInstance(child_in_parent, NodeView)
        self.assertEqual(child_in_parent.state, NodeState.Running)
        self.assertGreater(child_in_parent.update_seqnum, child_view_before.update_seqnum)


class TestRuntimeWatchTimeout(unittest.TestCase):
    def test_watch_timeout_returns_none_without_update(self) -> None:
        start_event = threading.Event()
        proceed_event = threading.Event()
        fn = _make_code_function(
            "watch_timeout_none",
            callable=_make_code_callable("done", start_event=start_event, proceed_event=proceed_event),
        )
        runtime = Runtime([fn], client_factories={})
        node = runtime.invoke(None, fn, {})

        first_view = node.watch()
        self.assertIsNotNone(first_view)
        assert first_view is not None  # type narrowing
        self.assertIn(first_view.state, {NodeState.Running, NodeState.Waiting})

        results: queue.Queue[Any] = queue.Queue()

        def watcher() -> None:
            results.put(runtime.watch(node, as_of_seq=first_view.update_seqnum, timeout=0.05))

        t = threading.Thread(target=watcher, daemon=True)
        t.start()
        t.join(timeout=1)
        self.assertFalse(t.is_alive())
        res = results.get(timeout=1)
        self.assertIsNone(res)

        # Cleanup
        proceed_event.set()
        node.result()

    def test_watch_timeout_zero_polling_and_fast_path(self) -> None:
        start_event = threading.Event()
        proceed_event = threading.Event()
        fn = _make_code_function(
            "watch_timeout_zero",
            callable=_make_code_callable("done", start_event=start_event, proceed_event=proceed_event),
        )
        runtime = Runtime([fn], client_factories={})
        node = runtime.invoke(None, fn, {})

        first_view = node.watch()
        self.assertIsNotNone(first_view)
        assert first_view is not None  # type narrowing

        # Zero-timeout behaves like a non-blocking poll: no update => None
        res_none = node.watch(as_of_seq=first_view.update_seqnum, timeout=0)
        self.assertIsNone(res_none)

        # Now complete the node to create a newer snapshot
        proceed_event.set()
        node.result()

        # Zero-timeout should return immediately with the newer snapshot (fast path)
        res_now = node.watch(as_of_seq=first_view.update_seqnum, timeout=0)
        self.assertIsNotNone(res_now)
        assert res_now is not None  # satisfy type checkers
        self.assertEqual(res_now.state, NodeState.Success)

class TestNodeViewStructure(unittest.TestCase):
    def test_node_view_children_is_tuple_and_preserves_order(self) -> None:
        invocation_order: List[int] = []

        def child_callable_factory(idx: int):
            def callable(ctx: RunContext) -> str:
                invocation_order.append(idx)
                return f"child-{idx}"

            return callable

        child1 = _make_code_function("child1", callable=child_callable_factory(1))
        child2 = _make_code_function("child2", callable=child_callable_factory(2))

        def parent_callable(ctx: RunContext) -> str:
            ctx.invoke(child1, {}).result()
            ctx.invoke(child2, {}).result()
            return "parent"

        parent_fn = _make_code_function("parent", callable=parent_callable, uses=[child1, child2])
        runtime = Runtime([parent_fn], client_factories={})
        parent_node = runtime.invoke(None, parent_fn, {})
        parent_node.result()

        view = runtime.get_view(parent_node.id)
        self.assertIsInstance(view.children, tuple)
        self.assertEqual(len(view.children), 2)
        self.assertEqual(invocation_order, [1, 2])
        self.assertEqual(view.children[0].id, parent_node.children[0].id)
        self.assertEqual(view.children[1].id, parent_node.children[1].id)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
