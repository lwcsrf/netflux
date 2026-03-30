import gc
import logging
import os
import queue
import sys
import tempfile
import threading
import time
import unittest
import weakref
from pathlib import Path
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
    TokenBill,
    TokenUsage,
    ModelTextPart,
    CancellationException,
)
from ..func_lib.bash_func import Bash, BashSession
from ..func_lib.text_editor_func import TextEditor
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
            cond=threading.Condition(runtime._lock),
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

    def test_invocable_functions_exposes_registered_functions(self) -> None:
        child = _make_code_function("child")
        parent = _make_code_function("parent", uses=[child])
        sibling = _make_code_function("sibling")
        runtime = Runtime([parent, sibling], client_factories={})

        self.assertEqual(runtime.invocable_functions, (parent, sibling, child))


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
                tool_use_id=None,
            ) -> None:
                super().__init__(ctx, id, fn, inputs, parent, cancel_event, client_factory, tool_use_id)
                type(self).last_client = None

            def run(self) -> None:
                type(self).last_client = self.client_factory()
                self.ctx.post_success("agent-output")

            @property
            def token_usage(self) -> TokenUsage:
                return TokenUsage()

            @property
            def provider(self) -> Provider:
                return Provider.Anthropic

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
            self.assertEqual(runtime.get_view(node.id).provider, Provider.Anthropic)

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

    def test_invoke_returns_root_in_error_state_when_thread_start_fails(self) -> None:
        fn = _make_code_function("root_start_failure")
        runtime = Runtime([fn], client_factories={})
        orig_start = Node.start

        def fail_start(node: Node) -> None:
            if node.fn.name != "root_start_failure":
                orig_start(node)
                return
            if node.thread is not None:
                return
            node.thread = threading.Thread(
                target=node.run_wrapper,
                name=f"netflux-node-{node.id}",
                daemon=True,
            )
            raise RuntimeError("can't start new thread")

        with patch("netflux.core.Node.start", new=fail_start), self.assertLogs(
            "netflux.runtime", level=logging.ERROR
        ):
            node = runtime.invoke(None, fn, {})

        self.assertEqual(node.state, NodeState.Error)
        self.assertTrue(node.done.is_set())
        self.assertIsNone(node.thread)
        self.assertIsInstance(node.exception, RuntimeError)
        with self.assertRaisesRegex(RuntimeError, "can't start new thread"):
            node.result()

        view = runtime.get_view(node.id)
        self.assertEqual(view.state, NodeState.Error)
        self.assertIsInstance(view.exception, RuntimeError)

    def test_invoke_returns_child_in_error_state_when_thread_start_fails(self) -> None:
        child_fn = _make_code_function("child_start_failure")
        captured_child: dict[str, Node] = {}

        def parent_callable(ctx: RunContext) -> str:
            child = ctx.invoke(child_fn, {})
            captured_child["node"] = child
            return "parent"

        parent_fn = _make_code_function("parent", callable=parent_callable, uses=[child_fn])
        runtime = Runtime([parent_fn], client_factories={})
        orig_start = Node.start

        def fail_start(node: Node) -> None:
            if node.fn.name != "child_start_failure":
                orig_start(node)
                return
            if node.thread is not None:
                return
            node.thread = threading.Thread(
                target=node.run_wrapper,
                name=f"netflux-node-{node.id}",
                daemon=True,
            )
            raise RuntimeError("can't start new thread")

        with patch("netflux.core.Node.start", new=fail_start), self.assertLogs(
            "netflux.runtime", level=logging.ERROR
        ):
            parent_node = runtime.invoke(None, parent_fn, {})
            self.assertEqual(parent_node.result(), "parent")

        self.assertEqual(len(parent_node.children), 1)
        child_node = parent_node.children[0]
        self.assertIs(child_node, captured_child["node"])
        self.assertEqual(parent_node.state, NodeState.Success)
        self.assertTrue(parent_node.done.is_set())
        self.assertEqual(child_node.state, NodeState.Error)
        self.assertTrue(child_node.done.is_set())
        self.assertIsNone(child_node.thread)
        self.assertIsInstance(child_node.exception, RuntimeError)
        with self.assertRaisesRegex(RuntimeError, "can't start new thread"):
            child_node.result()

        child_view = runtime.get_view(child_node.id)
        self.assertEqual(child_view.state, NodeState.Error)
        self.assertIsInstance(child_view.exception, RuntimeError)

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

    def test_terminal_node_closes_its_session_bag_values(self) -> None:
        closed = threading.Event()

        class Closeable:
            def close(self) -> None:
                closed.set()

        def callable(ctx: RunContext) -> str:
            ctx.get_or_put(SessionScope.Self, "ns", "key", Closeable)
            return "done"

        fn = _make_code_function("cleanup", callable=callable)
        runtime = Runtime([fn], client_factories={})
        node = runtime.invoke(None, fn, {})

        self.assertEqual(node.result(), "done")
        assert node.thread is not None
        node.thread.join(timeout=1)
        self.assertTrue(closed.wait(timeout=1))
        self.assertTrue(node.session_bag._closed)

    def test_result_waits_for_terminal_cleanup_to_finish(self) -> None:
        close_started = threading.Event()
        closed = threading.Event()

        class Closeable:
            def close(self) -> None:
                close_started.set()
                time.sleep(0.1)
                closed.set()

        def callable(ctx: RunContext) -> str:
            ctx.get_or_put(SessionScope.Self, "cleanup.wait", "resource", Closeable)
            return "done"

        fn = _make_code_function("cleanup_waits", callable=callable)
        runtime = Runtime([fn], client_factories={})
        node = runtime.invoke(None, fn, {})

        self.assertEqual(node.result(), "done")
        self.assertTrue(close_started.is_set())
        self.assertTrue(closed.is_set())
        self.assertTrue(node.session_bag._closed)

    def test_terminal_node_cleans_up_self_bag_across_terminal_outcomes(self) -> None:
        for outcome in ("success", "error", "cancel"):
            with self.subTest(outcome=outcome):
                closed = threading.Event()
                started = threading.Event()

                class Closeable:
                    def __init__(self) -> None:
                        self.close_calls = 0

                    def close(self) -> None:
                        self.close_calls += 1
                        closed.set()

                resource = Closeable()
                cancel_event = threading.Event() if outcome == "cancel" else None

                def callable(ctx: RunContext) -> str:
                    ctx.get_or_put(SessionScope.Self, "cleanup.self", "resource", lambda: resource)
                    started.set()
                    if outcome == "success":
                        return "done"
                    if outcome == "error":
                        raise RuntimeError("boom")
                    while not ctx.cancel_requested():
                        time.sleep(0.01)
                    raise CancellationException("stop")

                fn = _make_code_function(f"cleanup_{outcome}", callable=callable)
                runtime = Runtime([fn], client_factories={})
                node = runtime.invoke(None, fn, {}, cancel_event=cancel_event)

                if outcome == "success":
                    self.assertEqual(node.result(), "done")
                elif outcome == "error":
                    with self.assertRaisesRegex(RuntimeError, "boom"):
                        node.result()
                else:
                    self.assertTrue(started.wait(timeout=1))
                    assert cancel_event is not None
                    cancel_event.set()
                    with self.assertRaises(CancellationException):
                        node.result()

                assert node.thread is not None
                node.thread.join(timeout=1)
                self.assertTrue(closed.wait(timeout=1))
                self.assertEqual(resource.close_calls, 1)
                self.assertTrue(node.session_bag._closed)
                self.assertEqual(node.session_bag._values, {})
                with self.assertRaisesRegex(RuntimeError, "terminally closed"):
                    node.session_bag.get_or_put("cleanup.self", "late", lambda: object())

    def test_terminal_agent_closes_parent_scoped_bash_session(self) -> None:
        bash_fn = Bash()
        agent_fn = _make_agent_function("agent_bash_cleanup", uses=[bash_fn])
        captured: dict[str, Any] = {}

        def linux_proc_identity(pid: int) -> Optional[tuple[str, str]]:
            stat_path = Path(f"/proc/{pid}/stat")
            try:
                stat = stat_path.read_text(encoding="utf-8")
            except FileNotFoundError:
                return None

            stat_tail_idx = stat.rfind(")")
            self.assertNotEqual(stat_tail_idx, -1, f"unexpected /proc/{pid}/stat format: {stat!r}")

            fields = stat[stat_tail_idx + 2:].split()
            self.assertGreaterEqual(
                len(fields),
                20,
                f"unexpected /proc/{pid}/stat field count: {len(fields)}",
            )
            state = fields[0]
            start_time = fields[19]
            return state, start_time

        class FakeAgentNode(AgentNode):
            @property
            def token_usage(self) -> TokenUsage:
                return TokenUsage()

            @property
            def provider(self) -> Provider:
                return Provider.Anthropic

            def run(self) -> None:
                child = self.ctx.invoke(
                    bash_fn,
                    {"command": "echo hello from bash", "session_id": 7},
                    tool_use_id="tool-1",
                )
                result = child.result()
                session = self.session_bag._values["bash.session"][bash_fn._bag_key(7)]
                proc = session._proc
                assert proc is not None
                captured["session"] = session
                captured["proc"] = proc
                captured["pid"] = proc.pid
                if sys.platform.startswith("linux") and Path("/proc").is_dir():
                    captured["linux_proc_identity_before_cleanup"] = linux_proc_identity(proc.pid)
                captured["parent_bag_populated_before_cleanup"] = bool(self.session_bag._values)
                captured["child_bag_empty"] = child.session_bag._values == {}
                self.ctx.post_success(result)

        with patch("netflux.runtime.get_AgentNode_impl", return_value=FakeAgentNode):
            runtime = Runtime([agent_fn], client_factories={Provider.Anthropic: lambda: object()})
            node = runtime.invoke(None, agent_fn, {})
            self.assertEqual(node.result().strip(), "hello from bash")
            assert node.thread is not None
            node.thread.join(timeout=2)

        session = captured["session"]
        proc = captured["proc"]
        pid = captured["pid"]
        self.assertIsInstance(session, BashSession)
        self.assertTrue(captured["parent_bag_populated_before_cleanup"])
        self.assertTrue(captured["child_bag_empty"])
        self.assertFalse(session.alive())
        self.assertIsNotNone(proc.returncode)
        self.assertIsNone(session._proc)
        self.assertIsNone(session._stdout_thread)
        self.assertFalse(session.requires_restart)
        self.assertFalse(session._alive_once_started)
        self.assertTrue(node.session_bag._closed)
        self.assertEqual(node.session_bag._values, {})

        if os.name == "posix":
            with self.assertRaises(ChildProcessError):
                os.waitpid(pid, os.WNOHANG)

        if sys.platform.startswith("linux") and Path("/proc").is_dir():
            before = captured["linux_proc_identity_before_cleanup"]
            self.assertIsNotNone(before)
            after = linux_proc_identity(pid)
            if after is not None:
                self.assertNotEqual(
                    after[1],
                    before[1],
                    f"/proc/{pid}/stat still identifies the original bash process after cleanup "
                    f"(state={after[0]!r})",
                )

    def test_terminal_root_clears_top_level_text_editor_lock_created_by_descendant(self) -> None:
        editor_fn = TextEditor()
        captured: dict[str, Any] = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "notes.txt"
            target.write_text("before\n", encoding="utf-8")

            def grandchild_callable(ctx: RunContext) -> str:
                editor_node = ctx.invoke(
                    editor_fn,
                    {
                        "command": "str_replace",
                        "path": str(target),
                        "old_str": "before\n",
                        "new_str": "after\n",
                    },
                )
                result = editor_node.result()
                top_bag = ctx.object_bags[SessionScope.TopLevel]
                lock_key = editor_fn._file_lock_key(target.resolve())
                lock = top_bag._values[editor_fn._FILE_LOCK_NAMESPACE][lock_key]
                captured["top_level_bag_id"] = id(top_bag)
                captured["lock_ref"] = weakref.ref(lock)
                captured["lock_keys_before_cleanup"] = tuple(
                    top_bag._values[editor_fn._FILE_LOCK_NAMESPACE].keys()
                )
                return result

            grandchild_fn = _make_code_function(
                "grandchild_editor_cleanup",
                callable=grandchild_callable,
                uses=[editor_fn],
            )

            def child_callable(ctx: RunContext) -> str:
                return ctx.invoke(grandchild_fn, {}).result()

            child_fn = _make_code_function(
                "child_editor_cleanup",
                callable=child_callable,
                uses=[grandchild_fn],
            )

            def root_callable(ctx: RunContext) -> str:
                return ctx.invoke(child_fn, {}).result()

            root_fn = _make_code_function(
                "root_editor_cleanup",
                callable=root_callable,
                uses=[child_fn],
            )

            runtime = Runtime([root_fn], client_factories={})
            root_node = runtime.invoke(None, root_fn, {})
            self.assertEqual(root_node.result(), "Replace successful.")
            assert root_node.thread is not None
            root_node.thread.join(timeout=1)

            self.assertEqual(target.read_text(encoding="utf-8"), "after\n")
            self.assertEqual(captured["top_level_bag_id"], id(root_node.session_bag))
            self.assertTrue(captured["lock_keys_before_cleanup"])
            self.assertTrue(root_node.session_bag._closed)
            self.assertEqual(root_node.session_bag._values, {})

            gc.collect()
            self.assertIsNone(captured["lock_ref"]())

    def test_descendant_can_use_ancestor_session_bags_while_ancestor_terminalization_is_deferred(self) -> None:
        root_returning = threading.Event()
        child_returning = threading.Event()
        grandchild_started = threading.Event()
        allow_grandchild_finish = threading.Event()
        observed: dict[str, Any] = {}

        def grandchild_callable(ctx: RunContext) -> str:
            grandchild_started.set()
            self.assertTrue(root_returning.wait(timeout=1))
            self.assertTrue(child_returning.wait(timeout=1))

            assert ctx.node is not None
            assert ctx.node.parent is not None
            root_node = ctx.node.parent.parent
            child_node = ctx.node.parent
            assert root_node is not None

            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                if (
                    root_node.state is NodeState.Running
                    and child_node.state is NodeState.Running
                    and not root_node.done.is_set()
                    and not child_node.done.is_set()
                ):
                    break
                time.sleep(0.01)
            else:
                self.fail("Ancestors did not remain Running while terminalization was deferred")

            top_bag = ctx.object_bags[SessionScope.TopLevel]
            parent_bag = ctx.object_bags[SessionScope.Parent]
            observed["root_bag_closed_before_access"] = top_bag._closed
            observed["child_bag_closed_before_access"] = parent_bag._closed
            observed["root_value"] = ctx.get_or_put(
                SessionScope.TopLevel,
                "deferred.cleanup",
                "root",
                lambda: "wrong-root",
            )
            observed["child_value"] = ctx.get_or_put(
                SessionScope.Parent,
                "deferred.cleanup",
                "child",
                lambda: "wrong-child",
            )

            self.assertTrue(allow_grandchild_finish.wait(timeout=5))
            return "grandchild-done"

        grandchild_fn = _make_code_function(
            "grandchild_deferred_bag",
            callable=grandchild_callable,
        )

        def child_callable(ctx: RunContext) -> str:
            ctx.get_or_put(
                SessionScope.Self,
                "deferred.cleanup",
                "child",
                lambda: "child-value",
            )
            ctx.invoke(grandchild_fn, {})
            child_returning.set()
            return "child-done"

        child_fn = _make_code_function(
            "child_deferred_bag",
            callable=child_callable,
            uses=[grandchild_fn],
        )

        def root_callable(ctx: RunContext) -> str:
            ctx.get_or_put(
                SessionScope.Self,
                "deferred.cleanup",
                "root",
                lambda: "root-value",
            )
            ctx.invoke(child_fn, {})
            root_returning.set()
            return "root-done"

        root_fn = _make_code_function(
            "root_deferred_bag",
            callable=root_callable,
            uses=[child_fn],
        )

        runtime = Runtime([root_fn], client_factories={})
        root_node = runtime.invoke(None, root_fn, {})

        self.assertTrue(grandchild_started.wait(timeout=1))
        self.assertTrue(root_returning.wait(timeout=1))
        self.assertTrue(child_returning.wait(timeout=1))

        self.assertEqual(root_node.state, NodeState.Running)
        self.assertFalse(root_node.done.is_set())
        child_node = root_node.children[0]
        grandchild_node = child_node.children[0]
        self.assertEqual(child_node.state, NodeState.Running)
        self.assertFalse(child_node.done.is_set())

        allow_grandchild_finish.set()

        self.assertEqual(root_node.result(), "root-done")
        self.assertEqual(child_node.result(), "child-done")
        self.assertEqual(grandchild_node.result(), "grandchild-done")

        assert root_node.thread is not None
        assert child_node.thread is not None
        assert grandchild_node.thread is not None
        root_node.thread.join(timeout=1)
        child_node.thread.join(timeout=1)
        grandchild_node.thread.join(timeout=1)

        self.assertFalse(observed["root_bag_closed_before_access"])
        self.assertFalse(observed["child_bag_closed_before_access"])
        self.assertEqual(observed["root_value"], "root-value")
        self.assertEqual(observed["child_value"], "child-value")
        self.assertTrue(root_node.session_bag._closed)
        self.assertTrue(child_node.session_bag._closed)
        self.assertTrue(grandchild_node.session_bag._closed)

    def test_terminal_root_implies_descendant_subtree_is_terminal(self) -> None:
        root_returning = threading.Event()
        child_returning = threading.Event()
        grandchild_started = threading.Event()
        allow_grandchild_finish = threading.Event()

        def grandchild_callable(ctx: RunContext) -> str:
            grandchild_started.set()
            self.assertTrue(allow_grandchild_finish.wait(timeout=5))
            return "grandchild-done"

        grandchild_fn = _make_code_function(
            "grandchild_subtree_terminal",
            callable=grandchild_callable,
        )

        def child_callable(ctx: RunContext) -> str:
            ctx.invoke(grandchild_fn, {})
            child_returning.set()
            return "child-done"

        child_fn = _make_code_function(
            "child_subtree_terminal",
            callable=child_callable,
            uses=[grandchild_fn],
        )

        def root_callable(ctx: RunContext) -> str:
            ctx.invoke(child_fn, {})
            root_returning.set()
            return "root-done"

        root_fn = _make_code_function(
            "root_subtree_terminal",
            callable=root_callable,
            uses=[child_fn],
        )

        runtime = Runtime([root_fn], client_factories={})
        root_node = runtime.invoke(None, root_fn, {})

        self.assertTrue(grandchild_started.wait(timeout=1))
        self.assertTrue(root_returning.wait(timeout=1))
        self.assertTrue(child_returning.wait(timeout=1))

        child_node = root_node.children[0]
        grandchild_node = child_node.children[0]

        self.assertEqual(root_node.state, NodeState.Running)
        self.assertEqual(child_node.state, NodeState.Running)
        self.assertEqual(grandchild_node.state, NodeState.Running)
        self.assertFalse(root_node.done.is_set())
        self.assertFalse(child_node.done.is_set())
        self.assertFalse(grandchild_node.done.is_set())

        allow_grandchild_finish.set()

        self.assertEqual(root_node.result(), "root-done")
        self.assertEqual(child_node.result(), "child-done")
        self.assertEqual(grandchild_node.result(), "grandchild-done")
        self.assertEqual(root_node.state, NodeState.Success)
        self.assertEqual(child_node.state, NodeState.Success)
        self.assertEqual(grandchild_node.state, NodeState.Success)
        self.assertTrue(root_node.done.is_set())
        self.assertTrue(child_node.done.is_set())
        self.assertTrue(grandchild_node.done.is_set())

    def test_watch_keeps_node_running_while_terminalization_is_blocked_on_child(self) -> None:
        child_started = threading.Event()
        root_returning = threading.Event()
        child_release = threading.Event()

        child_fn = _make_code_function(
            "child_watch_running",
            callable=_make_code_callable(
                "child-done",
                start_event=child_started,
                proceed_event=child_release,
            ),
        )

        def root_callable(ctx: RunContext) -> str:
            ctx.invoke(child_fn, {})
            root_returning.set()
            return "root-done"

        root_fn = _make_code_function(
            "root_watch_running",
            callable=root_callable,
            uses=[child_fn],
        )

        runtime = Runtime([root_fn], client_factories={})
        root_node = runtime.invoke(None, root_fn, {})

        self.assertTrue(child_started.wait(timeout=1))
        self.assertTrue(root_returning.wait(timeout=1))
        time.sleep(0.05)

        running_view = runtime.get_view(root_node.id)
        self.assertEqual(running_view.state, NodeState.Running)
        self.assertFalse(root_node.done.is_set())
        self.assertIsNone(root_node.watch(as_of_seq=running_view.update_seqnum, timeout=0.05))
        self.assertEqual(runtime.get_view(root_node.id).state, NodeState.Running)

        child_release.set()
        self.assertEqual(root_node.result(), "root-done")
        self.assertEqual(runtime.get_view(root_node.id).state, NodeState.Success)

    def test_terminalization_waits_for_direct_children_across_terminal_outcomes(self) -> None:
        for outcome, expected_state in (
            ("success", NodeState.Success),
            ("error", NodeState.Error),
            ("cancel", NodeState.Canceled),
        ):
            with self.subTest(outcome=outcome):
                child_started = threading.Event()
                child_release = threading.Event()
                cancel_event = threading.Event() if outcome == "cancel" else None

                child_fn = _make_code_function(
                    f"child_wait_{outcome}",
                    callable=_make_code_callable(
                        "child-done",
                        start_event=child_started,
                        proceed_event=child_release,
                    ),
                )

                def parent_callable(ctx: RunContext) -> str:
                    ctx.invoke(child_fn, {})
                    if outcome == "success":
                        return "parent-done"
                    if outcome == "error":
                        raise RuntimeError("parent boom")
                    while not ctx.cancel_requested():
                        time.sleep(0.01)
                    raise CancellationException("parent cancel")

                parent_fn = _make_code_function(
                    f"parent_wait_{outcome}",
                    callable=parent_callable,
                    uses=[child_fn],
                )

                runtime = Runtime([parent_fn], client_factories={})
                parent_node = runtime.invoke(None, parent_fn, {}, cancel_event=cancel_event)

                self.assertTrue(child_started.wait(timeout=1))
                if cancel_event is not None:
                    cancel_event.set()
                time.sleep(0.05)

                self.assertEqual(parent_node.state, NodeState.Running)
                self.assertFalse(parent_node.done.is_set())
                self.assertEqual(len(parent_node.children), 1)
                child_node = parent_node.children[0]
                self.assertFalse(child_node.done.is_set())

                child_release.set()

                if outcome == "success":
                    self.assertEqual(parent_node.result(), "parent-done")
                elif outcome == "error":
                    with self.assertRaisesRegex(RuntimeError, "parent boom"):
                        parent_node.result()
                else:
                    with self.assertRaises(CancellationException):
                        parent_node.result()

                self.assertEqual(parent_node.state, expected_state)
                self.assertTrue(parent_node.done.is_set())
                self.assertTrue(child_node.done.is_set())

    def test_agent_node_preserves_explicit_terminal_outcome_against_wrapper_exception(self) -> None:
        agent_fn = _make_agent_function("agent_terminal_immutability")

        class FakeAgentNode(AgentNode):
            def run(self) -> None:
                self.ctx.post_success("early")
                raise RuntimeError("late boom")

            @property
            def token_usage(self) -> TokenUsage:
                return TokenUsage()

            @property
            def provider(self) -> Provider:
                return Provider.Anthropic

        with patch("netflux.runtime.get_AgentNode_impl", return_value=FakeAgentNode):
            runtime = Runtime([agent_fn], client_factories={Provider.Anthropic: lambda: object()})
            with self.assertLogs("netflux.runtime", level=logging.ERROR) as captured:
                node = runtime.invoke(None, agent_fn, {})
                self.assertEqual(node.result(), "early")

            assert node.thread is not None
            node.thread.join(timeout=1)

        self.assertEqual(node.state, NodeState.Success)
        self.assertIsNone(node.exception)
        self.assertTrue(
            any("post_exception" in msg and "has no effect and is ignored" in msg for msg in captured.output)
        )

    def test_agent_node_ignores_post_terminal_transcript_update(self) -> None:
        agent_fn = _make_agent_function("agent_transcript_immutability")

        class FakeAgentNode(AgentNode):
            def run(self) -> None:
                self.ctx.post_success("early")
                self.transcript.append(ModelTextPart(text="late"))
                self.ctx.post_transcript_update()

            @property
            def token_usage(self) -> TokenUsage:
                return TokenUsage()

            @property
            def provider(self) -> Provider:
                return Provider.Anthropic

        with patch("netflux.runtime.get_AgentNode_impl", return_value=FakeAgentNode):
            runtime = Runtime([agent_fn], client_factories={Provider.Anthropic: lambda: object()})
            with self.assertLogs("netflux.runtime", level=logging.ERROR) as captured:
                node = runtime.invoke(None, agent_fn, {})
                self.assertEqual(node.result(), "early")

            assert node.thread is not None
            node.thread.join(timeout=1)

        view = runtime.get_view(node.id)
        self.assertEqual(view.state, NodeState.Success)
        self.assertEqual(view.transcript, ())
        self.assertTrue(
            any(
                "post_transcript_update" in msg and "has no effect and is ignored" in msg
                for msg in captured.output
            )
        )

    def test_code_node_preserves_explicit_terminal_outcome(self) -> None:
        cases = (
            (
                "success_then_return",
                lambda ctx: (ctx.post_success("early"), "late")[1],
                NodeState.Success,
                "early",
                None,
            ),
            (
                "exception_then_return",
                lambda ctx: (ctx.post_exception(RuntimeError("early boom")), "late")[1],
                NodeState.Error,
                None,
                RuntimeError,
            ),
            (
                "cancel_then_return",
                lambda ctx: (
                    ctx.post_cancel(CancellationException("early cancel")),
                    "late",
                )[1],
                NodeState.Canceled,
                None,
                CancellationException,
            ),
            (
                "success_then_raise",
                lambda ctx: (_ for _ in ()).throw(RuntimeError("late boom")),
                NodeState.Success,
                "early",
                None,
            ),
        )

        for name, body, expected_state, expected_output, expected_exc_type in cases:
            with self.subTest(case=name):
                def callable(ctx: RunContext) -> Any:
                    if name == "success_then_raise":
                        ctx.post_success("early")
                    return body(ctx)

                fn = _make_code_function(name, callable=callable)
                runtime = Runtime([fn], client_factories={})
                with self.assertLogs("netflux.runtime", level=logging.ERROR) as captured:
                    node = runtime.invoke(None, fn, {})

                    if expected_exc_type is None:
                        self.assertEqual(node.result(), expected_output)
                    elif expected_exc_type is RuntimeError:
                        with self.assertRaisesRegex(RuntimeError, "early boom"):
                            node.result()
                    else:
                        with self.assertRaisesRegex(CancellationException, "early cancel"):
                            node.result()

                    assert node.thread is not None
                    node.thread.join(timeout=1)
                self.assertEqual(node.state, expected_state)
                self.assertTrue(any("has no effect and is ignored" in msg for msg in captured.output))

    def test_invoke_rejects_child_after_explicit_terminal_post_success_or_exception(self) -> None:
        child_fn = _make_code_function("late_child")
        cases = (
            (
                "success",
                lambda ctx: ctx.post_success("early"),
                NodeState.Success,
                "early",
                None,
            ),
            (
                "exception",
                lambda ctx: ctx.post_exception(RuntimeError("early boom")),
                NodeState.Error,
                None,
                RuntimeError,
            ),
        )

        for name, post_terminal, expected_state, expected_output, expected_exc_type in cases:
            with self.subTest(case=name):
                def parent_callable(ctx: RunContext) -> str:
                    post_terminal(ctx)
                    with self.assertRaisesRegex(RuntimeError, "terminal node"):
                        ctx.invoke(child_fn, {})
                    return "late"

                parent_fn = _make_code_function(
                    f"terminal_parent_{name}",
                    callable=parent_callable,
                    uses=[child_fn],
                )
                runtime = Runtime([parent_fn], client_factories={})
                with self.assertLogs("netflux.runtime", level=logging.ERROR) as captured:
                    parent_node = runtime.invoke(None, parent_fn, {})

                    if expected_exc_type is None:
                        self.assertEqual(parent_node.result(), expected_output)
                    else:
                        with self.assertRaisesRegex(RuntimeError, "early boom"):
                            parent_node.result()

                assert parent_node.thread is not None
                parent_node.thread.join(timeout=1)
                self.assertEqual(parent_node.state, expected_state)
                self.assertEqual(parent_node.children, [])
                self.assertTrue(any("has no effect and is ignored" in msg for msg in captured.output))

    def test_invoke_rejects_terminal_agent_child_before_construction(self) -> None:
        agent_fn = _make_agent_function("late_agent")
        observed = {"factory_calls": 0, "agent_inits": 0}

        class FakeAgentNode(AgentNode):
            def __init__(
                self,
                ctx: RunContext,
                id: int,
                fn: Function,
                inputs: dict[str, Any],
                parent: Optional[Node],
                cancel_event=None,
                client_factory=None,
                tool_use_id=None,
            ) -> None:
                observed["agent_inits"] += 1
                super().__init__(ctx, id, fn, inputs, parent, cancel_event, client_factory, tool_use_id)
                assert client_factory is not None
                self.client = client_factory()

            def run(self) -> None:
                self.ctx.post_success("agent-output")

            @property
            def token_usage(self) -> TokenUsage:
                return TokenUsage()

            @property
            def provider(self) -> Provider:
                return Provider.Anthropic

        def factory() -> object:
            observed["factory_calls"] += 1
            return object()

        def parent_callable(ctx: RunContext) -> str:
            ctx.post_success("early")
            with self.assertRaisesRegex(RuntimeError, "terminal node"):
                ctx.invoke(agent_fn, {})
            return "late"

        parent_fn = _make_code_function(
            "terminal_parent_agent_child",
            callable=parent_callable,
            uses=[agent_fn],
        )

        with patch("netflux.runtime.get_AgentNode_impl", return_value=FakeAgentNode):
            runtime = Runtime([parent_fn], client_factories={Provider.Anthropic: factory})
            parent_node = runtime.invoke(None, parent_fn, {})
            self.assertEqual(parent_node.result(), "early")

        assert parent_node.thread is not None
        parent_node.thread.join(timeout=1)
        self.assertEqual(parent_node.children, [])
        self.assertEqual(observed["agent_inits"], 0)
        self.assertEqual(observed["factory_calls"], 0)


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

    def test_post_running_transitions_waiting_to_running_and_notifies(self) -> None:
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

        runtime.post_running(node)
        thread.join(timeout=1)
        self.assertFalse(thread.is_alive())
        updated = results.get(timeout=1)
        self.assertEqual(node.state, NodeState.Running)
        self.assertEqual(updated.state, NodeState.Running)
        self.assertGreater(updated.update_seqnum, initial_view.update_seqnum)

    def test_post_running_is_noop_when_already_running(self) -> None:
        runtime, node = self._make_runtime_with_dummy_node()
        runtime.post_running(node)
        initial_view = runtime.get_view(node.id)

        runtime.post_running(node)

        self.assertEqual(node.state, NodeState.Running)
        self.assertEqual(runtime.get_view(node.id), initial_view)
        self.assertIsNone(runtime.watch(node.id, as_of_seq=initial_view.update_seqnum, timeout=0.01))

    def test_post_running_ignores_terminal_node(self) -> None:
        runtime, node = self._make_runtime_with_dummy_node()
        runtime.post_success(node, "ok")
        initial_view = runtime.get_view(node.id)

        with self.assertLogs("netflux.runtime", level=logging.ERROR) as captured:
            runtime.post_running(node)

        self.assertEqual(node.state, NodeState.Success)
        self.assertEqual(node.outputs, "ok")
        self.assertTrue(node.done.is_set())
        self.assertEqual(runtime.get_view(node.id), initial_view)
        self.assertTrue(
            any("post_running" in msg and "no effect and is ignored" in msg for msg in captured.output)
        )

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

    def test_terminal_posts_ignore_later_terminal_transitions(self) -> None:
        cases = (
            (
                "success_then_exception",
                lambda runtime, node: runtime.post_success(node, "ok"),
                lambda runtime, node: runtime.post_exception(node, RuntimeError("late boom")),
                NodeState.Success,
                "ok",
                None,
                "post_exception",
            ),
            (
                "error_then_cancel",
                lambda runtime, node: runtime.post_exception(node, RuntimeError("boom")),
                lambda runtime, node: runtime.post_cancel(node, CancellationException("late cancel")),
                NodeState.Error,
                None,
                RuntimeError,
                "post_cancel",
            ),
            (
                "canceled_then_success",
                lambda runtime, node: runtime.post_cancel(node, CancellationException("cancel")),
                lambda runtime, node: runtime.post_success(node, "late"),
                NodeState.Canceled,
                None,
                CancellationException,
                "post_success",
            ),
        )

        for name, first_post, second_post, expected_state, expected_output, expected_exc_type, ignored_call in cases:
            with self.subTest(case=name):
                runtime, node = self._make_runtime_with_dummy_node()
                first_post(runtime, node)
                first_exception = node.exception

                with self.assertLogs("netflux.runtime", level=logging.ERROR) as captured:
                    second_post(runtime, node)

                self.assertEqual(node.state, expected_state)
                self.assertEqual(node.outputs, expected_output)
                if expected_exc_type is None:
                    self.assertIsNone(node.exception)
                else:
                    self.assertIs(node.exception, first_exception)
                    self.assertIsInstance(node.exception, expected_exc_type)
                self.assertTrue(node.done.is_set())
                self.assertTrue(
                    any(ignored_call in msg and "has no effect and is ignored" in msg for msg in captured.output)
                )

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
        with self.assertRaises(TypeError):
            child_view_before.transcript_child_map[0] = child_view_before

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
        with self.assertRaises(TypeError):
            parent_view_after.transcript_child_map[0] = child_in_parent


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

    def test_total_tree_token_bill_groups_by_provider(self) -> None:
        agent_fn = _make_agent_function("agent")

        class FakeAnthropicAgentNode(AgentNode):
            @property
            def token_usage(self) -> TokenUsage:
                return TokenUsage(
                    input_tokens_cache_read=1,
                    input_tokens_cache_write=2,
                    input_tokens_regular=3,
                    output_tokens_total=4,
                )

            @property
            def provider(self) -> Provider:
                return Provider.Anthropic

            def run(self) -> None:
                self.ctx.post_success("anthropic")

        class FakeGeminiAgentNode(AgentNode):
            @property
            def token_usage(self) -> TokenUsage:
                return TokenUsage(
                    input_tokens_cache_read=10,
                    input_tokens_regular=20,
                    output_tokens_total=30,
                )

            @property
            def provider(self) -> Provider:
                return Provider.Gemini

            def run(self) -> None:
                self.ctx.post_success("gemini")

        def fake_impl(provider: Provider) -> type[AgentNode]:
            if provider == Provider.Anthropic:
                return FakeAnthropicAgentNode
            if provider == Provider.Gemini:
                return FakeGeminiAgentNode
            raise ValueError(provider)

        def parent_callable(ctx: RunContext) -> str:
            default_node = ctx.invoke(agent_fn, {})
            gemini_node = ctx.invoke(agent_fn, {}, provider=Provider.Gemini)
            self.assertEqual(default_node.result(), "anthropic")
            self.assertEqual(gemini_node.result(), "gemini")
            return "parent"

        parent_fn = _make_code_function("parent", callable=parent_callable, uses=[agent_fn])

        with patch("netflux.runtime.get_AgentNode_impl", side_effect=fake_impl):
            runtime = Runtime(
                [parent_fn],
                client_factories={
                    Provider.Anthropic: lambda: object(),
                    Provider.Gemini: lambda: object(),
                },
            )
            parent_node = runtime.invoke(None, parent_fn, {})
            self.assertEqual(parent_node.result(), "parent")

        view = runtime.get_view(parent_node.id)
        self.assertIsNone(view.provider)
        self.assertEqual(
            view.total_tree_token_bill(),
            {
                Provider.Anthropic: TokenBill(
                    input_tokens_cache_read=1,
                    input_tokens_cache_write=2,
                    input_tokens_regular=3,
                    output_tokens_total=4,
                ),
                Provider.Gemini: TokenBill(
                    input_tokens_cache_read=10,
                    input_tokens_cache_write=0,
                    input_tokens_regular=20,
                    output_tokens_total=30,
                ),
            },
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
