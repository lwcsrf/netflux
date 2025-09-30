import sys
import threading
import time
import types
import unittest
from typing import Any, Dict

from ..core import (
    CodeFunction,
    NoParentSessionError,
    RunContext,
    SessionBag,
    SessionScope,
)
from ..runtime import Runtime


class TestSessionBag(unittest.TestCase):
    def test_session_bag_get_or_put_creates_and_caches(self) -> None:
        """The factory is run exactly once and the cached object is reused."""

        bag = SessionBag()
        factory_calls: list[int] = []

        def factory() -> object:
            factory_calls.append(1)
            return object()

        first = bag.get_or_put("ns", "key", factory)
        second = bag.get_or_put("ns", "key", factory)

        self.assertEqual(len(factory_calls), 1)
        self.assertIs(first, second)

    def test_session_bag_is_thread_safe_single_factory_execution(self) -> None:
        """Concurrent callers racing for the same entry see a single factory invocation."""

        bag = SessionBag()
        created: list[object] = []
        call_count = 0
        call_count_lock = threading.Lock()
        barrier = threading.Barrier(5)

        def factory() -> object:
            nonlocal call_count
            with call_count_lock:
                call_count += 1
            # Give other threads a chance to race before we return.
            time.sleep(0.01)
            obj = object()
            created.append(obj)
            return obj

        results: list[object] = []

        def worker() -> None:
            barrier.wait()
            results.append(bag.get_or_put("ns", "key", factory))

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(call_count, 1)
        self.assertEqual(len(created), 1)
        self.assertEqual(len(results), 5)
        for obj in results:
            self.assertIs(obj, results[0])


class TestRunContextSessionBags(unittest.TestCase):
    def _build_runtime(self, root_fn: CodeFunction) -> Runtime:
        return Runtime(specs=[root_fn], client_factories={})

    def test_get_or_put_requires_initialized_bags(self) -> None:
        """RunContext.get_or_put fails when bags have not been initialised."""

        class _StubNode:
            parent = None

        ctx = RunContext(runtime=None, node=_StubNode())  # type: ignore[arg-type]

        with self.assertRaisesRegex(RuntimeError, "SessionBags have not been initialized"):
            ctx.get_or_put(SessionScope.Self, "ns", "key", lambda: object())

    def test_get_or_put_parent_scope_requires_parent(self) -> None:
        """Requesting the parent scope on a root node raises ``NoParentSessionError``."""

        class _StubNode:
            parent = None

        ctx = RunContext(runtime=None, node=_StubNode())  # type: ignore[arg-type]
        ctx.object_bags = {
            SessionScope.TopLevel: SessionBag(),
            SessionScope.Self: SessionBag(),
        }

        with self.assertRaises(NoParentSessionError):
            ctx.get_or_put(SessionScope.Parent, "ns", "key", lambda: object())

    def test_top_level_scope_is_shared_across_descendants(self) -> None:
        """The top-level SessionBag instance is reused by all descendants."""

        child_results: Dict[str, Any] = {}

        def child_callable(ctx: RunContext) -> Dict[str, object]:
            child_top = ctx.get_or_put(SessionScope.TopLevel, "bag", "shared", lambda: object())
            child_self = ctx.get_or_put(SessionScope.Self, "bag", "self", lambda: object())
            child_results.update({"top": child_top, "self": child_self})
            return child_results

        child_fn = CodeFunction(
            name="child",
            desc="child",
            args=[],
            callable=child_callable,
        )

        def root_callable(ctx: RunContext) -> Dict[str, Any]:
            root_top = ctx.get_or_put(SessionScope.TopLevel, "bag", "shared", lambda: object())
            child_node = ctx.invoke(child_fn, {})
            child_output = child_node.result()
            return {"root_top": root_top, "child": child_output}

        root_fn = CodeFunction(
            name="root",
            desc="root",
            args=[],
            callable=root_callable,
            uses=[child_fn],
        )

        runtime = self._build_runtime(root_fn)
        ctx = runtime.get_ctx()
        node = ctx.invoke(root_fn, {})
        outputs = node.result()

        assert isinstance(outputs, dict)
        root_top = outputs["root_top"]
        child_top = outputs["child"]["top"]

        self.assertIs(root_top, child_top)
        self.assertIs(outputs["child"]["self"], child_results["self"])

    def test_self_parent_top_level_distinctions_for_deep_tree(self) -> None:
        """Each scope resolves to the expected bag in a root → child → grandchild tree."""

        def grandchild_callable(ctx: RunContext) -> Dict[str, object]:
            return {
                "top": ctx.get_or_put(SessionScope.TopLevel, "bag", "shared", lambda: object()),
                "parent": ctx.get_or_put(SessionScope.Parent, "bag", "self", lambda: object()),
                "self": ctx.get_or_put(SessionScope.Self, "bag", "self", lambda: object()),
            }

        grandchild_fn = CodeFunction(
            name="grandchild",
            desc="grandchild",
            args=[],
            callable=grandchild_callable,
        )

        def child_callable(ctx: RunContext) -> Dict[str, Any]:
            child_top = ctx.get_or_put(SessionScope.TopLevel, "bag", "shared", lambda: object())
            child_self = ctx.get_or_put(SessionScope.Self, "bag", "self", lambda: object())
            child_parent = ctx.get_or_put(SessionScope.Parent, "bag", "self", lambda: object())
            grandchild_node = ctx.invoke(grandchild_fn, {})
            return {
                "top": child_top,
                "self": child_self,
                "parent": child_parent,
                "grandchild": grandchild_node.result(),
            }

        child_fn = CodeFunction(
            name="child",
            desc="child",
            args=[],
            callable=child_callable,
            uses=[grandchild_fn],
        )

        def root_callable(ctx: RunContext) -> Dict[str, Any]:
            root_top = ctx.get_or_put(SessionScope.TopLevel, "bag", "shared", lambda: object())
            root_self = ctx.get_or_put(SessionScope.Self, "bag", "self", lambda: object())
            child_node = ctx.invoke(child_fn, {})
            return {
                "top": root_top,
                "self": root_self,
                "child": child_node.result(),
            }

        root_fn = CodeFunction(
            name="root",
            desc="root",
            args=[],
            callable=root_callable,
            uses=[child_fn],
        )

        runtime = self._build_runtime(root_fn)
        top_ctx = runtime.get_ctx()
        node = top_ctx.invoke(root_fn, {})
        outputs = node.result()

        assert isinstance(outputs, dict)
        root_top = outputs["top"]
        root_self = outputs["self"]
        child_data = outputs["child"]
        child_top = child_data["top"]
        child_self = child_data["self"]
        child_parent = child_data["parent"]
        grandchild_data = child_data["grandchild"]

        # Top-level bag shared throughout the tree.
        self.assertIs(root_top, child_top)
        self.assertIs(root_top, grandchild_data["top"])

        # Parent scope resolves to immediate parent's self bag.
        self.assertIs(child_parent, root_self)
        self.assertIs(grandchild_data["parent"], child_self)

        # Each node's self bag is unique to that node.
        self.assertIsNot(root_self, child_self)
        self.assertIsNot(child_self, grandchild_data["self"])
        self.assertIsNot(root_self, grandchild_data["self"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
