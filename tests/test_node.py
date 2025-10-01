import threading
import time
import unittest

from ..core import CodeFunction, NodeState
from ..runtime import Runtime


class TestNodeLifecycle(unittest.TestCase):
    def _make_runtime(self, fn: CodeFunction) -> Runtime:
        return Runtime([fn], client_factories={})

    def test_node_success_returns_outputs(self):
        """node.result() should yield the exact outputs that the CodeFunction returned."""

        def _impl(ctx):  # pragma: no cover - invoked via node thread
            return {"status": "ok", "value": 42}

        fn = CodeFunction(name="success_fn", desc="returns value", args=[], callable=_impl)
        runtime = self._make_runtime(fn)

        node = runtime.invoke(None, fn, inputs={})
        result = node.result()

        self.assertEqual(result, {"status": "ok", "value": 42})
        self.assertEqual(node.state, NodeState.Success)
        self.assertIsNone(node.exception)

    def test_node_exception_raises_exception(self):
        """Ensure a CodeFunction exception propagates the original instance through node.result()."""

        raised: list[Exception] = []

        class CustomError(RuntimeError):
            pass

        def _impl(ctx):  # pragma: no cover - invoked via node thread
            exc = CustomError("boom")
            raised.append(exc)
            raise exc

        fn = CodeFunction(name="error_fn", desc="raises", args=[], callable=_impl)
        runtime = self._make_runtime(fn)

        node = runtime.invoke(None, fn, inputs={})
        with self.assertRaises(CustomError) as cm:
            node.result()

        self.assertIs(cm.exception, raised[0])
        self.assertEqual(node.state, NodeState.Error)
        self.assertIs(node.exception, raised[0])

    def test_node_wait_and_is_done_flags(self):
        """node.wait() should block until completion and node.is_done should mirror the Event state."""

        started = threading.Event()
        release = threading.Event()

        def _impl(ctx):  # pragma: no cover - invoked via node thread
            started.set()
            release.wait(timeout=5)
            return "done"

        fn = CodeFunction(name="wait_fn", desc="waits", args=[], callable=_impl)
        runtime = self._make_runtime(fn)

        node = runtime.invoke(None, fn, inputs={})

        self.assertTrue(started.wait(timeout=1), "function never started")
        self.assertFalse(node.is_done)

        waiter_finished = threading.Event()

        def _waiter():
            node.wait()
            waiter_finished.set()

        waiter_thread = threading.Thread(target=_waiter)
        waiter_thread.start()

        time.sleep(0.05)
        self.assertFalse(waiter_finished.is_set(), "wait() returned before completion")

        release.set()
        waiter_thread.join(timeout=1)
        self.assertTrue(waiter_finished.is_set(), "wait() did not return after completion")
        self.assertTrue(node.is_done)
        self.assertEqual(node.result(), "done")

    def test_node_watch_proxies_runtime_watch(self):
        """node.watch should block until Runtime emits a later update_seqnum."""

        release = threading.Event()

        def _impl(ctx):  # pragma: no cover - invoked via node thread
            release.wait(timeout=5)
            return "final"

        fn = CodeFunction(name="watch_fn", desc="observed", args=[], callable=_impl)
        runtime = self._make_runtime(fn)

        node = runtime.invoke(None, fn, inputs={})

        first_view = node.watch(as_of_seq=0)
        self.assertIsNotNone(first_view)
        assert first_view is not None  # type narrowing
        first_seq = first_view.update_seqnum
        self.assertGreater(first_seq, 0)
        self.assertIn(first_view.state, {NodeState.Waiting, NodeState.Running})

        release.set()

        final_view = node.watch(as_of_seq=first_seq)
        self.assertIsNotNone(final_view)
        assert final_view is not None  # type narrowing
        self.assertGreater(final_view.update_seqnum, first_seq)
        self.assertEqual(final_view.state, NodeState.Success)
        self.assertEqual(final_view.outputs, "final")


if __name__ == "__main__":
    unittest.main()
