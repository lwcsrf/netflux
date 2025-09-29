# tests/test_runcontext_and_session_plan.py
"""
Planning-only scaffolding for RunContext and SessionBag behavior.
"""

import unittest
# from netflux.core import RunContext, SessionBag, SessionScope, NoParentSessionError
# from netflux.runtime import Runtime
# from netflux.providers import Provider


class TestSessionBag(unittest.TestCase):
    def test_session_bag_get_or_put_creates_and_caches(self):
        """Call get_or_put twice with same namespace/key; assert factory ran once and both calls return the same object."""
        pass

    def test_session_bag_is_thread_safe_single_factory_execution(self):
        """Spin multiple threads racing on same namespace/key; assert exactly one factory invocation and all threads receive same instance."""
        pass


class TestRunContextSessionBags(unittest.TestCase):
    def test_get_or_put_requires_initialized_bags(self):
        """Construct RunContext without object_bags set; calling get_or_put should raise RuntimeError about uninitialized bags."""
        pass

    def test_get_or_put_parent_scope_requires_parent(self):
        """Bind a RunContext to a root node and call get_or_put(SessionScope.Parent,...); expect NoParentSessionError."""
        pass

    def test_top_level_scope_is_shared_across_descendants(self):
        """Create a top-level node and a child; fetch an object via TopLevel in both contexts and assert identity equality (shared bag)."""
        pass

    def test_self_parent_top_level_distinctions_for_deep_tree(self):
        """Create root -> child -> grandchild; verify each scope resolves to different bags where appropriate and parent scope matches immediate parent."""
        pass


if __name__ == "__main__":
    unittest.main()
