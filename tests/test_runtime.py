# tests/test_runtime_plan.py
"""
Planning-only scaffolding for Runtime:
- client factory validation
- registration (BFS over .uses), duplicate name handling
- invocation paths (CodeFunction vs AgentFunction)
- session bag initialization
"""

import unittest
# from netflux.runtime import Runtime
# from netflux.core import FunctionArg, CodeFunction, AgentFunction, RunContext, NodeState, NodeView
# from netflux.providers import Provider, get_AgentNode_impl


class TestRuntimeClientFactories(unittest.TestCase):
    def test_validate_client_factories_type_checks_keys_and_values(self):
        """Pass mapping with non-Provider key and non-callable value; expect TypeError for each from validate_client_factories."""
        pass


class TestRuntimeRegistration(unittest.TestCase):
    def test_rejects_duplicate_function_names_in_seeds(self):
        """Provide two distinct Function instances with same name in initial specs; expect ValueError during Runtime construction."""
        pass

    def test_rejects_duplicate_function_names_across_transitives(self):
        """Make a dependency D with same name as different instance already registered; expect ValueError during BFS traversal."""
        pass


class TestRuntimeInvocation(unittest.TestCase):
    def test_invoke_rejects_unregistered_function(self):
        """Create a Function not included in Runtime; Runtime.invoke(None, fn, ...) should raise ValueError about missing registration."""
        pass

    def test_invoke_rejects_name_collision_with_different_instance(self):
        """Register Function F; later create new Function with same name but different identity; expect ValueError when invoking the latter."""
        pass

    def test_invoke_disallows_provider_override_for_code_function(self):
        """Attempt Runtime.invoke(caller=None, code_fn, inputs, provider=SomeProvider) and assert ValueError."""
        pass

    def test_invoke_creates_and_starts_code_node(self):
        """Invoke a trivial CodeFunction; assert returned node is CodeNode, node.state transitions to Running then finishes; thread is set."""
        pass

    def test_invoke_creates_agent_node_with_provider_impl_and_factory(self):
        """Monkeypatch providers.get_AgentNode_impl to return a FakeAgentNode; register a dummy factory; invoke AgentFunction and assert FakeAgentNode constructed."""
        pass

    def test_invoke_links_parent_child_relationship(self):
        """Invoke a CodeFunction that internally uses RunContext.invoke to call another Function; after both complete, assert parent.children contains child in order."""
        pass

    def test_invoke_initializes_session_bags(self):
        """Upon node creation, assert the RunContext has SessionScope.TopLevel and SessionScope.Self; for non-root, also Parent; verify identity relationships."""
        pass


class TestRuntimeObservability(unittest.TestCase):
    def test_list_toplevel_views_returns_snapshots(self):
        """Create one or more top-level tasks; call list_toplevel_views and assert NodeView instances reflect latest states."""
        pass

    def test_watch_blocks_until_newer_seq(self):
        """Call Runtime.watch(node, as_of_seq=current_seq); in a background thread, advance node state; assert watch returns a view with higher update_seqnum."""
        pass

class TestRuntimeStateTransitions(unittest.TestCase):
    def test_post_status_update_mutates_state_and_notifies(self):
        """Call Runtime.post_status_update(node, NodeState.Running) and assert node.state updated and watchers notified (via watch)."""
        pass

    def test_post_success_sets_outputs_and_marks_done(self):
        """Call Runtime.post_success(node, outputs); assert outputs stored, state=Success, node.done is set, and NodeView updated to include outputs."""
        pass

    def test_post_exception_sets_exception_and_logs(self):
        """Call Runtime.post_exception(node, exc); assert exception stored, state=Error, node.done set; NodeView updated to include the exception."""
        pass

    def test_publish_tree_update_refreshes_ancestors(self):
        """Create parent->child; change child; assert parent's NodeView.children tuple reflects updated child view (and update_seqnum advanced)."""
        pass

class TestNodeViewStructure(unittest.TestCase):
    def test_node_view_children_is_tuple_and_preserves_order(self):
        """Build a parent with two sequential child invocations; fetch NodeView; assert children is a tuple and the order matches invocation order."""
        pass

if __name__ == "__main__":
    unittest.main()
