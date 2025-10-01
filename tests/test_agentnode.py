import unittest

from ..core import (
    AgentFunction,
    CodeFunction,
    FunctionArg,
    RunContext,
    AgentNode,
    TokenUsage,
    ModelProviderException,
)
from ..providers import Provider
from ..runtime import Runtime

class _FakeAgentNode(AgentNode):
    """Minimal AgentNode for tests. Default run() is a no-op; tests can subclass/override."""

    def __init__(
        self,
        ctx: RunContext,
        id: int,
        fn: AgentFunction,
        inputs: dict,
        parent=None,
        cancel_event=None,
        client_factory=None,
    ) -> None:
        super().__init__(ctx, id, fn, inputs, parent, cancel_event, client_factory)

    @property
    def token_usage(self) -> TokenUsage:
        return TokenUsage()

    def run(self) -> None:  # pragma: no cover - behavior overridden per test when needed
        # By default, succeed immediately with a placeholder result
        self.ctx.post_success({"ok": True})

class TestAgentNodeUtilities(unittest.TestCase):
    def test_build_user_text_formats_inputs(self):
        """Create AgentFunction with user_prompt_template like 'Hello {name}'; build AgentNode with inputs dict {'name': 'Ada'}; assert build_user_text returns 'Hello Ada'.
        This is to guard against changes that might accidentally remove the substitution behavior."""
        fn = AgentFunction(
            name="greet",
            desc="Test agent",
            args=[FunctionArg("name", str)],
            system_prompt="system",
            user_prompt_template="Hello {name}",
            uses=[],
            default_model=Provider.Anthropic,
        )

        # Minimal RunContext; Runtime is not used by build_user_text
        ctx = RunContext(runtime=None, node=None)  # type: ignore[arg-type]
        node = _FakeAgentNode(ctx, 1, fn, {"name": "Ada"}, parent=None, client_factory=lambda: None)

        self.assertEqual(node.build_user_text(), "Hello Ada")

    def test_build_user_text_raises_on_missing_placeholder(self):
        """Omit a required placeholder key in inputs; build_user_text should raise KeyError.
        This is to guard against possibility of missing arguments being silently ignored as non-exception."""
        # Intentionally declare no args so missing placeholder is only caught at format time
        fn = AgentFunction(
            name="greet",
            desc="Test agent",
            args=[],
            system_prompt="system",
            user_prompt_template="Hello {name}",
            uses=[],
            default_model=Provider.Anthropic,
        )

        ctx = RunContext(runtime=None, node=None)  # type: ignore[arg-type]
        node = _FakeAgentNode(ctx, 1, fn, {}, parent=None, client_factory=lambda: None)

        with self.assertRaises(KeyError):
            _ = node.build_user_text()

    def test_invoke_tool_function_requires_registered_tool(self):
        """AgentFunction.uses includes only tool 'x'; call invoke_tool_function('y', ...) and assert RuntimeError lists available tool names."""
        # Define a trivial tool 'x'
        def tool(ctx: RunContext) -> str:
            return "ok"

        tool_fn = CodeFunction(
            name="x",
            desc="tool x",
            args=[],
            callable=tool,
        )

        agent_fn = AgentFunction(
            name="agent",
            desc="agent",
            args=[],
            system_prompt="sys",
            user_prompt_template="noop",
            uses=[tool_fn],
            default_model=Provider.Anthropic,
        )

        ctx = RunContext(runtime=None, node=None)  # type: ignore[arg-type]
        agent = _FakeAgentNode(ctx, 1, agent_fn, {}, parent=None, client_factory=lambda: None)

        with self.assertRaises(RuntimeError) as cm:
            agent.invoke_tool_function("y", {})

        msg = str(cm.exception)
        self.assertIn("unknown tool: 'y".lower(), msg.lower())
        self.assertIn("x", msg)  # Available tools should be listed

    def test_run_wrapper_wraps_unexpected_exception_as_modelprovider(self):
        """Subclass AgentNode.run to raise a generic Exception that emulates a provider SDK raise; run_wrapper should post a ModelProviderException containing provider class, agent name, and inner exception info."""
        # Create an AgentNode subclass that raises from run()
        class ExplodingAgentNode(_FakeAgentNode):
            def run(self) -> None:  # type: ignore[override]
                raise RuntimeError("boom")

        # Build an AgentFunction
        fn = AgentFunction(
            name="exploder",
            desc="",
            args=[],
            system_prompt="sys",
            user_prompt_template="Hello",
            uses=[],
            default_model=Provider.Anthropic,
        )

        # Prepare a Runtime and monkeypatch providers.get_AgentNode_impl indirectly via Runtime's provider cache
        # by overriding get_AgentNode_impl to always return our ExplodingAgentNode
        # Patch the reference used inside Runtime module
        from .. import runtime as runtime_mod  # type: ignore[assignment]

        original_get_impl = runtime_mod.get_AgentNode_impl
        runtime_mod.get_AgentNode_impl = lambda provider: ExplodingAgentNode  # type: ignore[assignment]
        try:
            rt = Runtime(specs=[fn], client_factories={Provider.Anthropic: (lambda: object())})
            ctx = rt.get_ctx()
            node = rt.invoke(ctx.node, fn, {})
            # Wait for completion and examine exception
            node.wait()
            self.assertIsNotNone(node.exception)
            self.assertIsInstance(node.exception, ModelProviderException)
            mpe: ModelProviderException = node.exception  # type: ignore[assignment]
            self.assertIs(mpe.provider, ExplodingAgentNode)
            self.assertEqual(mpe.agent_name, fn.name)
            # Ensure the inner exception details are preserved
            self.assertIsInstance(mpe.inner_exception, RuntimeError)
            self.assertIn("boom", str(mpe.inner_exception))
        finally:
            runtime_mod.get_AgentNode_impl = original_get_impl

if __name__ == "__main__":
    unittest.main()
