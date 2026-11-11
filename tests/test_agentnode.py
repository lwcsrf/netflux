import unittest
from importlib import import_module
from types import MappingProxyType
from unittest.mock import patch

from ..core import (
    AgentFunction,
    CodeFunction,
    FunctionArg,
    RunContext,
    AgentNode,
    TokenUsage,
    ModelTextPart,
    ToolUsePart,
    ModelProviderException,
    Continuation,
    TranscriptPosition,
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
        tool_use_id=None,
    ) -> None:
        super().__init__(ctx, id, fn, inputs, parent, cancel_event, client_factory, tool_use_id)

    @property
    def token_usage(self) -> TokenUsage:
        return TokenUsage()

    @property
    def provider(self) -> Provider:
        return Provider.Anthropic

    def run(self) -> None:  # pragma: no cover - behavior overridden per test when needed
        # By default, succeed immediately with a placeholder result
        self.transcript.append(ModelTextPart(text="ok"))
        self.ctx.post_success("ok")

class TestAgentNodeUtilities(unittest.TestCase):
    def test_intermediate_events_publish_stable_snapshots(self):
        fn = AgentFunction(
            name="events", desc="", args=[], system_prompt="system",
            user_prompt_template="prompt", uses=[], default_model=Provider.Anthropic,
        )
        runtime = Runtime([fn], client_factories={Provider.Anthropic: lambda: None})
        # Drive the existing fake node synchronously.
        with patch("netflux.runtime.get_AgentNode_impl", return_value=_FakeAgentNode), \
                patch.object(_FakeAgentNode, "start"):
            node: _FakeAgentNode = runtime.invoke(None, fn, {})  # type: ignore[assignment]
        node.ctx.post_running()
        node.transcript.append(ModelTextPart("A🙂"))
        node.ctx.post_transcript_update()
        before = runtime.get_view(node.id)
        failure = RuntimeError("retry")
        node.record_retriable_exception(failure, attempt=0)
        during = node.watch(as_of_seq=before.update_seqnum, timeout=0)
        assert during is not None
        self.assertEqual(during.transcript, before.transcript)
        event, = during.intermediate_events
        self.assertIs(event.exception, failure)  # type: ignore[attr-defined]
        self.assertEqual(event.position, node.transcript_end_position)
        self.assertEqual(event.position, TranscriptPosition(0, 2))
        node.record_continuation()
        node.transcript[-1] = ModelTextPart("A🙂\nB")
        node.ctx.post_success(node.transcript[-1].text)
        final = runtime.get_view(node.id)
        self.assertEqual(final.intermediate_events, (event, Continuation(event.position)))
        self.assertEqual(before.intermediate_events, ())
        self.assertEqual(during.intermediate_events, (event,))
        self.assertEqual(during.transcript, (ModelTextPart("A🙂"),))

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
            agent.invoke_tool_function("y", {}, "test-tool-use-id")

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

class TestAgentProviderToolSelection(unittest.TestCase):
    @staticmethod
    def _agent(name, *, args=(), uses=(), recursive=False, allow_models=None,
               default_model=Provider.Anthropic):
        return AgentFunction(
            name=name, desc=name, args=args, system_prompt="system",
            user_prompt_template="prompt", uses=uses, uses_recursion=recursive,
            allow_models=allow_models, default_model=default_model,
        )

    def test_tool_provider_dispatch_preserves_arguments_and_transcript(self):
        leaf = self._agent(
            "leaf", args=[FunctionArg("task", str)], allow_models=[Provider.Gemini, Provider.OpenAI],
            default_model=Provider.OpenAI,
        )
        root_fn = self._agent(
            "root", args=[FunctionArg("task", str)], uses=[leaf], recursive=True,
            allow_models=[Provider.Anthropic, Provider.Gemini],
        )
        records = []

        def fake_impl(selected_provider):
            class FakeAgentNode(_FakeAgentNode):
                @property
                def provider(self):
                    return selected_provider

                def run(self):
                    if self.agent_fn is leaf:
                        output = self.provider.value
                    else:
                        calls = [("leaf", {}), ("leaf", {"provider": None})]
                        if self.parent is None:
                            calls += [
                                ("leaf", {"provider": "OpenAI"}),
                                ("root", {"provider": "Anthropic"}),
                            ]
                        outputs = []
                        for index, (name, extra_args) in enumerate(calls):
                            raw_args = {"task": str(index), **extra_args}
                            original = dict(raw_args)
                            tool_id = f"call-{index}"
                            part = ToolUsePart(tool_id, name, MappingProxyType(raw_args))
                            self.transcript.append(part)
                            description = self.func_args(self.func_map[name])[-1].desc
                            child = self.invoke_tool_function(name, raw_args, tool_id)
                            records.append((original, child, part, description))
                            outputs.append(child.result())
                        output = " ".join(outputs)
                    self.transcript.append(ModelTextPart(output))
                    self.ctx.post_success(output)

            return FakeAgentNode

        with patch("netflux.runtime.get_AgentNode_impl", side_effect=fake_impl):
            runtime = Runtime(
                [root_fn],
                client_factories={Provider.Anthropic: object, Provider.Gemini: object, Provider.OpenAI: object},
            )
            root = runtime.invoke(
                None, root_fn, {"task": "start"}, provider=Provider.Gemini, max_agent_levels=3,
            )
            self.assertEqual(root.result(), "Gemini Gemini OpenAI OpenAI OpenAI")

        self.assertEqual(len(records), 6)
        for original, child, part, description in records:
            with self.subTest(tool_use_id=child.tool_use_id, parent=child.parent.id):
                # The transcript mapping aliases the original tool arguments.
                self.assertEqual(dict(part.args), original)
                self.assertEqual(child.inputs, {"task": original["task"]})
                self.assertEqual(child.tool_use_id, part.tool_use_id)
                if original.get("provider") is None:
                    # Keep core's provider_arg_spec()/func_args() promise consistent with
                    # Runtime.invoke() for omitted/null providers, covering inheritance
                    # and default fallback. Fixed result expectations above guard against
                    # both implementations agreeing on the wrong provider.
                    behavior = "inherit" if child.parent is root else "use default"
                    self.assertTrue(description.endswith(
                        f"Omit or null: will {behavior} {child.provider.value}."
                    ))
        self.assertIs(root.children[-1].provider, Provider.Anthropic)

    def test_provider_spec_validation_and_code_argument_isolation(self):
        with self.assertRaisesRegex(ValueError, "provider"):
            self._agent("collision", args=[FunctionArg("provider", str)])
        for invalid in ([], [Provider.Gemini, "Anthropic"]):
            with self.subTest(allow_models=invalid), self.assertRaises(ValueError):
                self._agent("invalid", allow_models=invalid)
        with self.assertRaises(AssertionError):
            self._agent("excluded_default", allow_models=[Provider.Gemini])
        choices = [Provider.Gemini]
        leaf = self._agent(
            "leaf", args=[FunctionArg("task", str)], allow_models=choices,
            default_model=Provider.Gemini,
        )
        choices.clear()
        self.assertEqual(leaf.allow_models, (Provider.Gemini,))
        code = CodeFunction(
            name="code", desc="code", args=[FunctionArg("provider", str)],
            callable=lambda ctx, *, provider: provider,
        )
        root_fn = self._agent("root", uses=[leaf, code], recursive=True)
        runtime = Runtime([root_fn], client_factories={Provider.Anthropic: object})
        self.assertEqual(root_fn.allow_models, tuple(Provider))
        with patch("netflux.runtime.get_AgentNode_impl", return_value=_FakeAgentNode), \
                patch.object(_FakeAgentNode, "start"):
            node = runtime.invoke(None, root_fn, {})
        spec = node.provider_arg_spec(leaf)
        self.assertEqual(spec.enum, {"Gemini"})
        self.assertTrue(spec.desc.endswith(
            "Omit or null: will use default Gemini."
        ))
        self.assertEqual(node.func_args(leaf), [*leaf.args, spec])
        self.assertEqual(node.func_args(root_fn)[-1].enum, {p.value for p in Provider})
        self.assertEqual([arg.name for arg in leaf.args], ["task"])
        self.assertIsNot(node.func_args(code), code.args)
        self.assertEqual(node.func_args(code), code.args)

        with patch.object(node.ctx, "invoke") as invoke:
            for value in ("Anthropic", "gemini", 42, [], Provider.Gemini):
                with self.subTest(provider=value):
                    raw_args = {"task": "work", "provider": value}
                    with self.assertRaisesRegex(ValueError, "provider"):
                        node.invoke_tool_function("leaf", raw_args, "invalid")
                    self.assertEqual(raw_args, {"task": "work", "provider": value})
                    invoke.assert_not_called()
            raw_args = {"provider": "ordinary business input"}
            self.assertIs(node.invoke_tool_function("code", raw_args, "code-use"), invoke.return_value)
            invoke.assert_called_once_with(code, raw_args, provider=None, tool_use_id="code-use")
            self.assertEqual(raw_args, {"provider": "ordinary business input"})
        raw_args = {"provider": "xAI"}
        with self.assertRaises(NotImplementedError):
            node.invoke_tool_function("root", raw_args, "unsupported-provider")
        self.assertEqual(raw_args, {"provider": "xAI"})

    def test_provider_tool_schemas_across_adapters(self):
        primitives = {
            "text": (str, "string"), "count": (int, "integer"),
            "ratio": (float, "number"), "enabled": (bool, "boolean"),
        }
        leaf = self._agent("leaf", args=[
            FunctionArg("task", str),
            FunctionArg("style", str, optional=True, enum={"brief", "full"}),
            *(FunctionArg(name, argtype, optional=True) for name, (argtype, _) in primitives.items()),
        ], allow_models=[Provider.Anthropic])
        null_args = {"task": "work", **dict.fromkeys(primitives)}
        self.assertEqual(leaf.validate_coerce_args(null_args), null_args)
        code = CodeFunction(
            name="code", desc="code", args=[FunctionArg("provider", str, enum={"local"})],
            callable=lambda ctx, *, provider: provider,
        )
        plain = CodeFunction(name="plain", desc="plain", args=[], callable=lambda ctx: None)
        root_fn = self._agent("root", uses=[leaf, code, plain], recursive=True)
        runtime = Runtime([root_fn], client_factories={Provider.Anthropic: object})
        for sdk, module_name, class_name in (
            ("anthropic", "anthropic", "AnthropicAgentNode"),
            ("google.genai", "gemini", "GeminiAgentNode"),
            ("openai", "openai", "OaiAgentNode"),
        ):
            with self.subTest(adapter=module_name):
                try:
                    import_module(sdk)
                except ModuleNotFoundError:
                    self.skipTest(f"optional SDK {sdk} is not installed")
                cls = getattr(import_module(f"netflux.providers.{module_name}"), class_name)
                # Exercise pure schema construction without creating SDK clients.
                node = object.__new__(cls)
                AgentNode.__init__(node, RunContext(runtime, None), 1, root_fn, {}, None, None, object)
                if module_name == "anthropic":
                    schemas = {tool["name"]: tool["input_schema"] for tool in node._build_tool_params()}
                elif module_name == "gemini":
                    schemas = {
                        declaration.name: declaration.parameters.model_dump(mode="json", exclude_none=True)
                        for tool in node._build_tool_params() for declaration in tool.function_declarations
                    }
                else:
                    schemas = {fn.name: node.make_function_tool(fn)["parameters"] for fn in root_fn.uses}
                for fn in (leaf, root_fn):
                    provider = schemas[fn.name]["properties"]["provider"]
                    choices = {p.value for p in fn.allow_models}
                    self.assertNotIn("provider", schemas[fn.name].get("required", []))
                    self.assertEqual(provider["description"], node.provider_arg_spec(fn).desc)
                    if module_name == "gemini":
                        self.assertTrue(provider["nullable"])
                        self.assertEqual(provider["type"], "STRING")
                        self.assertEqual(set(provider["enum"]), choices)
                    else:
                        self.assertEqual(set(provider["type"]), {"string", "null"})
                        self.assertEqual(set(provider["enum"]), choices | {None})
                self.assertEqual(schemas["leaf"]["required"], ["task"])
                style = schemas["leaf"]["properties"]["style"]
                if module_name == "gemini":
                    self.assertTrue(style["nullable"])
                else:
                    self.assertIn("null", style["type"])
                    self.assertIn(None, style["enum"])
                for name, (_, json_type) in primitives.items():
                    with self.subTest(argument=name):
                        prop = schemas["leaf"]["properties"][name]
                        if module_name == "gemini":
                            self.assertTrue(prop["nullable"])
                            self.assertEqual(prop["type"], json_type.upper())
                        else:
                            self.assertEqual(set(prop["type"]), {json_type, "null"})
                self.assertEqual(schemas["code"]["required"], ["provider"])
                self.assertEqual(schemas["code"]["properties"]["provider"]["enum"], ["local"])
                self.assertEqual(schemas["plain"].get("properties", {}), {})
                self.assertEqual([arg.name for arg in leaf.args], ["task", "style", *primitives])


if __name__ == "__main__":
    unittest.main()
