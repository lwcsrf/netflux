import sys
import types
import unittest

from ..core import AgentFunction, CodeFunction, FunctionArg


def _make_noop_code_function(name: str) -> CodeFunction:
    def _body(ctx):
        return None

    return CodeFunction(
        name=name,
        desc="noop",
        args=(),
        callable=_body,
    )


class TestAgentFunctionConstruction(unittest.TestCase):
    def test_agent_function_disallows_duplicate_tool_names(self):
        """Construct AgentFunction.uses with two Functions sharing the same name; expect ValueError about duplicate tool names."""

        tool_a = _make_noop_code_function("duplicate")
        tool_b = _make_noop_code_function("duplicate")

        with self.assertRaisesRegex(ValueError, "duplicate tool names"):
            AgentFunction(
                name="agent",
                desc="test",
                args=(),
                system_prompt="system",
                user_prompt_template="user",
                uses=[tool_a, tool_b],
            )


class TestCodeFunctionSignatureValidation(unittest.TestCase):
    def test_code_function_requires_runcontext_first_positional(self):
        """Attempt CodeFunction with callable whose first parameter isn't a positional RunContext; expect TypeError."""

        def bad_callable(*, value: str) -> None:  # missing RunContext
            del value

        with self.assertRaisesRegex(TypeError, "first parameter must be a RunContext"):
            CodeFunction(
                name="bad",
                desc="",
                args=(FunctionArg("value", str),),
                callable=bad_callable,
            )

    def test_code_function_forbids_varargs_and_kwargs(self):
        """Callable uses *args/**kwargs -> expect TypeError at construction time."""

        def bad_varargs(ctx, *values):  # type: ignore[no-untyped-def]
            del ctx, values

        with self.assertRaisesRegex(TypeError, "\\*args/\\*\\*kwargs are not allowed"):
            CodeFunction(
                name="bad_varargs",
                desc="",
                args=(),
                callable=bad_varargs,
            )

        def bad_kwargs(ctx, **values):  # type: ignore[no-untyped-def]
            del ctx, values

        with self.assertRaisesRegex(TypeError, "\\*args/\\*\\*kwargs are not allowed"):
            CodeFunction(
                name="bad_kwargs",
                desc="",
                args=(),
                callable=bad_kwargs,
            )

    def test_code_function_requires_keyword_only_parameters(self):
        """Callable defines positional params after RunContext -> expect TypeError requiring KEYWORD_ONLY params."""

        def bad_callable(ctx, value):  # positional parameter after RunContext
            del ctx, value

        with self.assertRaisesRegex(TypeError, "must be keyword-only"):
            CodeFunction(
                name="bad",
                desc="",
                args=(FunctionArg("value", str),),
                callable=bad_callable,
            )

    def test_code_function_signature_must_match_arg_names_and_order(self):
        """Provide FunctionArg list ['a','b'] but callable has (*, b, a) -> TypeError about mismatch."""

        def bad_callable(ctx, *, b: str, a: str):
            del ctx, a, b

        with self.assertRaisesRegex(
            TypeError,
            r"callable signature args \['b', 'a'\] do not match spec \['a', 'b'\]",
        ):
            CodeFunction(
                name="mismatch",
                desc="",
                args=(FunctionArg("a", str), FunctionArg("b", str)),
                callable=bad_callable,
            )

    def test_code_function_optional_arg_requires_default_none(self):
        """Optional FunctionArg but callable default != None -> TypeError; default must be None exactly."""

        def bad_callable(ctx, *, maybe: str = "default"):
            del ctx, maybe

        with self.assertRaisesRegex(TypeError, "optional arg 'maybe' default must be None"):
            CodeFunction(
                name="optional",
                desc="",
                args=(FunctionArg("maybe", str, optional=True),),
                callable=bad_callable,
            )

    def test_code_function_required_arg_must_not_have_default(self):
        """Required FunctionArg has a default in callable -> TypeError."""

        def bad_callable(ctx, *, required: str = "value"):
            del ctx, required

        with self.assertRaisesRegex(TypeError, "required arg 'required' must not have a default"):
            CodeFunction(
                name="required",
                desc="",
                args=(FunctionArg("required", str),),
                callable=bad_callable,
            )

    def test_code_function_disallows_duplicate_use_names(self):
        """Create two different Functions with the same name in uses; expect ValueError about duplicate names."""

        tool_a = _make_noop_code_function("dup")
        tool_b = _make_noop_code_function("dup")

        def body(ctx):
            return None

        with self.assertRaisesRegex(ValueError, "duplicate uses names"):
            CodeFunction(
                name="orchestrator",
                desc="",
                args=(),
                callable=body,
                uses=[tool_a, tool_b],
            )


if __name__ == "__main__":
    unittest.main()
