import unittest
from ..core import (
    FunctionArg, Function, AgentFunction, CodeFunction,
    UserTextPart, ModelTextPart, ToolUsePart, ToolResultPart, ThinkingBlockPart
)
from ..providers import Provider




class TestAgentFunctionConstruction(unittest.TestCase):
    def test_agent_function_disallows_duplicate_tool_names(self):
        """Construct AgentFunction.uses with two Functions sharing the same name; expect ValueError about duplicate tool names."""
        pass


class TestCodeFunctionSignatureValidation(unittest.TestCase):
    def test_code_function_requires_runcontext_first_positional(self):
        """Attempt CodeFunction with callable whose first parameter isn't a positional RunContext; expect TypeError."""
        pass

    def test_code_function_forbids_varargs_and_kwargs(self):
        """Callable uses *args/**kwargs -> expect TypeError at construction time."""
        pass

    def test_code_function_requires_keyword_only_parameters(self):
        """Callable defines positional params after RunContext -> expect TypeError requiring KEYWORD_ONLY params."""
        pass

    def test_code_function_signature_must_match_arg_names_and_order(self):
        """Provide FunctionArg list ['a','b'] but callable has (*, b, a) -> TypeError about mismatch."""
        pass

    def test_code_function_optional_arg_requires_default_none(self):
        """Optional FunctionArg but callable default != None -> TypeError; default must be None exactly."""
        pass

    def test_code_function_required_arg_must_not_have_default(self):
        """Required FunctionArg has a default in callable -> TypeError."""
        pass

    def test_code_function_disallows_duplicate_use_names(self):
        """Create two different Functions with the same name in uses; expect ValueError about duplicate names."""
        pass




if __name__ == "__main__":
    unittest.main()
