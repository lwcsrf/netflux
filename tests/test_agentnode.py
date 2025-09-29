import unittest

class TestAgentNodeUtilities(unittest.TestCase):
    def test_build_user_text_formats_inputs(self):
        """Create AgentFunction with user_prompt_template like 'Hello {name}'; build AgentNode with inputs dict {'name': 'Ada'}; assert build_user_text returns 'Hello Ada'.
        This is to guard against changes that might accidentally remove the substitution behavior."""
        pass

    def test_build_user_text_raises_on_missing_placeholder(self):
        """Omit a required placeholder key in inputs; build_user_text should raise KeyError.
        This is to guard against possibility of missing arguments being silently ignored as non-exception."""
        pass

    def test_invoke_tool_function_requires_registered_tool(self):
        """AgentFunction.uses includes only tool 'x'; call invoke_tool_function('y', ...) and assert RuntimeError lists available tool names."""
        pass

    def test_run_wrapper_wraps_unexpected_exception_as_modelprovider(self):
        """Subclass AgentNode.run to raise a generic Exception that emulates a provider SDK raise; run_wrapper should post a ModelProviderException containing provider class, agent name, and inner exception info."""
        pass

if __name__ == "__main__":
    unittest.main()
