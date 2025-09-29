import unittest
from ..core import (
    FunctionArg, Function, AgentFunction, CodeFunction,
    UserTextPart, ModelTextPart, ToolUsePart, ToolResultPart, ThinkingBlockPart
)
from ..providers import Provider




class TestFunctionBaseValidateArgs(unittest.TestCase):
    def test_validate_coerce_args_rejects_unknown_arg(self):
        """Call Function.validate_coerce_args with an unexpected key; expect ValueError listing unknown argument names."""
        pass

    def test_validate_coerce_args_requires_missing_required(self):
        """Omit a required arg; expect ValueError listing missing names."""
        pass

    def test_validate_coerce_args_allows_omitted_optional(self):
        """Define optional arg; omit it; expect no error and returned mapping excludes the key."""
        pass

    def test_validate_coerce_args_coerces_boolean_strings(self):
        """Provide 'true'/'false' strings for a bool arg; expect returned dict with actual bools."""
        pass

    def test_validate_coerce_args_rejects_non_boolean_string(self):
        """Provide a non-coercible string (e.g., 'yes') for bool arg; expect ValueError from validate_value complaining about type."""
        pass


class TestFunctionArgValidation(unittest.TestCase):
    def test_function_arg_rejects_unsupported_type(self):
        """Construct FunctionArg with an argtype not in {str,int,float,bool} (e.g., list) and assert ValueError."""
        pass

    def test_function_arg_enum_requires_string_type(self):
        """Attempt enum on non-str argtype (e.g., int) and assert ValueError explaining enum only for str."""
        pass

    def test_function_arg_enum_requires_nonempty_all_strings(self):
        """Give empty set or a set with non-strings and assert ValueError; check message mentions non-empty and string-only expectation."""
        pass

    def test_validate_value_allows_none_when_optional(self):
        """Create optional FunctionArg and call validate_value(None); expect no exception."""
        pass

    def test_validate_value_rejects_none_when_required(self):
        """Create required FunctionArg and call validate_value(None); expect ValueError stating arg is required."""
        pass

    def test_validate_value_rejects_incorrect_type(self):
        """For int arg, pass a float (or for float pass int/bool per exactness rules) and assert ValueError with type detail."""
        pass

    def test_validate_value_enforces_bool_exact_type(self):
        """For bool arg, pass 1 (int) -> ValueError; pass True (bool) -> ok. For int/float arg, pass bool -> ValueError."""
        pass

    def test_validate_value_enforces_enum_membership(self):
        """For str arg with enum, pass a value not in enum and assert ValueError listing allowed values."""
        pass