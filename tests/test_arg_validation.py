import unittest

from ..core import Function, FunctionArg

class _DummyFunction(Function):
    def __init__(self, args: list[FunctionArg]):
        super().__init__(name="dummy", desc="", args=args)

    @property
    def uses(self) -> list[Function]:
        return []


class TestFunctionBaseValidateArgs(unittest.TestCase):
    def test_validate_coerce_args_rejects_unknown_arg(self):
        """Call Function.validate_coerce_args with an unexpected key; expect ValueError listing unknown argument names."""
        fn = _DummyFunction([FunctionArg("alpha", str)])
        with self.assertRaisesRegex(
            ValueError,
            r"Unknown arg\(s\) for dummy: \['extra'\]; expected \['alpha'\]",
        ):
            fn.validate_coerce_args({"alpha": "ok", "extra": 1})

    def test_validate_coerce_args_requires_missing_required(self):
        """Omit a required arg; expect ValueError listing missing names."""
        fn = _DummyFunction([FunctionArg("required", int)])
        with self.assertRaisesRegex(
            ValueError,
            r"Missing required arg\(s\) for dummy: \['required'\]",
        ):
            fn.validate_coerce_args({})

    def test_validate_coerce_args_allows_omitted_optional(self):
        """Define optional arg; omit it; expect no error and returned mapping excludes the key."""
        fn = _DummyFunction(
            [
                FunctionArg("required", str),
                FunctionArg("maybe", int, optional=True),
            ]
        )
        result = fn.validate_coerce_args({"required": "value"})
        self.assertEqual(result, {"required": "value"})

    def test_validate_coerce_args_coerces_boolean_strings(self):
        """Provide 'true'/'false' strings for a bool arg; expect returned dict with actual bools."""
        fn = _DummyFunction(
            [
                FunctionArg("flag_true", bool),
                FunctionArg("flag_false", bool),
            ]
        )
        coerced = fn.validate_coerce_args(
            {"flag_true": " True ", "flag_false": "FALSE"}
        )
        self.assertEqual(coerced["flag_true"], True)
        self.assertEqual(coerced["flag_false"], False)

    def test_validate_coerce_args_rejects_non_boolean_string(self):
        """Provide a non-coercible string (e.g., 'yes') for bool arg; expect ValueError from validate_value complaining about type."""
        fn = _DummyFunction([FunctionArg("flag", bool)])
        with self.assertRaisesRegex(
            ValueError,
            r"Arg 'flag' expects bool, got str",
        ):
            fn.validate_coerce_args({"flag": "yes"})


class TestFunctionArgValidation(unittest.TestCase):
    def test_function_arg_rejects_unsupported_type(self):
        """Construct FunctionArg with an argtype not in {str,int,float,bool} (e.g., list) and assert ValueError."""
        with self.assertRaisesRegex(
            ValueError,
            r"Arg 'bad' has unsupported type",
        ):
            FunctionArg("bad", list)  # type: ignore[arg-type]

    def test_function_arg_enum_requires_string_type(self):
        """Attempt enum on non-str argtype (e.g., int) and assert ValueError explaining enum only for str."""
        with self.assertRaisesRegex(
            ValueError,
            r"enum constraint is only supported for str args",
        ):
            FunctionArg("choice", int, enum={"one", "two"})  # type: ignore[arg-type]

    def test_function_arg_enum_requires_nonempty_all_strings(self):
        """Give empty set or a set with non-strings and assert ValueError; check message mentions non-empty and string-only expectation."""
        with self.subTest(case="empty"):
            with self.assertRaisesRegex(
                ValueError,
                r"enum must be a set of string literals",
            ):
                FunctionArg("option", str, enum=set())
        with self.subTest(case="non_string"):
            with self.assertRaisesRegex(
                ValueError,
                r"enum values must be strings",
            ):
                FunctionArg("option", str, enum={"ok", 1})  # type: ignore[arg-type]

    def test_validate_value_allows_none_when_optional(self):
        """Create optional FunctionArg and call validate_value(None); expect no exception."""
        arg = FunctionArg("maybe", str, optional=True)
        arg.validate_value(None)

    def test_validate_value_rejects_none_when_required(self):
        """Create required FunctionArg and call validate_value(None); expect ValueError stating arg is required."""
        arg = FunctionArg("need", str)
        with self.assertRaisesRegex(
            ValueError,
            r"Arg 'need' is required and cannot be None",
        ):
            arg.validate_value(None)

    def test_validate_value_rejects_incorrect_type(self):
        """For int arg, pass a float (or for float pass int/bool per exactness rules) and assert ValueError with type detail."""
        arg = FunctionArg("count", int)
        with self.assertRaisesRegex(
            ValueError,
            r"Arg 'count' expects int, got float",
        ):
            arg.validate_value(3.14)

    def test_validate_value_enforces_bool_exact_type(self):
        """For bool arg, pass 1 (int) -> ValueError; pass True (bool) -> ok. For int/float arg, pass bool -> ValueError."""
        bool_arg = FunctionArg("flag", bool)
        with self.assertRaisesRegex(
            ValueError,
            r"Arg 'flag' expects bool, got int",
        ):
            bool_arg.validate_value(1)
        bool_arg.validate_value(True)

        int_arg = FunctionArg("amount", int)
        with self.assertRaisesRegex(
            ValueError,
            r"Arg 'amount' expects int, got bool",
        ):
            int_arg.validate_value(True)

        float_arg = FunctionArg("ratio", float)
        with self.assertRaisesRegex(
            ValueError,
            r"Arg 'ratio' expects float, got bool",
        ):
            float_arg.validate_value(False)

    def test_validate_value_enforces_enum_membership(self):
        """For str arg with enum, pass a value not in enum and assert ValueError listing allowed values."""
        arg = FunctionArg("mode", str, enum={"read", "write"})
        with self.assertRaisesRegex(
            ValueError,
            r"Arg 'mode' must be one of: read, write; got 'delete'",
        ):
            arg.validate_value("delete")


if __name__ == "__main__":
    unittest.main()
