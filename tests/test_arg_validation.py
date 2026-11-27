import unittest
from enum import Enum

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

    def test_enum_input_resolves_to_declared_spelling(self):
        mode = FunctionArg("mode", str, enum={"Straße"})
        inputs = {"mode": "STRASSE"}
        self.assertEqual(mode.resolve_value(inputs["mode"]), "Straße")
        self.assertEqual(_DummyFunction([mode]).validate_coerce_args(inputs), {"mode": "Straße"})
        self.assertEqual(inputs, {"mode": "STRASSE"})

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
        """An unrecognized boolean string raises ValueError."""
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

    def test_function_arg_rejects_casefold_collisions_at_construction(self):
        with self.assertRaisesRegex(ValueError, "unique after casefolding"):
            FunctionArg("mode", str, enum={"Straße", "STRASSE"})

    def test_string_enum_choice_resolves_to_plain_string(self):
        class Mode(str, Enum):
            READ_ONLY = "ReadOnly"

        arg = FunctionArg("mode", str, enum={Mode.READ_ONLY})
        resolved = arg.resolve_value("readonly")
        self.assertEqual(resolved, "ReadOnly")
        self.assertIs(type(resolved), str)

    def test_non_string_enum_choice_is_rejected(self):
        class Mode(Enum):
            READ_ONLY = "ReadOnly"

        with self.assertRaisesRegex(ValueError, "enum values must be strings"):
            FunctionArg("mode", str, enum={Mode.READ_ONLY})  # type: ignore[arg-type]

    def test_function_arg_stores_immutable_enum_choices(self):
        declared_choices = {"ReadOnly"}
        mode = FunctionArg("mode", str, enum=declared_choices)
        self.assertIsInstance(mode.enum, frozenset)
        declared_choices.add("READONLY")
        self.assertEqual(mode.enum, {"ReadOnly"})

    def test_resolve_value_allows_none_when_optional(self):
        arg = FunctionArg("maybe", str, optional=True)
        self.assertIsNone(arg.resolve_value(None))

    def test_resolve_value_rejects_none_when_required(self):
        arg = FunctionArg("need", str)
        with self.assertRaisesRegex(
            ValueError,
            r"Arg 'need' is required and cannot be None",
        ):
            arg.resolve_value(None)

    def test_resolve_value_coerces_integral_float_to_int(self):
        arg = FunctionArg("count", int)
        for value, expected in [(3.0, 3), (-2.0, -2), (-0.0, 0)]:
            with self.subTest(value=value):
                resolved = arg.resolve_value(value)
                self.assertIs(type(resolved), int)
                self.assertEqual(resolved, expected)

    def test_resolve_value_rejects_fractional_and_nonfinite_floats_for_int_arg(self):
        arg = FunctionArg("count", int)
        for value in [3.14, float("nan"), float("inf"), float("-inf")]:
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, r"Arg 'count' expects int, got float"):
                    arg.resolve_value(value)

    def test_resolve_value_distinguishes_booleans_from_numbers(self):
        bool_arg = FunctionArg("flag", bool)
        with self.assertRaisesRegex(
            ValueError,
            r"Arg 'flag' expects bool, got int",
        ):
            bool_arg.resolve_value(1)
        self.assertIs(bool_arg.resolve_value(True), True)

        int_arg = FunctionArg("amount", int)
        with self.assertRaisesRegex(
            ValueError,
            r"Arg 'amount' expects int, got bool",
        ):
            int_arg.resolve_value(True)

        float_arg = FunctionArg("ratio", float)
        with self.assertRaisesRegex(
            ValueError,
            r"Arg 'ratio' expects float, got bool",
        ):
            float_arg.resolve_value(False)

    def test_resolve_value_enforces_enum_membership(self):
        arg = FunctionArg("mode", str, enum={"read", "write"})
        with self.assertRaisesRegex(
            ValueError,
            r"Arg 'mode' must be one of: read, write; got 'delete'",
        ):
            arg.resolve_value("delete")


if __name__ == "__main__":
    unittest.main()
