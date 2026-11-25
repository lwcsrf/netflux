# Case-insensitive enum arguments

To accept `ViEw` for the text editor and `oPeNaI` for an agent's provider, Netflux should resolve each string to its declared spelling before execution. The production change fits in `core.py` and `tui/tui.py`. All three implemented provider adapters already use shared dispatch. The editor needs the resolved value because its command handler branches on the literal `view`. [Editor dispatch](../func_lib/text_editor_func.py#L109-L128)

## Where to resolve the value

| Entry point | Proposed change |
| --- | --- |
| Python and model-supplied function arguments | Give `FunctionArg` a value-returning resolver. Have the public `validate_value()` delegate to it and keep its existing `None` return contract. In `Function.validate_coerce_args()`, store the resolved value in the copied argument dictionary after the existing type coercions. Both runtime invocation and node construction pass through this method, so resolving an already canonical value must give the same result. [Argument validation](../core.py#L159-L241), [runtime](../runtime.py#L179), [node construction](../core.py#L668-L672) |
| Agent tool calls, including recursion | Resolve the injected `provider` against the callee's `allow_models` before passing it to `Provider(...)`. Omission and explicit null retain inheritance and default selection. A code function can declare its own business argument named `provider` and receive it as an ordinary argument. [Shared tool dispatch](../core.py#L990-L1035) |
| TUI launch form | Use the same resolver for enum fields in `_parse_arg()`, which currently rejects mixed case before invoking the runtime. The dedicated provider picker already accepts mixed case. [Form parsers](../tui/tui.py#L903-L935) |

Canonical values then reach Python callables, agent prompt templates and node inputs. Resolve on the invocation copy so the caller's dictionary, native provider replay and `ToolUsePart.args` retain the spelling the model actually supplied. [Callable](../core.py#L815-L819), [prompt](../core.py#L960-L968), [provider contract](../README.md#L748-L750)

## Matching and compatibility

Prefer an exact match first. Otherwise compare strings with `str.casefold()` and return the sole matching declaration. Today a user can declare both `US` and `us`. Both exact inputs should keep working, while `Us` should raise a `ValueError` naming the ambiguous choices. Unknown values should report the allowed spellings. Searching the current set on the fallback path also respects callers that mutate it after construction. [Enum declaration](../core.py#L131-L157), [Python case folding](https://docs.python.org/3/library/stdtypes.html#str.casefold)

Apply this rule only to values of declared string enums. Whitespace and punctuation remain significant, including in enum literals. Function names, argument keys, free text and the typed Python `provider=Provider.OpenAI` option keep their existing contracts.

Five demo `--provider` flags have separate lowercase-only argparse choices. CLI parity would take a `type=str.casefold` conversion on those flags. [Apply diff](../demos/apply_diff.py#L333), [performance optimization](../demos/perf_opt.py#L470), [puzzle](../demos/puzzle.py#L279), [reasoning continuity](../demos/reasoning_continuity.py#L101), [built-in stress tests](../demos/stress_builtins.py#L324)

## Provider behavior

Offline probes with Anthropic, OpenAI and Gemini reproduced the same rejection for `oPeNaI` with `OpenAI` allowed. Their SDK schema builders all emitted sorted, declared spellings. Anthropic and OpenAI include JSON null in optional enums. Gemini uses a string enum with `nullable=True`. xAI has a provider value that shared dispatch can resolve, but selecting it reaches an unimplemented adapter. [Anthropic](../providers/anthropic.py#L635-L658), [OpenAI](../providers/openai.py#L923-L953), [Gemini](../providers/gemini.py#L701-L725), [xAI](../providers/__init__.py#L23-L36)

Keep the canonical choices in tool schemas. An external validator enforcing a JSON Schema `enum` requires an exact spelling. The proposed resolver would accept variants that reach local dispatch. [JSON Schema enum validation](https://json-schema.org/draft/2020-12/json-schema-validation#section-6.1.2), [string equality](https://json-schema.org/draft/2020-12/json-schema-core#section-4.2.2)

## Tests for an implementation PR

- Exercise exact, mixed-case, Unicode, ambiguous and unknown values. Cover wrong types, whitespace, omitted optional values and explicit null. [Argument tests](../tests/test_arg_validation.py)
- Invoke the editor through `Runtime` with `ViEw` and a mixed-case path. Assert the callable succeeds, node inputs contain `view`, and the caller's arguments retain both original strings. Submit the same command through the TUI. [Editor tests](../tests/test_text_editor.py), [TUI tests](../tests/tui/test_tui_controllers.py)
- Move lowercase `gemini` from the existing rejection test into a success case. Exercise a mixed-case recursive override, an excluded provider, and a code function's own `provider` argument. Assert the selected child provider, original transcript arguments, and tool-use ID. [Agent dispatch tests](../tests/test_agentnode.py#L207-L325)
- Check the three SDK schema builders offline for canonical literals, optional nullability and the callee's provider allowlist. Run these checks with the provider extras installed. [CI dependencies](../.github/workflows/tests.yml)
