# Case-insensitive enum arguments

`FunctionArg("command", str, enum={"view", "create"})` accepts `"ViEw"` and gives the function `"view"`. Netflux uses `str.casefold()` on every string enum input, including inputs that already use a declared spelling. Whitespace and punctuation remain significant.

## Declaration

`FunctionArg.__post_init__()` checks the enum alongside the existing argument type checks. The options must form a nonempty set of strings, and each option must have a unique casefolded spelling. For example, both `{"US", "us"}` and `{"Straße", "STRASSE"}` raise `ValueError` when the argument is created. The argument freezes the choices and indexes each declared spelling by its casefolded value. Later changes to the caller's set cannot introduce a collision. [Declaration checks](../core.py#L148-L170)

## Trace of a tool call

Suppose a model calls the text editor with `{"command": "ViEw", "path": "ReadMe.md"}`.

1. The provider adapter records the original arguments for transcript and replay, then calls `AgentNode.invoke_tool_function()`. Shared dispatch finds the function by name, copies its arguments, and calls `RunContext.invoke()`. [Anthropic](../providers/anthropic.py#L497), [OpenAI](../providers/openai.py#L785), [Gemini](../providers/gemini.py#L536)
2. `RunContext.invoke()` forwards the call to `Runtime.invoke()`. Before creating a node, the runtime calls `fn.validate_coerce_args(inputs)`. Direct Python calls through `ctx.invoke()` enter the same path here. [Run context](../core.py#L519), [runtime invocation](../runtime.py#L179)
3. `Function.validate_coerce_args()` checks for unknown or missing argument names, applies existing scalar coercions, and passes each supplied value to `FunctionArg.resolve_value()`. After the null and type checks, **the resolver looks up `"ViEw".casefold()` and returns the declared `"view"`.** An unknown literal raises `ValueError` listing the choices. The unconstrained path stays `"ReadMe.md"`. The public `validate_value()` delegates to the resolver and returns `None` on success. [Per-argument call](../core.py#L256), [enum lookup](../core.py#L203-L210)
4. `Node.__init__()` runs the same validation when storing `node.inputs`. Resolving `"view"` again produces `"view"`. `CodeNode.run()` passes those inputs to the callable, so the editor takes its `view` branch. An agent function likewise formats its prompt from the resolved inputs. The model's original `"ViEw"` remains visible in `ToolUsePart.args`. [Node inputs](../core.py#L687), [callable](../core.py#L834), [agent prompt](../core.py#L976), [editor dispatch](../func_lib/text_editor_func.py#L119)

## Provider selection and the TUI

For an agent tool call, shared dispatch extracts the optional `provider` from the copied arguments *before* calling `RunContext.invoke()`. It uses the same resolver with the callee's `allow_models`, so `"oPeNaI"` selects `Provider.OpenAI` when allowed. An excluded provider raises `ValueError`. Omission or explicit null uses the existing inheritance and default rules. A code function that declares its own `provider` argument receives it through ordinary argument validation. The TUI launch form also calls the resolver for function arguments. [Agent tool dispatch](../core.py#L1035-L1050), [TUI form](../tui/tui.py#L923)

Provider schemas advertise the declared spellings. Anthropic and OpenAI add JSON null for optional enums, while Gemini marks them nullable. A client enforcing JSON Schema validates the advertised literals exactly. Case-insensitive matching applies when the input reaches Netflux. xAI has a provider value but its adapter is unimplemented. [Anthropic schema](../providers/anthropic.py#L641), [OpenAI schema](../providers/openai.py#L932), [Gemini schema](../providers/gemini.py#L705), [xAI dispatch](../providers/__init__.py#L35), [JSON Schema enum validation](https://json-schema.org/draft/2020-12/json-schema-validation#section-6.1.2)
