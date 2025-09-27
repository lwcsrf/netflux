# Defines:
#   - ViewFile (CodeFunction): reads a file by absolute path
#   - AgentA   (AgentFunction): uses ViewFile via tool call to read & explain the file
#   - Outer    (CodeFunction):  invokes AgentA and returns its text result

import os
from typing import Optional

# Import the framework (unchanged).
from ..core import (
    FunctionArg,
    CodeFunction,
    AgentFunction,
    Provider,
    RunContext,
    NodeView,
)
from ..runtime import Runtime

# This test file is the demo file.
DEMO_FILE_ABS_PATH: Optional[str] = os.path.abspath(__file__)

# ------------ CodeFunction: ViewFile ------------
def view_file_callable(ctx: RunContext, *, path: str) -> str:
    """
    Read the file at `path` and return its text content.
    Enforces absolute path; decodes as UTF-8 with replacement.
    Truncates extremely large files to keep the LLM payload reasonable.
    """
    import io

    if not os.path.isabs(path):
        raise ValueError(f"ViewFile: expected absolute path, got: {path!r}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"ViewFile: file does not exist: {path}")

    # Read safely as text.
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = f.read()

    # Trim if huge (simple guardrail)
    MAX_CHARS = 50_000
    if len(data) > MAX_CHARS:
        head = data[:45_000]
        tail = data[-4_000:]
        data = (
            head
            + "\n\n[...TRUNCATED...]\n\n"
            + tail
        )
    return data

ViewFile = CodeFunction(
    name="ViewFile",
    desc="Read a source file by absolute path and return its text contents.",
    args=[FunctionArg("path", str, "Absolute file path to read")],
    callable=view_file_callable,
)

# ------------ AgentFunction: AgentA ------------
AgentA = AgentFunction(
    name="AgentA",
    desc="Explain what is happening inside a given source code file.",
    args=[FunctionArg("filepath", str, "Absolute file path to analyze")],
    system_prompt=(
        "You are AgentA, a careful code explainer.\n"
        "Rules:\n"
        "- You must call the tool `ViewFile` exactly once at the start to fetch the file contents using the provided absolute path.\n"
        "- Do not invent code; rely only on the tool result.\n"
        "- After reading, produce a concise explanation in Markdown with sections:\n"
        "  Overview, Main Components, Execution Flow, Notable Patterns, Potential Risks.\n"
        "- If the tool errors, explain the error briefly and stop.\n"
    ),
    user_prompt_template=(
        "Analyze the code in the file at this absolute path:\n"
        "{filepath}\n\n"
        "First, call the tool `ViewFile` with the argument `path` set to the filepath above.\n"
        "Only after reading the tool's output should you write your explanation."
    ),
    uses=[ViewFile],
    default_model=Provider.Anthropic
)

# ------------ CodeFunction: Outer (invokes AgentA) ------------
def outer_callable(ctx: RunContext) -> str:
    """
    Simple orchestrator that invokes AgentA on DEMO_FILE_ABS_PATH
    and returns AgentA's final text as its own output.
    """
    child = ctx.invoke(AgentA, {"filepath": DEMO_FILE_ABS_PATH})
    result_text = child.result() or ""
    return result_text

Outer = CodeFunction(
    name="Outer",
    desc="Invoke AgentA on a fixed absolute file path and return its text result.",
    args=[],  # no args; uses the constant path above
    callable=outer_callable,
)

# ------------ Run the end-to-end task ------------
def pretty_tree(view: NodeView, indent: int = 0) -> None:
    pad = "  " * indent
    print(f"{pad}- [{view.state.value}] {view.fn.name} (id={view.id})")
    for child in view.children:
        pretty_tree(child, indent + 1)

def main():
    print("=== netflux demo: end-to-end ===")
    print(f"File to analyze: {DEMO_FILE_ABS_PATH}")

    # Register all functions with the Runtime
    runtime = Runtime(specs=[ViewFile, AgentA, Outer])

    # Kick off the top-level task (Outer)
    ctx = runtime.get_ctx()
    root = ctx.invoke(Outer, {})

    # Wait and collect result text
    output = root.result()

    print("\n=== AgentA Output ===\n")
    if isinstance(output, str):
        print(output)
    else:
        print(str(output))

    print("\n=== Execution Tree ===")
    latest: NodeView = runtime.watch(root)
    pretty_tree(latest)

if __name__ == "__main__":
    main()
