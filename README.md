
[![PyPI - Version](https://img.shields.io/pypi/v/netflux.svg)](https://pypi.org/project/netflux/)
[![Tests](https://img.shields.io/github/actions/workflow/status/lwcsrf/netflux/tests.yml?branch=master&label=tests)](https://github.com/lwcsrf/netflux/actions/workflows/tests.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="center">
    <img src="https://raw.githubusercontent.com/lwcsrf/netflux/master/assets/banner.png" alt="netflux banner" width="500">
</p>


**netflux** is a minimalist Python framework for authoring custom agentic applications. Its core idea is simple but powerful: **treat agents exactly like functions** — with **inputs**, **outputs**, **composition** (by calling other functions), and **side‑effects** on stateful structures.

Our goal is a framework that is:

* semantically flexible enough to express **workflows** or **dynamic, open‑ended problem solvers**, or any hybrid of the two.
* **Agents are first‑class functions.** They take typed arguments, return results, can call other functions, and leave a trace of their work.
* **Task Decomposition by design:** Compose higher-level behavior from cohesive, reusable building blocks—mirroring how we structure libraries and helper functions in traditional programming. This hierarchy is key to building reliable agents with current LLM limitations.
* **`Exception`s for agents**: let agents raise or bubble up exceptions, or attempt to handle and recover, just like traditional code.

---

## Quick-Start Demos & Development

To build an agentic app on `netflux`, just add the [pypi dependency](https://pypi.org/project/netflux/) to your project. The `demos/` are a good guide to rapidly getting started.

To run the `demos/`, you will need the provider-specific dependencies installed.
Once you make a venv, see `demos/README.md` to run any demo.

```
python -m venv .venv
source .venv/bin/activate

# Install the library in "editable" mode (`-e`), meaning your source code changes
# are immediately reflected. It also installs the `test` and `all` dependency groups,
# which include `pytest` and all the provider SDKs (Anthropic, Gemini, etc).
pip install -e .[test,all]

# Run all tests.
pytest tests/ -v
```

## The central idea: `Function`

Everything in netflux is a `Function`. There are two concrete kinds:

* **`CodeFunction`** — Deterministic Python code (your callable) with a declared signature. Think basic utilities, orchestrators, and agent decorators.

  *Example utility:* `TextEditor` (`func_lib/text_editor.py`) provides file viewing and editing commands.

* **`AgentFunction`** — An LLM‑backed function with a schema (arguments), a system prompt, and an initial user prompt template. Under the hood it runs an **agent loop** and can call other Functions in-between thinking. We casually say *“Agent”* to mean an instance of `AgentFunction`.

  *Example agent:* `ApplyDiffPatch` (`func_lib/apply_diff.py`) applies unified diff patches (including diffs fenced in markdown) to files. It focuses on **applying** a patch — tolerating small whitespace/indentation drift and other minor fuzz — while keeping **patch creation** as a separate concern. This separation lets you review diffs or delegate patch generation to a different agent, avoiding context window dilution for the patch producer (an example of what we mean by "task decomposition").

Because both kinds are just `Function`s, **any Function can call any Function**:

* code → code (classic programming),
* code → agent,
* agent → code,
* agent → agent.

Calling an agent is simply **invoking a function** whose implementation happens to be an LLM reasoning-with-tools loop.

> When we say an agent “invokes tools,” that’s physically the LLM issuing tool calls (Anthropic calls it "tools", Gemini calls it "functions"); *semantically* in netflux, those tools are just `Function`s made available to the agent. They may map to `AgentFunction`s or `CodeFunction`s.

> **Why treat agents like functions?**
> Because then composition is uniform: **code can call code or agents; agents can call code or other agents**. Classic programming already covers “code → code”; netflux enables the other three combinations in a principled way that looks consistent.

---

## What the framework gives you

* A **clear convention** for specifying agents, workflows, orchestrators, utilities, etc (your building blocks)
* A **minimal execution runtime** for running, monitoring, debugging, and tracing agent instances.
* Library of well-tested built-ins (`func_lib`).

---

## Task decomposition, by design

Good programs decompose behavior into cohesive functions. We encourage the same pattern for agents:

* A high‑level `AgentFunction` **breaks work into sub‑tasks** and delegates to more specialized sub‑`Function`s (including other agents).
* This disciplined decomposition matters for agents because **focused sub‑tasks with deliberate, limited context** are often the bottleneck to reliable LLM execution.
* **Circular references are forbidden** and **recursion is disallowed** to prevent runaway scenarios.

There is **no notion of “top” vs. “non‑top”** functions. *Any* `Function` (code or agent) can be invoked as the top‑level entry from your app, or it can be a deeply nested sub‑task inside a broader agent.

Long‑running, workflow‑like agents typically act as **orchestrators**, breaking problems into sub‑tasks that can change as progress is made.

---

## What defines an Agent

Every `AgentFunction` specifies:

* **Invocation schema**: typed args, so it can be called like a regular function. Seen by other agents.
* **Short description**: purpose and usage explanation. Seen by other agents.
* **System prompt** (usually static).
* **First (and only) user turn** (templated; substitutions using the input args).
* **How to inject specifics**—typically string substitution, but any deterministic transform is fine as long as args ⇒ concrete prompt is well‑defined.
* **Allowed `Function`s** it may call (task decomposition, sub‑agents, actuators i.e. leaf tools).
* Opt-in to the built‑in **`RaiseException`** function so the agent can proactively signal failure by raising an `AgentException`.

> **Design note:** *The agent’s logical reasoning replaces a function’s fixed code body. Otherwise, we treat agents and code functions uniformly—which is the foundation of netflux.*

### Example: `find_bug_agent`

`find_bug_agent` is an instance of `AgentFunction` (notice that usually it is not even necessary to subtype `AgentFunction`). It is an agent that inspects one or more source files in a repository, given an error message, searches for likely root causes, and **writes a short report under `/tmp`**. It returns the **absolute path** to the report file it created.

> This is an example of task decomposition: the same agent need not be concerned with both RCA'ing the bug and resolving it, in case either or both tasks require substantial effort. This minimizes the context rot and dilution that one sub-task would have on the other, much like how a human organization may delegate these tasks separately.

It uses:

* the built‑in `text_editor` tool to read/write files,
* the built‑in `raise_exception` to fail honestly when appropriate.
* a `repo_search` function (impl not shown for brevity),

```python
find_bug_agent = AgentFunction(
    name="find_bug_agent",
    desc="Inspects an error message (with as many details as possible e.g. stacktrace), "
         "searches workspace for context, and authors an in-depth bug report. "
         "Returns the absolute path to the report written under `/tmp`. "
         "Raises if details are insufficient, has trouble exploring the workspace, or lacks confidence."
    args=[
        FunctionArg("root", str, "Absolute path to the project or a finer-scoped dir subtree."),
        FunctionArg("error_message", str, "Observed error/stack trace or description of failing behavior."),
    ],
    system_prompt=(
        "You are an extremely thorough bug investigator agent (non-conversational). "
        "When invoked with an investigation, run autonomously for an extended period "
        "to explore the project subtree provided, and use extreme critical thinking to "
        # ... (mirror the intent of the `desc`; make sure your prompts are a superset)
    ),
    user_prompt_template=(
        # User prompt is usually composed of the instance-specific details made from `args`.
        "Target project subtree: {root}\n"
        "Observed error: {error_message}\n---\n"
    ),
    # Here we can list any instances of `AgentFunction` or `CodeFunction`.
    # Each callee's schematized entrypoint is automatically exposed to the agent as a tool/function.
    uses=[text_editor, repo_search, raise_exception],
    # Every LLM has types of tasks it is top-ranked for. Set a default or
    # set based on your availability constraints. Can override on each `ctx.invoke()`.
    default_model=Provider.Gemini,
)
```

---

## What exactly is a `CodeFunction`?

Many frameworks use the word *tool* for what we call **leaf `CodeFunction`s**: file viewers, string replacers, shell runners, etc. These are your lowest‑level building blocks. But higher‑level `CodeFunction`s are just as important:

* **Deterministic orchestrators**: fixed logic that fans out work to one or more `AgentFunction`s (or to other `CodeFunction`s via direct call, or via the runtime just for observability).
* **Decorators/wrappers around agents** that enhance behavior. The prime example is **`Ensemble`** (`func_lib/ensemble.py`): it decorates any `AgentFunction`; when invoked, it launches multiple independent agent runs (optionally across providers) and then forces a follow-up reconciliation of the alternative responses for enhanced results.

### Example: `fix_bug_workflow` (a `CodeFunction` orchestrator)

This orchestrator does two steps determinstically in order:

1. Invoke **`find_bug_agent`** (wrapped in an **Ensemble**) to produce a report path under `/tmp`.
2. Invoke **`bugfixer_agent`** to generate a **unified diff**.

But the second agent does another step one or more times:

3. Agent→agent: have `bugfixer_agent` write the diff to a file and invoke the built‑in **`apply_diff_patch`** agent itself to apply it. If `apply_diff_patch` raises, `bugfixer_agent` sees the exception, can revise the diff, and retry.

First we define another agent needed, the `bugfixer_agent`, and then we complete the workflow.

```python
bugfixer_agent = AgentFunction(
    name="bugfixer_agent",
    desc=(
        "Given a bug report and project (sub-)dir, emit a minimal unified diff (`git`-like) that fixes the issue, "
        "save it to a file, and apply it by calling the built-in `apply_diff_patch` agent."
    ),
    args=[
        FunctionArg("root", str, "Absolute path to the project or a finer-scoped dir subtree."),
        FunctionArg("bug_report", str, "Absolute path to the bug report."),
    ],
    system_prompt=(
        # Author a prompt that clearly instructs the agent to **plan** a fix
        # and consider alternatives, before proceeding to implement the fix.
        # This usually leads to much better results on difficult problems.
        # Emphasize: no workarounds, minimal changes but keeping cohesive architecture, etc.
        # Then instruct to use the `apply_diff_patch` as sub-agent to apply the diff.
        # This lets this agent focus on the implementation instead of getting bogged down
        # by file editor calls and without needing whitespace perfection (sub-agent handles well).
        # If `apply_diff_patch` fails, this agent can re-try.
    ),
    user_prompt_template=(
        "Target project subtree: {root}\n"
        "Bug report path: {bug_report}\n---\n"
    ),
    uses=[text_editor, apply_diff_patch, raise_exception],
    default_model=Provider.Anthropic,
)

# Wrap `find_bug_agent` with an Ensemble-of-Answers for extra reliability.
find_bug_ensemble = Ensemble(
    agent=find_bug_agent,
    instances={Provider.Anthropic: 1, Provider.Gemini: 3},
    allow_fail={Provider.Anthropic: 0, Provider.Gemini: 1},
)

def _fix_bug_workflow(ctx: RunContext, *, root: str, error_message: str) -> str:
    # 1) Find the bug (ensembled)
    report_node = ctx.invoke(find_bug_ensemble, {
        "root": root,
        "error_message": error_message,
    })
    report_path = report_node.result()  # path to report returned by the agent

    # 2) Fix the bug: generate a unified diff, write it to a file, and apply it by invoking `apply_diff_patch`.
    fix_node = ctx.invoke(bugfixer_agent, {
        "root": root,
        "bug_report": report_path,
    })
    summary = fix_node.result()  # e.g. "Target successfully patched N hunks." (and/or patch path)

    return f"{summary}\nReport: {report_path}"

# `CodeFunction`s are often just defined as instances, but sometimes it is convenient to define
# them as subclasses of `CodeFunction` (see `func_lib/text_editor.py` example).

fix_bug_workflow = CodeFunction(
    name="fix_bug_workflow",
    desc="Find the bug (ensembled), generate a minimal diff, and apply it.",
    args=[
        FunctionArg("root", str, "Absolute path to the project or a finer-scoped dir subtree."),
        FunctionArg("error_message", str, "Observed error (stack trace or message)."),
    ],
    callable=_fix_bug_workflow,
    # For `CodeFunction`s, the `uses` should still be populated because it
    # helps enforce function hierarchy to prevent risk of agent causing
    # infinite cycles or recursion runaway.
    uses=[find_bug_ensemble, bugfixer_agent, apply_diff_patch],
)
```

---

## `runtime`: definition and execution

We model each `Function` invocation as a **task** executed by a **`Node`**.

A `Node` represents the state and history of a call both **while it is executing** and **after it completes**.

> We often use the terms "function invocation", "task", and `Node` interchangeably.

* **`CodeNode`** runs a `CodeFunction` (a single Python call).
* **`AgentNode`** runs an `AgentFunction` (the provider‑specific LLM loop), **tracking**:

  * the **ordered sequence** of child `Function` calls it makes,
  * a **full transcript** of the LLM session (user/model messages, tool calls and results, and thinking blocks when available) for traceability,
  * **token usage** accumulated throughout the loop.

A **tree of `Node`s** represents a top‑level task and all of its sub‑tasks. This tree persists **after completion** (until you delete it) so you can **debug and trace** what happened.

At any point, if you snapshot the call hierarchy, it looks like a traditional **call stack**—except a frame may be an **agent** instead of a piece of deterministic code. Deeper frames tend to be **more specialized**, and at the bottom you’ll typically find **leaf `CodeFunction`s** (e.g., file IO, text replacement, running a shell command).

> **Key perspectives**
>
> * **The logical reasoning of an agent replaces the fixed code logic of a function, but otherwise we treat them the same.**
> * At any moment, a snapshot of the invocation tree reads like a **traditional call stack**—except that stack frames can be *agents* or *code*.
> * Highly specialized agents often use only **leaf tools** (e.g., read/write files) or **no tools at all** (analysis‑only). Such agents appear **deeper** in the call stack.

---

## Observability: `NodeView` and snapshots

External consumers (e.g., your UI) do **not** read `Node` objects directly—those mutate as tasks run. Instead, you consume **`NodeView`**, an **immutable, consistent, atomic snapshot** of a node and its entire subtree at a single global version.

A minimal **watch loop** facility is provided for event-driven UI and looks like this:

```python
from netflux.core import NodeState

prev = 0
while True:
    view = node.watch(as_of_seq=prev)   # blocks until there is a newer snapshot
    prev = view.update_seqnum
    
    # read view.state / view.outputs / view.exception / view.children safely
    print(f"[{view.update_seqnum}] node={view.id} state={view.state.name}")
    
    if view.state in TerminalNodeStates:
        break
```

This ensures your UI only sees **consistent** views of the task tree.

---

## `SessionBag` & Objects

In OOP, methods are functions that mutate an object’s state. netflux supports similar patterns with a **`SessionBag`**, a scoped object store with the **lifetime of a task** (and handy access to parent/root scopes). There are two important use cases in mind:

* `Function`s can stash and retrieve objects to act like methods scoped to their parent or root ancestor. A good example of this is an agent that uses `bash` to perform its task, where it needs to persist a `BashSession` across command invocations (in this case, the `BashSession` is kept in the agent-scope `SessionBag` and the children invocations retrieve it).
* `Function`s may use the `SessionBag` objects to pass input/output across sub‑tasks without serializing through text.

---

## Concurrency (fan‑out)

A task and its sub‑tasks form a **tree** where each node’s children are ordered by invocation (child edges record the call sequence). Like `Future`s, you can **launch multiple children in parallel** and defer collecting each `node.result()` until you’re ready to block. For `AgentFunction`s, this also works when the underlying model supports **parallel tool calling**.

---

## Top‑level tasks vs. sub‑tasks

A **top‑level task** is any `Function` call initiated by your app, which is external to and consuming the framework (e.g., a web handler or CLI tool). It can be a coarse orchestrator or a fine‑grained utility; **there’s no special status**—any `Function` can be called from the top or from deep inside a tree.

Long‑running, workflow‑like agents usually act as **orchestrators**, dynamically redefining sub‑tasks as progress is made. Highly specialized agents tend to live deeper in the stack and may use only **leaf `CodeFunction`s** (or even **no tools** if purely analytical). However, nothing stops your from invoking specialized agents directly to create top-level tasks — the framework is agnostic to this.

---

## Exceptions

Another core idea is that `Exception`s flow **fluidly** through `Function`s as in regular programming:

* A `CodeFunction` can `raise` for ordinary reasons (bad args, invariants, business logic). They may be raised by the framework during argument validation. Or they may be raised by the function during biz logic execution. They will be presented to agents in function call results. Most LLM providers support some way to indicate error, and we put the `Exception` stringification in the response (`providers/` extensions must do this correctly).
* An `AgentFunction` can **decide to fail** by calling the built‑in `raise_exception()`, which raises an **`AgentException`**. Example reasons include missing context, unavailable sub‑functions, irrecoverable or repeated child errors, or determining the task is unsolvable. This **reduces hallucinations** by encouraging honest failure.
* Downstream code can catch and handle exceptions, or let them **bubble** to the caller. This is true for both `CodeFunction`s via normal try/catch, and `AgentFunction`s which can be instructed how to handle various problems or not. The smarter models get, the more they can handle exceptions autonomously, provided the messages are sufficiently descriptive.

See the detailed [Exception Model](#exception-model) below for guidance on when agents should raise, bubble, or retry.

---

## Cooperative Cancellation

**Cooperative Cancellation** uses cancellation token chaining, similar to that seen in languages like C# (`CancellationToken`s) and Go (`Context`s). For now we simply use `mp.Event` for these. Since tasks can be long-running, especially when they are agents, it's imperative to be able to timely interrupt entire trees or sub-trees to save resources when agents are not going in the desired direction or progress is not meeting time deadlines, and also just for responsive user experience.

By "cooperative", we refer to the pattern of cancellation chaining that requires framework consumers to properly adhere to the pattern in order to get the benefit. This means:

* New `providers/` extensions (`AgentNode` subtypes) should check for cancellation at opportune times (before invoking children; before initial remote model invocation or before following up with function results). `AgentNode`s should `post_cancel()` in their agent loops and then simply exit the loop (return), or alternatively they can raise `cancellationException`. See `providers/anthropic.py` for an example.
* `CodeFunction` callables should similarly be responsive to `self.is_cancel_requested()` and simply raise `cancellationException` as opportune times.
* Always check for cancellation before invoking children tasks.
* Always collect running children (e.g. block on each child `node.result()`) before responding to a cancellation request.
* If an agent loop or callable is able to determine a success/exception outcome at or near the same time it would respond to cancellation, it should always prioritize concluding with success/exception instead of reacting to the cancellation request. This is because the work was done anyway, so you want the transcripts to show whatever was actually done at the time of cancellation.

---

## Providers

Providers are **subtypes of `AgentNode`** that bridge the framework’s pattern to each model’s SDK (Anthropic, Gemini, etc.). It is the **driver** that runs the agent loop and manages function calls, forwarding them through the framework. Adding a new provider means implementing a new `AgentNode` subtype. See more details below on writing a new `providers/` extension.

### Token accounting

Each agent instance **tracks token usage** over its lifetime (updated on every request/response), including input tokens (e.g., cache hits/writes vs. regular), and output tokens (e.g., reasoning vs. final text where available). See the section below for the full accounting fields and how to access them.

---

## A tiny end‑to‑end example

Below we **reuse** the earlier definitions:

* `find_bug_agent` (AgentFunction)
* `bugfixer_agent` (AgentFunction)
* `find_bug_ensemble` (CodeFunction as decorator/wrapper around AgentFunction)
* `fix_bug_workflow` (CodeFunction as simple static workflow orchestrator)
* built‑ins: `text_editor`, `apply_diff_patch`, `raise_exception`

…and wire them up in a minimal end‑to‑end run. We also show a **simple watcher** using `NodeView` to print progress.

```python
# --- Imports from netflux ---
from netflux.core import NodeState
from netflux.providers import Provider
from netflux.runtime import Runtime

# Built-ins
from netflux.func_lib.text_editor import text_editor           # CodeFunction (leaf tool)
from netflux.func_lib.apply_diff import apply_diff_patch       # AgentFunction (built-in)
from netflux.func_lib.raise_exception import raise_exception   # CodeFunction (to raise AgentException)
from netflux.func_lib.ensemble import Ensemble                 # CodeFunction decorator

# Introduced earlier in the sections above (already defined):
#   - repo_search (CodeFunction)
#   - find_bug_agent (AgentFunction)
#   - bugfixer_agent (AgentFunction)
#   - find_bug_ensemble = Ensemble(agent=find_bug_agent, ...)
#   - fix_bug_workflow (CodeFunction orchestrator)

# Demo auth factories (reads api keys for Anthropic & Gemini from file).
# Consumer must always specify the factory functions to create the LLM SDK clients
# since this configures endpoint, authorization mechanism, etc.
from netflux.demos.client_factory import CLIENT_FACTORIES

# Register everything we intend to use.
runtime = Runtime(
    specs=[
        # Our app's custom building blocks:
        repo_search, find_bug_agent, bugfixer_agent, find_bug_ensemble, fix_bug_workflow,
        # Built-ins we depend on:
        text_editor, apply_diff_patch, raise_exception
    ],
    client_factories=CLIENT_FACTORIES,
)

# Invoke the top-level `CodeFunction` task.
# We also could directly invoke any of the agents if we wanted.
root = runtime.get_ctx().invoke(
    fix_bug_workflow,
    {"root": "/repos/my_repo/sub/problem_library", "error_message": "..."}
)

# Optional: simple watcher to show progress (consistent snapshots via NodeView).
prev = 0
while True:
    view = root.watch(as_of_seq=prev)
    prev = view.update_seqnum

    print(f"[{view.update_seqnum}] node={view.id} fn={view.fn.name} state={view.state.name}")
    for child in view.children:
        print(f"  └─ child id={child.id} fn={child.fn.name} state={child.state.name}")

    if view.state in TerminalNodeStates:
        break

# Finally, print the result or surface the exception.
try:
    print(root.result())  # blocks until done; returns output or raises the exception.
except Exception as e:
    print(f"Workflow failed: {e}")
```

This refined example shows:

* **agent → code** (`find_bug_agent` and `fix_bug_agent` use `text_editor`, `repo_search`, and `raise_exception`)
* **code → agent** (the `fix_bug_workflow` orchestrator calls both agents and the built-in `apply_diff_patch`)
* **agent → agent** (`fix_bug_agent` invoking `apply_diff_patch` one or more times)
* **decorator CodeFunction** (`Ensemble`) wrapping an agent to improve reliability
* **watcher loop** using `NodeView` to print **consistent** progress updates

---

# Tips & Tricks

## Context Engineering

- The framework tries to make it easy to do effective context engineering. Usually higher-level agents will have the role of orchestration or workflow. Lower-level agents will solve more concrete patterns of problem. As LLMs become more sophisticated, a single agent can take on a broader set of responsibilities (more Functions as its disposal, and longer-running). Partition sub-tasks as fine-grained as needed but don't over-partition unnecessarily as this can degrade your evals.
- user prompt: (1) all the agent-specific context of the generic problem background (even if common to all instances this is still not system prompt), (2) the specific problem instance the agent is being invoked to do now.
- may help to be slightly repetitive of system prompt elements in user prompt to get better adherence of critical instructions.
- system prompts kept focused, relevant, small and stable: agent’s role declaration, generic task explanation, non‑negotiable rules/guardrails, output contract, meticulosity, verbosity/brevity, tool‑use policy (steer how often and when to use certain tools, beyond tool schema).
    - "you focus on performance optimization of the algorithm already select; do not propose new algorithms, just optimize impl using the one chosen."
    - "you must use tools to test performance and confirm speedups. You cannot just be speculative -- your results need to be backed up by numbers and you can admit lack of improvement."
- agents may use files for input(s)/output(s). Input filepaths would be given as args, and output filepaths may be returned (Write File tool used prior to returning).
    - Allows the same intermediate output to be re-used by multiple Functions without needing to repeat tokens.

## Misc

* structured outputs are discouraged since LLMs are sophisticated enough to parse unstructured outputs from their sub-tasks. However, sometimes strict structured outputs are critical, and this can be enforced by defining `CodeFunction`s where the arguments are the schema and the Callable performs serialization and/or verification, depending on the reason for the structured output, and returns empty or provides a filepath with the serialized data, etc. You can leverage the framework's Exceptions Model to propagate an Exception if verification fails.
* When authoring agents, place the common prompt before the specifics of the task instance. This is because LLMs are known to pay greater attention to the beginning and end of the context window. Particularly, when giving background information, whatever most heavily will influence the specific actions the agent will take should be placed closer to the end of the prompt.
* `human_in_loop()`
    - becomes blocking for human input. Human can interject and this content will present forward guidance in the "function" output.
    - implement the hook to human UI using whatever mechanism particular to your application.
    - various reasons why model may choose to invoke:
        a. sign-off at key points
        b. lacking confidence and need guidance on the task
        c. on the verge of raise_exception() and seeking opinion of what to try before doing so.

---

# Entities & Architecture

## `Function`

`Function` is the central abstraction whereby code or agents are both abstracted as merely being kinds of function calls. Specification / metadata describing the agent or code.

* Concrete subtypes must override abstract property `uses() -> List[Function]`, specifying any `Function`s that can be `invoke()`d by the `Function`.
* `AgentFunction`: can be invoked by any `Function`.
    * user subtypes to define their own agents (could use abstract properties that user must override).
    * subtype must specify: input vars, system prompt, templated user prompt (var substitution). Each var may be given as strings or filepaths (upon instantiation of the agent, files would have to be loaded and then substitution done by the runner infra instead of asking the agent to do it).
    * subtype specifies `uses: List[Function]` — the `Function`s that the agent may invoke.
* `CodeFunction`: can also be invoked by any `Function`.
    * some framework built-in subtypes (`Ensemble`, `ThinkMoreDecorator`).
    * mostly user subtypes to define any plain python functions that do some deterministic logic, intended to be invoked most often by `AgentFunction`s or as the top-level request, to coordinate sub-agents doing sub-tasks.
    * may also invoke another `CodeFunction` within their code although this will be less common.
    * points to a python function Callable. First arg is a `RunContext` which is used to invoke the framework to run a `Function`.
    * spec gives the arguments (names, types, description) without the `RunContext`. Framework will later check that the Callable matches the spec + the `RunContext` arg present. For now only allow basic primitive types (string, int, float, bool). Use python primitives to indicate types.
    * in user python code (inside the Callable), user can invoke other `Function` by doing this:
        * invoke another `CodeFunction` via:
            * Just call the callable directly (regular python code calling a function); framework does not see this happening and it's perfectly allowed.
                * in the case of `Ensemble`, user could theoretically use: `<Ensemble instance>.callable` after they have one.
                * Pass the same `RunContext` through.
                * No need to include the invoked function in the `uses()` property.
            * Use the framework runner infra to invoke, via `RunContext.invoke()`. Possible invocation of a `CodeFunction` must be declared in `uses`.
        * invoke an `AgentFunction`:
            * use the framework runner infra to invoke, via `RunContext.invoke()`. Possible invocation of an `AgentFunction` must be declared in `uses`.

## `Runtime`

`Runtime` is the top-level runner responsible for execution of trees and managing their state.

* Framework-provided object encapsulating all runner infra
* Created with a collection of user-defined `Function`s (hierarchy) that may be invoked directly or indirectly; Author defines `Function`s fully before creating a `Runtime`.
    * `runtime = Runtime(specs: List[Function], client_factories: Mapping[Provider, Callable[[], Any]])`
        * `client_factories` are factory functions that return an instance of the client type expected by each provider.
            * Used in `AgentNode` provider-specific subtypes.
            * Support pluggable configuration and authentication mechanisms unique to the consumer's app.
    * `runtime.get_ctx() -> RunContext`: return a special `RunContext` that is outside the scope of any Task (`Function` invocation).
* During registration, the runtime automatically performs a **BFS over each Function's `uses` graph** to discover and register all transitively referenced Functions. Consumers may seed with a partial set; transitives are added automatically. Duplicate names that point to different Function instances are rejected.
* Responsible for creating trees of `Node`s that execute `Function`s.
    * `RunContext.invoke()` posts `Function` invocations to the `Runtime`.
    * `Runtime` creates child `Node` for the invocation, updates the relationship in the `Node` caller, and updates its own `Node`-indexing data structures.
    * `Runtime` drives the child `Node` to start when resources are available (i.e. agent concurrency control is managed by the runtime).
        * `CodeNode` will always be started immediately.
* Provides consumer interface for querying trees of `Node`s.
    * e.g. visualization
    * `list_toplevel_views() -> List[NodeView]`: get consistent snapshot of all root tasks
    * `get_view(node_id: int) -> NodeView`: get latest snapshot for any node without blocking
    * `watch(node: Node | int, as_of_seq: int = 0, *, timeout: Optional[float] = None) -> Optional[NodeView]`: block until newer snapshot available; if `timeout` elapses, returns `None` (read more about `NodeView`s below).
    * To prevent race conditions, consumers should use `Runtime` to query state via `NodeView`s and do top-level invocations.

## `RunContext`

`RunContext` is a common framework interface used by both framework consumers and framework internal impl to invoke `Function`s.

* serves at least as the interface for:
    1. top-level task invocation; called by an app that is consuming the framework and a collection of `Function`s (the app or someone else may define these); access via `Runtime.get_ctx()`.
    2. python code for user-defined or framework-builtin `CodeFunction`s that invoke other `Function`s.
    3. when some framework component needs to handle agents doing tool calls, that component delegates invocation to the `RunContext`.
    * e.g. they all use: `ctx.invoke(fn: Function, args: Dict[str, Any], provider: Optional[Provider] = None) -> Node`
* every `Function` invocation has a `RunContext` given to it, providing the interface, but also tracking the particular `Function` using it.
    * when a `Function` invokes another `Function` (including when framework handles `AgentFunction` invoking any `Function` via tool call), the `RunContext` knows its associated invoking `Node` (identity of the caller) and causes creation of the invoked `Node`.
        * this information is used to construct the directed edges relationships of the `Node` tree. A single top-level task invocation is the parent `Node` of a tree.
* Each top-level `ctx.invoke()` (by consuming app) initiates one tree of `Node`s where the parent `Node` of the tree is the top-level Task.
    * Each top-level Task is an independent tree with `Node`s disjoint from those originating from other top-level tasks.
    * Each top-level Task may originate from the invocation of **any** `Function` that was registered with the `Runtime` constructor, thus being coarse-grained tasks or fine-grained tasks at the top level.
        * General idea: Fine-grained top-level Tasks would appear as shallow trees, that may be comparable to the deepest subtrees of a coarse-grained top-level Task that decomposes into the former -- the latter being a broader-scope task that needs to solve the former's scope of problem perhaps as a mere sub-sub-Task.
* Fields:
    * `node: Optional[Node]`: a reference to the particular `Node` identifying this specific `Function` invocation. `None` for top-level contexts.
    * `runtime: Runtime`: a reference to the shared `Runtime`.
    * `object_bags: Dict[SessionScope, SessionBag]`: references to session bags accessible at different scopes.
    * `cancel_event: Optional[Event]`: cooperative cancellation token inherited from the caller unless explicitly overridden by the caller.
* Methods:
    * `invoke(fn: Function, args: Dict[str, Any], provider: Optional[Provider] = None, cancel_event: Optional[Event] = None) -> Node`: invoke a `Function`, optionally overriding the cancellation scope, and return the created `Node`.
    * `post_status_update(state: NodeState)`: update the current node's status.
    * `post_success(outputs: Any)`: mark the current node as successful with given outputs.
    * `post_exception(exception: Exception)`: mark the current node as failed with given exception.
    * `post_cancel()`: mark the current node as terminally canceled.
    * `cancel_requested() -> bool`: helper to check whether the associated cancellation token has been triggered. This does not mean that the `Node` is already canceled and in canceled state -- it means there is active signaled *intention* to cancel.
* Narrow Scope: `RunContext` is just a mechanism to pass on `Function` invocation directives to the `Runtime` to act on them.

## `Node`

`Node` is an core abstract object that represents the invocation of a `Function` (which we also call a "Task").

* `AgentNode`: represents and manages the state and running of an `AgentFunction` invocation.
    * `AnthropicAgentNode`
        * particular implementation when the `AgentFunction` is invoked with Anthropic LLM (e.g. Opus 4.1).
    * `GeminiAgentNode`
        * particular implementation when the `AgentFunction` is invoked with Gemini LLM (e.g. Gemini Pro 2.5).
    * Tracks history of LLM session thus far (which it also uses in tool cycle when doing follow-up request)
        * Subtypes `AnthropicAgentNode` and `GeminiAgentNode` store and use the SDK-specific types in their internal impl.
    * `node.get_transcript() -> List[TranscriptPart]`
        * Subtypes must implement; they must convert the SDK-specific types in the transcription they are tracking to the framework-common `TranscriptPart`s. They never convert types in the reverse direction.
        * For external observers (UIs, tools), prefer `NodeView.transcript: tuple[TranscriptPart, ...]` which is an immutable snapshot captured at publish time. `node.get_transcript()` returns a copy of the live list and should not be used concurrently from outside the node’s thread.
    * Child `Function` invocations are tracked in `node.children: List[Node]` property.
        * Always ordered to reflect the sequence in which `Function`s were invoked. 
        * For consumers outside the framework, use `NodeView.children: tuple[NodeView, ...]` instead to access child information safely.
    * Has states (Waiting, Running, Success, Error, Canceled) but also sub-state including tool use (`Function` invocation) that it is waiting on.
    * `AgentNode` is completed once it returns final assistant text or the model decides to `RaiseException` (if it has been given as an option).
    * `TokenUsage` cumulative accounting must be reportable by every `AgentNode` and kept up to date throughout the agent loop (updated on every request/response iteration).
        * Subtype implementations must use the provider SDK's token usage meta to track the accumulation.
* `CodeNode`: represents and manages the state and running of a `CodeFunction` invocation.
    * Simpler than `AgentFunction` because it is just a function call (unlike LLM session complexity). Authors invoke `Function`s directly from within the `Callable`.
    * `CodeNode` is completed once it either returns or raises.
* Fields / Properties:
    * `id: int`: monotonically increasing unique identifier when Node is to be used as key in any lookup. This is one and the same as "task id".
    * `fn: Function`: which `Function` the `Node` is an instance of.
    * `inputs: Dict[str, Any]`: What the inputs were for the invocation.
    * `outputs: Optional[Any]`: What the output(s) were from the run (if finished). Usually just an unstructured string.
    * `exception: Optional[Exception]`: the exception, if there was an exception.
    * `state: NodeState`: (Waiting, Running, Success, Error, Canceled) enum
    * `children: List[Node]`: ordered list of child `Function` invocations made by this `Node`.
    * **Note**: External consumers should access this information through `NodeView` instead of `Node` directly to avoid race conditions.
    * For agents, `NodeView.usage` is a deep-copied snapshot and `NodeView.transcript` is an immutable tuple (empty tuple for `CodeNode`).

## `TranscriptPart`

`TranscriptPart` represents each of the parts that are common to the model-specific SDKs in concept. The subtypes:

* `UserTextPart`
* `ModelTextPart`
* `ToolUsePart`
* `ToolResultPart`
* `ThinkingBlockPart`
    * including both redacted and non-redacted
    * includes `signature` field for thinking block signatures
* On every follow-up call, replay the full history in original order.

## `Ensemble`

* Is a `CodeFunction` that decorates any `AgentFunction` to do parallel independent invocations followed by reconciliation.
    * First phase: each `AgentFunction` call proceeds as normal, with args forwarded for normal user prompt substitution.
    * Second phase: same system prompt and substituted user prompt; append to the user prompt each of the completions along with a reconciliation instruction.
* Given any `AgentFunction`, mostly users will construct one from built-in factory facility (ctor directly):
    * `Ensemble(agent: AgentFunction, instances: Dict[Provider, int], name: Optional[str] = None, reconcile_by: Optional[Provider] = None)`
        * `instances`: how many parallel invocations of `AgentFunction` to do with each model.
    * User uses this when defining their `Function`s.
* Automatically has a valid inner Callable like any `CodeFunction` that does the ensembling phases.

## `SessionBag`

* Collection of arbitrary objects that may be read, mutated, and persisted by `Function`s.
* Each `Node` created introduces a `SessionBag` with its lifetime. The `Node` and its children can access the bag.
    * Thus, the `Node` can also access its parent's `SessionBag`, if it has a parent.
* Each `Node` can also access the `SessionBag` of the root `Node`.
* `SessionScope`: enum of lifetime scopes, each of which would refer to a different `SessionBag` that a `Node` can access:
    * `TopLevel`: Lifetime envelopes all Nodes in a top-level tree. This would give the root `Node`'s bag.
    * `Parent`: Lifetime of the `Node`'s parent `Node`. This would give the parent's bag.
    * `Self`: Lifetime of the `Node` itself. This gives the `Node` access to its own bag.
        * The main application of this is for a `Function` to receive results from its children and to act as a scratchpad.
* Mechanism to do object-oriented programming
    * `Function` operates on an object and thus can behave like a method.
    * `Function` can accept arguments that refer to objects; pass data between `Node`s by in-memory strong types instead of requiring ser/des or free-form text.
* Mechanism for `Function` to own its own objects and invoke `Function`s that read/create/mutate them.
    * Example: an `AgentFunction` needs its own persistent Bash session (e.g. process tree, env vars, vars, cwd)
        * To be used at random points over its lifetime
        * Example: launch executables asynchronously (in terminal background; running locally on client); retrieve results later after doing other steps.
* `RunContext` of a `Node` carries the references to the three scopes of `SessionBag`s.
    * For a root `Node`, the `TopLevel` bag is the same as the `Self` bag. Trying to access the non-existent `Parent` bag raises `NoParentSessionError`.
    * For children of the root, the `Parent` bag is the same as the `TopLevel` bag.
    * Any deeper `Node`s will find the bags of the three `SessionScope`s to be different.
    * `RunContext.get_or_put(scope: SessionScope, namespace: str, key: str, factory: Callable[[], Any]) -> Any`
        * To simplify, this is the only mechanism to be used by `Function`s for access. Concurrency-safe in case of parallel `Function` invocations. Simplify by invoking `factory` under the lock since not high-frequency.
        * `Function` implementations should cooperate to use descript namespaces and keys, composed of static string constants and instance numbers if multiplicity is possible.
    * `Runtime` is responsible for creating `SessionBag` with each `Node` and propagating references to new descendants.
* Framework will currently rely on ref counting, garbage collection, and self-disposing object behavior (author responsibility).
    * `Runtime` destruction, or explicit user request to delete a finished tree, will induce disposal of all `SessionBag`-referenced objects and their resources.
    * This keeps objects alive long past their usable scope (potential resource leak), but is very worth the debuggability for finished subtrees. We can make this more configurable in the future (e.g. mandatory finalizers and dispose on `Node` completion).

## Exception Model

* Any `Function` can raise or bubble up an `Exception` at any point while running.
    * For `CodeFunction`, this is just for the vanilla reasons:
        * `raise TException(..)` in the `Callable`, e.g. due to contract breakage, bad args, business logic, assertion failure, etc.
        * Bubble-up: its `Callable` invokes a regular function that in turn raises and the `Callable` is unable to handle it or recover.
    * For `AgentFunction`, this is the agent making a proactive intelligent decision that it wants to raise an `Exception`.
        * All the reasons in classical programming, but also:
            * Agent is unable to do as directed because:
                * lacks context or key knowledge
                * lacks the sub-`Function`s it needs (leaf tools or sub-agents) due to author error
                * sub-agent (invoked `AgentFunction`) is not behaving as expected on a sub-task
                * an invoked child `Function` has raised, and it's unclear how to handle or it's recurring, and there are no alternatives or the alternatives have already been tried.
        * An agent may be given guidance on:
            * when and which `Exception`s from children `Function`s to recover from, versus when to bubble them up.
            * when to decide the given task is unsolvable and give up by raising.
        * Encourage the agent to declare failure to reduce the rate of hallucination.

### Framework support mechanisms:

* Framework built-in `RaiseException` (is-a `Function` subtype) intended to be provided to agents in their `AgentFunction.uses` definition, by voluntary opt-in from the agent author.
    * The spec instructs usage directives like:
        * Bubbling up an `Exception` it can't solve? Include an inner exception type and inner msg inside of the `msg` arg.
        * No alternatives worked? Very briefly describe what was tried.
        * Missing information or context, don't know how to solve, etc? Describe this very briefly for the caller in case a follow-up attempt could address this.
    * Implementation of the `RaiseException` callable is a one-liner: raise the `AgentException`. (Assume `CodeFunction` never invokes it).
* Differentiation of agent vs. service/infra faults:
    * `AgentException`: used when an agent decides to invoke `raise_exception(msg)` by its own volition, for any reason.
        * includes: faulting agent's name and instance id (`Node.id`).
    * `ModelProviderException`: used when an `AgentNode` implementation (`providers/`) fails for any reason.
        * Always unrelated to the agent's task and never caused by an agent.
        * includes: provider class name, name of agent being processed when provider faulted, instance id (`Node.id`), inner exception object.
        * Examples:
            * provider `AgentNode` malimplementation (not following protocol; not using SDK correctly)
            * connection socket broken or can't open
            * authentication / configuration
            * provider is overloaded, client is being rate-limited, any kind of load shedding or quota issue
            * core framework bug (developer regression) e.g. during `ctx.invoke(..)`.
            * other faults by the remote provider service
* Accessing `Node.result()` will either return the `Function` output (usually a string) if it was successful, or will raise the `Exception` from that invocation if there is one (similar to Futures in many languages), regardless of the invoked `Function`'s subtype.
* Provider-specific `AgentNode` subtypes shall implement this contract:
    * When collecting `Function` invocation results from `ctx.invoke(..).result()`, expect the possibility of an `Exception` being raised and always catch it.
        * Pass on a string representation of the `Exception`'s type and message (with details but never too verbose and never with stacktrace) back to the LLM in the regular follow-up tool cycle, and flag the fault if the provider's SDK has an explicit field for that. Some LLMs are fine-tuned to pay attention to the error flag but most will understand the `Exception` string properly anyway especially if the detail is present.
        * Includes `ValueError` for built-in argument type checking (LLM can respond by re-trying).
    * Implement backoff-retry around SDK `Exception`s that are known to be transient only.
    * Intercept `RaiseException` calls and use `ctx.post_exception(e)` where `e` is an instance of `AgentException`. Then exit the run loop.
    * Allow any other unexpected `Exception` to bubble past `run()`. The supertype `AgentNode` will wrap it in a `ModelProviderException` with context.
    * A batch of parallel tool calls may result in 0, 1, or more of them succeeding or excepting and this is normal.
    * When a model issues a batch of tool calls and one of them is RaiseException (unusual), honor the model's intent and propagate `AgentException` to end the agent loop after the whole batch is attempted.
* `CodeFunction` authors guidance:
    * Be aware that `Node.result()` from invoked functions may raise.
    * Ensure raisable `Exception`s from the Callable have descript type names and sufficient detail. If bubbling, sometimes this requires try-catch interception just to augment details (e.g. is the error pertaining to an input or output) and then re-raising.
    * Consider `AgentException`: may be difficult to handle statically; consider: retry, change provider. If repeatable, wrap attempts, augment context, and bubble up.
    * Consider `ModelProviderException`: Avoid backoff-retry to prevent multi-layer retry. Log in case of bug. Augment context and re-raise.
* `AgentFunction` authors guidance:
    * Add the built-in `RaiseException` to the `AgentFunction.uses` property to enlist in `AgentException`s.
    * Provide additional guidance on when to raise, when to bubble up, (or how hard to retry alternatives first) in the system and user prompt. Iterate through trial and error. This will be very specific to the agent's purpose and scope.
    * Strongly consider instructing the model to invoke `human_in_loop()` **before** considering `raise_exception()`.

## `NodeView`

`NodeView` is an immutable, consistent snapshot of a `Node` and, through reference, its subtree. Use it for observation; do not read live `Node` fields from UIs or other threads.

- Fields
  - `children: tuple[NodeView, ...]` — immutable ordered children
  - `usage: Optional[TokenUsage]` — deep-copied snapshot for agents
  - `transcript: tuple[TranscriptPart, ...]` — immutable transcript snapshot for agents (empty for `CodeNode`)
  - `update_seqnum: int` — global sequence when this view was produced
  - Plus core fields: `id`, `fn`, `inputs`, `state`, `outputs`, `exception`, `started_at`, `ended_at`

- What triggers a new `NodeView`
  - Node creation and linking into the tree
  - Status changes (`post_status_update`), success/exception/cancel
  - Transcript appends (agents call `post_transcript_update()` after each append)

- Consistency model (origin-only live rebuild)
  - On each change, the origin node’s `NodeView` is rebuilt from the live `Node` under a global lock. Each ancestor gets a fresh `NodeView` by reusing its previous snapshot fields and recomputing only `children` from current child `NodeView`s. No live ancestor fields are read.
  - Implication: an ancestor’s `usage`/`transcript` reflect the last time that ancestor itself published; child changes do not refresh them. Only `children` changes propagate up.

- Immutability guarantees
  - `children` and `transcript` are tuples in the snapshot. Provider code appends immutable `TranscriptPart`s; `ToolUsePart.args` are stored as immutable mappings; `TokenUsage` is deep-copied at snapshot time.

- Watching for updates
  - Use `node.watch(as_of_seq=prev_seq)` to block until a newer snapshot is available, then set `prev_seq = view.update_seqnum` for the next iteration.
  - `runtime.list_toplevel_views()` returns a consistent snapshot of all root `NodeView`s; `runtime.get_view(node_id)` fetches the latest view without blocking.

## Deferred Features

This is a bucket list of nice-to-haves.

* Concurrency control
    * Limit the number of `AgentNode`s in the agent loop concurrently, keyed by `Provider`.
* Replayability, Pausability, Interruptibility, cancellation
    * Pre-requisites:
        * Restart-ability of tree from the state where any `Node` was just created.
        * Serializability of `Node` tree state.
        * cancellation, Pause `NodeState`.
* `NodeState.WaitingOnFunction`
* Async and Futures
    * `RunContext.invoke()` -> return Future and generally use async chaining.
* Streaming SDK usage in model provider `AgentNode` run loops.
    * For observability, debuggability.
    * Compress results into the SDK's full native type blocks after each block is done streaming.
* Fully migrate to Event-Driven Architecture.
    * Single event loop per Runtime; remove the per-`Node` thread.
* Smarter caching based on past `Function` instance statistics.
* Cycle & Recursion Prohibition
    * Enforce at `Runtime` construction.
    * Ensure each `Function` has legal references to other `Function`s.
    * Reject if not a DAG.
    * Reject if function type annotations do not match the spec in `CodeFunction`.

---
