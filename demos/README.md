### Authentication setup

The demos use client factories defined in `netflux/demos/client_factory.py`.
By default it reads API keys from the following files in this directory:

- `anthropic.key`
- `gemini.key`
- `openai.key`

Create each file and paste your API key as the only line of text. If you use a different
authentication flow, update the callables in `client_factory.py` before running the demos.

OpenAI, Anthropic, and Gemini have built-in providers, and the client factories are for demo purposes. These can be re-used as real client factories in your app.
Only the selected provider's SDK and key are needed.

### Interactive console viewer

The interactive demos use `ConsoleRender.run(node)` directly. The renderer owns the
terminal session, streams live `NodeView` updates, handles keyboard navigation, and leaves
the completed tree open for browsing until you quit.

### Gauntlet (`puzzle.py`)

The LLM needs to solve a series of puzzles and can't advance to the next stage until it has solved this stage's puzzle.
No limits in how many guesses it gets per stage, but it needs to get the correct answer to get the key required to advance to the next stage.

The `puzzle` demo also serves the purpose of proving that the provider is capable of a single continuous reasoning chain that envelopes the multi cycles of tool use.

`python -m netflux.demos.puzzle --provider={openai,anthropic,gemini}`

### Reasoning Continuity (`reasoning_continuity.py`)

Attempts to empirically prove correct reasoning continuity across function calls
through the agent's own experience, correlated with the source code of the provider
and framework it knows it is currently running in. The agent receives its exact
runtime identity and source locations, and may use a nonce test or another method to
correlate the source code against its own experience.

`python -m netflux.demos.reasoning_continuity --provider={openai,anthropic,gemini}`

### Performance Optimizer (`perf_opt.py`)

Profiles, critically analyzes, and iteratively optimizes a Python code target.
Uses a combination of cProfile and critical reasoning. Produces intermediate profiling and analysis
reports, and a final report summarizing changes and measured performance gains.

`python -m netflux.demos.perf_opt --provider={openai,anthropic,gemini}`

### Apply Diff (`apply_diff.py`)

Applies a multi-file unified diff (within a markdown changes doc) to a temporary workspace using the built-in `apply_diff_patch` agent. The patch exercises multiple operations: multi-hunk edits, add, delete, and rename, plus a filename containing spaces. Optionally, it can first run an intentionally failing patch to exercise rollback semantics.
The script prints the workspace path, streams a live view of the agent’s work, and then verifies that all expected file changes were applied.

Run:

`python -m netflux.demos.apply_diff --provider={openai,anthropic,gemini} [--fail-first]`

### Built-in Stress Tests (`stress_builtins.py`)

Choose one of three test modes: `bash`, `image`, or `tree`.

- `bash`: runs a relatively bare `AgentFunction` whose role is to stress-test the built-in `bash` tool that covers a superset of how any agent might use the function.
- `image`: inspect an existing asset, refine an SVG through at least ten visual revisions that require using `view_image` of the rasterized SVG, exercise downsizing and re-encoding of a large image, then derive more probes after looking at the `view_image` source and try them.
- `tree`: recursively delegate with a three-agent-level limit and relay evidence of the deepest-level warning and rejected fourth level invocation attempt.

Optionally, pass `--custom-instruction` to steer the run toward a narrower behavior or failure mode you want to probe. Only `tree` exposes an agent as a function, and does so recursively, and its delegates use the selected provider too.

The script creates a disposable workspace, changes into it before invoking the agent, streams a live tree view, and leaves the workspace on a tmp volume for inspectability.

Run:

`python -m netflux.demos.stress_builtins {bash,image,tree} --provider={openai,anthropic,gemini} [--custom-instruction "..."]`

## TUI (`tui.py`)

The puzzle, bash/image stress tests, performance optimizer, and apply-diff demos
are also available as top-level Functions in the interactive TUI.
It lets you launch multiple tree roots from one terminal session and switch between their live or completed execution trees.

Run:

`python -m netflux.demos.tui`

Each tree launch chooses its provider in the TUI launch form. For `AgentFunction` roots, the provider field defaults to that `AgentFunction`'s `default_model`; changing it overrides only that top-level invoke.
