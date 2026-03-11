### Authentication setup

The demos use client factories defined in `netflux/demos/client_factory.py`.
By default it reads API keys from the following files in this directory:

- `anthropic.key`
- `gemini.key`

Create each file and paste your API key as the only line of text. If you use a different
authentication flow, update the callables in `client_factory.py` before running the demos.

### Interactive console viewer

The interactive demos use `ConsoleRender.run(node)` directly. The renderer owns the
terminal session, streams live `NodeView` updates, handles keyboard navigation, and leaves
the completed tree open for browsing until you quit.

### Gauntlet (`puzzle.py`)

The LLM needs to solve a series of puzzles and can't advance to the next stage until it has solved this stage's puzzle.
No limits in how many guesses it gets per stage, but it needs to get the correct answer to get the key required to advance to the next stage.

The `puzzle` demo also serves the purpose of proving that the provider is capable of a single continuous reasoning chain that envelopes the multi cycles of tool use.

`python3 -m netflux.demos.puzzle --provider={gemini,anthropic}`

### Performance Optimizer (`perf_opt.py`)

Profiles, critically analyzes, and iteratively optimizes a Python code target.
Uses a combination of cProfile and critical reasoning. Produces intermediate profiling and analysis
reports, and a final report summarizing changes and measured performance gains.

`python3 -m netflux.demos.perf_opt --provider={gemini,anthropic}`

### Apply Diff (`apply_diff.py`)

Applies a multi-file unified diff (within a markdown changes doc) to a temporary workspace using the built-in `apply_diff_patch` agent. The patch exercises multiple operations: multi-hunk edits, add, delete, and rename, plus a filename containing spaces. Optionally, it can first run an intentionally failing patch to exercise rollback semantics.
The script prints the workspace path, streams a live view of the agent’s work, and then verifies that all expected file changes were applied.

Run:

`python3 -m netflux.demos.apply_diff --provider={gemini,anthropic} [--fail-first]`

### Bash Stress (`bash_stress.py`)

Runs a relatively bare `AgentFunction` whose job is to stress-test the built-in `bash` tool the way an agent would actually use it. By default it performs a broad progression of shell workflows, starting simple and becoming more sophisticated. Optionally, pass `--custom-instruction` to steer the run toward a narrower behavior or failure mode you want to probe.

The script creates a disposable workspace, changes into it before invoking the agent, streams a live tree view, and leaves the workspace on disk so you can inspect what the agent did afterward.

Run:

`python3 -m netflux.demos.bash_stress --provider={gemini,anthropic} [--custom-instruction "..."]`
