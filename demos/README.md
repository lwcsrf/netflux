### Authentication setup

The demos use client factories defined in `netflux/demos/auth_factory.py`.
By default it reads API keys from the following files in this directory:

- `anthropic.key`
- `gemini.key`

Create each file and paste your API key as the only line of text. If you use a different
authentication flow, update the callables in `auth_factory.py` before running the demos.

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
