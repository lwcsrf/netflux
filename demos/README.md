### Authentication setup

The demos and tests use a shared client factory defined in `netflux/demos/auth_factory.py`.
By default it reads API keys from the following files in this directory:

- `anthropic.key`
- `gemini.key`

Create each file and paste your API key as the only line of text. If you prefer a different
authentication flow, update the callables in `auth_factory.py` and rerun the demos.

### Gauntlet

The LLM needs to solve a series of puzzles and can't advance to the next stage until it has solved this stage's puzzle.
No limits in how many guesses it gets per stage, but it needs to get the correct answer to get the key required to advance to the next stage.

The `puzzle` demo also serves the purpose of proving that the provider is capable of a single continuous reasoning chain that envelopes the multi cycles of tool use.

`python3 -m netflux.demos.puzzle --provider={gemini,anthropic}`
