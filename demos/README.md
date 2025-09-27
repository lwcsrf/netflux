### Gauntlet

The LLM needs to solve a series of puzzles and can't advance to the next stage until it has solved this stage's puzzle.
No limits in how many guesses it gets per stage, but it needs to get the correct answer to get the key required to advance to the next stage.

The `puzzle` demo also serves the purpose of proving that the provider is capable of a single continuous reasoning chain that envelopes the multi cycles of tool use.

`python3 -m skynet.demos.puzzle -provider={gemini,anthropic}`
