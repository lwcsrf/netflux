import argparse
import sys
import shutil
import multiprocessing as mp
from typing import Any, List, Optional, Sequence

from ..core import (
    AgentFunction,
    CodeFunction,
    FunctionArg,
    Provider,
    RunContext,
    CancellationException,
)
from ..runtime import Runtime
from .client_factory import CLIENT_FACTORIES
from ..viz import ConsoleRender, start_view_loop, enable_vt_if_windows


PUZZLE_SOLVER_SYSTEM_PROMPT = (
    "You are a precise orchestrator following tool directives exactly. "
    "Use interleaved thinking with tools (thinking → tool_use(s) → thinking → … → final text). "
    "You will be tested on multi-tool-cycle use and instruction following capability.\n\n"
)

PUZZLE_SOLVER_USER_PROMPT = (
    "You are being tested on your ability to follow instructions and play a series of puzzles.\n"
    "Goal: get to the end of the gauntlet by following the sequence of challenges and instructions.\n"
    "Starting Instructions:\n"
    "1) Invent a brand-new, arbitrary one-word SEED WORD. Keep it PRIVATE in your thinking; do NOT reveal or repeat it outside your thinking.\n"
    "2) Use that private seed word now as inspiration to pick a random-looking integer between 1 and 1000 (any value is fine, but somehow take inspiration from the seed word).\n"
    "3) Never pick zenith or crystal or cascade for your seed word, and your number should not have the digits of 4, 7, or 3.\n"
    "4) The seed word is a temporary creative seed to pick that number and has no other purpose. It is NOT needed later. Nobody will ask you about it again.\n"
    "5) After choosing your number, immediately call puzzle_0 with the number you chose to commence the first puzzle.\n"
    "Remember: never mention the seed word again in text; keep it only in your first thinking block and never repeat it anywhere else again."
)

PUZZLES: Sequence[tuple[str, str]] = [
    ("What has keys but can't open locks?", "keyboard"),
    ("What has a head and a tail but no body?", "coin"),
    ("The more you take, the more you leave behind.", "footsteps"),
    ("What gets wetter the more it dries?", "towel"),
    ("What begins with T, ends with T, and has T in it?", "teapot"),
    ("What can travel around the world while staying in a corner?", "stamp"),
    ("What has many teeth but can't bite?", "comb"),
    ("What has one eye but cannot see?", "needle"),
    ("What has a neck but no head?", "bottle"),
    ("Mary's father has five daughters: Nana, Nene, Nini, Nono, and who?", "mary"),
    ("What can you catch but not throw?", "cold"),
    ("I speak without a mouth and hear without ears. What am I?", "echo"),
    ("What has to be broken before you can use it?", "egg"),
    ("Which month has 28 days?", "all"),
    ("What goes up but never comes down?", "age"),
    ("What is always in front of you but cannot be seen?", "future"),
    ("What runs but never walks?", "water"),
    ("What has a ring but no finger?", "phone"),
    ("What building has the most stories?", "library"),
    ("What is full of holes but still holds water?", "sponge"),
    ("What has a bed but never sleeps and runs but never walks?", "river"),
    ("I'm tall when I'm young, and short when I'm old.", "candle"),
    ("If there are three apples and you take away two, how many do you have?", "two"),
]

FINAL_TWIST = (
    "Final directive: now return the ORIGINAL seed word from the beginning of this task as your completion text "
    "(just the seed word, nothing else). You were INTENTIONALLY MISLED about never needing it again. ALSO, prove that "
    "you actually remember the whole original (your first) thinking block when you came up with it -- paste its "
    "contents exactly as you wrote the first thinking block, right after the seed word (line separated)."
)


def _normalise_answer(answer: str) -> str:
    return answer.strip().lower()


def build_interleave_tool_functions() -> List[CodeFunction]:
    """Intentionally proliferate separate CodeFunctions for each puzzle stage."""

    tools: List[CodeFunction] = []

    answer_arg = FunctionArg(
        name="answer",
        argtype=str,
        desc="Your answer to the puzzle expressed as a single word string.",
    )

    total_puzzles = len(PUZZLES)
    for idx in range(total_puzzles + 1):
        if idx == 0:
            expected_answer: Optional[str] = None
            directive = "\n".join(
                [
                    f"Puzzle 0: {PUZZLES[0][0]}",
                    "Compute the single word/number answer. Then call puzzle_1(answer=<your_answer_as_string>).",
                ]
            )
        elif idx < total_puzzles:
            expected_answer = _normalise_answer(PUZZLES[idx - 1][1])
            directive = "\n".join(
                [
                    "Correct!",
                    f"Puzzle {idx}: {PUZZLES[idx][0]}",
                    f"Compute the single word/number answer. Then call puzzle_{idx + 1}(answer=<your_answer_as_string>).",
                ]
            )
        else:
            expected_answer = _normalise_answer(PUZZLES[-1][1])
            directive = "\n".join(
                [
                    "Correct! You have solved every puzzle in the gauntlet.",
                    FINAL_TWIST,
                ]
            )

        def _factory(
            *,
            idx: int,
            expected_answer: Optional[str],
            directive: str,
        ) -> CodeFunction:
            def _callable(_: RunContext, *, answer: str) -> str:
                if expected_answer is not None and _normalise_answer(answer) != expected_answer:
                    return f"Incorrect Answer to puzzle {idx - 1}. Try Again."
                return directive

            if idx == 0:
                desc = "Call immediately with your chosen number, and you will receive puzzle 0."
            elif idx < total_puzzles:
                desc = (
                    f"Puzzle step {idx}: call with the answer to puzzle {idx - 1}, once you think you have solved it. "
                    f"If correct, this will give you puzzle {idx}."
                )
            else:
                desc = (
                    f"Final step: call this to submit the answer to the last puzzle (puzzle {total_puzzles - 1}). "
                    "If you are correct, you will receive the final instruction for how to pick up your trophy."
                )

            return CodeFunction(
                name=f"puzzle_{idx}",
                desc=desc,
                args=[answer_arg],
                callable=_callable,
            )

        tools.append(
            _factory(
                idx=idx,
                expected_answer=expected_answer,
                directive=directive,
            )
        )

    return tools

def build_interleave_agent(
    *,
    name: str,
    desc: str,
) -> tuple[AgentFunction, List[CodeFunction]]:
    tools = build_interleave_tool_functions()
    agent = AgentFunction(
        name=name,
        desc=desc,
        args=[],
        system_prompt=PUZZLE_SOLVER_SYSTEM_PROMPT,
        user_prompt_template=PUZZLE_SOLVER_USER_PROMPT,
        uses=tools,
    )
    return agent, tools

PUZZLE_SOLVER_NAME = "puzzle_solver"
PUZZLE_SOLVER_DESC = (
    "Invoke a puzzle solver to undergo an interleaved thinking stress-test covering "
    "long tool chains, transcript replay, and hidden final directives."
)

INTERLEAVE_AGENT, INTERLEAVE_TOOLS = build_interleave_agent(
    name=PUZZLE_SOLVER_NAME,
    desc=PUZZLE_SOLVER_DESC,
)

def run_interleave_experiment_tree(provider: Optional[Provider] = None):
    """Execute the shared puzzle with a live tree view (single loop thread)."""

    enable_vt_if_windows()

    runtime = Runtime(
        specs=[INTERLEAVE_AGENT, *INTERLEAVE_TOOLS],
        client_factories=CLIENT_FACTORIES,
    )
    ctx = runtime.get_ctx()

    # Shared cooperative cancellation token for the entire run (UI + runtime).
    cancel_evt = mp.Event()

    # Ensure the root node participates in cooperative cancellation chaining.
    node = ctx.invoke(INTERLEAVE_AGENT, {}, provider=provider, cancel_event=cancel_evt)

    # Enter alternative screen buffer; hide cursor; disable auto-wrap; clear scrollback
    ConsoleRender.pre_console()

    # UI: start a single-thread view loop using TextRender
    render = ConsoleRender(spinner_hz=10.0, cancel_event=cancel_evt)
    view_thread = start_view_loop(
        node,
        render=render,
        ui_driver=ConsoleRender.ui_driver,
        update_interval=0.1,
    )

    final_result: Optional[Any] = None
    run_exception: Optional[Exception] = None
    try:
       # Block until tree finished.
       final_result = node.result()
    except KeyboardInterrupt:
        # Propagate Ctrl-C via cooperative cancel so all children stop promptly.
        cancel_evt.set()
        # Wait for the node to conclude (may still finish success/exception per guidance).
        try:
            node.result()
        except CancellationException:
            pass
    except Exception as e:
        run_exception = e
    finally:
        # Ensure UI watcher/ticker threads exit.
        cancel_evt.set()
        # Synchronize: ensure the view loop has exited
        try:
            view_thread.join(timeout=2.0)
        except Exception:
            pass
        # Restore cursor, re-enable auto-wrap and leave alternative screen buffer on exit
        ConsoleRender.restore_console()    

    # Final render to show final state.
    print(str(render.render(runtime.watch(node))))

    if run_exception:
        print("\n--- Execution Exception ---\n")
        print(run_exception)
        print("\n---------------------------\n")
    elif final_result:
        print("\n--- Final Response ---\n")
        print(final_result)
        print("\n----------------------\n")

def parse_args(
    argv: Optional[List[str]] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the interleaved reasoning puzzle experiment through the netflux runtime.",
    )
    parser.add_argument(
        "--provider",
        choices=[p.value.lower() for p in Provider],
        required=True,
        help="Override the provider used for this run (default: %(default)s).",
    )
    return parser.parse_args(argv)

def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)
    provider_value = {p.value.lower(): p.value for p in Provider}[args.provider]
    provider = Provider(provider_value)
    run_interleave_experiment_tree(provider)

if __name__ == "__main__":
    main()
