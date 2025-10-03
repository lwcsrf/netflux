import argparse
import io
import sys
import time
import traceback
import platform
import tempfile
import multiprocessing as mp
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core import AgentFunction, CancellationException, CodeFunction, FunctionArg, Provider, RunContext
from ..runtime import Runtime
from ..viz import ConsoleRender, start_view_loop
from .auth_factory import CLIENT_FACTORIES
from ..func_lib.text_editor import text_editor
from ..func_lib.ensemble import Ensemble

ULTRATHINK_PROMPT = (
"""
You are part of an expert applied science team working on a component that will run on a spacecraft
during missions where human lives are at stake. Thus, please be extremely thorough,
critical, meticulous, and thoughtful in your work, as we have zero failure tolerance.
You should be extremely liberal in reasoning tokens and take as much time as needed
to carry out exhaustive analysis. <thinking_level>**Think ultra-hard.**</thinking_level>
"""
)

class PerfProfiler(CodeFunction):
    def __init__(self):
        super().__init__(
            name="profile_perf",
            desc=(
"""Execute and profile a self-contained Python file using cProfile, then write a
plaintext report (path is returned). The file must include both:
- setup for representative test data
- invocation of the target entrypoint using that data

Behavior:
- Sets up the profiler wrapper.
- Uses exec() on the file's contents.
- Captures wall clock time and cProfile stats (sorted by cumulative time).
- Writes a human-readable report including top hot spots.

Assume profiling and execution is done using the relevant project's venv.
For example, if the code uses numpy, pandas, or any non-stdlib dependency at all,
assume it will be run in a venv where those are installed."""
            ),
            args=[
                FunctionArg("code_path", str, 
                            "Absolute path to python file under profiling (contains setup + invocation)."),
                FunctionArg("report_path", str, 
                            "Absolute filepath for the output report."),
            ],
            callable=self._profile_perf,
        )

    def _profile_perf(
        self,
        ctx: RunContext, *,
        code_path: str,
        report_path: str,
    ) -> str:
        """
        Execute and profile a self-contained Python file using cProfile, then write a
        plaintext report (path is returned). The file must include both:
        - setup for representative test data
        - invocation of the target entrypoint on that data

        Behavior:
        - Starts the profiler, then execs the file's contents (like `python file.py`).
        - Captures wall clock time and cProfile stats (sorted by cumulative time).
        - Writes a human-readable report including top hot spots.

        Notes:
        - No subprocess or venv is used; runs in-process. If your code requires isolation
        or a specific interpreter, use a wrapper file that shells out, then point here.
        """
        import cProfile
        import pstats

        p = Path(code_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"code_path not found: {p}")
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            raise RuntimeError(f"Unable to read code_path: {p}: {e}")

        # Prepare output path
        out_file = Path(report_path).expanduser().resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        pr = cProfile.Profile()
        wall_t0 = time.perf_counter()

        # Isolate globals for exec; emulate module run
        g: Dict[str, Any] = {
            "__file__": str(p),
            "__name__": "__main__",
            "__package__": None,
            "__cached__": None,
        }

        # Capture stdout/stderr produced by the program to surface in the report
        stdout_cap = io.StringIO()
        stderr_cap = io.StringIO()
        real_stdout, real_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = stdout_cap, stderr_cap
        exec_exception: Optional[str] = None

        try:
            pr.enable()
            code_obj = compile(src, str(p), "exec")
            exec(code_obj, g, g)
            pr.disable()
        except SystemExit as e:  # if the code calls sys.exit(), treat as normal completion
            pr.disable()
            exec_exception = f"SystemExit: {e}"
        except Exception:
            pr.disable()
            exec_exception = traceback.format_exc()
        finally:
            sys.stdout, sys.stderr = real_stdout, real_stderr

        wall_elapsed = time.perf_counter() - wall_t0

        # Build report
        s = io.StringIO()
        s.write("# PERF PROFILE REPORT\n")
        s.write(f"CODE_PATH: {p}\n")
        s.write(f"PYTHON: {platform.python_version()} ({sys.executable})\n")
        s.write(f"PLATFORM: {platform.platform()}\n")
        s.write(f"SCRIPT_WALL_CLOCK_S: {wall_elapsed:.6f}\n")
        if exec_exception:
            s.write("\n== PROGRAM EXCEPTION ==\n")
            s.write(exec_exception.rstrip() + "\n")

        # cProfile stats
        s.write("\n== CPROFILE (top 50 by cumulative time) ==\n")
        ps_io = io.StringIO()
        ps = pstats.Stats(pr, stream=ps_io)
        ps.strip_dirs().sort_stats("cumulative").print_stats(50)
        s.write(ps_io.getvalue())

        # Hotspots table (func key -> (cc, nc, tt, ct))
        s.write("\n== HOTSPOTS (cumtime desc) ==\n")
        stats = ps.stats  # type: ignore[attr-defined]
        rows: List[tuple[str, float, float, int, int]] = []
        for (filename, line, func_name), (cc, nc, tt, ct, _callers) in stats.items():
            rows.append((f"{func_name} ({filename}:{line})", ct, tt, cc, nc))
        rows.sort(key=lambda r: r[1], reverse=True)
        for name, ct, tt, cc, nc in rows[:50]:
            s.write(f"- {name}: cum={ct:.6f}s, tot={tt:.6f}s, calls={nc}/{cc}\n")

        # Program stdout/stderr
        out = stdout_cap.getvalue()
        err = stderr_cap.getvalue()
        if out:
            s.write("\n== PROGRAM STDOUT ==\n")
            s.write(out.rstrip() + "\n")
        if err:
            s.write("\n== PROGRAM STDERR ==\n")
            s.write(err.rstrip() + "\n")

        out_file.write_text(s.getvalue(), encoding="utf-8")
        return f"Report written to: {out_file}"

perf_profiler = PerfProfiler()


perf_reasoner = AgentFunction(
    name="perf_reasoner",
    desc=(
        "Analyze a Python implementation and propose performance improvements. Reads the file, then writes a "
        "detailed theoretical bottleneck analysis and optimization plan to a report file."
    ),
    args=[
        FunctionArg("code_path", str, "Absolute path to implementation to review."),
        FunctionArg("report_path", str, "Absolute path to write the analysis report (new file)."),
    ],
    system_prompt=(
        f"{ULTRATHINK_PROMPT}\n"
        "You are a critical performance engineer.\n"
        "- Identify algorithmic and data-structure inefficiencies, hot loops, I/O hotspots, and Pythonism issues.\n"
        "- Recommend concrete code-level changes with rationale and expected impact.\n"
        "- Propose a quick micro-benchmark or representative input for validation.\n"
        "Return exactly: 'See the report here: <report_path>'.\n"
    ),
    user_prompt_template=(
        "INPUTS:\n"
        "code_path: {code_path}\n"
        "report_path: {report_path}\n"
        "---\n"
        "Task: produce an extremely thorough analysis and write it to `report_path`. "
        "Provide the final confirmation message when done.\n"
    ),
    uses=[text_editor],
    default_model=Provider.Anthropic,
)

perf_critical_reasoner = Ensemble(
    agent=perf_reasoner,
    instances={
        Provider.Anthropic: 1,
        Provider.Gemini: 2,
    },
    allow_fail={
        Provider.Anthropic: 0,
        Provider.Gemini: 1,
    },
    name="perf_critical_reasoner",
)

perf_optimizer = AgentFunction(
    name="perf_optimizer",
    desc=(
        "Iteratively optimize a Python implementation using profiling and critical reasoning. "
        "Adds a test scaffold (if missing), empirically profiles, analytically reasons, and rewrites the code for performance. "
        "Re-evaluates to measure improvements, and repeats the process until exhaustion or plateau. "
        "Produces a final report with the optimized implementation, analysis of changes, and performance gains."
    ),
    args=[
        FunctionArg("input_code_path", str, "Absolute path to the initial implementation to optimize."),
        FunctionArg("scope_instruction", str, "Scope of optimization, e.g. 'optimize expensive_compute() and associated code'."),
        FunctionArg("scratch_dir", str, "Working dir to store intermediate artifacts, candidate code iterations, and intermediate/final reports."),
        FunctionArg("max_iters", int, "Maximum optimization iterations."),
    ],
    system_prompt=(
        f"{ULTRATHINK_PROMPT}\n"
        "ROLE: You are a critical performance engineer.\n\n"
        "CONTRACT:\n"
        "- All inputs and outputs are file paths. Write every artifact to `scratch_dir`.\n"
        "- For profiling, the `code_path` must include setup + invocation of the target on representative data. Include\n"
        "  enough input volume and repeated calls to yield rich and stable profiling (good stack/call stats).\n"
        "- If the original `input_code_path` content lacks scaffold, create a new scaffolded file in `scratch_dir` by appending a bottom section:\n"
        "  `if __name__ == '__main__':  # setup test data -> call target -> maybe print result`.\n"
        "- On each iteration, maintain a candidate file that is ready-to-profile (implementation + scaffold). The scaffold\n"
        "  should generate substantial data or loop over varied inputs to increase sample size.\n"
        "- In parallel at the start of each iteration: call {profile_perf.name} and {perf_critical_reasoner.name} "
        "  on the latest iteration candidate to collect ideas for how to make it more performant.\n"
        "- Read both reports. Next, you will spend the bulk of your time thinking super-critically about both reports.\n"
        "- Once you have analyzed the reports, synthesize a new implementation in a new file in `scratch_dir`.\n"
        "- Ensure the new candidate has proper scaffold for profiling.\n"
        "- Use robust unique filenames for all artifacts to avoid collisions: append a hyphen + hex suffix to\n"
        "  each output filepath you choose (e.g. profile reports, reasoner reports, candidates, final report).\n"
        "- Iterate until you hit a clear plateau: once remaining ideas stop improving the profile and it's evident no\n"
        "  further material gains are available, stop attempting further improvements, or stop when max_iters is reached.\n"
        "- Finally, produce a CLEAN file in `scratch_dir`: including:\n"
        "  - \"Explanation of Changes\" section\n"
        "    - identify all bottlenecks identified\n"
        "    - how they were addressed\n"
        "  - \"Performance Gains\" summary\n"
        "    - quantify improvements over each iteration.\n"
        "    - quantify overall improvement from original to final.\n"
        "  - \"Iteration Log\"\n"
        "    - summarize what the iterations were"
        "      - what input path was used for each function call involved (e.g. {profile_perf.name}, {perf_critical_reasoner.name})\n"
        "      - what output path was produced for each function call involved\n"
        "  - \"New Implementation\"\n"
        "    - Just the final code to replace the original that was at `input_code_path`.\n\n"
        "Return its absolute path.\n\n"
        "MEASUREMENT & RECONCILIATION:\n"
        "- Parse the paths returned by the functions you invoke and read them to inform your iteration decisions.\n"
        "- Consider both the performance profiling metrics and the critical reasoning analysis when drafting iterations. "
        " The critical reasoner may identify issues that are not visible in the profile because of the particular input data. "
        " Thus, weigh its analysis carefully and consider if the next iteration's setup should be adjusted to surface bottlenecks "
        " currently not visible in the profile.\n"
    ),
    user_prompt_template=(
        "INPUTS\n"
        "input_code_path: {input_code_path}\n"
        "scope_instruction: {scope_instruction}\n"
        "scratch_dir: {scratch_dir}\n"
        "max_iters: {max_iters}\n"
        "Execute the plan now.\n"
    ),
    uses=[text_editor, perf_profiler, perf_critical_reasoner],
    default_model=Provider.Anthropic,
)


def make_demo_workspace() -> Dict[str, str]:
    scratch_dir = Path(tempfile.mkdtemp(prefix="netflux_perf_opt_")).resolve()
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # Baseline implementation without scaffold
    baseline_code = (
        "def is_prime(x: int) -> bool:\n"
        "    if x < 2:\n"
        "        return False\n"
        "    for d in range(2, x):\n"
        "        if x % d == 0:\n"
        "            return False\n"
        "    return True\n\n"
        "def sum_primes(n: int = 15000) -> int:\n"
        "    s = 0\n"
        "    for i in range(2, n):\n"
        "        if is_prime(i):\n"
        "            s += i\n"
        "    return s\n"
    )
    impl_path = scratch_dir / "impl_baseline.py"
    impl_path.write_text(baseline_code, encoding="utf-8")

    return {
        "scratch_dir": str(scratch_dir),
        "input_code_path": str(impl_path),
    }


def run_perf_optimizer_tree(provider: Optional[Provider] = None) -> str:
    ws = make_demo_workspace()

    runtime = Runtime(
        specs=[perf_optimizer, perf_critical_reasoner, perf_profiler, text_editor],
        client_factories=CLIENT_FACTORIES,
    )

    ctx = runtime.get_ctx()
    cancel_evt = mp.Event()

    node = ctx.invoke(
        perf_optimizer,
        {
            "input_code_path": ws["input_code_path"],
            "scope_instruction": "Optimize sum_primes() including helpers if beneficial.",
            "scratch_dir": ws["scratch_dir"],
            "max_iters": 5,
        },
        provider=provider,
        cancel_event=cancel_evt,
    )

    def _writer(frame: str) -> None:
        sys.stdout.write("\x1b[H\x1b[2J")
        sys.stdout.write(frame)
        sys.stdout.write("\n")
        sys.stdout.flush()

    _ = start_view_loop(
        node,
        cancel_evt,
        render=ConsoleRender(spinner_hz=10.0),
        ui_driver=_writer,
        update_interval=0.1,
    )

    try:
        final_path = node.result()
    except KeyboardInterrupt:
        cancel_evt.set()
        print("\nCancellation requested, waiting for tasks to stop...\n")
        # Will raise CancellationException to terminal.
        final_path = node.result()
    finally:
        cancel_evt.set()

    return str(final_path)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the performance optimizer demo.",
    )
    parser.add_argument(
        "--provider",
        choices=[p.value.lower() for p in Provider],
        required=True,
        help="Choose the provider to use for this run.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    provider_value = {p.value.lower(): p.value for p in Provider}[args.provider]
    provider = Provider(provider_value)
    report_path = run_perf_optimizer_tree(provider)
    
    # Read and print the content of the final report
    try:
        report_file = Path(report_path)
        if report_file.exists():
            report_content = report_file.read_text(encoding="utf-8")
            print(f"""
{'=' * 80}
FINAL REPORT PATH
{'=' * 80}
{report_path}
{'=' * 80}
FINAL REPORT CONTENT
{'=' * 80}
{report_content}
{'=' * 80}
""")
        else:
            print(f"\nWarning: Report file not found at {report_path}")
    except Exception as e:
        print(f"\nError reading report file: {e}")


if __name__ == "__main__":
    main()
