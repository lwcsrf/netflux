import argparse
import io
import contextlib
import os
import sys
import shutil
import time
import platform
import tempfile
import subprocess
import threading
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional

from ..core import AgentFunction, CodeFunction, FunctionArg, NodeState, Provider, RunContext, CancellationException
from ..runtime import Runtime
from ..viz import ConsoleRender, start_view_loop, enable_vt_if_windows
from .client_factory import CLIENT_FACTORIES
from ..func_lib.text_editor import text_editor
from ..func_lib.raise_exception import raise_exception
 

ULTRATHINK_PROMPT = (
"""
<specify_task_importance>
You are part of an expert applied science team working on a component that will run on a spacecraft
during missions where human lives are at stake. Thus, please be extremely thorough,
critical, meticulous, and thoughtful in your work, as we have zero failure tolerance.
Your analysis should show both breadth and depth. Consider all the trade-offs and alternatives
that could be made.
</specify_task_importance>

<specify_thinking_level>
**Think ultra-hard.**
</specify_thinking_level>
"""
)

class PerfProfiler(CodeFunction):
    def __init__(self):
        super().__init__(
            name="perf_profile",
            desc=(
"""Execute and profile a self-contained Python file using cProfile, then write a
plaintext report (path is returned). The file must include both:
- setup for representative test data
- invocation of the target entrypoint using that data

Behavior:
- Sets up the profiler wrapper.
- Launches target script in subprocess via cProfile.
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
            callable=self._perf_profile,
        )

    def _perf_profile(
        self,
        ctx: RunContext, *,
        code_path: str,
        report_path: str,
    ) -> str:
        import pstats

        # Resolve input/output paths
        p = Path(code_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"code_path not found: {p}")

        out_file = Path(report_path).expanduser().resolve()
        out_file.parent.mkdir(parents=True, exist_ok=True)

        # Prepare stats file path (same dir as report for easy cleanup)
        # Place stats next to the report; avoid with_suffix() to support suffix-less filenames
        stats_path = out_file.parent / (out_file.name + ".pstats")

        # Environment: preserve current interpreter/venv and sys.path
        env = os.environ.copy()
        parent_paths = [s for s in sys.path if isinstance(s, str) and s]
        existing_pp = env.get("PYTHONPATH", "")
        merged: List[str] = []
        seen = set()
        for entry in (existing_pp.split(os.pathsep) if existing_pp else []) + parent_paths:
            if entry and entry not in seen:
                merged.append(entry)
                seen.add(entry)
        if merged:
            env["PYTHONPATH"] = os.pathsep.join(merged)
        env["PYTHONUNBUFFERED"] = "1"

        # Command: run target under cProfile, write stats to file
        cmd = [
            sys.executable,
            "-m",
            "cProfile",
            "-o",
            str(stats_path),
            str(p),
        ]

        # Start subprocess with capturing pipes
        cwd = str(p.parent)
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        # Drainers to avoid deadlocks on large outputs
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        def _drain(stream, buf):
            for chunk in iter(lambda: stream.readline(), ""):
                buf.write(chunk)

        t_out = threading.Thread(
            target=_drain, args=(proc.stdout, stdout_buf), name="profile-stdout", daemon=True)  # type: ignore[arg-type]
        t_err = threading.Thread(
            target=_drain, args=(proc.stderr, stderr_buf), name="profile-stderr", daemon=True)  # type: ignore[arg-type]

        wall_t0 = time.perf_counter()
        t_out.start()
        t_err.start()

        # Cancellation-aware wait loop
        while True:
            if ctx.cancel_requested():
                term_err: Optional[BaseException] = None
                try:
                    proc.terminate()
                    try:
                        proc.wait(timeout=1.5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=1.0)
                except BaseException as e:
                    term_err = e
                finally:
                    t_out.join(timeout=1.0)
                    t_err.join(timeout=1.0)
                if term_err:
                    raise CancellationException(
                        f"Requested to cancel during profiling (cleanup error: {type(term_err).__name__}: {term_err})"
                    )
                raise CancellationException("Requested to cancel during profiling.")
            ret = proc.poll()
            if ret is not None:
                break
            time.sleep(0.05)

        wall_elapsed = time.perf_counter() - wall_t0
        # Ensure drainers finish after process exits
        t_out.join(timeout=2.0)
        t_err.join(timeout=2.0)

        returncode = proc.returncode if proc.returncode is not None else -1
        out = stdout_buf.getvalue()
        err = stderr_buf.getvalue()

        # Build report
        s = io.StringIO()
        s.write("# PERF PROFILE REPORT\n")
        s.write(f"CODE_PATH: {p}\n")
        s.write(f"WORKING_DIR: {p.parent}\n")
        s.write(f"PYTHON: {platform.python_version()} ({sys.executable})\n")
        s.write(f"PLATFORM: {platform.platform()}\n")
        s.write(f"SCRIPT_WALL_CLOCK_S: {wall_elapsed:.6f}\n")
        s.write(f"EXIT_CODE: {returncode}\n")

        # cProfile stats (if available)
        s.write("\n== CPROFILE (top 50 by cumulative time) ==\n")
        ps = None
        ps_io = io.StringIO()
        stats_available = False
        try:
            if stats_path.exists() and stats_path.stat().st_size > 0:
                ps = pstats.Stats(str(stats_path), stream=ps_io)
                ps.strip_dirs().sort_stats("cumulative").print_stats(50)
                stats_available = True
        except Exception as e:
            ps_io.write(f"[WARNING] Failed to load pstats: {e}\n")
        s.write(ps_io.getvalue())
        if not stats_available:
            s.write("[INFO] cProfile stats unavailable (program may have exited before profile wrote).\n")

        # Hotspots (if stats available)
        s.write("\n== HOTSPOTS (cumtime desc) ==\n")
        if stats_available and ps is not None:
            try:
                stats = ps.stats  # type: ignore[attr-defined]
                rows: List[tuple[str, float, float, int, int]] = []
                for (filename, line, func_name), (cc, nc, tt, ct, _callers) in stats.items():
                    rows.append((f"{func_name} ({filename}:{line})", ct, tt, cc, nc))
                rows.sort(key=lambda r: r[1], reverse=True)
                for name, ct, tt, cc, nc in rows[:50]:
                    s.write(f"- {name}: cum={ct:.6f}s, tot={tt:.6f}s, calls={nc}/{cc}\n")
            except Exception as e:
                s.write(f"[WARNING] Failed to compute hotspots: {e}\n")
        else:
            s.write("[INFO] Hotspots unavailable due to missing stats.\n")

        # Non-zero exit diagnostic (optional)
        if returncode not in (0, None):
            s.write("\n== PROGRAM NON-ZERO EXIT ==\n")
            s.write(f"Return code: {returncode}\n")

        # Program stdout/stderr
        if out:
            s.write("\n== PROGRAM STDOUT ==\n")
            s.write(out.rstrip() + "\n")
        if err:
            s.write("\n== PROGRAM STDERR ==\n")
            s.write(err.rstrip() + "\n")

        out_file.write_text(s.getvalue(), encoding="utf-8")

        # Cleanup stats file
        with contextlib.suppress(FileNotFoundError, PermissionError, OSError):
            if stats_path.exists():
                # missing_ok available on 3.8+; use exists() guard for portability
                stats_path.unlink()

        return f"Perf Profile Report written to: {out_file}"

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
        "Do interleaved reasoning between your function calls. "
        "You are a critical performance engineer.\n"
        "## Task\n"
        "- Read the full contents of the file at `code_path`. You do not need to list or explore any directories.\n"
        "- Identify algorithmic and data-structure inefficiencies, hot loops, I/O hotspots, "
        "cache or working set inefficiencies, and Pythonism issues.\n"
        "- Recommend concrete code-level changes with rationale and expected impact.\n"
        "- Propose a quick micro-benchmark or representative input for validation.\n"
        "- Write your report to a new file at the provided `report_path`.\n"
        "- Only open the `code_path` file to do your analysis.\n"
        "- Your job is solely to be a theoretical analyst, not to execute anything yourself.\n"
        "**Do not read any other files in the same scratch directory or you might get confused by unrelated task artifacts**."
        " ## Return\n"
        "Return final message: 'Critical Analysis Report written to: {report_path}'.\n"
        " ## Exceptions\n"
        f"Only raise an exception via the {raise_exception.name} function if the `code_path` file is: "
        "unreadable; missing; clearly corrupt; non-sensical; unrelated to code."
    ),
    user_prompt_template=(
        "INPUTS:\n"
        "code_path: {code_path}\n"
        "report_path: {report_path}\n"
        "---\n"
        "Task: produce an extremely thorough analysis, as per above instructions, and write it to `report_path`. "
        "Finish with the final confirmation message (confirming your report's absolute filepath) when you're done.\n"
    ),
    uses=[text_editor, raise_exception],
    default_model=Provider.Anthropic,
)

perf_optimizer = AgentFunction(
    name="perf_optimizer",
    desc=(
        "Iteratively optimize a Python implementation using profiling and critical reasoning. "
        "Adds a test scaffold (if missing), empirically profiles, analytically reasons, and rewrites the code for performance. "
        "Re-evaluates to measure improvements, and repeats the process until exhaustion or plateau. "
        "Produces a final report with the optimized implementation, analysis of changes, and performance gains. "
        "Returns its filepath."
    ),
    args=[
        FunctionArg("input_code_path", str, "Absolute path to the initial implementation to optimize."),
        FunctionArg("scope_instruction", str, "Scope of optimization, e.g. 'optimize expensive_compute() and associated code'."),
        FunctionArg("scratch_dir", str, "Working dir to store intermediate artifacts, candidate code iterations, and intermediate/final reports."),
        FunctionArg("max_iters", int, "Maximum optimization iterations."),
    ],
    system_prompt=(
        f"{ULTRATHINK_PROMPT}\n"
        "Do interleaved reasoning between your function calls. "
        "<role>You are a critical performance engineer.</role>\n\n"
        "<instructions>\n"
        "- All inputs and outputs are file paths. Write every artifact to `scratch_dir`.\n"
        "- Take care to not have off-by-one symbol errors when passing filepaths.\n"
        "- For profiling, the `code_path` must include setup + invocation of the target on representative data. Include\n"
        "  enough input volume and repeated calls to yield rich and stable profiling (good stack/call stats).\n"
        "- If the original `input_code_path` content lacks scaffold, create a new scaffolded file in `scratch_dir` by appending a bottom section:\n"
        "  `if __name__ == '__main__':  # setup test data -> call target -> maybe print result`.\n"
        "- On each iteration, maintain a candidate file that is ready-to-profile (implementation + scaffold). The scaffold\n"
        "  should generate substantial data or loop over varied inputs to increase sample size.\n"
        f"- In parallel at the start of each iteration: call {perf_profiler.name} and {perf_reasoner.name} "
        "  on the latest iteration candidate to collect ideas for how to make it more performant.\n"
        f"- For each iteration you MUST pass distinct `report_path` values to BOTH {perf_profiler.name} and "
        f"  {perf_reasoner.name} to avoid collisions.\n"
        "- Read both reports. Next, you will spend the bulk of your time thinking super-critically about both reports.\n"
        "- Once you have analyzed the reports, synthesize a new implementation in a new file in `scratch_dir`.\n"
        "- Ensure the new candidate has proper scaffold for profiling.\n"
        "- Use robust unique filenames for all artifacts to avoid collisions: append a hyphen + hex suffix to "
        "  each output filepath you choose (e.g. profile reports, reasoner reports, candidates, final report).\n"
        "- Import/Packaging rules (critical):\n"
        "  - The file at `input_code_path` may belong to an installed package (editable install). When you create a new "
        "    candidate in `scratch_dir`, executing it with the profiler uses `exec(...)` under `__name__='__main__'` "
        "    and `__package__=None`. Therefore, RELATIVE IMPORTS WILL BREAK (e.g., `from ..parentmodule import a`).\n"
        "  - You MUST rewrite all relative imports, such as `from .x import y` or "
        "    `from ..parentmodule import a` into ABSOLUTE imports anchored at the original top-level package. "
        "    The top-level package should be in the venv and thus discoverable on sys.path.\n"
        f" - Fail early using {raise_exception.name} if this is not working out.\n"
        "  - Example: if `input_code_path` is `/a/b/c/mypkg/subpkg/impl.py`, then `from ..parentmodule import a as a1` "
        "    MUST become `from mypkg.parentmodule import a as a1`.\n"
        "- Use the perf profile report to also detect source code errors that may cause runtime failures, by looking at "
        "  the captured stderr and exception sections. If you find issues, fix them in the new candidate. "
        "- If a bug was already present in the original code and it can't be fixed after some attempts, use {raise_exception.name} to fail.\n"
        "- If the functions you invoke raise an exception, re-check your paths and the code input.\n"
        "- If you encounter a new bug in your candidate code, iterate to fix it by using the stderr of the profiler (read its report) as an executor. "
        " You can do that multiple times in a row if necessary, otherwise consider backtracking on your current direction.\n"
        "- Iterate until you hit a clear plateau: once remaining ideas stop improving the profile and it's evident no "
        "  further material gains are available, stop attempting further improvements, or stop when max_iters is reached.\n"
        "<reconciliation>\n"
        "- Parse the paths returned by the functions you invoke and read them to inform your iteration decisions.\n"
        "- Consider both the performance profiling metrics and the critical reasoning analysis when drafting iterations. "
        "  The critical reasoner may identify issues that are not visible in the profile because of the particular input data. "
        "  Thus, weigh its analysis carefully and consider if the next iteration's setup should be adjusted to surface bottlenecks "
        "  currently not visible in the profile.\n"
        "</reconciliation>\n"        
        "<final_deliverable>\n"
        "- Lastly, PRODUCE A FINAL REPORT IN `scratch_dir` CONTAINING EXACTLY:\n"
        "  1. \"Analysis & Explanation of Changes\" section\n"
        "    - Identify all bottlenecks discovered over the whole process.\n"
        "    - How they were each addressed (changes in algorithm / data structures / libraries used; "
        " cache efficiency; working set reduction; refactoring, etc).\n"
        "    - Ground all claims in both sound reasoning and empirical evidence from the profiling reports.\n"
        "  2. \"Performance Gains\" numerical summary\n"
        "    - Quantify improvements over each iteration.\n"
        "    - Quantify overall improvement from original to final.\n"
        "  3. \"Iteration Log\"\n"
        "    - Summarize what the iterations were. For each iteration, include:\n"
        f"     - What input path was used for each function/tool call involved (e.g. {perf_profiler.name}, {perf_reasoner.name}).\n"
        "      - What output path was produced for each function/tool call involved.\n"
        "  - \"New Implementation\"\n"
        "    - The final code to replace the original code that was at `input_code_path` (not including scaffold additions).\n\n"
        "</final_deliverable>\n"
        "<final_output>\n"
        "Return a string containing the absolute filepath of your final report.\n"
        "Your final completion text should be only this filepath, nothing else. Save commentary for the report.\n"
        "Example final completion text: '/{scratch_dir}/FINAL_REPORT-3f5e2a9c.md'. Not even a message, just a path.\n"
        "**This must be written explicitly as assistant/model text output, not just implied by the function result.**\n"
        "</final_output>\n"
        "</instructions>\n\n"

    ),
    user_prompt_template=(
        "## Inputs\n"
        "input_code_path: {input_code_path}\n"
        "scope_instruction: {scope_instruction}\n"
        "scratch_dir: {scratch_dir}\n"
        "max_iters: {max_iters}\n"
        "Execute the plan now.\n"
    ),
    uses=[text_editor, perf_profiler, perf_reasoner, raise_exception],
    default_model=Provider.Anthropic,
)


def make_demo_workspace() -> Dict[str, str]:
    scratch_dir = Path(tempfile.mkdtemp(prefix="perf_opt_")).resolve()

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


def run_perf_optimizer_tree(provider: Optional[Provider] = None) -> Optional[str]:
    enable_vt_if_windows()
    ws = make_demo_workspace()

    # Ensure the agents uses the same provider as the top-level optimizer
    # for the purpose of this simple demo.
    if provider:
        perf_optimizer.default_model = provider
        perf_reasoner.default_model = provider

    runtime = Runtime(
        specs=[perf_optimizer, perf_reasoner, perf_profiler, text_editor],
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
            "max_iters": 3,
        },
        provider=provider,
        cancel_event=cancel_evt,
    )

    # Enter alternate screen buffer, hide cursor, disable wrap, clear scrollback
    ConsoleRender.pre_console()

    # UI: start a single-thread background view loop to console.
    render = ConsoleRender(spinner_hz=10.0, cancel_event=cancel_evt)
    view_thread = start_view_loop(
        node,
        render=render,
        ui_driver=ConsoleRender.ui_driver,
        update_interval=0.1,
    )

    final_path: Optional[str] = None
    try:
       # Block until tree finished.
       final_path = str(node.result())
    except KeyboardInterrupt:
        # Propagate Ctrl-C via cooperative cancel so all children stop promptly.
        cancel_evt.set()
    except:
        # Handle below.
        pass
    
    # Wait for terminal state.
    node.wait()
    # Ensure UI watcher/ticker threads exit.
    cancel_evt.set()
    # Synchronize: ensure the view loop has exited
    try:
        view_thread.join(timeout=2.0)
    except Exception:
        pass
    # Restore cursor, re-enable auto-wrap and leave alternative screen buffer on exit
    ConsoleRender.restore_console()

    # Final render to show final state (regardless of success, error, canceled).
    print(str(render.render(runtime.watch(node))))

    if node.state == NodeState.Error:
        print(f"\nError: {node.exception}")
    elif node.state == NodeState.Canceled:
        print("\nCanceled.")
    elif node.state == NodeState.Success:
        print(f"\nSuccess. Final report at: {final_path}")

    return final_path


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
    if not report_path:
        return
    report_path = report_path.strip()

    try:
        report_file = Path(report_path)
        if report_file.exists():
            report_content = report_file.read_text(encoding="utf-8")
            print(f"""
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
