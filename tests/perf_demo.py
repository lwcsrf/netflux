#!/usr/bin/env python3
from __future__ import annotations

import sys
import os
import time
import textwrap
import tempfile
import subprocess
import threading
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Import your framework core (with your own GeminiAgentNode / AnthropicAgentNode)
# ──────────────────────────────────────────────────────────────────────────────
from .. import core
from ..runtime import Runtime
from ..demos.auth_factory import CLIENT_FACTORIES

# ──────────────────────────────────────────────────────────────────────────────
# Simple tracing utilities
# ──────────────────────────────────────────────────────────────────────────────
def _ts() -> str:
    return time.strftime("%H:%M:%S")

def trace(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)

def _indent(depth: int) -> str:
    return "  " * depth


# ──────────────────────────────────────────────────────────────────────────────
# CodeFunctions
# ──────────────────────────────────────────────────────────────────────────────
def _read_text_file(ctx: core.RunContext, *, filepath: str) -> str:
    p = Path(filepath)
    if not p.exists():
        raise FileNotFoundError(str(p))
    s = p.read_text(encoding="utf-8", errors="replace")
    trace(f"CodeFunction read_text_file: read {len(s)} chars from {p}")
    return s

ReadTextFile = core.CodeFunction(
    name="read_text_file",
    desc="Read a UTF-8 text file and return its contents.",
    args=[core.FunctionArg("filepath", str, "Absolute path to a text file")],
    callable=_read_text_file,
)

def _write_text_file(ctx: core.RunContext, *, filepath: str, content: str, overwrite: bool) -> str:
    p = Path(filepath)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {p}")
    p.write_text(content, encoding="utf-8")
    trace(f"CodeFunction write_text_file: wrote {len(content)} chars to {p}")
    return str(p.resolve())

WriteTextFile = core.CodeFunction(
    name="write_text_file",
    desc="Write text content to a file. Returns the absolute path.",
    args=[
        core.FunctionArg("filepath", str, "Absolute output file path"),
        core.FunctionArg("content", str, "Text content to write"),
        core.FunctionArg("overwrite", bool, "Set true to overwrite if the file exists"),
    ],
    callable=_write_text_file,
)

def _venv_python(venv_path: str) -> str:
    vp = Path(venv_path)
    if sys.platform.startswith("win"):
        py = vp / "Scripts" / "python.exe"
    else:
        py = vp / "bin" / "python"
    if not py.exists():
        raise FileNotFoundError(f"Python interpreter not found in venv: {py}")
    return str(py)

def _perf_profile_tool(ctx: core.RunContext, *, test_script: str, venv_path: str, target: str) -> str:
    """
    - Syntax-checks the test_script in the venv
    - Profiles it with cProfile in the venv
    - Filters stats to the target function and its callees
    - Writes a rich plaintext report to /tmp/... and returns that path (string)
    """
    py = _venv_python(venv_path)
    test_script_path = Path(test_script).resolve()
    if not test_script_path.exists():
        raise FileNotFoundError(f"test_script not found: {test_script_path}")

    out_txt = Path(tempfile.mkstemp(prefix="netflux_profile_", suffix=".txt")[1])
    trace(f"CodeFunction PerfProfileTool: compile-check {test_script_path}")
    comp = subprocess.run(
        [py, "-m", "py_compile", str(test_script_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        cwd=str(test_script_path.parent),
    )
    if comp.returncode != 0:
        report = []
        report.append("# PERF PROFILE REPORT (COMPILE ERROR)")
        report.append(f"TEST_SCRIPT: {test_script_path}")
        report.append(f"TARGET_FUNC: {target}")
        report.append("")
        report.append("== PY_COMPILE STDERR ==")
        report.append(comp.stderr.rstrip())
        out_txt.write_text("\n".join(report), encoding="utf-8")
        trace(f"CodeFunction PerfProfileTool: compile error, wrote report {out_txt}")
        return str(out_txt)

    wrapper = r"""
import sys, runpy, cProfile, pstats, io, time, traceback, platform
from pathlib import Path

def main():
    test_script = Path(sys.argv[1]).resolve()
    target_name = sys.argv[2]
    out_path = Path(sys.argv[3]).resolve()
    try:
        pr = cProfile.Profile()
        wall_t0 = time.perf_counter()
        pr.enable()
        runpy.run_path(str(test_script), run_name="__main__")
        pr.disable()
        wall_elapsed = time.perf_counter() - wall_t0

        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s)
        ps.strip_dirs()
        stats = ps.stats  # func -> (cc, nc, tt, ct, callers)

        call_graph = {}
        for callee, (_, _, _, _, callers) in stats.items():
            for caller in callers.keys():
                call_graph.setdefault(caller, set()).add(callee)

        target_candidates = [f for f in stats.keys() if f[2] == target_name]
        target_key, best_ct = None, -1.0
        for f in target_candidates:
            ct = stats[f][3]
            if ct > best_ct:
                target_key, best_ct = f, ct

        with out_path.open("w", encoding="utf-8") as fh:
            fh.write("# PERF PROFILE REPORT\n")
            fh.write(f"HOST: {platform.node()}  PY: {sys.version.split()[0]}  PLATFORM: {platform.platform()}\n")
            fh.write(f"TEST_SCRIPT: {test_script}\n")
            fh.write(f"TARGET_FUNC: {target_name}\n")
            fh.write(f"SCRIPT_WALL_CLOCK_S: {wall_elapsed:.6f}\n")
            fh.write("\n")

            if not target_key:
                fh.write("TARGET_STATUS: NOT_FOUND\n")
                fh.write("NOTE: Target name not found in cProfile stats. Showing global top by CUMTIME.\n\n")
                rows = []
                for f, (cc, nc, tt, ct, callers) in stats.items():
                    rows.append((ct, tt, nc, cc, f))
                rows.sort(reverse=True)
                fh.write("GLOBAL_TOP_BY_CUMTIME:\n")
                fh.write("ct_s  tt_s  nc  cc  file:line func\n")
                for (ct, tt, nc, cc, f) in rows[:50]:
                    fh.write(f"{ct:8.6f}  {tt:8.6f}  {nc:6d} {cc:6d}  {f[0]}:{f[1]} {f[2]}\n")
                return

            fh.write("TARGET_STATUS: FOUND\n")
            tgt_cc, tgt_nc, tgt_tt, tgt_ct, _ = stats[target_key]
            fh.write("METRICS:\n")
            fh.write(f"  target.cumtime_s: {tgt_ct:.6f}\n")
            fh.write(f"  target.tottime_s: {tgt_tt:.6f}\n")
            fh.write(f"  target.calls: {tgt_nc}\n")
            fh.write("\n")

            reachable = set()
            stack = [target_key]
            while stack:
                cur = stack.pop()
                if cur in reachable: continue
                reachable.add(cur)
                for callee in call_graph.get(cur, ()):
                    if callee not in reachable:
                        stack.append(callee)

            rows = []
            for f in reachable:
                cc, nc, tt, ct, _ = stats[f]
                rows.append((ct, tt, nc, cc, f))
            rows.sort(reverse=True)

            fh.write("REACHABLE_TOP_BY_CUMTIME (TARGET and callees):\n")
            fh.write("ct_s     tt_s     nc     cc     file:line func\n")
            for (ct, tt, nc, cc, f) in rows[:80]:
                fh.write(f"{ct:8.6f} {tt:8.6f} {nc:6d} {cc:6d}  {f[0]}:{f[1]} {f[2]}\n")

            fh.write("\nRAW_PSTATS_SUMMARY:\n")
            ps.sort_stats('cumulative').print_stats(40)
            fh.write(s.getvalue())

    except SystemExit:
        raise
    except Exception:
        with out_path.open("w", encoding="utf-8") as fh:
            fh.write("# PERF PROFILE REPORT (RUNTIME ERROR)\n")
            fh.write(f"TEST_SCRIPT: {test_script}\n")
            fh.write(f"TARGET_FUNC: {target_name}\n\n")
            fh.write(traceback.format_exc())

if __name__ == "__main__":
    main()
"""
    wrapper_path = Path(tempfile.mkstemp(prefix="netflux_profwrap_", suffix=".py")[1])
    wrapper_path.write_text(wrapper, encoding="utf-8")

    trace("CodeFunction PerfProfileTool: running profiler wrapper in venv")
    run = subprocess.run(
        [_venv_python(venv_path), str(wrapper_path), str(test_script_path), str(target), str(out_txt)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        cwd=str(test_script_path.parent),
    )
    if run.stdout.strip():
        trace("CodeFunction PerfProfileTool (stdout): " + run.stdout.strip())
    if run.stderr.strip():
        trace("CodeFunction PerfProfileTool (stderr): " + run.stderr.strip())
    trace(f"CodeFunction PerfProfileTool: wrote report {out_txt}")
    return str(out_txt)

PerfProfileTool = core.CodeFunction(
    name="PerfProfileTool",
    desc=("Profiles a Python test script inside the given venv; filters stats to the target function "
          "and its callees; writes a rich plaintext report to /tmp and returns its absolute path."),
    args=[
        core.FunctionArg("test_script", str, "Absolute path to a self-contained test script"),
        core.FunctionArg("venv_path", str, "Absolute path to a Python virtual environment"),
        core.FunctionArg("target", str, "Name of the function to profile"),
    ],
    callable=_perf_profile_tool,
)


# ──────────────────────────────────────────────────────────────────────────────
# AgentFunctions
# ──────────────────────────────────────────────────────────────────────────────
PerfReasoningAgent = core.AgentFunction(
    name="PerfReasoningAgent",
    desc="Analyze profiling text and/or source code to identify bottlenecks and propose concrete edits.",
    args=[
        core.FunctionArg("profile_text", str, "Plaintext profile report contents ('' if unavailable)"),
        core.FunctionArg("source_text", str, "Function source code ('' if unavailable)"),
    ],
    system_prompt=textwrap.dedent("""
        You are a performance engineer. Be surgical and specific.
        Extract hard insights from the profile and/or source: name exact hot functions/loops/lines,
        and propose small, testable code edits that preserve semantics. Explain why each change helps
        and how to measure it. Avoid vague tips.
    """).strip(),
    user_prompt_template=textwrap.dedent("""
        PROFILE REPORT:
        ---------------
        {profile_text}

        SOURCE:
        -------
        {source_text}

        TASK:
        - Identify the dominant bottlenecks (precise, grounded in numbers).
        - Recommend concrete edits and why they help.
        - Suggest how to verify improvements (metrics).
    """).strip(),
    uses=[ReadTextFile],
    default_model=core.Provider.Anthropic,
)

PerfImprovementAgent = core.AgentFunction(
    name="PerfImprovementAgent",
    desc="Iteratively profile → reason → rewrite → re-profile until gains taper off.",
    args=[
        core.FunctionArg("function_name", str, "Name of the function under optimization"),
        core.FunctionArg("initial_source", str, "Initial source code for that function"),
        core.FunctionArg("function_file_path", str, "Path to write the function code"),
        core.FunctionArg("test_script_path", str, "Path to the bench/test script"),
        core.FunctionArg("venv_path", str, "Path to the Python virtual environment"),
        core.FunctionArg("max_iters", int, "Maximum iterations"),
        core.FunctionArg("improvement_threshold_pct", float, "Stop if speedup < this percent vs previous"),
    ],
    system_prompt=textwrap.dedent("""
        ROLE: Performance Orchestrator.

        LOOP UNTIL DONE:
          1) Ensure the current function is written to {function_file_path} using write_text_file(..., overwrite=True).
          2) Call PerfProfileTool(test_script={test_script_path}, venv_path={venv_path}, target={function_name});
             it returns a PATH STRING to a plaintext report.
          3) Call read_text_file(filepath=that_path) to load the report text.
          4) Call PerfReasoningAgent(profile_text=<report>, source_text=read_text_file({function_file_path}))
             to decide concrete code edits.
          5) Overwrite {function_file_path} with your improved version using write_text_file(overwrite=True).
          6) Re-profile and compare "target.cumtime_s" and "SCRIPT_WALL_CLOCK_S" to the prior iteration.

        STOPPING RULES:
          - Stop if you've reached the iteration budget.
          - Stop if speedup vs previous iteration is less than improvement_threshold_pct.
          - If the profile shows errors or target not found, explain and stop gracefully.

        OUTPUT EACH ITERATION:
          - Iteration header
          - Old vs new metrics (target.cumtime_s, SCRIPT_WALL_CLOCK_S)
          - Edits applied (short bullet list)

        FINAL OUTPUT:
          - Baseline vs final metrics
          - Total speedup
          - Key edits that mattered
    """).strip(),
    user_prompt_template=textwrap.dedent("""
        INPUTS
        ------
        function_name: {function_name}
        function_file_path: {function_file_path}
        test_script_path: {test_script_path}
        venv_path: {venv_path}
        max_iters: {max_iters}
        improvement_threshold_pct: {improvement_threshold_pct}

        INITIAL BASELINE SOURCE
        -----------------------
        {initial_source}

        Begin now. Follow the tool calling plan exactly.
    """).strip(),
    uses=[PerfProfileTool, ReadTextFile, WriteTextFile, PerfReasoningAgent],
    default_model=core.Provider.Anthropic,
)


# ──────────────────────────────────────────────────────────────────────────────
# Demo workspace
# ──────────────────────────────────────────────────────────────────────────────
def make_demo_workspace() -> Dict[str, str]:
    """
    Creates a temporary workspace with:
      - venv
      - a 'work' directory
      - placeholder path for the function module (agent will write it)
      - a bench script that imports the function and runs it once
    Returns a dict with paths and the baseline function source.
    """
    root = Path(tempfile.mkdtemp(prefix="netflux_demo_")).resolve()
    trace(f"Workspace: {root}")

    venv_dir = root / "venv"
    trace("Creating venv...")
    subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
    trace(f"Venv ready: {venv_dir}")

    workdir = root / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    trace(f"Workdir: {workdir}")

    function_file_path = workdir / "target_module.py"  # the agent overwrites this each iter

    baseline_function = textwrap.dedent("""
        def is_prime(x: int) -> bool:
            if x < 2:
                return False
            for d in range(2, x):
                if x % d == 0:
                    return False
            return True

        def sum_primes(n: int = 7000) -> int:
            s = 0
            for i in range(2, n):
                if is_prime(i):
                    s += i
            return s
    """).strip()

    bench_script = textwrap.dedent(f"""
        import sys, time
        from pathlib import Path

        target_dir = Path("{function_file_path.parent.as_posix()}")
        if str(target_dir) not in sys.path:
            sys.path.insert(0, str(target_dir))

        import target_module  # must define sum_primes()

        if __name__ == "__main__":
            t0 = time.perf_counter()
            val = target_module.sum_primes(7000)
            elapsed = time.perf_counter() - t0
            print(f"RESULT={{val}}  ELAPSED_S={{elapsed:.6f}}")
    """).strip()

    bench_path = workdir / "bench_sum_primes.py"
    bench_path.write_text(bench_script, encoding="utf-8")
    trace(f"Wrote bench script: {bench_path}")

    return dict(
        venv_path=str(venv_dir),
        workdir=str(workdir),
        function_file_path=str(function_file_path),
        test_script_path=str(bench_path),
        baseline_function=baseline_function,
    )


# ──────────────────────────────────────────────────────────────────────────────
# NodeView-based runtime monitor
# ──────────────────────────────────────────────────────────────────────────────
def monitor_with_nodeview(root: core.Node) -> None:
    """
    Uses NodeView.watch() to monitor the runtime tree and prints:
      - Node state transitions and updates
      - New children as they are created
      - Agent transcript changes
    """
    # Track what we've already seen and reported
    seen_nodes: Set[int] = set()
    prev_states: Dict[int, core.NodeState] = {}
    prev_child_counts: Dict[int, int] = {}
    prev_transcript_sizes: Dict[int, int] = {}

    def summarize_transcript_part(p: core.TranscriptPart) -> str:
        cls = p.__class__.__name__
        if isinstance(p, core.UserTextPart):
            return f"UserTextPart: {p.text[:140].replace(chr(10),' ')}..."
        if isinstance(p, core.ModelTextPart):
            return f"ModelTextPart: {p.text[:140].replace(chr(10),' ')}..."
        if isinstance(p, core.ToolUsePart):
            keys = ", ".join(p.args.keys())
            return f"ToolUsePart: {p.tool_name} (args: {keys})"
        if isinstance(p, core.ToolResultPart):
            out = str(p.outputs)
            if isinstance(p.outputs, str):
                out = p.outputs[:140].replace(chr(10), ' ')
            return f"ToolResultPart: {p.tool_name} (error={p.is_error}) -> {out}..."
        if isinstance(p, core.ThinkingBlockPart):
            kind = "redacted" if p.redacted else "visible"
            return f"ThinkingBlockPart ({kind}) len={len(p.content)} sig={p.signature[:12]}..."
        return cls

    def compute_depths_from_view(view: core.NodeView, depths: Optional[Dict[int, int]] = None, current_depth: int = 0) -> Dict[int, int]:
        if depths is None:
            depths = {}
        depths[view.id] = current_depth
        for child_view in view.children:
            compute_depths_from_view(child_view, depths, current_depth + 1)
        return depths

    def process_view(view: core.NodeView, depths: Dict[int, int]) -> None:
        """Process a single NodeView and log any changes."""
        nid = view.id
        depth = depths.get(nid, 0)

        # Check if this is a new node we haven't seen
        if nid not in seen_nodes:
            seen_nodes.add(nid)
            prev_states[nid] = view.state
            prev_child_counts[nid] = len(view.children)
            trace(_indent(depth) + f"Node #{nid} CREATED: fn={view.fn.name} state={view.state.value}")

        # Check for state transitions
        last_state = prev_states.get(nid)
        if last_state != view.state:
            prev_states[nid] = view.state
            trace(_indent(depth) + f"Node #{nid} STATE: {last_state.value if last_state else 'N/A'} -> {view.state.value}")

        # Check for new children
        cur_child_count = len(view.children)
        prev_count = prev_child_counts.get(nid, 0)
        if cur_child_count != prev_count:
            new_count = cur_child_count - prev_count
            prev_child_counts[nid] = cur_child_count
            if new_count > 0:
                # Report new children
                for child_view in view.children[-new_count:]:
                    trace(_indent(depth) + f"Node #{nid} CHILD ADDED -> Node #{child_view.id} (fn={child_view.fn.name})")

        # Check transcript updates for AgentNodes (if we can access the actual node)
        # For now, we'll skip transcript monitoring since NodeView doesn't include transcript info
        # This could be enhanced if needed by accessing the actual node

        # Recursively process children
        for child_view in view.children:
            process_view(child_view, depths)

    trace("NodeView Monitor: started")
    as_of_seq = 0

    try:
        while not root.is_done:
            # Watch for updates using NodeView
            view = root.watch(as_of_seq=as_of_seq+1)
            as_of_seq = view.update_seqnum

            # Compute depths and process the entire tree
            depths = compute_depths_from_view(view)
            process_view(view, depths)

    except Exception as e:
        trace(f"NodeView Monitor: error while monitoring: {e}")

    # Final snapshot after completion
    final_view = root.watch(as_of_seq=as_of_seq+1)
    depths = compute_depths_from_view(final_view)
    trace("NodeView Monitor: root completed, final snapshot:")

    def final_walk(view: core.NodeView, depths: Dict[int, int]):
        d = depths.get(view.id, 0)
        trace(_indent(d) + f"Node #{view.id} FINAL state={view.state.value} fn={view.fn.name}")
        for child_view in view.children:
            final_walk(child_view, depths)

    final_walk(final_view, depths)
    trace("NodeView Monitor: stopped")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    # Workspace
    ws = make_demo_workspace()

    # Runtime + registration
    trace("Registering Functions/Agents in Runtime...")
    runtime = Runtime(
        specs=[
            # Tools
            ReadTextFile,
            WriteTextFile,
            PerfProfileTool,
            # Agents
            PerfReasoningAgent,
            PerfImprovementAgent,
        ],
        client_factories=CLIENT_FACTORIES,
    )
    trace("Registration complete.")

    # Top-level invocation (Gemini by default, handled by your core)
    trace("Creating top-level RunContext and invoking PerfImprovementAgent...")
    ctx = runtime.get_ctx()
    root = ctx.invoke(
        PerfImprovementAgent,
        {
            "function_name": "sum_primes",
            "initial_source": ws["baseline_function"],
            "function_file_path": ws["function_file_path"],
            "test_script_path": ws["test_script_path"],
            "venv_path": ws["venv_path"],
            "max_iters": 3,
            "improvement_threshold_pct": 10.0,
        },
    )

    # Start NodeView-based monitoring in a separate thread
    trace("Starting NodeView-based monitoring...")
    monitor_thread = threading.Thread(
        target=monitor_with_nodeview,
        args=(root,),
        name="netflux-nodeview-monitor",
        daemon=True
    )
    monitor_thread.start()

    # Wait for completion & show final report
    trace("Waiting for top-level node to finish...")
    root.wait()
    trace("Top-level node finished. Fetching result...")
    try:
        final_report = root.result()
    except Exception as ex:
        final_report = f"[ERROR] {type(ex).__name__}: {ex}"

    print("\n" + "=" * 80)
    print("FINAL REPORT FROM PerfImprovementAgent")
    print("=" * 80)
    print(final_report)
    print("=" * 80)

    # Let monitor thread finish its final snapshot
    monitor_thread.join(timeout=2.0)
    trace("Demo complete.")


if __name__ == "__main__":
    main()
