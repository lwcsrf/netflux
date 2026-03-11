import argparse
import multiprocessing as mp
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from ..core import AgentFunction, FunctionArg, NodeState, Provider
from ..func_lib.bash import bash
from ..func_lib.raise_exception import raise_exception
from ..runtime import Runtime
from ..viz import ConsoleRender


BASH_STRESS_SYSTEM_PROMPT = (
    "You are a non-conversational agent whose sole purpose is to stress-test the built-in "
    "`bash` function the way a capable agent would really use it.\n\n"
    "Core objective:\n"
    "- Determine whether the bash tool behaves reasonably across a broad range of realistic "
    "agent workflows.\n"
    "- If a `custom_instruction` is provided, prioritize it while still keeping the task a "
    "bash stress test.\n"
    "- If no custom instruction is provided, choose your own diverse suite and continue until "
    "satisfied; aim for roughly 100 bash tool invocations unless a clear issue appears sooner.\n\n"
    "Progression of testing:\n"
    "1. Start simple — basic filesystem and shell orientation:\n"
    "   - pwd, ls -la, mkdir -p, touch, cat, echo, cp, mv, rm\n"
    "   - quoting (single, double, mixed, escaped), paths with spaces, relative navigation\n"
    "   - env var setting and reading, cwd persistence across calls\n"
    "   - basename, dirname, realpath\n"
    "   - uname, which, command -v (system inspection)\n"
    "2. Move to medium-complexity agent patterns:\n"
    "   - grep -rn, grep -rl, grep -c, grep with regex; find with -name, -type, -newer, -exec\n"
    "   - sort, uniq, wc -l, head -n, tail -n, cut, tr, paste\n"
    "   - pipes, redirects (>, >>, 2>&1, &>), globbing, brace expansion ({a,b,c}), tilde expansion\n"
    "   - exploring files, generating files, validating outputs\n"
    "   - xargs patterns: find ... | xargs grep, find ... -print0 | xargs -0\n"
    "   - chaining with &&, ||, ; for conditional execution and fallback patterns\n"
    "   - exit code checking: $?, || true, if command; then ...; fi\n"
    "   - tee for writing to file and stdout simultaneously\n"
    "   - tree (if available) or find-based directory tree printing\n"
    "3. Realistic agent workflow simulations:\n"
    "   - scaffold a project: mkdir -p src/utils tests; create config files, source files, a README\n"
    "   - write code to files using heredocs (cat > file.py << 'EOF'), then run the code (python3 file.py)\n"
    "   - read back file content, verify it matches expectations (diff, cmp, or grep assertions)\n"
    "   - search-and-replace in files: sed -i for in-place edits, then verify the edit landed\n"
    "   - text processing pipelines: awk column extraction, sed transformations, multi-step pipes\n"
    "   - write a test script, run it, check results — the edit-run-verify loop agents do constantly\n"
    "   - python3 -c one-liners for calculations, json processing, string manipulation\n"
    "   - git init, git add, git commit, git log --oneline, git diff (the full git workflow)\n"
    "   - archive operations: tar czf, tar xzf, verifying contents\n"
    "   - checksums: md5sum or sha256sum to verify file integrity\n"
    "   - du -sh, df -h for disk usage inspection\n"
    "   - creating and sourcing a .env file or shell config\n"
    "4. Sophisticated shell behavior:\n"
    "   - heredocs (with and without quoting the delimiter), here-strings (<<<)\n"
    "   - shell variables, arrays, associative arrays, arithmetic $(( ))\n"
    "   - functions, conditionals (if/elif/else, case), loops (for, while, until)\n"
    "   - reading a file line-by-line: while IFS= read -r line; do ...; done < file\n"
    "   - subshells, command substitution $(), nested substitution\n"
    "   - background jobs (&) and wait\n"
    "   - a second `session_id` only when you intentionally want independent state\n"
    "   - `restart=true` only when useful to test restart semantics or recover from a prior timeout/failure\n"
    "   - multi-line compound commands spanning a single tool call (functions + loops + conditionals together)\n"
    "5. Robustness and edge cases:\n"
    "   - Unicode and special characters in file names and content (accented chars, CJK, emoji, etc.)\n"
    "   - filenames with spaces, quotes, newlines, glob characters (*, ?) — proper quoting discipline\n"
    "   - large output that approaches or exceeds the tool's truncation limit (~40k chars); verify truncation note\n"
    "   - custom `timeout_sec` with a short timeout and a quick command to confirm it works\n"
    "   - `set -e` / `set -o errexit` persistence across calls — set it, run a succeeding command, verify it sticks\n"
    "   - stdout redirection resilience: `exec 1>/dev/null`, then run another command and verify the tool still returns output\n"
    "   - process substitution (`<(...)`, `>(...)`)\n"
    "   - traps (`trap '...' EXIT`) — verify they don't break the tool's own sentinel mechanism\n"
    "   - symlinks: create, follow, resolve with `readlink`\n"
    "   - permission handling: `chmod`, attempt to read an unreadable file\n"
    "   - nested script execution: write a `.sh` file, `chmod +x`, execute it\n"
    "   - binary / non-UTF-8 output (e.g. `head -c 64 /dev/urandom | xxd`)\n"
    "   - empty command output (command that succeeds but prints nothing)\n"
    "   - very long single-line output vs. many short lines\n"
    "   - commands with embedded newlines in arguments\n"
    "   - rapidly successive short commands to test session throughput\n"
    "6. Use a few controlled negative probes when useful:\n"
    "   - commands that fail because of a missing file, grep miss, or similar normal shell outcomes\n"
    "   - writing to a read-only location, killing a nonexistent PID\n"
    "   - syntax errors: verify the tool reports them cleanly rather than hanging\n"
    "   - distinguish expected command-level failure from a bash tool malfunction\n\n"
    "Important rules:\n"
    "- You will be given the absolute path to a temporary workspace created specifically for this demo.\n"
    "- Do your play under that workspace tree. Keep your bash activity scoped there unless a command inherently "
    "queries ambient system facts like `pwd`, `uname`, or similar harmless inspection.\n"
    "- You may freely create, edit, rename, and delete files inside that workspace. Proliferate your own example files and dirs.\n"
    "- Use `bash` as the primary tool.\n"
    "- Reuse the same bash session sequentially unless deliberately testing separate session state.\n"
    "- Do not report an issue merely because a command returned a non-zero exit code when that outcome was expected.\n"
    "- A reportable issue is something clearly unreasonable: state not persisting when it should, output corruption or "
    "sentinel leakage, broken quoting/path handling, restart semantics failing, commands hanging unreasonably, or "
    "obvious tool contract violations.\n"
    f"- On a clear malfunction or unreasonable behavior, stop immediately and call `{raise_exception.name}` with a "
    "concise report that includes the triggering command, observed behavior, and why it is unreasonable.\n"
    "- Avoid pointless repetition once a category is clearly covered, except where repetition helps expose flakiness.\n\n"
    "If everything seems reasonable, finish with a concise human-readable report stating:\n"
    "- whether a clear issue was found\n"
    "- the major categories exercised\n"
    "- notable commands or patterns that worked\n"
    "- whether a custom instruction was followed\n"
)

bash_stress_agent = AgentFunction(
    name="bash_stress_agent",
    desc=(
        "Stress-tests the built-in bash function through a long sequence of realistic shell workflows. "
        "An optional `custom_instruction` can steer the stress test toward specific behaviors."
    ),
    args=[
        FunctionArg(
            "workspace",
            str,
            "Absolute path to the temporary workspace created for this stress run.",
        ),
        FunctionArg(
            "custom_instruction",
            str,
            "Optional extra directive describing what to emphasize while stress-testing bash.",
            optional=True,
        ),
    ],
    system_prompt=BASH_STRESS_SYSTEM_PROMPT,
    user_prompt_template=(
        "Run the bash stress test now.\n\n"
        "I created this temporary workspace for you to play around with:\n"
        "{workspace}\n"
        "Do your bash exploration and file mutations under that directory tree.\n\n"
        "Custom instruction:\n"
        "{custom_instruction}\n"
        "If the custom instruction is `None`, choose your own long, diverse bash stress suite.\n"
        "Start with simple behaviors, then progress to more sophisticated ones. "
        f"Stop immediately and call `{raise_exception.name}` if you find a clear bash-tool malfunction or unreasonable behavior.\n"
        "If all goes well, end with normal summary text describing the breadth and depth of what you tried.\n"
    ),
    uses=[bash, raise_exception],
    default_model=Provider.Anthropic,
)
def make_demo_workspace() -> Path:
    return Path(tempfile.mkdtemp(prefix="netflux-bash-stress-")).resolve()


def run_bash_stress_tree(
    provider: Optional[Provider] = None,
    *,
    custom_instruction: Optional[str] = None,
) -> Path:
    from .client_factory import CLIENT_FACTORIES

    workspace = make_demo_workspace()
    print(f"Workspace: {workspace}")

    runtime = Runtime(
        specs=[bash_stress_agent, bash, raise_exception],
        client_factories=CLIENT_FACTORIES,
    )
    ctx = runtime.get_ctx()
    cancel_evt = mp.Event()

    invoke_args = {
        "workspace": str(workspace),
        "custom_instruction": custom_instruction,
    }

    cwd_save = Path.cwd()
    try:
        os.chdir(workspace)
        node = ctx.invoke(
            bash_stress_agent,
            invoke_args,
            provider=provider,
            cancel_event=cancel_evt,
        )

        render = ConsoleRender(spinner_hz=10.0, cancel_event=cancel_evt)

        final_result: Optional[str] = None
        run_exception: Optional[Exception] = None
        try:
            render.run(node)
        except Exception:
            pass

        node.wait()
        cancel_evt.set()

        try:
            final_result = str(node.result())
        except Exception as ex:
            run_exception = ex

        print(str(render.render(runtime.watch(node))))

        if node.state == NodeState.Success:
            print("\n--- Final Report ---\n")
            if final_result:
                print(final_result)
            print("\n--------------------\n")
        elif node.state == NodeState.Error:
            print("\n--- Bash Issue Detected ---\n")
            if run_exception:
                print(run_exception)
            print("\n---------------------------\n")
        elif node.state == NodeState.Canceled:
            print("\nCanceled.\n")

        return workspace
    finally:
        os.chdir(cwd_save)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the bash stress-test demo.",
    )
    parser.add_argument(
        "--provider",
        choices=[p.value.lower() for p in Provider],
        required=True,
        help="Choose the provider to use for this run.",
    )
    parser.add_argument(
        "--custom-instruction",
        help="Optional extra directive for the bash stress agent.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    provider_value = {p.value.lower(): p.value for p in Provider}[args.provider]
    provider = Provider(provider_value)
    run_bash_stress_tree(
        provider,
        custom_instruction=args.custom_instruction,
    )


if __name__ == "__main__":
    main()
