import argparse
import multiprocessing as mp
import os
import tempfile
from pathlib import Path
from typing import List, Optional

from ..core import AgentFunction, FunctionArg, NodeState, Provider
from ..func_lib import bash, raise_exception, status_update, view_image
from ..runtime import Runtime
from ..tui import ConsoleRender


BASH_STRESS_SYSTEM_PROMPT = (
    "You are a non-conversational agent whose sole purpose is to stress-test the built-in "
    "`bash` function the way a very capable agent would really use it.\n\n"
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
    "   - run multiple independent `session_id`s in parallel and verify their state stays isolated\n"
    "   - `restart=true` only when useful to test restart semantics or recover from a prior timeout/failure\n"
    "   - multi-line compound commands spanning a single tool call (functions + loops + conditionals together)\n"
    "5. Robustness and edge cases:\n"
    "   - Unicode and special characters in file names and content (accented chars, CJK, emoji, etc.)\n"
    "   - filenames with spaces, quotes, newlines, glob characters (*, ?) — proper quoting discipline\n"
    "   - large output that approaches or exceeds the tool's truncation limit (~40k chars); verify truncation note\n"
    "   - custom `timeout_sec` with a short timeout and a quick command to confirm it works\n"
    "   - child-shell `set -e` / `set -o errexit` usage (for example via `bash script.sh`) should behave like an ordinary command failure\n"
    "   - `set -x` / `set -o xtrace` behavior — verify commands still complete and follow-up calls still work; "
    "it is acceptable if xtrace output exposes some internal transport/delimiter text\n"
    "   - fd redirection: persistent `exec 1>/dev/null`, and explicit restoration after saving; command-scoped `cmd 1>/x/y`; restart session also back to normal FDs\n"
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
    "- Direct `set -e` / `set -o errexit` inside the persistent session is unsupported; if you probe it and the session "
    "fails, do not treat that alone as a bash-tool bug.\n"
    "- For `set -x` / xtrace specifically, seeing some internal transport or delimiter text in the traced output is "
    "acceptable by itself; only treat hangs, broken completion, or follow-up session breakage as issues.\n"
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
    uses=[bash, raise_exception, status_update],
    default_model=Provider.Anthropic,
)
SOURCE_ROOT = Path(__file__).resolve().parents[1]
STRESS_RULES = (
    "You are a non-conversational agent stress-testing the framework. Prioritize a supplied "
    "custom_instruction over the default suite below; None means run the default suite. "
    "Keep generated files and mutations in temporary dir workspace. "
    "Report observed evidence, failures and limitations honestly; expected probe errors are not bugs. "
    f"On a clear malfunction, call `{raise_exception.name}` with a concise reproducer and observations.\n\n"
)

IMAGE_STRESS_SYSTEM_PROMPT = STRESS_RULES + (
    "Stress-test the built-in view_image using bash to create and rasterize your own SVGs. "
    "Complete these stages in order; do not read view_image source until stage 4.\n"
    f"1. Call view_image on {SOURCE_ROOT / 'assets/banner.png'}. Describe what you actually see: "
    "colors, shapes, distinctive features and composition; give your opinion of the image.\n"
    "2. Design an SVG poster of a floating glass greenhouse at dusk, with visible plants behind "
    "translucent panes, overlapping structural ribs, a spiral walkway, water reflections and "
    "three readable callout labels with noncrossing leaders. Aim for balanced spacing, clean "
    "occlusion, coherent lighting and legible text with no clipping or unintended overlaps. "
    "Write down visual acceptance criteria, render an initial PNG and inspect it with view_image. "
    "Then perform at least 10 numbered revision cycles: predict an improvement, meaningfully alter "
    "the SVG, rasterize to PNG, call view_image, and describe the observed change and remaining defects "
    "before the next edit. SVG source suggests appearance; only the rendered image confirms it. "
    "Keep each SVG/PNG version; do not batch unseen revisions or count identical renders. Continue "
    "until a final view_image inspection confirms every criterion, reporting any unresolved defects. "
    "Do not go beyond 20 iterations.\n"
    "3. Create a second SVG: an optical calibration sheet with colored wedges, fine concentric "
    "rings, gradients and corner labels. Rasterize it at 5000x4000 (20 megapixels) and save a "
    "single-frame BMP. Record its source dimensions and encoding, then pass that original BMP "
    "directly to view_image. Verify its returned status explicitly reports BOTH downsizing and "
    "re-encoding; report delivered dimensions/MIME and visually confirm layout, colors and "
    "large labels still look as expected, noting any fine detail lost.\n"
    f"4. Only after stages 1–3, read {SOURCE_ROOT / 'func_lib/view_image.py'}. Choose and execute "
    "at least three additional corner cases motivated by that implementation. Explain expected "
    "versus observed results, distinguishing clean rejections from tool malfunctions. Finish with "
    "a concise report of visual findings, revision count, conversion evidence, extra probes and artifact paths.\n"
    "Additional instructions:\n"
    "- Provide a final verdict on how each stage went (e.g. SUCCESS or FAIL). Mark a stage as a "
    "failure if, for example, there was no way to carry out a feedback loop of viewing, updating "
    "the drawing and re-rasterizing.\n"
    "- Provide a final verdict on the successful integration of the view_image built-in function "
    "by reflecting on the context window and how seamlessly iterations were carried out. "
    "Did the media appear exactly as intended?\n"
    "- Work in your own /tmp directory. You may see other agents concurrently working on this "
    "problem. Make sure you don't spoil the integrity of the task by looking at other agents' "
    "work — stick to your own /tmp directory.\n"
    "- Once done, put the final SVG series of improvements into a final directory under your "
    "tmp directory and give its clear, absolute path in your response.\n"
    "- Report any problems whatsoever with view_image efficacy: we want to know about it!\n"
)

TREE_STRESS_SYSTEM_PROMPT = STRESS_RULES + (
    "Stress-test recursive agent delegation; the top invocation sets a maximum of 3 agent levels. "
    "Unless instructed not to (as a lower-level delegate), use targeted searches and read only "
    f"relevant portions of {SOURCE_ROOT / 'core.py'} and {SOURCE_ROOT / 'runtime.py'} to learn "
    "self-recursion, level counting/inheritance, the actual depth-limit exception type and how "
    "the deepest-level system warning is injected.\n"
    "Then invoke yourself recursively to reach the deepest allowed agent. Give each delegate "
    "the framework facts you learned, its level, a precise task and reporting instructions; "
    "explicitly tell it not to reread the framework and to follow the supplied instructions. "
    "The deepest agent must quote the warning actually present in its own system prompt, "
    "attempt one more self-invocation to exercise the limit, record the actual exception type "
    "and message, and verify a simple bash call still works afterward. The expected rejection "
    "is a successful probe; do not retry it. Each parent must wait for and relay its child's "
    "evidence so the root's final report confirms the reached and rejected levels, warning "
    "visibility and exception match, with observed evidence rather than source-based predictions.\n"
)

CUSTOM_INSTRUCTION_ARGS = [
    FunctionArg("custom_instruction", str, "Optional override of the stress-test goal.", optional=True),
]

image_stress_agent = AgentFunction(
    name="image_stress_agent",
    desc="Stress-tests view_image through SVG revisions, large-image conversion and source-guided probes.",
    args=CUSTOM_INSTRUCTION_ARGS,
    system_prompt=IMAGE_STRESS_SYSTEM_PROMPT,
    user_prompt_template="Run the image stress test.\nCustom instruction: {custom_instruction}\n",
    uses=[bash, view_image, raise_exception, status_update],
)

tree_stress_agent = AgentFunction(
    name="tree_stress_agent",
    desc="Stress-tests recursive agent depth limits, warnings and reporting through delegates.",
    args=CUSTOM_INSTRUCTION_ARGS,
    system_prompt=TREE_STRESS_SYSTEM_PROMPT,
    user_prompt_template="Run the agent tree stress test.\nCustom instruction: {custom_instruction}\n",
    uses=[bash, raise_exception, status_update],
    uses_recursion=True,
)

STRESS_AGENTS = {"bash": bash_stress_agent, "image": image_stress_agent, "tree": tree_stress_agent}


def make_demo_workspace(mode: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"netflux-{mode}-stress-")).resolve()


def run_stress_tree(
    mode: str,
    provider: Optional[Provider] = None,
    *,
    custom_instruction: Optional[str] = None,
) -> Path:
    from .client_factory import CLIENT_FACTORIES

    agent = STRESS_AGENTS[mode]
    if mode == "tree":
        # Delegates use the definition's default, not the root's provider override.
        agent.default_model = provider or Provider.Anthropic
    workspace = make_demo_workspace(mode)
    print(f"Workspace: {workspace}")

    runtime = Runtime(
        specs=[agent],
        client_factories=CLIENT_FACTORIES,
    )
    ctx = runtime.get_ctx()
    cancel_evt = mp.Event()

    invoke_args = {
        "custom_instruction": custom_instruction,
    }
    if mode == "bash":
        invoke_args["workspace"] = str(workspace)

    cwd_save = Path.cwd()
    try:
        os.chdir(workspace)
        node = ctx.invoke(
            agent,
            invoke_args,
            provider=provider,
            cancel_event=cancel_evt,
            max_agent_levels=3 if mode == "tree" else None,
        )

        render = ConsoleRender(spinner_hz=10.0)

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

        if node.state == NodeState.Success:
            print("\n--- Final Report ---\n")
            if final_result:
                print(final_result)
            print("\n--------------------\n")
        elif node.state == NodeState.Error:
            print(f"\n--- {mode.title()} Issue Detected ---\n")
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
        description="Stress-test built-in bash, image viewing or recursive agent trees.",
    )
    parser.add_argument("mode", choices=STRESS_AGENTS, help="Choose the stress test to run.")
    parser.add_argument(
        "--provider",
        choices=[p.value.lower() for p in Provider],
        required=True,
        help="Choose the provider to use for this run.",
    )
    parser.add_argument(
        "--custom-instruction",
        help="Optional custom instruction passed to the top-level stress agent.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    provider_value = {p.value.lower(): p.value for p in Provider}[args.provider]
    provider = Provider(provider_value)
    run_stress_tree(
        args.mode,
        provider,
        custom_instruction=args.custom_instruction,
    )


if __name__ == "__main__":
    main()
