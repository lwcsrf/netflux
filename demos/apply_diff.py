import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from ..core import AgentFunction, CodeFunction, FunctionArg, NodeState, Provider
from ..runtime import Runtime
from ..viz import ConsoleRender, start_view_loop
from .client_factory import CLIENT_FACTORIES
from ..func_lib.apply_diff import apply_diff_patch


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _mk_workspace() -> Path:
    root = Path(tempfile.mkdtemp(prefix="netflux-applydiff-"))
    # Seed files to exercise: edit, add, delete, rename, multi-hunk, and a path with spaces
    _write(
        root / "src/module.py",
        """
# demo module for patch stress
import math
import time

def fib(n: int) -> list[int]:
    a, b = 0, 1
    out = []
    for _ in range(n):
        out.append(a)
        a, b = b, a + b
    return out

def greet(name: str) -> str:
    return f"Hello, {name}!"

def main():
    print('Hello from module')

if __name__ == '__main__':
    main()
""".lstrip(),
    )

    _write(
        root / "docs/notes.md",
        """
# Notes

This is a sample notes file.

- item A
- item B
- item C
""".lstrip(),
    )

    _write(root / "delete_me.txt", "line1\nline2\nline3\n")
    _write(root / "old_name.txt", "Old contents line 1\nline 2\nline 3\n")
    _write(root / "spaced file.txt", "hello world\n")
    return root


def _unified_diff_for_text(
    a_lines: List[str],
    b_lines: List[str],
    a_path: str,
    b_path: str,
) -> str:
    import difflib

    # Use classic unified diff header with a/ and b/ prefixes
    diff = difflib.unified_diff(
        a_lines,
        b_lines,
        fromfile=f"a/{a_path}",
        tofile=f"b/{b_path}",
        lineterm="\n",
        n=3,
    )
    return "".join(diff)


def _make_success_patch_doc(root: Path) -> str:
    """Build a markdown document containing multiple ```diff blocks that together:
    - edit existing files with multiple hunks
    - add a new file
    - delete a file
    - rename a file (and tweak its first line)
    - modify a filename with spaces
    """

    # 1) Multi-hunk edit for src/module.py
    orig_mod = (root / "src/module.py").read_text(encoding="utf-8").splitlines(keepends=True)
    new_mod_text = (
        """
# demo module for patch stress (patched)
import math
import sys
import time

def fib(n: int) -> list[int]:
    # slight tweak: keep semantics, add comment
    a, b = 0, 1
    out = []
    for _ in range(n):
        out.append(a)
        a, b = b, a + b
    return out

def greet(name: str) -> str:
    return f"Hello, {name}!"

def main():
    print('Hello from module (updated)')

if __name__ == '__main__':
    main()
""".lstrip()
    )
    new_mod = new_mod_text.splitlines(keepends=True)
    module_diff = _unified_diff_for_text(orig_mod, new_mod, "src/module.py", "src/module.py")

    # 2) Edit docs/notes.md (single hunk)
    orig_notes = (root / "docs/notes.md").read_text(encoding="utf-8").splitlines(keepends=True)
    new_notes_text = (
        """
# Notes

This is a sample notes file (patched).

- item A
- item B
- item C
- item D
""".lstrip()
    )
    new_notes = new_notes_text.splitlines(keepends=True)
    notes_diff = _unified_diff_for_text(orig_notes, new_notes, "docs/notes.md", "docs/notes.md")

    # 3) Modify a path with spaces
    orig_spaced = (root / "spaced file.txt").read_text(encoding="utf-8").splitlines(keepends=True)
    new_spaced = "hello patched world\n".splitlines(keepends=True)
    spaced_diff = _unified_diff_for_text(orig_spaced, new_spaced, "spaced file.txt", "spaced file.txt")

    # 4) Add new file new_dir/added.txt
    add_block = (
        "diff --git a/new_dir/added.txt b/new_dir/added.txt\n"
        "new file mode 100644\n"
        "index 0000000..e69de29\n"
        "--- /dev/null\n"
        "+++ b/new_dir/added.txt\n"
        "@@ -0,0 +1,4 @@\n"
        "+First line\n"
        "+Second line\n"
        "+Third line\n"
        "+Fourth line\n"
    )

    # 5) Delete file delete_me.txt
    delete_block = (
        "diff --git a/delete_me.txt b/delete_me.txt\n"
        "deleted file mode 100644\n"
        "index e69de29..0000000\n"
        "--- a/delete_me.txt\n"
        "+++ /dev/null\n"
        "@@ -1,3 +0,0 @@\n"
        "-line1\n"
        "-line2\n"
        "-line3\n"
    )

    # 6) Rename old_name.txt -> renamed/new_name.txt and tweak first line
    rename_block = (
        "diff --git a/old_name.txt b/renamed/new_name.txt\n"
        "similarity index 100%\n"
        "rename from old_name.txt\n"
        "rename to renamed/new_name.txt\n"
        "--- a/old_name.txt\n"
        "+++ b/renamed/new_name.txt\n"
        "@@ -1,3 +1,3 @@\n"
        "-Old contents line 1\n"
        "+Old contents line 1 (renamed)\n"
        " line 2\n"
        " line 3\n"
    )

    # Compose as a markdown document with multiple diff blocks and interspersed commentary
    doc = (
        "# Changeset (stress demo)\n\n"
        "Below are multiple patches across files. Only the fenced diff blocks should be processed.\n\n"
        "```diff\n" + module_diff + "```\n\n"
        "Some commentary here, ignore me.\n\n"
        "```diff\n" + notes_diff + spaced_diff + "```\n\n"
        "More commentary.\n\n"
        "```diff\n" + add_block + delete_block + rename_block + "```\n"
    )
    return doc


def _make_failure_patch_doc(root: Path) -> str:
    """Build a changes doc that makes a valid edit first, then includes a bad hunk
    referencing a non-existent file to force a rollback. This exercises the agent's
    atomic transaction semantics in its prompt.
    """
    orig_notes = (root / "docs/notes.md").read_text(encoding="utf-8").splitlines(keepends=True)
    new_notes = (
        """
# Notes

This is a sample notes file (bad run).

- item A
- item B
- item C
""".lstrip()
    ).splitlines(keepends=True)
    notes_diff = _unified_diff_for_text(orig_notes, new_notes, "docs/notes.md", "docs/notes.md")

    # Invalid target path
    bad_block = (
        "diff --git a/DOES_NOT_EXIST.txt b/DOES_NOT_EXIST.txt\n"
        "index e69de29..0000000\n"
        "--- a/DOES_NOT_EXIST.txt\n"
        "+++ b/DOES_NOT_EXIST.txt\n"
        "@@ -1,1 +1,1 @@\n"
        "-foo\n"
        "+bar\n"
    )

    return (
        "# Intentionally failing patch to test rollback\n\n"
        "```diff\n" + notes_diff + bad_block + "```\n"
    )


def _run_applydiff(
    provider: Provider,
    patch_doc: str,
    *,
    chdir: Optional[Path] = None,
) -> Tuple[NodeState, Optional[str], Optional[Exception]]:
    # Optionally set working directory so relative paths in diffs are correct
    cwd_save: Optional[Path] = None
    if chdir is not None:
        cwd_save = Path.cwd()
        os.chdir(chdir)

    try:
        runtime = Runtime(
            specs=[apply_diff_patch],
            client_factories=CLIENT_FACTORIES,
        )
        ctx = runtime.get_ctx()

        node = ctx.invoke(
            apply_diff_patch,
            {"diff_content": patch_doc},
            provider=provider,
        )

        # Simple live view in console
        ConsoleRender.pre_console()
        render = ConsoleRender(spinner_hz=10.0)
        view_thread = start_view_loop(
            node,
            render=render,
            ui_driver=ConsoleRender.ui_driver,
            update_interval=0.1,
        )

        result_text: Optional[str] = None
        exc: Optional[Exception] = None
        try:
            result_text = str(node.result())
        except Exception as e:
            exc = e

        node.wait()
        try:
            view_thread.join(timeout=1.0)
        except Exception:
            pass
        ConsoleRender.restore_console()

        # Final static frame
        print(str(render.render(runtime.watch(node))))

        return node.state, result_text, exc
    finally:
        if cwd_save is not None:
            os.chdir(cwd_save)


def _verify_success(root: Path) -> Tuple[bool, List[str]]:
    problems: List[str] = []

    # Added file
    add_path = root / "new_dir/added.txt"
    if not add_path.exists():
        problems.append(f"missing added file: {add_path}")
    else:
        content = add_path.read_text(encoding="utf-8")
        if "Fourth line" not in content:
            problems.append("added file content mismatch")

    # Deleted file
    if (root / "delete_me.txt").exists():
        problems.append("delete_me.txt was not deleted")

    # Renamed file
    if (root / "old_name.txt").exists():
        problems.append("old_name.txt still exists after rename")
    renamed_path = root / "renamed/new_name.txt"
    if not renamed_path.exists():
        problems.append("renamed/new_name.txt missing")
    else:
        if "renamed" not in renamed_path.read_text(encoding="utf-8"):
            problems.append("renamed file content mismatch")

    # Edits
    mod_path = root / "src/module.py"
    txt = mod_path.read_text(encoding="utf-8")
    if "(patched)" not in txt or "Hello from module (updated)" not in txt or "import sys" not in txt:
        problems.append("src/module.py edits not fully applied")

    notes_path = root / "docs/notes.md"
    if "(patched)" not in notes_path.read_text(encoding="utf-8"):
        problems.append("docs/notes.md edit missing")

    if "patched world" not in (root / "spaced file.txt").read_text(encoding="utf-8"):
        problems.append("spaced file change missing")

    return (len(problems) == 0), problems


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stress-test apply_diff_patch by applying a multi-file changeset.",
    )
    parser.add_argument(
        "--provider",
        choices=[p.value.lower() for p in Provider],
        required=True,
        help="Choose the provider to use for this run.",
    )
    parser.add_argument(
        "--fail-first",
        action="store_true",
        help="First run an intentionally failing patch to exercise rollback.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    provider_value = {p.value.lower(): p.value for p in Provider}[args.provider]
    provider = Provider(provider_value)

    root = _mk_workspace()
    print(f"Workspace: {root}")

    if args.fail_first:
        print("\nRunning failure scenario to test rollback ...\n")
        fail_doc = _make_failure_patch_doc(root)
        state, _, exc = _run_applydiff(provider, fail_doc, chdir=root)
        print(f"Failure run state: {state}")
        if state != NodeState.Error:
            print("Warning: expected an error state for the failing patch, but it did not error.")
        if exc:
            print(f"Observed exception (from agent): {exc}")

        # Spot-check that previous files remain unmodified (rollback). A strict check would snapshot beforehand,
        # but for brevity, check that delete_me.txt still exists and notes.md does not include the 'bad run' text.
        if not (root / "delete_me.txt").exists():
            print("Warning: delete_me.txt missing after failed run (rollback may not have occurred).")
        if "(bad run)" in (root / "docs/notes.md").read_text(encoding="utf-8"):
            print("Warning: docs/notes.md appears modified after failed run (rollback may not have occurred).")

    # Success scenario
    print("\nRunning success scenario ...\n")
    patch_doc = _make_success_patch_doc(root)
    state, result_text, exc = _run_applydiff(provider, patch_doc, chdir=root)
    print(f"Final state: {state}")
    if result_text:
        print(f"Result: {result_text}")
    if exc:
        print(f"Error: {exc}")

    if state == NodeState.Success:
        ok, problems = _verify_success(root)
        if ok:
            print("Verification: OK")
        else:
            print("Verification problems:")
            for p in problems:
                print(f"- {p}")


if __name__ == "__main__":
    main()

