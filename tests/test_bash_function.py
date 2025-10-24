import tempfile
import unittest
from pathlib import Path
from typing import Dict, Optional

from ..core import RunContext, SessionBag, SessionScope
from ..func_lib.bash import Bash, BashSession, BashCommandTimeoutException


class _DummyNode:
    def __init__(self, parent: Optional["_DummyNode"] = None) -> None:
        self.parent = parent
        self.session_bag = SessionBag()


class TestBashFunctionCommands(unittest.TestCase):
    def setUp(self) -> None:
        self.top_bag = SessionBag()
        self.parent_node = _DummyNode()
        self.child_node = _DummyNode(parent=self.parent_node)
        self.ctx = RunContext(runtime=None, node=self.child_node)  # type: ignore[arg-type]
        self.ctx.object_bags = {
            SessionScope.TopLevel: self.top_bag,
            SessionScope.Parent: self.parent_node.session_bag,
            SessionScope.Self: self.child_node.session_bag,
        }
        self.bash = Bash()

    def tearDown(self) -> None:
        bag_values: Dict[str, Dict[str, object]] = getattr(self.parent_node.session_bag, "_values", {})
        for namespace in bag_values.values():
            for obj in namespace.values():
                if isinstance(obj, BashSession):
                    proc = obj._proc
                    obj._terminate_group_if_alive()
                    if proc is not None:
                        if proc.stdin:
                            try:
                                proc.stdin.close()
                            except Exception:
                                pass
                        if proc.stdout:
                            try:
                                proc.stdout.close()
                            except Exception:
                                pass
                        if proc.stderr:
                            try:
                                proc.stderr.close()
                            except Exception:
                                pass
                    thread = obj._stdout_thread
                    if thread is not None and thread.is_alive():
                        thread.join(timeout=0.5)

    def test_heredoc_without_trailing_newline_creates_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "hello.txt"
            command = "\n".join(
                [
                    f"cat <<'EOF' > {target}",
                    "hello from heredoc",
                    "EOF",
                ]
            )

            output = self.bash._call(self.ctx, command=command, session_id=0)

            self.assertEqual(output.strip(), "")
            self.assertTrue(target.exists())
            self.assertEqual(target.read_text(), "hello from heredoc\n")

    def test_multiple_heredocs_and_follow_up_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            first = Path(tmpdir) / "first.txt"
            second = Path(tmpdir) / "second.txt"
            command = "\n".join(
                [
                    f"cat <<'ONE' > {first}",
                    "alpha",
                    "ONE",
                    f"cat <<'TWO' > {second}",
                    "beta",
                    "TWO",
                    f"paste -d',' {first} {second}",
                ]
            )

            output = self.bash._call(self.ctx, command=command, session_id=0)

            self.assertTrue(output.startswith("alpha,beta"))
            self.assertEqual(output.strip(), "alpha,beta")
            self.assertEqual(first.read_text(), "alpha\n")
            self.assertEqual(second.read_text(), "beta\n")

    def test_heredoc_with_tab_stripping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "tabs.txt"
            command = "\n".join(
                [
                    f"cat <<-'EOF' > {target}",
                    "\tline one",
                    "\tline two",
                    "EOF",
                    f"cat {target}",
                ]
            )

            output = self.bash._call(self.ctx, command=command, session_id=0)

            self.assertIn("line one\nline two", output)
            self.assertEqual(output.strip(), "line one\nline two")
            self.assertEqual(target.read_text(), "line one\nline two\n")

    def test_heredoc_inside_if_block(self) -> None:
        command = "\n".join(
            [
                "if true; then",
                "  cat <<'EOF'",
                "branch body",
                "EOF",
                "else",
                "  echo skipped",
                "fi",
            ]
        )

        output = self.bash._call(self.ctx, command=command, session_id=0)

        self.assertEqual(output.strip(), "branch body")

    def test_heredoc_piped_into_filter(self) -> None:
        command = "\n".join(
            [
                "cat <<'EOF' | sed 's/foo/bar/'",
                "foo fighters",
                "EOF",
            ]
        )

        output = self.bash._call(self.ctx, command=command, session_id=0)

        self.assertEqual(output.strip().splitlines(), ["bar fighters"])

    def test_heredoc_with_variable_expansion(self) -> None:
        command = "\n".join(
            [
                "NAME=netflux",
                "cat <<EOF",
                "hello $NAME",
                "EOF",
            ]
        )

        output = self.bash._call(self.ctx, command=command, session_id=0)

        self.assertEqual(output.strip(), "hello netflux")

    def test_subshell_chaining_captures_both_outputs(self) -> None:
        command = "({ echo outer; ( echo inner 1>&2 ); } && echo done)"

        output = self.bash._call(self.ctx, command=command, session_id=0)

        # Filter out empty lines because the subshell chaining may produce blank lines in the output.
        lines = [line for line in output.strip().splitlines() if line]
        self.assertEqual(lines, ["outer", "inner", "done"])

    def test_long_running_command_respects_timeout(self) -> None:
        with self.assertRaisesRegex(BashCommandTimeoutException, "Command timed out"):
            # Sleep longer than the overridden timeout to force the BashFunction wrapper to report.
            self.bash._call(self.ctx, command="sleep 2", session_id=1, timeout_sec=1)

    def test_background_job_and_wait(self) -> None:
        command = "\n".join(
            [
                "sleep 0.1 &",
                "pid=$!",
                "wait \"$pid\"",
                "echo done",
            ]
        )

        output = self.bash._call(self.ctx, command=command, session_id=0)

        self.assertEqual(output.strip(), "done")

    def test_trailing_comments_and_blank_lines(self) -> None:
        command = "\n".join(
            [
                "echo hi # trailing comment",
                "",
                "# pure comment line",
                "echo bye",
            ]
        )

        output = self.bash._call(self.ctx, command=command, session_id=0)

        self.assertEqual([line for line in output.strip().splitlines()], ["hi", "bye"])

    def test_brace_expansion_and_globbing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir)
            command = "\n".join(
                [
                    f"pushd {target} >/dev/null",
                    "touch file{1..3}.txt",
                    "printf '%s\\n' file?.txt",
                    "popd >/dev/null",
                ]
            )

            output = self.bash._call(self.ctx, command=command, session_id=0)

            self.assertEqual(
                [line for line in output.strip().splitlines()],
                ["file1.txt", "file2.txt", "file3.txt"],
            )

    def test_here_string_and_process_substitution(self) -> None:
        command = "\n".join(
            [
                "cat <<< 'alpha'",
                "diff <(printf 'one\\n') <(printf 'one\\n')",
                "echo done",
            ]
        )

        output = self.bash._call(self.ctx, command=command, session_id=0)

        self.assertEqual([line for line in output.strip().splitlines()], ["alpha", "done"])

    def test_set_e_and_pipefail_restored(self) -> None:
        command = "\n".join(
            [
                "set -e",
                "set -o pipefail",
                "echo start",
                "false || true",
                "echo after",
            ]
        )

        output = self.bash._call(self.ctx, command=command, session_id=2)

        self.assertEqual([line for line in output.strip().splitlines()], ["start", "after"])

    def test_trap_exit_in_subshell_runs(self) -> None:
        command = "( trap 'echo cleanup' EXIT; echo work )"

        output = self.bash._call(self.ctx, command=command, session_id=0)

        self.assertEqual([line for line in output.strip().splitlines()], ["work", "cleanup"])

    def test_function_definition_and_call(self) -> None:
        command = "\n".join(
            [
                "greet() { echo \"hello $1\"; }",
                "greet netflux",
            ]
        )

        output = self.bash._call(self.ctx, command=command, session_id=0)

        self.assertEqual(output.strip(), "hello netflux")

    def test_source_with_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script = Path(tmpdir) / "helper.sh"
            script.write_text("say() { echo sourced; }\n")
            command = "\n".join(
                [
                    f"pushd {tmpdir} >/dev/null",
                    ". ./helper.sh",
                    "say",
                    "popd >/dev/null",
                ]
            )

            output = self.bash._call(self.ctx, command=command, session_id=0)

            self.assertEqual(output.strip(), "sourced")

if __name__ == "__main__":
    unittest.main()
