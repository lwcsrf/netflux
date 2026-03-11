import os
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Optional

from ..core import RunContext, SessionBag, SessionScope
from ..func_lib.bash import (
    Bash,
    BashSession,
    BashException,
    BashCommandTimeoutException,
    BashNonZeroExitCodeException,
    BashSessionCrashedException,
    BashRequiresRestartException,
)


def _bash_path(p) -> str:
    """Convert a path to a form usable inside bash.

    On Windows (MSYS2/Git Bash) this turns ``C:\\Users\\foo`` into
    ``/c/Users/foo`` so that the persistent bash process can resolve it.
    On POSIX systems the path is returned unchanged.
    """
    s = str(p)
    if os.name == "nt" and len(s) >= 2 and s[1] == ":":
        drive = s[0].lower()
        rest = s[2:].replace("\\", "/")
        return f"/{drive}{rest}"
    return s


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
            bp = _bash_path(target)
            command = "\n".join(
                [
                    f"cat <<'EOF' > {bp}",
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
            bp_first = _bash_path(first)
            bp_second = _bash_path(second)
            command = "\n".join(
                [
                    f"cat <<'ONE' > {bp_first}",
                    "alpha",
                    "ONE",
                    f"cat <<'TWO' > {bp_second}",
                    "beta",
                    "TWO",
                    f"paste -d',' {bp_first} {bp_second}",
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
            bp = _bash_path(target)
            command = "\n".join(
                [
                    f"cat <<-'EOF' > {bp}",
                    "\tline one",
                    "\tline two",
                    "EOF",
                    f"cat {bp}",
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
            bp = _bash_path(tmpdir)
            command = "\n".join(
                [
                    f"pushd {bp} >/dev/null",
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
            bp = _bash_path(tmpdir)
            command = "\n".join(
                [
                    f"pushd {bp} >/dev/null",
                    ". ./helper.sh",
                    "say",
                    "popd >/dev/null",
                ]
            )

            output = self.bash._call(self.ctx, command=command, session_id=0)

            self.assertEqual(output.strip(), "sourced")

    def test_no_sentinel_leakage_across_sequential_commands(self) -> None:
        """Sentinel must never appear in command output, even across rapid sequential calls."""
        import re as _re
        sentinel_pat = _re.compile(r"__NETFLUX_BASH_DONE__")

        for i in range(20):
            # Use a pipe (the pattern that originally triggered the bug).
            out1 = self.bash._call(self.ctx, command="echo 'hello world' | cat", session_id=0)
            self.assertNotRegex(out1, sentinel_pat, f"Sentinel leaked on piped command, iteration {i}")

            # Immediate follow-up, echoing a unique string.
            out2 = self.bash._call(self.ctx, command=f"echo 'check-{i}'", session_id=0)
            self.assertNotRegex(out2, sentinel_pat, f"Sentinel leaked on follow-up command, iteration {i}")
            self.assertEqual(out2.strip(), f"check-{i}")

    def test_no_sentinel_leakage_with_stderr_redirect(self) -> None:
        """Reproduce the original report: piped command with 2>/dev/null followed by more commands."""
        import re as _re
        sentinel_pat = _re.compile(r"__NETFLUX_BASH_DONE__")

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "sample.txt"
            # Create a small file.
            target.write_text("\n".join(f"line {n}" for n in range(1, 52)) + "\n")
            bp = _bash_path(target)

            for _ in range(10):
                command = "\n".join([
                    f"cat {bp} 2>/dev/null | head -5",
                    'echo "---"',
                    f"wc -l < {bp}",
                ])
                output = self.bash._call(self.ctx, command=command, session_id=0)
                self.assertNotRegex(output, sentinel_pat, "Sentinel leaked into output")
                lines = [l for l in output.strip().splitlines() if l]
                self.assertIn("---", lines)
                self.assertTrue(lines[-1].strip() == "51", f"Expected 51 lines, got: {lines[-1]}")

    def test_set_x_does_not_break_sentinel_parsing(self) -> None:
        output = self.bash._call(self.ctx, command="set -x\necho hi", session_id=0)
        self.assertIn("hi", output.splitlines())

        follow_up = self.bash._call(self.ctx, command="echo after", session_id=0)
        self.assertIn("after", follow_up.splitlines())

    def test_alias_set_does_not_pollute_wrapper(self) -> None:
        self.bash._call(self.ctx, command="alias set='echo aliasset'", session_id=0)
        output = self.bash._call(self.ctx, command="echo hi", session_id=0)
        self.assertEqual(output.strip(), "hi")

    def test_function_named_printf_does_not_break_sentinel_parsing(self) -> None:
        self.bash._call(self.ctx, command="printf() { echo hijacked; }", session_id=0)
        output = self.bash._call(self.ctx, command="echo hi", session_id=0)
        self.assertEqual(output.strip(), "hi")


class TestBashEdgeCases(unittest.TestCase):
    """Edge-case tests for unusual but plausible Bash session scenarios."""

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

    # ── State persistence ──

    def test_cwd_persists_across_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            bp = _bash_path(tmpdir)
            self.bash._call(self.ctx, command=f"cd {bp}", session_id=0)
            output = self.bash._call(self.ctx, command="pwd", session_id=0)
            self.assertIn(os.path.basename(tmpdir), output)

    def test_env_var_persists_across_calls(self) -> None:
        self.bash._call(self.ctx, command="export NF_TEST_VAR=hello_netflux", session_id=0)
        output = self.bash._call(self.ctx, command="echo $NF_TEST_VAR", session_id=0)
        self.assertEqual(output.strip(), "hello_netflux")

    def test_shell_function_persists_across_calls(self) -> None:
        self.bash._call(self.ctx, command="nf_fn() { echo fn_output; }", session_id=0)
        output = self.bash._call(self.ctx, command="nf_fn", session_id=0)
        self.assertEqual(output.strip(), "fn_output")

    def test_alias_persists_across_calls(self) -> None:
        self.bash._call(self.ctx, command="alias nfalias='echo aliased'", session_id=0)
        output = self.bash._call(self.ctx, command="nfalias", session_id=0)
        self.assertEqual(output.strip(), "aliased")

    def test_separate_sessions_independent_state(self) -> None:
        self.bash._call(self.ctx, command="export X=session10", session_id=10)
        self.bash._call(self.ctx, command="export X=session11", session_id=11)
        out10 = self.bash._call(self.ctx, command="echo $X", session_id=10)
        out11 = self.bash._call(self.ctx, command="echo $X", session_id=11)
        self.assertEqual(out10.strip(), "session10")
        self.assertEqual(out11.strip(), "session11")

    # ── Restart behavior ──

    def test_restart_clears_env(self) -> None:
        self.bash._call(self.ctx, command="export EPHEMERAL=123", session_id=0)
        self.bash._call(self.ctx, command=None, restart=True, session_id=0)
        output = self.bash._call(self.ctx, command='echo "${EPHEMERAL:-unset}"', session_id=0)
        self.assertEqual(output.strip(), "unset")

    def test_restart_with_command(self) -> None:
        self.bash._call(self.ctx, command="export BEFORE=yes", session_id=0)
        output = self.bash._call(
            self.ctx, command='echo "${BEFORE:-gone}"', restart=True, session_id=0
        )
        self.assertEqual(output.strip(), "gone")

    def test_restart_only_returns_message(self) -> None:
        result = self.bash._call(self.ctx, command=None, restart=True, session_id=0)
        self.assertIn("restarted", result)

    def test_new_session_does_not_source_inherited_bash_env(self) -> None:
        from unittest import mock

        with tempfile.TemporaryDirectory() as tmpdir:
            marker = Path(tmpdir) / "bash_env_marker.txt"
            bash_env = Path(tmpdir) / "bash_env.sh"
            bash_env.write_text(
                "\n".join(
                    [
                        "export FROM_BASH_ENV=1",
                        f"echo sourced > {_bash_path(marker)}",
                    ]
                )
                + "\n"
            )

            with mock.patch.dict(os.environ, {"BASH_ENV": str(bash_env)}):
                output = self.bash._call(
                    self.ctx,
                    command='echo "${FROM_BASH_ENV:-unset}"',
                    session_id=0,
                )

            self.assertEqual(output.strip(), "unset")
            self.assertFalse(marker.exists())

    def test_new_session_does_not_import_inherited_shellopts(self) -> None:
        from unittest import mock

        with mock.patch.dict(os.environ, {"SHELLOPTS": "xtrace"}):
            output = self.bash._call(self.ctx, command="echo hi", session_id=0)

        self.assertEqual(output.strip(), "hi")
        self.assertNotIn("__nf_was_e", output)
        self.assertNotIn("__NETFLUX_BASH_DONE__", output)

    def test_new_session_does_not_import_inherited_bash_functions(self) -> None:
        from unittest import mock

        with mock.patch.dict(os.environ, {"BASH_FUNC_nffn%%": "() { echo imported_fn; }"}):
            output = self.bash._call(
                self.ctx,
                command='type nffn >/dev/null 2>&1; echo "$?"',
                session_id=0,
            )

        self.assertEqual(output.strip(), "1")

    # ── fd 254 resilience ──

    def test_exec_stdout_to_dev_null_sentinel_survives(self) -> None:
        """Redirecting fd 1 to /dev/null must not break sentinel delivery on fd 254."""
        output = self.bash._call(
            self.ctx, command="exec 1>/dev/null; echo invisible", session_id=0
        )
        self.assertNotIn("invisible", output)
        # Session still works (fd 1 remains /dev/null but sentinel on 254 is fine)
        output2 = self.bash._call(self.ctx, command="echo still_invisible", session_id=0)
        self.assertNotIn("still_invisible", output2)

    # ── exit behavior ──

    def test_exit_kills_session(self) -> None:
        """Running `exit` inside the command block kills bash; session should crash."""
        with self.assertRaises((BashSessionCrashedException, BashCommandTimeoutException)):
            self.bash._call(self.ctx, command="exit 0", session_id=20)

    def test_subshell_exit_does_not_kill_session(self) -> None:
        """(exit N) only exits the subshell; parent shell remains healthy."""
        with self.assertRaises(BashNonZeroExitCodeException) as cm:
            self.bash._call(self.ctx, command="(exit 42)", session_id=0)
        self.assertEqual(cm.exception.exit_code, 42)
        # Session still alive
        output = self.bash._call(self.ctx, command="echo alive", session_id=0)
        self.assertEqual(output.strip(), "alive")

    # ── Input validation ──

    def test_empty_command_rejected(self) -> None:
        with self.assertRaises(BashException):
            self.bash._call(self.ctx, command="", session_id=0)
        with self.assertRaises(BashException):
            self.bash._call(self.ctx, command="   \n  ", session_id=0)

    def test_zero_timeout_rejected(self) -> None:
        with self.assertRaises(BashException):
            self.bash._call(self.ctx, command="echo hi", timeout_sec=0, session_id=0)

    def test_negative_timeout_rejected(self) -> None:
        with self.assertRaises(BashException):
            self.bash._call(self.ctx, command="echo hi", timeout_sec=-5, session_id=0)

    # ── Non-zero exit codes ──

    def test_nonzero_exit_carries_code_and_output(self) -> None:
        with self.assertRaises(BashNonZeroExitCodeException) as cm:
            self.bash._call(
                self.ctx, command="echo before_error; sh -c 'exit 42'", session_id=0
            )
        self.assertEqual(cm.exception.exit_code, 42)
        self.assertIn("before_error", cm.exception.output)

    def test_session_healthy_after_nonzero_exit(self) -> None:
        """A non-zero exit should NOT mark the session as requiring restart."""
        with self.assertRaises(BashNonZeroExitCodeException):
            self.bash._call(self.ctx, command="false", session_id=0)
        output = self.bash._call(self.ctx, command="echo healthy", session_id=0)
        self.assertEqual(output.strip(), "healthy")

    def test_multiline_parse_errors_do_not_require_restart(self) -> None:
        cases = {
            "missing_fi": "if true; then\n  echo hi",
            "unclosed_single_quote": "echo 'hello",
        }

        for sid, (name, command) in enumerate(cases.items(), start=70):
            with self.subTest(name=name):
                with self.assertRaises(BashNonZeroExitCodeException) as cm:
                    self.bash._call(self.ctx, command=command, session_id=sid, timeout_sec=2)
                self.assertEqual(cm.exception.exit_code, 2)
                self.assertIn("<command>", cm.exception.output)

                follow_up = self.bash._call(self.ctx, command="echo alive", session_id=sid, timeout_sec=2)
                self.assertEqual(follow_up.strip(), "alive")

    def test_eof_terminated_constructs_do_not_poison_session(self) -> None:
        cases = [
            ("missing_heredoc_terminator", "cat <<'EOF'\nhello", ["<command>", "hello"]),
            ("trailing_backslash", "echo hello \\", ["hello"]),
        ]

        for sid, (name, command, expected_parts) in enumerate(cases, start=80):
            with self.subTest(name=name):
                output = self.bash._call(self.ctx, command=command, session_id=sid, timeout_sec=2)
                for expected_part in expected_parts:
                    self.assertIn(expected_part, output)

                follow_up = self.bash._call(self.ctx, command="echo alive", session_id=sid, timeout_sec=2)
                self.assertEqual(follow_up.strip(), "alive")

    # ── Comment-only / no-op commands ──

    def test_comment_only_command(self) -> None:
        output = self.bash._call(self.ctx, command="# just a comment", session_id=0)
        self.assertEqual(output.strip(), "")

    def test_multiline_blank_command(self) -> None:
        with self.assertRaises(BashException):
            self.bash._call(self.ctx, command="\n\n\n", session_id=0)

    def test_accidental_stdin_reader_gets_eof_not_wrapper_input(self) -> None:
        output = self.bash._call(self.ctx, command="cat", session_id=44, timeout_sec=2)
        self.assertEqual(output.strip(), "")

    def test_read_builtin_does_not_consume_wrapper_lines(self) -> None:
        output = self.bash._call(
            self.ctx,
            command="\n".join(
                [
                    'if read var; then echo "read:$var"; else echo "eof"; fi',
                    'printf "var=<%s>\\n" "${var:-}"',
                ]
            ),
            session_id=45,
            timeout_sec=2,
        )
        self.assertEqual(
            [line for line in output.splitlines() if line],
            ["eof", "var=<>"],
        )

    # ── Output truncation ──

    def test_output_truncated_at_limit(self) -> None:
        """Output exceeding MAX_OUTPUT_CHARS should be truncated with a notice."""
        output = self.bash._call(self.ctx, command="seq 1 10000", session_id=0)
        self.assertIn("Output truncated", output)
        # Total should be roughly MAX + length of the truncation notice
        self.assertLessEqual(
            len(output), BashSession.MAX_OUTPUT_CHARS + 200
        )

    # ── set -e wrapper semantics ──

    def test_output_exactly_at_limit_not_marked_truncated(self) -> None:
        output = self.bash._call(
            self.ctx,
            command=r"head -c 40000 /dev/zero | tr '\0' x",
            session_id=0,
        )
        self.assertEqual(len(output), BashSession.MAX_OUTPUT_CHARS)
        self.assertNotIn("Output truncated", output)

    def test_crash_after_exact_limit_not_marked_truncated(self) -> None:
        with self.assertRaises(BashSessionCrashedException) as cm:
            self.bash._call(
                self.ctx,
                command="head -c 40000 /dev/zero | tr '\\0' x\nexit 0",
                session_id=0,
            )
        self.assertNotIn("Output truncated", str(cm.exception))

    def test_user_newline_past_limit_is_marked_truncated(self) -> None:
        output = self.bash._call(
            self.ctx,
            command=r"head -c 40000 /dev/zero | tr '\0' x; printf '\n'",
            session_id=0,
        )
        self.assertIn("Output truncated", output)

    def test_set_e_restored_between_calls(self) -> None:
        """The wrapper suppresses -e during execution for safety (prevents the
        shell from dying on failed commands) but preserves it at the shell level
        between calls.  Because set +e runs before each command, the user's code
        observes -e as off — this is the documented design trade-off.
        Verify the session stays healthy and the wrapper's save/restore cycle
        does not corrupt other shell state."""
        self.bash._call(self.ctx, command="set -e", session_id=30)
        self.bash._call(self.ctx, command="echo middle", session_id=30)
        # The wrapper always does set +e before the user's command for safety,
        # so $- will NOT contain 'e' inside the command (by design).
        output = self.bash._call(self.ctx, command='case $- in *e*) echo YES;; *) echo NO;; esac', session_id=30)
        self.assertEqual(output.strip(), "NO")
        # Session is still healthy after the set -e round-trip
        output2 = self.bash._call(self.ctx, command="echo still_ok", session_id=30)
        self.assertEqual(output2.strip(), "still_ok")

    def test_set_e_wrapper_prevents_premature_exit(self) -> None:
        """Even with set -e, the wrapper's set +e prevents the shell from dying
        mid-block. Both echos should run."""
        self.bash._call(self.ctx, command="set -e", session_id=31)
        # Without the wrapper's set +e, `false` would kill the shell.
        output = self.bash._call(
            self.ctx, command="echo before; false; echo after", session_id=31
        )
        self.assertIn("before", output)
        self.assertIn("after", output)

    # ── Timeout → restart recovery ──

    def test_timeout_requires_restart_then_recovers(self) -> None:
        sid = 40
        with self.assertRaises(BashCommandTimeoutException):
            self.bash._call(self.ctx, command="sleep 5", session_id=sid, timeout_sec=1)
        # Next call should refuse (requires restart)
        with self.assertRaises(BashRequiresRestartException):
            self.bash._call(self.ctx, command="echo hi", session_id=sid)
        # Restart
        self.bash._call(self.ctx, command=None, restart=True, session_id=sid)
        # Should work now
        output = self.bash._call(self.ctx, command="echo recovered", session_id=sid)
        self.assertEqual(output.strip(), "recovered")

    def test_timeout_hint_mentions_noexec_only_when_pattern_matches(self) -> None:
        hint = "Ensure you avoid 'set -n' / 'set -o noexec' in commands run in the session"

        for sid, command in enumerate(("set -n", "set -o noexec"), start=47):
            with self.subTest(command=command):
                with self.assertRaises(BashCommandTimeoutException) as cm:
                    self.bash._call(self.ctx, command=command, session_id=sid, timeout_sec=1)
                self.assertIn(hint, str(cm.exception))

        with self.assertRaises(BashCommandTimeoutException) as cm:
            self.bash._call(self.ctx, command="sleep 5", session_id=49, timeout_sec=1)
        self.assertNotIn(hint, str(cm.exception))

    def test_terminate_closes_process_pipes(self) -> None:
        session = BashSession(41)
        session.start()
        proc = session._proc

        self.assertIsNotNone(proc)
        session._terminate_group_if_alive()

        thread = session._stdout_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=0.5)

        assert proc is not None
        self.assertTrue(proc.stdin is None or proc.stdin.closed)
        self.assertTrue(proc.stdout is None or proc.stdout.closed)
        self.assertTrue(proc.stderr is None or proc.stderr.closed)

    def test_legacy_internal_var_names_do_not_collide_with_wrapper(self) -> None:
        self.bash._call(
            self.ctx,
            command="readonly __nf_ec=7; readonly __nf_was_e=1",
            session_id=42,
        )
        output = self.bash._call(self.ctx, command="echo hi", session_id=42)
        self.assertEqual(output.strip(), "hi")

    def test_set_u_and_legacy_internal_var_names_do_not_crash_wrapper(self) -> None:
        output = self.bash._call(
            self.ctx,
            command="set -u; unset __nf_was_e; echo body",
            session_id=43,
        )
        self.assertEqual(output.strip(), "body")
        follow_up = self.bash._call(self.ctx, command="echo alive", session_id=43)
        self.assertEqual(follow_up.strip(), "alive")

    def test_per_call_internal_vars_do_not_accumulate_in_session(self) -> None:
        sid = 44
        count_cmd = r"compgen -A variable | grep -Ec '^__nf_.*_[0-9a-f]{32}$' || true"

        baseline = self.bash._call(self.ctx, command=count_cmd, session_id=sid)
        self.assertEqual(baseline.strip(), "1")

        for i in range(10):
            output = self.bash._call(self.ctx, command=f"echo run-{i}", session_id=sid)
            self.assertEqual(output.strip(), f"run-{i}")

        after = self.bash._call(self.ctx, command=count_cmd, session_id=sid)
        self.assertEqual(after.strip(), "1")

    @unittest.skipUnless(os.name == "nt", "Windows-specific process tree termination")
    def test_windows_terminate_uses_ctrl_break_before_taskkill(self) -> None:
        from unittest import mock
        from ..func_lib import bash as bash_mod

        session = BashSession(45)
        proc = mock.Mock()
        proc.pid = 1234
        proc.poll.return_value = None
        proc.wait.return_value = None
        proc.stdin = None
        proc.stdout = None
        proc.stderr = None
        session._proc = proc

        with mock.patch.object(bash_mod.subprocess, "run") as run_mock:
            session._terminate_group_if_alive()

        proc.send_signal.assert_called_once_with(bash_mod.signal.CTRL_BREAK_EVENT)
        run_mock.assert_not_called()
        proc.wait.assert_called_once_with(timeout=2)
        self.assertIsNone(session._proc)

    @unittest.skipUnless(os.name == "nt", "Windows-specific process tree termination")
    def test_windows_terminate_falls_back_to_taskkill_tree(self) -> None:
        from unittest import mock
        from ..func_lib import bash as bash_mod

        session = BashSession(46)
        proc = mock.Mock()
        proc.pid = 1234
        proc.poll.return_value = None
        proc.wait.return_value = None
        proc.send_signal.side_effect = OSError("no console")
        proc.stdin = None
        proc.stdout = None
        proc.stderr = None
        session._proc = proc

        with mock.patch.object(bash_mod.subprocess, "run") as run_mock:
            session._terminate_group_if_alive()

        run_mock.assert_called_once_with(
            ["taskkill", "/PID", "1234", "/T", "/F"],
            stdout=bash_mod.subprocess.DEVNULL,
            stderr=bash_mod.subprocess.DEVNULL,
            check=False,
            timeout=5,
        )
        proc.wait.assert_called_once_with(timeout=2)
        self.assertIsNone(session._proc)


class TestBashDiscovery(unittest.TestCase):
    def test_find_bash_skips_windowsapps_stub(self) -> None:
        from unittest import mock
        from ..func_lib import bash as bash_mod

        stub = r"C:\Users\andre\AppData\Local\Microsoft\WindowsApps\bash.exe"
        real = r"C:\Program Files\Git\bin\bash.exe"
        with mock.patch.object(bash_mod.os, "name", "nt"), \
             mock.patch.object(bash_mod.os.path, "isfile", return_value=False), \
             mock.patch.object(bash_mod, "_windows_bash_candidates", return_value=[stub, real]), \
             mock.patch.object(bash_mod, "_bash_works", side_effect=lambda path: path == real):
            self.assertEqual(BashSession._find_bash(), real)


class TestBashSyntaxPreflight(unittest.TestCase):
    def test_syntax_check_flags_hard_parse_errors_only(self) -> None:
        session = BashSession(0)
        bad_host, bad_bash = BashSession._write_command_file("if true; then\n  echo hi")
        warn_host, warn_bash = BashSession._write_command_file("cat <<'EOF'\nhello")

        try:
            bad = session._syntax_check_command_file(bad_host, bad_host, bad_bash)
            self.assertIsNotNone(bad)
            assert bad is not None
            bad_output, bad_code = bad
            self.assertEqual(bad_code, 2)
            self.assertIn("<command>", bad_output)

            warn = session._syntax_check_command_file(warn_host, warn_host, warn_bash)
            self.assertIsNone(warn)
        finally:
            for path in (bad_host, warn_host):
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass


class TestBashChunkedReader(unittest.TestCase):
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

    def test_large_no_newline_output_does_not_deadlock(self) -> None:
        output = self.bash._call(
            self.ctx,
            command="head -c 20000 /dev/zero | tr '\\0' x; echo done",
            session_id=60,
            timeout_sec=3,
        )
        self.assertIn("done", output)

    def test_large_inline_script_executes_and_persists_state(self) -> None:
        sid = 65
        command = "\n".join([f"v{i}={i}" for i in range(5000)] + ['echo "${v4999}"'])

        output = self.bash._call(self.ctx, command=command, session_id=sid, timeout_sec=10)
        self.assertEqual(output.strip(), "4999")

        follow_up = self.bash._call(self.ctx, command='echo "${v0}-${v4999}"', session_id=sid, timeout_sec=3)
        self.assertEqual(follow_up.strip(), "0-4999")

    def test_binary_pipe_path_preserves_plain_output(self) -> None:
        output = self.bash._call(self.ctx, command="printf 'hello\\n'", session_id=62)
        self.assertEqual(output, "hello\n\n")

    def test_binary_pipe_path_normalizes_crlf_output(self) -> None:
        output = self.bash._call(
            self.ctx,
            command="printf 'hello\\r\\nbye\\r'; echo done",
            session_id=64,
        )
        self.assertEqual(output, "hello\nbye\ndone\n\n")

    def test_binary_pipe_path_decodes_utf8_output(self) -> None:
        output = self.bash._call(
            self.ctx,
            command="printf '\\303\\251\\303\\251'; echo done",
            session_id=63,
        )
        self.assertEqual(output, "\u00e9\u00e9done\n\n")

    def test_small_chunk_reader_handles_split_utf8_and_sentinel(self) -> None:
        from unittest import mock

        with mock.patch.object(BashSession, "READ_CHUNK_BYTES", 1):
            output = self.bash._call(
                self.ctx,
                command="printf '\\303\\251\\303\\251\\303\\251'; echo done",
                session_id=61,
            )
        self.assertIn("\u00e9\u00e9\u00e9done", output)


if __name__ == "__main__":
    unittest.main()
