import os
import platform
import re
import queue
import signal
import subprocess
import threading
import time
import uuid
from typing import Optional, Tuple, List

from ..core import CodeFunction, FunctionArg, RunContext, SessionScope, NoParentSessionError

class BashException(Exception):
    """Base class for Bash tool failures."""

class BashCommandTimeoutException(BashException):
    """Raised when a command did not complete (sentinel not seen) within timeout."""

class BashNonZeroExitCodeException(BashException):
    """Raised when the command completed with a non-zero exit code."""
    def __init__(self, exit_code: int, output: str):
        super().__init__(f"Exit code: {exit_code}\n--- output ---\n{output}")
        self.exit_code = exit_code
        self.output = output

class BashRequiresRestartException(BashException):
    """Raised when the session has been marked as requiring a restart."""

class BashSessionRaceException(BashException):
    """Raised when the same Bash session would be used in parallel."""

class BashSessionCrashedException(BashException):
    """(Ultra-rare?) Raised when the bash process is dead and cannot be used."""

class BashSession:
    """
    Owns a persistent /bin/bash process with UTF-8 pipes, a process group (POSIX),
    and reader threads. Commands are executed by sending to stdin and reading
    until a unique sentinel line appears on stdout.

    One-command-at-a-time gate for this session. The CodeFunction should
    `try_acquire()`/`release()` around *any* use (execute/restart) to guarantee
    end-to-end exclusivity.
    """

    DEFAULT_TIMEOUT_SEC = 120
    MAX_OUTPUT_CHARS = 40_000

    def __init__(self, session_id: int) -> None:
        self.session_id = session_id
        self._proc: Optional[subprocess.Popen[str]] = None
        self._q: queue.Queue[str] = queue.Queue()
        self._stdout_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()         # serialize commands per session
        self.requires_restart: bool = False   # set true on timeout, etc.
        self._alive_once_started: bool = False

    def start(self) -> None:
        if platform.system().lower() not in ("linux", "darwin"):
            raise BashException("Bash tool currently supports Unix-like systems only.")

        if self._proc and self._proc.poll() is None:
            return

        # Ensure clean slate.
        self._terminate_group_if_alive()

        preexec = os.setsid if os.name == "posix" else None
        creationflags = 0

        self._proc = subprocess.Popen(  # noqa: S603, S607 (we intentionally run bash)
            ["/bin/bash", "--noprofile", "--norc", "-s"],  # read from stdin; skip user rc/profile
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # line-buffered
            encoding="utf-8",
            errors="replace",
            preexec_fn=preexec,
            creationflags=creationflags,
        )
        self._alive_once_started = True
        self.requires_restart = False
        self._start_readers()

        # Set pipeline semantics, disable history, quiet PS1, and block until ready.
        self._write(
            "set -o pipefail; export HISTFILE=/dev/null; export PS1=''; "
            "shopt -s expand_aliases; echo __NETFLUX_BASH_READY__\n"
        )
        # Require readiness; if not reached, mark for restart and fail.
        if not self._drain_until("__NETFLUX_BASH_READY__", timeout=10):
            self.requires_restart = True
            self._terminate_group_if_alive()
            raise BashSessionCrashedException("Bash session failed to reach ready state; tool must be restarted.")

    def restart(self) -> None:
        # Caller must hold the session lock via try_acquire().
        self._terminate_group_if_alive()
        # Join reader threads but don't stress over it.
        try:
            if self._stdout_thread is not None:
                self._stdout_thread.join(timeout=0.25)
        except Exception:
            pass
        self._proc = None
        self._stdout_thread = None
        # Clear any pending.
        self._drain_queue_nowait()
        self.start()

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _terminate_group_if_alive(self) -> None:
        if not self._proc:
            return
        try:
            if self._proc.poll() is None:
                if os.name == "posix":
                    try:
                        os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
                    except Exception:
                        # Fall back to process terminate if group kill fails
                        self._proc.terminate()
                else:
                    self._proc.terminate()
                try:
                    self._proc.wait(timeout=2)
                except Exception:
                    if os.name == "posix":
                        try:
                            os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                        except Exception:
                            pass
                    else:
                        self._proc.kill()
        finally:
            self._proc = None

    def _start_readers(self) -> None:
        assert self._proc and self._proc.stdout
        def pump(pipe, label: str):
            try:
                for line in iter(pipe.readline, ""):
                    self._q.put(line)
            except Exception:
                # Mark session unhealthy if the reader crashes (broken pipe, decode error, etc.).
                self.requires_restart = True

        self._stdout_thread = threading.Thread(
            target=pump, args=(self._proc.stdout, "stdout"), name=f"bash-{self.session_id}-stdout", daemon=True
        )
        self._stdout_thread.start()

    # -------- I/O helpers --------
    def _write(self, s: str) -> None:
        if not (self._proc and self._proc.stdin):
            raise BashSessionCrashedException("Bash process is not available.")
        self._proc.stdin.write(s)
        self._proc.stdin.flush()

    def _drain_queue_nowait(self) -> None:
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            return

    def _drain_until(self, token: str, timeout: float) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            try:
                line = self._q.get(timeout=0.05)
                if token in line:
                    return True
            except queue.Empty:
                pass
        return False

    # -------- external concurrency control --------
    def try_acquire(self, timeout_sec: float = 0.0) -> bool:
        """Attempt to lock this session for exclusive use (optionally waiting)."""
        if timeout_sec > 0:
            return self._lock.acquire(timeout=timeout_sec)
        return self._lock.acquire(blocking=False)

    def release(self) -> None:
        """Release the session lock if held by this thread."""
        if self._lock.locked():
            self._lock.release()

    # -------- execute --------
    def execute(self, command: str, timeout_sec: Optional[int]) -> Tuple[str, int, str]:
        """
        Returns (combined_output, exit_code, sentinel_id).
        Raises BashCommandTimeoutException if sentinel not observed within timeout.
        The returned output is stdout+stderr (sentinel trimmed).
        """
        if self.requires_restart:
            raise BashRequiresRestartException(
                "Bash session is marked as requiring restart due to a previous timeout or failure. "
                "Please call the tool with restart=true."
            )

        if not self.alive():
            if self._alive_once_started:
                # Was started before but is now dead → require explicit restart.
                raise BashSessionCrashedException("Bash session has crashed; tool must be restarted.")
            self.start()

        # Clear any buffered leftovers before running the next command.
        self._drain_queue_nowait()

        # Single sentinel line that also carries the exit code (Y's format).
        sentinel = f"__NETFLUX_BASH_DONE__{uuid.uuid4().hex}"
        # Ensure the user command ends with a newline so here-doc delimiters stand alone.
        cmd = command if command.endswith("\n") else command + "\n"
        # Make sentinel resilient to `set -e` and common fd redirections; print to both stdout and stderr.
        block = (
            "was_e=false; case $- in *e*) was_e=true;; esac\n"
            # The curly braces are intentionally placed on their own lines to ensure heredoc delimiters
            # are recognized correctly by bash. Do not merge them with other lines.
            "set +e; {\n"
            f"{cmd}"
            "}\n"
            "__nf_ec=$?\n"
            "$was_e && set -e\n"
            f"printf '\\n{sentinel} %d\\n' \"$__nf_ec\"\n"
            f"printf '\\n{sentinel} %d\\n' \"$__nf_ec\" 1>&2\n"
        )
        try:
            self._write(block)
        except Exception as e:
            self.requires_restart = True
            raise BashSessionCrashedException(
                f"Bash stdin is not writable (session exited?): {e}. Tool must be restarted."
            ) from e

        # Read until we see the sentinel; preserve arrival order (stdout+stderr interleaving).
        effective_timeout = self.DEFAULT_TIMEOUT_SEC if (timeout_sec is None) else float(timeout_sec)
        deadline = time.time() + effective_timeout
        chunks: List[str] = []
        collected = 0
        exit_code: Optional[int] = None
        found = False

        while time.time() < deadline:
            try:
                line = self._q.get(timeout=0.05)
            except queue.Empty:
                # Also detect process death while waiting
                if self._proc and self._proc.poll() is not None:
                    self.requires_restart = True
                    partial = "".join(chunks)
                    if collected >= self.MAX_OUTPUT_CHARS:
                        partial += (
                            f"\n\n... Output truncated (showing {self.MAX_OUTPUT_CHARS} characters; "
                            f"limit {self.MAX_OUTPUT_CHARS}) ..."
                        )
                    raise BashSessionCrashedException(
                        "Bash process terminated unexpectedly; tool must be restarted.\n\n"
                        "--- partial output ---\n"
                        f"{partial}"
                    )
                continue

        
            line = line or ""
            if sentinel in line:
                # Parse exit code from the sentinel line (format: "<sentinel> <code>")
                m = re.search(rf"{re.escape(sentinel)}\s+(-?\d+)\s*$", line)
                exit_code = int(m.group(1)) if m else None
                found = True
                # Do NOT append the sentinel line to output.
                break

            if collected < self.MAX_OUTPUT_CHARS:
                space = self.MAX_OUTPUT_CHARS - collected
                if space > 0:
                    slice_text = line if len(line) <= space else line[:space]
                    chunks.append(slice_text)
                    collected += len(slice_text)

        if not found:
            # Timed out; mark session as requiring restart and stop activity.
            self.requires_restart = True
            self._terminate_group_if_alive()
            partial = "".join(chunks)
            if collected >= self.MAX_OUTPUT_CHARS:
                partial += (
                    f"\n\n... Output truncated (showing {self.MAX_OUTPUT_CHARS} characters; "
                    f"limit {self.MAX_OUTPUT_CHARS}) ..."
                )
            raise BashCommandTimeoutException(
                f"Command timed out after {int(effective_timeout)} seconds. "
                f"The session has been marked as requiring restart. "
                "Invoke the tool again with restart=true to continue.\n\n"
                f"--- partial output ---\n"
                f"{partial}"
            )

        combined = "".join(chunks)
        if collected >= self.MAX_OUTPUT_CHARS:
            combined += (
                f"\n\n... Output truncated (showing {self.MAX_OUTPUT_CHARS} characters; "
                f"limit {self.MAX_OUTPUT_CHARS}) ..."
            )

        if exit_code is None:
            # Sentinel observed but exit code unparsable → mark unhealthy and force restart.
            self.requires_restart = True
            self._terminate_group_if_alive()
            raise BashSessionCrashedException(
                "Malformed sentinel line; tool must be restarted."
            )
        return combined, exit_code, sentinel


class Bash(CodeFunction):
    """
    Stateful Bash tool.
    - Persistent shell state per {agent, session_id} across function/tool calls.
    - Non-interactive commands only. Use `restart=true` to reset the environment.
    - Success: returns stdout+stderr (concatenated) preserving interleaving order.
      Failure: raises with exit code and combined output.
    - Timeout: default 120s; marks the session as requiring restart.
    - Output is truncated to MAX_OUTPUT_CHARS characters per command with a clear note.
    """

    desc = (
        "Execute shell commands in a persistent bash session (owned and used "
        "by only *you*, the agent caller).\n"
        "* Maintains state (cwd, env, vars) across calls until restarted.\n"
        "* Non-interactive commands only (no TTY prompts).\n"
        "* heredocs and multi-line commands supported.\n"
        "* Success returns merged stdout/stderr, preserving interleaved order.\n"
        "* On non-zero exit, raises exception with exit code and the stdout/stderr.\n"
        "* Timeout (default 120s) raises and marks the session as requiring restart.\n"
        f"* Output truncated to {BashSession.MAX_OUTPUT_CHARS} characters per call.\n"
        "* Avoid concurrent calls to the same session (use different `session_id`s if necessary).\n"
        "* Restarts are session_id specific and do not affect other sessions.\n"
        "* Avoid commands that may produce excessive/irrelevant output to prevent context pollution.\n"
    )

    args = [
        FunctionArg("command", str, "bash command to run (required unless `restart=true`).", optional=True),
        FunctionArg("restart", bool, "Set true to restart the session (default false).", optional=True),
        FunctionArg("session_id", int, "Optional session selector (default 0). Use small ints.", optional=True),
        FunctionArg("timeout_sec", int, "Optional per-command timeout seconds (default 120).", optional=True),
    ]

    def __init__(self):
        super().__init__(
            name="bash",
            desc=self.desc,
            args=self.args,
            callable=self._call,
            uses=[],
        )

    # For insignificant concurrent use of the same session_id, before complaining.
    LOCK_WAIT_SEC = 3.0

    @staticmethod
    def _bag_key(session_id: int) -> str:
        return f"session:{session_id}"

    def _call(
        self,
        ctx: RunContext,
        *,
        command: Optional[str] = None,
        restart: Optional[bool] = None,
        session_id: Optional[int] = None,
        timeout_sec: Optional[int] = None,
    ) -> str:
        if timeout_sec is not None and timeout_sec <= 0:
            raise BashException("`timeout_sec` must be a positive integer when provided.")
        sid = int(session_id) if session_id is not None else 0
        do_restart = bool(restart)

        # Bash utility must be invoked from within an Agent (Parent scope required).
        try:
            session: BashSession = ctx.get_or_put(
                SessionScope.Parent,
                namespace="bash.session",
                key=self._bag_key(sid),
                factory=lambda: BashSession(sid),
            )
        except NoParentSessionError as exc:
            raise BashException(
                "Bash must be invoked from within an agent (Parent SessionBag is required)."
            ) from exc

        # Acquire exclusive access for the entire operation (restart and/or execute).
        if not session.try_acquire(timeout_sec=self.LOCK_WAIT_SEC):
            raise BashSessionRaceException(
                "Bash session is busy with another command; you are not "
                "supposed to call the same session concurrently!")
        try:
            if do_restart:
                session.restart()
                # If a command is also provided, continue to execute it after restart.
                if command is None or command.strip() == "":
                    return f"tool has been restarted (session_id={sid})"

            if command is None or command.strip() == "":
                raise BashException("`command` is required unless `restart=true`.")

            combined, exit_code, _ = session.execute(command, timeout_sec=timeout_sec)
            if exit_code != 0:
                # Raise with rich detail per contract (agent layer will stringify this)
                raise BashNonZeroExitCodeException(exit_code, combined)
            return combined
        finally:
            session.release()


# Built-in global singleton for author reference.
bash = Bash()
