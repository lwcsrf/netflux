import io
import codecs
import os
import re
import queue
import shutil
import shlex
import signal
import subprocess
import tempfile
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
    READ_CHUNK_BYTES = 4096
    PENDING_TAIL_CHARS = 8192

    def __init__(self, session_id: int) -> None:
        self.session_id = session_id
        self._proc: Optional[subprocess.Popen[bytes]] = None
        self._q: queue.Queue[str] = queue.Queue()
        self._stdout_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()         # serialize commands per session
        self.requires_restart: bool = False   # set true on timeout, etc.
        self._alive_once_started: bool = False

    @staticmethod
    def _find_bash() -> str:
        """Locate a usable bash executable, or raise."""
        if os.path.isfile("/bin/bash"):
            return "/bin/bash"
        if os.name == "nt":
            for path in _windows_bash_candidates():
                if _bash_works(path):
                    return path
        else:
            found = shutil.which("bash")
            if found:
                return found
        raise BashException(
            "Cannot locate a bash executable. "
            "On Windows, install Git for Windows (includes Git Bash) and ensure it is on PATH."
        )

    def start(self) -> None:
        if self._proc and self._proc.poll() is None:
            return

        # Ensure clean slate.
        self._terminate_group_if_alive()

        bash_path = self._find_bash()
        preexec = os.setsid if os.name == "posix" else None
        # On Windows, create a new process group so we can cleanly terminate the tree.
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0

        self._proc = subprocess.Popen(  # noqa: S603, S607 (we intentionally run bash)
            [bash_path, "--noprofile", "--norc", "-s"],  # read from stdin; skip user rc/profile
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=-1,
            preexec_fn=preexec,
            creationflags=creationflags,
            env=_sanitized_bash_env(),
        )
        self._alive_once_started = True
        self.requires_restart = False
        self._start_readers()

        # Set pipeline semantics, disable history, quiet PS1, and block until ready.
        # Save the original stdout pipe to fd 254 for resilient sentinel delivery in execute(),
        # and prove that writes through fd 254 actually work before declaring the session ready.
        self._write(
            "set -o pipefail; export HISTFILE=/dev/null; export PS1=''; "
            "shopt -s expand_aliases; exec 254>&1; "
            "builtin printf '__NETFLUX_BASH_READY__\\n' >&254\n"
        )
        # Require readiness; if not reached, mark for restart and fail.
        if not self._drain_until("__NETFLUX_BASH_READY__", timeout=3):
            self.requires_restart = True
            self._terminate_group_if_alive()
            raise BashSessionCrashedException(
                "Bash session failed to initialize fd 254 or failed to reach ready state; "
                "tool must be restarted. If this repeats, this bash function is probably not going "
                "to work on this host environment."
            )

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
        proc = self._proc
        if not proc:
            return
        try:
            if proc.poll() is None:
                if os.name == "posix":
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    except Exception:
                        # Fall back to process terminate if group kill fails
                        proc.terminate()
                else:
                    ctrl_break = getattr(signal, "CTRL_BREAK_EVENT", None)
                    if ctrl_break is not None:
                        try:
                            proc.send_signal(ctrl_break)
                        except Exception:
                            ctrl_break = None
                    if ctrl_break is None:
                        try:
                            subprocess.run(
                                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                check=False,
                                timeout=5,
                            )
                        except Exception:
                            proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    if os.name == "posix":
                        try:
                            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                        except Exception:
                            pass
                    else:
                        try:
                            subprocess.run(
                                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                check=False,
                                timeout=5,
                            )
                        except Exception:
                            proc.kill()
        finally:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                if stream is None:
                    continue
                try:
                    stream.close()
                except Exception:
                    pass
            self._proc = None

    def _start_readers(self) -> None:
        assert self._proc and self._proc.stdout
        def pump(pipe, label: str):
            decoder = io.IncrementalNewlineDecoder(
                codecs.getincrementaldecoder("utf-8")("replace"),
                translate=True,
            )
            try:
                while True:
                    chunk = pipe.read1(self.READ_CHUNK_BYTES)
                    if not chunk:
                        tail = decoder.decode(b"", final=True)
                        if tail:
                            self._q.put(tail)
                        return
                    text = decoder.decode(chunk)
                    if text:
                        self._q.put(text)
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
        self._proc.stdin.write(s.encode("utf-8", errors="replace"))
        self._proc.stdin.flush()

    def _drain_queue_nowait(self) -> None:
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            return

    @staticmethod
    def _command_file_path(path: str) -> str:
        """Convert a host path to a form the bash process can source.

        On Windows, translates native paths (e.g. C:\\tmp\\foo.sh) to the
        MSYS/Git-Bash mount format (/c/tmp/foo.sh) that bash understands.
        No python stdlib exists for this conversion.
        """
        if os.name == "nt" and len(path) >= 2 and path[1] == ":":
            drive = path[0].lower()
            rest = path[2:].replace("\\", "/")
            return f"/{drive}{rest}"
        return path

    @staticmethod
    def _normalize_command_output(text: str, *paths: str) -> str:
        """Scrub internal temp-file paths from command output before surfacing it.
        Bash errors may reference the temp .sh file we staged the command in;
        replacing those paths with ``<command>`` keeps the output clean.

        Args:
            text:   Raw stdout+stderr from the bash process.
            *paths: Temp-file paths used to stage the command (native + MSYS forms).

        Each path is expanded into backslash and forward-slash variants, then
        all occurrences are replaced with the literal ``<command>`` (longest-first
        to avoid partial matches).
        """
        if not text:
            return text
        variants = set()
        for path in paths:
            if not path:
                continue
            variants.add(path)
            variants.add(path.replace("\\", "/"))
        for variant in sorted(variants, key=len, reverse=True):
            text = text.replace(variant, "<command>")
        return text

    @staticmethod
    def _write_command_file(command: str) -> Tuple[str, str]:
        fd, path = tempfile.mkstemp(prefix="netflux-bash-", suffix=".sh")
        try:
            payload = command if command.endswith("\n") else command + "\n"
            with os.fdopen(fd, "wb") as f:
                f.write(payload.encode("utf-8", errors="replace"))
        except Exception:
            try:
                os.close(fd)
            except Exception:
                pass
            _silent_unlink(path)
            raise
        return path, BashSession._command_file_path(path)

    @staticmethod
    def _timeout_hint_for_command(command: str) -> str:
        # Keep this as a post-timeout heuristic only: a raw text scan is too imprecise
        # to block execution because these strings may legitimately appear inside heredocs,
        # generated script content, comments, or commands run in a child bash process. If found,
        # these may be the cause of the timeout, so we give the agent a fighting chance to try again.
        if "set -n" not in command and "set -o noexec" not in command:
            return ""
        return (
            "\nEnsure you avoid 'set -n' / 'set -o noexec' in commands run in the session "
            "(okay in scripts run via a separate bash process)."
        )

    def _drain_until(self, token: str, timeout: float) -> bool:
        end = time.time() + timeout
        pending = ""
        keep = max(len(token) * 2, 256)
        while time.time() < end:
            try:
                pending += self._q.get(timeout=0.05)
                if token in pending:
                    return True
                if len(pending) > keep:
                    pending = pending[-keep:]
            except queue.Empty:
                pass
        return False

    def _read_until_sentinel(
        self,
        sentinel: str,
        timeout_sec: Optional[int],
        *,
        normalize_paths: Tuple[str, ...] = (),
        timeout_hint: str = "",
    ) -> Tuple[str, int]:
        # Read until we see the sentinel; preserve arrival order (stdout+stderr interleaving).
        effective_timeout = self.DEFAULT_TIMEOUT_SEC if (timeout_sec is None) else float(timeout_sec)
        deadline = time.time() + effective_timeout
        chunks: List[str] = []
        collected = 0
        truncated = False
        overflow_newlines = 0
        exit_code: Optional[int] = None
        found = False
        pending = ""

        def append_output(text: str) -> None:
            nonlocal collected, truncated, overflow_newlines
            if not text:
                return
            if collected >= self.MAX_OUTPUT_CHARS:
                if text.strip("\n"):
                    truncated = True
                else:
                    overflow_newlines += len(text)
                return
            space = self.MAX_OUTPUT_CHARS - collected
            overflow = text[space:]
            if overflow:
                if overflow.strip("\n"):
                    truncated = True
                else:
                    overflow_newlines += len(overflow)
            slice_text = text if len(text) <= space else text[:space]
            chunks.append(slice_text)
            collected += len(slice_text)

        while time.time() < deadline:
            try:
                pending += self._q.get(timeout=0.05)
            except queue.Empty:
                # Also detect process death while waiting
                if self._proc and self._proc.poll() is not None:
                    self.requires_restart = True
                    append_output(pending)
                    partial = self._normalize_command_output("".join(chunks), *normalize_paths)
                    if truncated or overflow_newlines > 0:
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

            while True:
                nl = pending.find("\n")
                if nl < 0:
                    break
                line = pending[:nl + 1]
                pending = pending[nl + 1:]
                # Only treat an exact sentinel line as command completion. This avoids
                # confusing xtrace/debug output that may contain the sentinel text.
                m = re.fullmatch(rf"{re.escape(sentinel)}\s+(-?\d+)", line.rstrip("\r\n"))
                if m:
                    exit_code = int(m.group(1))
                    found = True
                    # Do not append the sentinel line to output.
                    break
                append_output(line)

            if found:
                break

            if len(pending) > self.PENDING_TAIL_CHARS:
                append_output(pending[:-self.PENDING_TAIL_CHARS])
                pending = pending[-self.PENDING_TAIL_CHARS:]

        if not found:
            # Timed out; mark session as requiring restart and stop activity.
            self.requires_restart = True
            self._terminate_group_if_alive()
            append_output(pending)
            partial = self._normalize_command_output("".join(chunks), *normalize_paths)
            if truncated or overflow_newlines > 0:
                partial += (
                    f"\n\n... Output truncated (showing {self.MAX_OUTPUT_CHARS} characters; "
                    f"limit {self.MAX_OUTPUT_CHARS}) ..."
                )
            raise BashCommandTimeoutException(
                f"Command timed out after {int(effective_timeout)} seconds. "
                f"The session has been marked as requiring restart. "
                f"Invoke the tool again with restart=true to continue."
                f"{timeout_hint}\n\n"
                f"--- partial output ---\n"
                f"{partial}"
            )

        combined = self._normalize_command_output("".join(chunks), *normalize_paths)
        if truncated or overflow_newlines > 1:
            combined += (
                f"\n\n... Output truncated (showing {self.MAX_OUTPUT_CHARS} characters; "
                f"limit {self.MAX_OUTPUT_CHARS}) ..."
            )

        if exit_code is None:
            self.requires_restart = True
            self._terminate_group_if_alive()
            raise BashSessionCrashedException(
                "Malformed sentinel line; tool must be restarted."
            )
        return combined, exit_code

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
                raise BashSessionCrashedException("Bash session has crashed; tool must be restarted.")
            self.start()

        self._drain_queue_nowait()

        token = uuid.uuid4().hex
        sentinel = f"__NETFLUX_BASH_DONE__{token}"
        was_e_var = f"__nf_was_e_{token}"
        ec_var = f"__nf_ec_{token}"
        timeout_hint = self._timeout_hint_for_command(command)
        try:
            script_path, bash_script_path = self._write_command_file(command)
        except Exception as e:
            raise BashException(f"Failed to stage bash command in a temp file: {e}") from e

        block = (
            f"{was_e_var}=false; case $- in *e*) {was_e_var}=true;; esac\n"
            "builtin set +e\n"
            f"builtin source {shlex.quote(bash_script_path)} </dev/null\n"
            f"{ec_var}=$?\n"
            "if [[ \"${" + was_e_var + ":-}\" == true ]]; then builtin set -e; fi\n"
            "builtin printf '\\n" + sentinel + " %d\\n' \"${" + ec_var + "}\" >&254\n"
            "builtin unset -v " + was_e_var + " " + ec_var + " 2>/dev/null || builtin true\n"
        )
        try:
            self._write(block)
            combined, exit_code = self._read_until_sentinel(
                sentinel,
                timeout_sec,
                normalize_paths=(script_path, bash_script_path),
                timeout_hint=timeout_hint,
            )
            return combined, exit_code, sentinel
        except Exception as e:
            if isinstance(e, BashException):
                raise
            self.requires_restart = True
            raise BashSessionCrashedException(
                f"Bash stdin is not writable (session exited?): {e}. Tool must be restarted."
            ) from e
        finally:
            _silent_unlink(script_path)


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
        "* Syntax errors are normal command failures and may leave earlier lines' side effects in place.\n"
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


def _sanitized_bash_env() -> dict[str, str]:
    """Strip inherited env vars that bash would use to inject startup behaviour.

    The child bash process inherits the parent environment (in which the agent
    harness is being run), which is mostly desirable (PATH, HOME, etc.).
    However, a few inherited variables let the parent silently alter the
    child's startup, so we remove them:
      - BASH_ENV / ENV      — sourced automatically by bash/sh on startup.
      - SHELLOPTS / BASHOPTS — force shell options onto the child process.
      - BASH_FUNC_*          — exported functions that could shadow commands.
    """
    env = dict(os.environ)
    blocked_exact = {"BASH_ENV", "ENV", "SHELLOPTS", "BASHOPTS"}

    for key in list(env):
        probe = key.upper() if os.name == "nt" else key
        if probe in blocked_exact or probe.startswith("BASH_FUNC_"):
            env.pop(key, None)

    return env


def _bash_works(path: Optional[str]) -> bool:
    if not path or not os.path.isfile(path):
        return False
    low = os.path.normcase(path)
    if os.name == "nt" and ("windowsapps" in low or low.endswith("\\system32\\bash.exe")):
        return False
    try:
        proc = subprocess.run(
            [path, "--noprofile", "--norc", "-lc", "printf ok"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=2,
            env=_sanitized_bash_env(),
        )
    except Exception:
        return False
    return proc.returncode == 0 and proc.stdout == "ok"


def _windows_bash_candidates() -> List[str]:
    seen, paths = set(), []
    for base in filter(None, os.environ.get("PATH", "").split(os.pathsep)):
        candidate = os.path.join(base, "bash.exe")
        key = os.path.normcase(os.path.normpath(candidate))
        if key not in seen:
            seen.add(key)
            paths.append(candidate)
    for root in filter(None, [os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")]):
        for suffix in (("Git", "bin", "bash.exe"), ("Git", "usr", "bin", "bash.exe")):
            candidate = os.path.join(root, *suffix)
            key = os.path.normcase(os.path.normpath(candidate))
            if key not in seen:
                seen.add(key)
                paths.append(candidate)
    return paths


def _silent_unlink(path: str) -> None:
    """Remove *path* ignoring any error (file missing, permissions, etc.)."""
    try:
        os.unlink(path)
    except Exception:
        pass
