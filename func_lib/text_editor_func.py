from pathlib import Path
from multiprocessing import Lock
import tempfile
import os
import re
import secrets
from typing import Set, Optional

from ..core import FunctionArg, CodeFunction, RunContext, SessionScope

class TextEditorException(Exception):
    """Raised for business-logic violations in the TextEditor tool."""

class TextEditor(CodeFunction):
    """
    A text editor tool for reading and modifying files inspired by and aligned with Anthropic's
    `str_replace_based_edit_tool` , but implemented and consumable as an ordinary `CodeFunction`
    independent of any Provider, so it can be used by all agents as a standard utility.
    """
    name: str = "text_editor"
    commands: set[str] = {"view", "str_replace", "create", "insert"}

    # Truncate on each `view` command output.
    max_characters: int = 36000

    _FILE_LOCK_NAMESPACE: str = "text_editor.file_lock"

    desc = (
        "View, create, and edit text files.\n"
        "Similar to Anthropic's `str_replace_based_edit_tool`.\n"  # Beneficial (for all providers); do not remove.
        "Commands:\n"
        "• `view`: View a file or list a directory (single level). "
        "Files show line numbers (format: `number|content`) and you can optionally "
        "specify a line range. For directories, subdirectories are listed with '/' appended to their names.\n"
        "List dirs only sparingly to avoid context pollution.\n"
        "Be selective when viewing files to prevent context explosion; use line ranges to limit output.\n"
        "• `str_replace`: Replace exact text in a file (`old_str` → `new_str`). Line numbers are"
        " for display only; do not include them in `old_str`.\n"
        "• `create`: Create a new file with given text content (overwrites disallowed).\n"
        "• `insert`: Insert text after a given line in a file.\n\n"
        "Arguments by command:\n"
        "• `view` → `path` (absolute or relative; file or directory). Optional: `view_start_line`, `view_end_line`. "
        "Examples (files only): (view_start_line=1, view_end_line=1) shows only the first line; "
        "(view_start_line=10, view_end_line=20) shows lines 10 through 20; "
        "set view_end_line=-1 to read through the last line."
        f"Truncation will occur after {max_characters} characters, with it clearly noted in the returned text.\n"
        "• `str_replace` → `path`, `old_str`, `new_str` (requires *exact* match including whitespace/indentation).\n"
        "• `create` → `path` (must not pre-exist), `file_text`.\n"
        "• `insert` → `path`, `insert_line`, `new_str`. "
        "Examples: `insert_line=0` inserts at the beginning of the file; "
        "`insert_line=1` inserts *after* the very first line (starting on the second line); "
        "`insert_line=30` inserts *after* the first 30 lines.\n"
        "Absolute paths are recommended to avoid ambiguity.\n"
        "String output:\n"
        "• Success confirmation (create, insert, str_replace)\n"
        "• Content requested (view)\n"
        "• Exception string (with detail; to help debug)\n"
    )

    args = [
        FunctionArg("command", str, "Which of the editor commands to run.",
                    enum=commands),
        FunctionArg("path", str,
                    "Relative or Absolute path to the target file or directory (required by all commands)."),

        # Command-specific parameters (globally optional; validated per command at runtime):
        FunctionArg("old_str", str, "Text to replace when command=str_replace (must match exactly).",
                    optional=True),
        FunctionArg("new_str", str, "Replacement text for `str_replace` (can be empty str); or inserted text for `insert`.",
                    optional=True),
        FunctionArg("file_text", str, "Full file contents when command=create.",
                    optional=True),
        FunctionArg("insert_line", int,
                    "1-indexed line number *after* which to insert when command=insert."
                    "Use 0 to insert at the start of the file.",
                    optional=True),
        FunctionArg("view_start_line", int,
                    "When command=view on a file, optional 1-indexed start line (inclusive) for partial view.",
                    optional=True,
        ),
        FunctionArg("view_end_line", int,
                    "When command=view on a file, optional 1-indexed end line (inclusive) for partial view. "
                    "Use -1 to read to end of file.",
                    optional=True,
        ),
    ]

    def __init__(self):
        super().__init__(
            name=self.name,
            desc=self.desc,
            args=self.args,
            callable=self.call,
            uses=[],
        )

    def call(
        self,
        ctx: RunContext, *,
        command: str,
        path: str,
        old_str: Optional[str] = None,
        new_str: Optional[str] = None,
        file_text: Optional[str] = None,
        insert_line: Optional[int] = None,
        view_start_line: Optional[int] = None,
        view_end_line: Optional[int] = None,
    ) -> str:
        # Rely on strong framework contracts for `command` enum, arg types, optional args.
        # Framework does not handle command-specific arg optionality -- command handlers do this.
        # We can do common extraneous arg checking here.
        args_provided = {"path"} | {k for k, v in [
            ("old_str", old_str), ("new_str", new_str), ("file_text", file_text),
            ("insert_line", insert_line), ("view_start_line", view_start_line),
            ("view_end_line", view_end_line)
        ] if v is not None}
        self._check_extraneous_args(args_provided, command)

        if command == "view":
            return self.handle_view(path, view_start_line, view_end_line)
        if command == "str_replace":
            return self.handle_str_replace(ctx, path, old_str, new_str)
        if command == "create":
            return self.handle_create(path, file_text)
        if command == "insert":
            return self.handle_insert(ctx, path, insert_line, new_str)

        raise NotImplementedError(f"TextEditor needs impl update for `command`: '{command!r}'")

    @staticmethod
    def _file_lock_key(p: Path) -> str:
        # Ensure consistent keying across common casing + separator conventions.
        return os.path.normcase(str(p))

    def _get_file_lock(self, ctx: RunContext, p: Path):
        return ctx.get_or_put(
            SessionScope.TopLevel,
            namespace=self._FILE_LOCK_NAMESPACE,
            key=self._file_lock_key(p),
            factory=Lock,
        )

    def handle_view(
        self,
        path: str,
        view_start_line: Optional[int],
        view_end_line: Optional[int],
    ) -> str:
        p = self._resolve_path(path)
        if not p.exists():
            raise FileNotFoundError(f"Path not found: {p}")

        # Directory listing.
        if p.is_dir():
            if view_start_line is not None:
                raise TextEditorException("Arg `view_start_line` was provided but the path is a directory.")
            if view_end_line is not None:
                raise TextEditorException("Arg `view_end_line` was provided but the path is a directory.")

            try:
                entries = sorted(p.iterdir(), key=lambda q: (not q.is_dir(), q.name.lower()))
                entries = [e for e in entries if e.name not in (".", "..")]
                listing = "\n".join(e.name + ("/" if e.is_dir() else "") for e in entries)

                if self._needs_truncation(listing):
                    listing = self._truncate_with_inline_warning(listing)
                    notice = "[WARNING: DIRECTORY LISTING TRUNCATED.]"
                    if listing.endswith(("\r\n", "\n", "\r")):
                        listing += notice
                    else:
                        listing += "\n" + notice
                return listing

            except PermissionError as exc:
                raise PermissionError(f"while listing directory '{p}': {exc}") from exc
            except OSError as exc:
                raise OSError(f"while listing directory '{p}': {exc}") from exc

        # File view
        if not p.is_file():
            raise TextEditorException(f"Path is neither directory nor regular file: {p}")
        if view_end_line == 0:
            raise TextEditorException("view_end_line cannot be 0; use -1 for end-of-file")
        if view_start_line is not None and view_start_line < 1:
            raise TextEditorException(f"view_start_line must be >= 1 (received {view_start_line})")
        if view_end_line is not None and view_end_line < -1:
            raise TextEditorException(f"view_end_line must be -1 or >= 1 (received {view_end_line})")

        # No range + full file (subject to truncation)
        if view_start_line is None and view_end_line is None:
            # Stream for full-file views to avoid large allocations; preserve prior output semantics.
            snippet, _, was_truncated = self._read_from_start_stream(p, 1)
            if was_truncated:
                snippet = self._truncate_with_inline_warning(snippet)
            numbered = self._add_line_numbers(snippet)
            if was_truncated:
                numbered = self._append_post_numbering_truncation_notice(numbered)
            return numbered

        # Normalize range
        start = 1 if view_start_line is None else view_start_line
        end = -1 if view_end_line is None else view_end_line
        if end != -1 and start > end:
            raise TextEditorException(
                f"Invalid view range (view_start_line={start}, view_end_line={end}). "
                "Require view_start_line <= view_end_line or view_end_line = -1."
            )

        # Stream when end is bounded; otherwise stream to EOF from start.
        if end != -1:
            snippet, last_line, was_truncated = self._read_range_stream(p, start, end)
            # Validate against actual file length observed during streaming
            if last_line == 0:
                # Empty file
                if start == 1 and end == 1:
                    return ""
                raise TextEditorException("File is empty")
            if start > last_line:
                raise TextEditorException(f"view_start_line {start} exceeds file line count ({last_line})")
            if end > last_line:
                raise TextEditorException(f"view_end_line {end} exceeds file line count ({last_line})")
            if was_truncated:
                snippet = self._truncate_with_inline_warning(snippet)
            numbered = self._add_line_numbers(snippet, start_at=start)
            if was_truncated:
                numbered = self._append_post_numbering_truncation_notice(numbered)
            return numbered
        else:
            snippet, last_line, was_truncated = self._read_from_start_stream(p, start)
            if last_line == 0:
                # Empty file
                if start == 1:
                    return ""
                raise TextEditorException("File is empty")
            if start > last_line:
                raise TextEditorException(f"view_start_line {start} exceeds file line count ({last_line})")
            if was_truncated:
                snippet = self._truncate_with_inline_warning(snippet)
            numbered = self._add_line_numbers(snippet, start_at=start)
            if was_truncated:
                numbered = self._append_post_numbering_truncation_notice(numbered)
            return numbered

    def handle_str_replace(
        self,
        ctx: RunContext,
        path: str,
        old_str: Optional[str],
        new_str: Optional[str],
    ) -> str:
        if old_str is None:
            raise TextEditorException(
                "Missing argument: `old_str`")
        if old_str == "":
            raise TextEditorException(
                "`old_str` must be non-empty; received an empty string.")
        if new_str is None:
            raise TextEditorException(
                "Missing argument: `new_str`. "
                "If intention was to delete `old_str`, use empty string for `new_str`.")

        p = self._resolve_path(path)
        if not p.exists():
            raise FileNotFoundError(f"Path not found: {p}")
        if p.is_dir():
            raise IsADirectoryError(f"Path is a directory: {p}")
        if not p.is_file():
            raise TextEditorException(f"Path is not a regular file: {p}")

        lock = self._get_file_lock(ctx, p)
        lock.acquire()
        try:
            # Mid-air guard: record mtime before we read
            mtime_before = self._stat_mtime_ns(p)
            content = self._read_text_preserve_eols(p)

            count = self._count_overlapping(content, old_str)
            if count == 0:
                # See if there is a likely reason for no match that an LLM might make.
                extra = ""
                if self._looks_line_numbered(old_str):
                    extra = ". Ensure line numbers not included in `old_str`."
                else:
                    # Extra guidance when line endings are the likely culprit.
                    if old_str and old_str.replace("\r\n", "\n") in content.replace("\r\n", "\n"):
                        extra += ". Possible CRLF/LF newline mismatch."
                raise TextEditorException("No match found for replacement" + extra)
            if count > 1:
                raise TextEditorException(
                    f"Found {count} matches for replacement text. "
                    "Provide more context to make it unique.")

            updated = content.replace(old_str, new_str, 1)

            # Mid-air guard just before write
            mtime_now = self._stat_mtime_ns(p)
            if (mtime_before is not None and mtime_now is not None) and (mtime_now != mtime_before):
                # Generally we expect agents will be instructed to use temp files and workspaces
                # to prevent conflict with other processes. This only catches a small subset of races.
                raise TextEditorException("File changed during edit; refresh your view and try again")

            if updated != content:
                self._atomic_write_text(p, updated)
        finally:
            lock.release()
        return "Replace successful."

    def handle_create(
        self,
        path: str,
        file_text: Optional[str],
    ) -> str:
        if file_text is None:
            raise TextEditorException("Missing argument: file_text")

        p = self._resolve_path(path)
        if p.exists():
            if p.is_dir():
                raise IsADirectoryError(f"Path is a directory: {p}")
            self._raise_create_conflict(p, file_text)

        # Race-safe create:
        # Use exclusive create ('x') so if another process creates the file between our
        # existence check and the open, the OS raises FileExistsError and we don't overwrite.
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except PermissionError as exc:
            raise PermissionError(f"while creating parent directory for '{p}': {exc}") from exc
        except OSError as exc:
            raise OSError(f"while creating parent directory for '{p}': {exc}") from exc

        try:
            with p.open("x", encoding="utf-8", errors="surrogateescape", newline="") as f:
                f.write(file_text)
        except FileExistsError as exc:
            self._raise_create_conflict(p, file_text, original_exc=exc)
        except PermissionError as exc:
            raise PermissionError(f"while creating file '{p}': {exc}") from exc
        except OSError as exc:
            raise OSError(f"while creating file '{p}': {exc}") from exc

        # Count lines consistently with other operations (treat each newline-terminated chunk as a line)
        line_count = len(file_text.splitlines(keepends=True))
        return f"Successfully wrote {line_count} lines to new file: {p}."

    def handle_insert(
        self,
        ctx: RunContext,
        path: str,
        insert_line: Optional[int],
        new_str: Optional[str],
    ) -> str:
        if insert_line is None:
            raise TextEditorException("Missing argument: insert_line")
        if insert_line < 0:
            raise TextEditorException(f"insert_line must be >= 0 (received {insert_line})")
        if new_str is None:
            raise TextEditorException("Missing argument: new_str")

        p = self._resolve_path(path)
        if not p.exists():
            raise FileNotFoundError(f"Path not found: {p}")
        if p.is_dir():
            raise IsADirectoryError(f"Path is a directory: {p}")
        if not p.is_file():
            raise TextEditorException(f"Path is not a regular file: {p}")

        lock = self._get_file_lock(ctx, p)
        lock.acquire()
        try:
            # Mid-air guard: record mtime before we read
            mtime_before = self._stat_mtime_ns(p)
            content = self._read_text_preserve_eols(p)

            lines = content.splitlines(keepends=True)
            n = len(lines)
            if insert_line > n:
                raise TextEditorException(f"insert_line {insert_line} is out of range (file has {n} lines)")

            before = "".join(lines[:insert_line])
            after = "".join(lines[insert_line:])
            updated = before + new_str + after

            # Mid-air guard just before write
            mtime_now = self._stat_mtime_ns(p)
            if (mtime_before is not None and mtime_now is not None) and (mtime_now != mtime_before):
                raise TextEditorException("File changed on disk; refresh your view and try again")

            if updated != content:
                self._atomic_write_text(p, updated)
        finally:
            lock.release()
        return "Insert successful."

    def _raise_create_conflict(
        self,
        p: Path,
        file_text: str,
        original_exc: Optional[Exception] = None,
    ) -> None:
        """
        Handle the case where a create would overwrite an existing path.
        We first attempt an alternate sibling file and then raise a FileExistsError
        describing what happened.

        This design lets agents preserve user changes without forcing the LLM to
        re-send the full file contents as another tool argument when resolving
        path conflicts, which helps avoid unnecessary context bloat.
        """
        alt_path, alt_exc = self._attempt_alternate_create(p, file_text)
        line_count = len(file_text.splitlines(keepends=True))

        if alt_exc is None and alt_path is not None:
            err = FileExistsError(
                f"File already exists: {p}. "
                f"Successfully wrote {line_count} lines instead to alternate path '{alt_path}' "
                "because overwrite is not allowed for command=create. "
                "You should rename or otherwise resolve the conflict (for example by moving or "
                "deleting one of these paths using other tools)."
            )
            if original_exc is not None:
                raise err from original_exc
            raise err

        if alt_path is not None and alt_exc is not None:
            raise FileExistsError(
                f"File already exists: {p}. "
                f"Also failed to write contents to alternate path '{alt_path}' while trying to "
                f"preserve your changes: {type(alt_exc).__name__}: {alt_exc}"
            ) from alt_exc

        err = FileExistsError(f"File already exists: {p}")
        if original_exc is not None:
            raise err from original_exc
        raise err

    def _attempt_alternate_create(
        self,
        original_path: Path,
        file_text: str,
        max_attempts: int = 5,
    ) -> tuple[Optional[Path], Optional[Exception]]:
        """
        Attempt to create a sibling file with a random 6-hex-digit suffix when the
        original path already exists and we do not allow overwrite.

        Returns (alternate_path, exc):
          - (Path, None) on success.
          - (Path, exc) if we tried an alternate path but failed with an error.
          - (None, exc) is not expected in normal operation but included for completeness.
        """
        last_exc: Optional[Exception] = None
        alt_path: Optional[Path] = None

        for _ in range(max_attempts):
            stem = original_path.stem
            suffix = original_path.suffix
            rand = secrets.token_hex(3)
            name = f"{stem}.{rand}{suffix}" if stem else f".{rand}{suffix}"
            alt_path = original_path.with_name(name)

            try:
                with alt_path.open("x", encoding="utf-8", errors="surrogateescape", newline="") as f:
                    f.write(file_text)
                return alt_path, None
            except FileExistsError as exc:
                last_exc = exc
                continue
            except PermissionError as exc:
                return alt_path, PermissionError(f"while creating alternate file '{alt_path}': {exc}")
            except OSError as exc:
                return alt_path, OSError(f"while creating alternate file '{alt_path}': {exc}")

        return alt_path, last_exc

    def _check_extraneous_args(self, args_provided: Set[str], command: str) -> None:
        allowed_by_cmd: dict[str, set[str]] = {
            "view": {"path", "view_start_line", "view_end_line"},
            "str_replace": {"path", "old_str", "new_str"},
            "create": {"path", "file_text"},
            "insert": {"path", "insert_line", "new_str"},
        }
        allowed = allowed_by_cmd.get(command, set())
        extraneous = args_provided - allowed
        if extraneous:
            raise TextEditorException(f"Extraneous arguments for command '{command}': {extraneous}")

    def _resolve_path(self, path: str) -> Path:
        if path is None or not str(path).strip():
            raise TextEditorException("Argument `path` is empty after stripping whitespace.")
        try:
            return Path(path).expanduser().resolve()
        except Exception as e:
            raise TextEditorException(f"Path fails to resolve to absolute path: {path!r}: {e}")

    def _needs_truncation(self, s: str) -> bool:
        """Return True when `s` exceeds the max character allowance."""
        lim = self.max_characters
        if lim is None:
            return False
        return lim > 0 and len(s) > lim

    def _truncate_with_inline_warning(self, s: str) -> str:
        """
        Truncate `s` at the limit and append the inline truncation warning on the last line.

        Example before numbering:
            "    if x < calcu[WARNING: OUTPUT TRUNCATED ABRUPTLY AFTER ...]"

        Call only when truncation happened (or is about to happen) so the warning suffix is applied
        exactly once. The function defensively trims to the limit in case the caller has not already
        done so.
        """
        lim = self.max_characters
        if lim is None or lim <= 0:
            return s
        warning = f"[WARNING: OUTPUT TRUNCATED ABRUPTLY AFTER {lim} CHARACTERS]"
        truncated = s[:lim] if self._needs_truncation(s) else s
        trailing_break = ""
        if truncated.endswith("\r\n"):
            truncated = truncated[:-2]
            trailing_break = "\r\n"
        elif truncated.endswith(("\n", "\r")):
            trailing_break = truncated[-1]
            truncated = truncated[:-1]
        return f"{truncated}{warning}{trailing_break}"

    def _append_post_numbering_truncation_notice(self, s: str) -> str:
        """
        Add the terminal warning once line numbers have been applied.
        There is good reason for this redundancy:
        - Make sure truncation within a line is clearly noted.
          Otherwise, the model may think that line was complete.
        - Make sure the first truncation warning is not interpreted as part of the line.
          That's why we need the second line.

        Example final output:
            34|    if x < calcu[WARNING: OUTPUT TRUNCATED ABRUPTLY AFTER ...]
            [WARNING: OUTPUT TRUNCATED. LINE ABOVE IS INCOMPLETE. RE-SCOPE VIEW RANGE IF NEEDED.]
        """
        notice = "[WARNING: OUTPUT TRUNCATED. LINE ABOVE IS INCOMPLETE. RE-SCOPE VIEW RANGE IF NEEDED.]"
        if s.endswith(("\r\n", "\n", "\r")):
            return s + notice
        return s + "\n" + notice

    def _add_line_numbers(self, s: str, *, start_at: int = 1) -> str:
        """
        Presentation-only: add 1-indexed line numbers using the format
        "{line_number}|{line_text}" with no padding and a pipe separator. `start_at`
        allows absolute numbering when viewing partial ranges.

        Applied strictly after truncation logic and after any warning injection.
        """
        if not s:
            return s
        # Preserve original EOLs by keeping ends when splitting.
        parts = s.splitlines(keepends=True)
        return "".join(f"{i}|{line}" for i, line in enumerate(parts, start=start_at))

    def _stat_mtime_ns(self, p: Path) -> Optional[int]:
        """
        Best-effort nanosecond mtime for mid-air collision detection.
        Returns None only when the platform does not provide a usable timestamp.
        """
        try:
            st = p.stat()
        except NotImplementedError:
            return None
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"while checking modification time for '{p}': {exc}") from exc
        except PermissionError as exc:
            raise PermissionError(f"while checking modification time for '{p}': {exc}") from exc
        except OSError as exc:
            raise OSError(f"while checking modification time for '{p}': {exc}") from exc

        mtime_ns = getattr(st, "st_mtime_ns", None)
        if mtime_ns is not None:
            return int(mtime_ns)

        mtime = getattr(st, "st_mtime", None)
        if mtime is None:
            return None

        try:
            return int(mtime * 1_000_000_000)
        except (OverflowError, TypeError, ValueError):
            return None

    def _read_text_preserve_eols(self, p: Path) -> str:
        """
        Read text while preserving exact newline characters and unknown bytes.
        - encoding='utf-8', errors='surrogateescape' keeps undecodable bytes round-trippable
          (decoded into U+DC80..U+DCFF) and writes them back as the original bytes.
        - newline='' preserves CRLF vs LF exactly.
        """
        if not p.exists():
            raise FileNotFoundError(f"Path not found: {p}")
        if not p.is_file():
            raise TextEditorException(f"Path is not a file: {p}")
        try:
            with p.open("r", encoding="utf-8", errors="surrogateescape", newline="") as f:
                return f.read()
        except PermissionError as exc:
            raise PermissionError(f"while reading '{p}': {exc}") from exc
        except OSError as exc:
            raise OSError(f"while reading '{p}': {exc}") from exc

    def _atomic_write_text(self, p: Path, data: str) -> None:
        """
        Atomic write:
          1) create temp file in same directory
          2) write with utf-8 + surrogateescape, fsync
          3) set permissions:
             - existing file: preserve original mode
             - new file: apply 0666 & ~umask (typical default; commonly 0644)
          4) os.replace to destination
          5) fsync parent directory (POSIX durability)
        """
        parent = p.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except PermissionError as exc:
            raise PermissionError(f"while preparing parent directory for '{p}': {exc}") from exc
        except OSError as exc:
            raise OSError(f"while creating parent directory for '{p}': {exc}") from exc

        tmp_fd = None
        tmp_path = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(prefix=f".{p.name}.", dir=str(parent))
            with os.fdopen(tmp_fd, "w", encoding="utf-8", errors="surrogateescape", newline="") as tmpf:
                tmp_fd = None  # fd passed to file object
                tmpf.write(data)
                tmpf.flush()
                os.fsync(tmpf.fileno())

            # Permissions: preserve when overwriting; otherwise use 0666 & ~umask for new files
            try:
                st = p.stat()
                os.chmod(tmp_path, st.st_mode)  # existing file: preserve
            except FileNotFoundError:
                # New file: set default permissions like 0666 masked by current umask
                um = os.umask(0)
                os.umask(um)
                os.chmod(tmp_path, 0o666 & ~um)

            os.replace(tmp_path, p)  # atomic on same volume
            tmp_path = None

            # Make the rename durable on POSIX (best-effort cross-platform)
            try:
                dir_fd = os.open(str(parent), getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except Exception:
                # Not supported (e.g., Windows) or restricted FS; best-effort only.
                pass
        except PermissionError as exc:
            raise PermissionError(f"while writing to '{p}': {exc}") from exc
        except OSError as exc:
            raise OSError(f"while writing to '{p}': {exc}") from exc
        finally:
            if tmp_fd is not None:
                try:
                    os.close(tmp_fd)
                except Exception:
                    pass
            if tmp_path is not None and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _count_overlapping(self, hay: str, needle: str) -> int:
        """
        Count occurrences allowing overlaps (e.g., 'ababa' find 'aba' => 2).
        Uses str.find in a loop; efficient in CPython (C-level scanning).
        """
        if not needle:
            return 0
        i = 0
        c = 0
        while True:
            i = hay.find(needle, i)
            if i == -1:
                break
            c += 1
            i += 1  # allow overlap
        return c

    def _looks_line_numbered(self, text: str) -> bool:
        """Heuristic: detect if `text` appears to include line numbers from the view output."""
        return bool(TextEditor._LINE_NUMBER_PATTERN.search(text))

    def _read_range_stream(self, p: Path, start: int, end_inclusive: int) -> tuple[str, int, bool]:
        """
        Stream only the requested 1-indexed inclusive [start, end_inclusive] lines.
        Returns (snippet, last_line_seen, truncated_by_max_characters).

        We enforce self.max_characters during streaming to avoid large allocations, but
        continue reading (without appending) until end_inclusive to validate bounds.
        """
        limit = getattr(self, "max_characters", None)
        limit = None if (limit is None or limit < 0) else int(limit)

        out_parts: list[str] = []
        out_len = 0
        was_truncated = False
        last_line = 0

        try:
            with p.open("r", encoding="utf-8", errors="surrogateescape", newline="") as f:
                for i, line in enumerate(f, 1):
                    last_line = i
                    if i < start:
                        continue
                    if not was_truncated:
                        if limit is None:
                            out_parts.append(line)
                        else:
                            need = limit - out_len
                            if need > 0:
                                seg = line[:need]
                                out_parts.append(seg)
                                out_len += len(seg)
                                if len(line) > need:
                                    was_truncated = True
                            else:
                                was_truncated = True
                    if i >= end_inclusive:
                        break
        except PermissionError as exc:
            raise PermissionError(f"while reading '{p}': {exc}") from exc
        except OSError as exc:
            raise OSError(f"while reading '{p}': {exc}") from exc

        return "".join(out_parts), last_line, was_truncated

    def _read_from_start_stream(self, p: Path, start: int) -> tuple[str, int, bool]:
        """
        Stream from 1-indexed line `start` to EOF.
        Returns (snippet, last_line_seen, truncated_by_max_characters).
        """
        limit = getattr(self, "max_characters", None)
        limit = None if (limit is None or limit < 0) else int(limit)

        out_parts: list[str] = []
        out_len = 0
        was_truncated = False
        last_line = 0

        try:
            with p.open("r", encoding="utf-8", errors="surrogateescape", newline="") as f:
                for i, line in enumerate(f, 1):
                    last_line = i
                    if i < start:
                        continue
                    if not was_truncated:
                        if limit is None:
                            out_parts.append(line)
                        else:
                            need = limit - out_len
                            if need > 0:
                                seg = line[:need]
                                out_parts.append(seg)
                                out_len += len(seg)
                                if len(line) > need:
                                    was_truncated = True
                            else:
                                was_truncated = True
                # EOF reached
        except PermissionError as exc:
            raise PermissionError(f"while reading '{p}': {exc}") from exc
        except OSError as exc:
            raise OSError(f"while reading '{p}': {exc}") from exc

        return "".join(out_parts), last_line, was_truncated

    # Heuristic for common numbering styles in the `view` output that LLM may
    # accidentally include in follow-up `old_str`: "1|", "1│", "1:", or "1\t" (cat -n).
    # We only use this to sometimes craft better error messages when no match found.
    _LINE_NUMBER_PATTERN = re.compile(r"(?:^|\n)\s*\d+\s*(?:[|│:]\s?|\t)")


# Built-in global singleton for author reference.
text_editor = TextEditor()
