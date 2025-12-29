import tempfile
import threading
import unittest
from pathlib import Path
from typing import Optional
from unittest.mock import patch

from ..core import RunContext, SessionBag, SessionScope
from ..func_lib.text_editor import TextEditor, TextEditorException


class _DummyNode:
    def __init__(self, parent: Optional["_DummyNode"] = None) -> None:
        self.parent = parent
        self.session_bag = SessionBag()


class TestTextEditorBasicOperations(unittest.TestCase):
    def setUp(self) -> None:
        node = _DummyNode()
        self.ctx = RunContext(runtime=None, node=node)  # type: ignore[arg-type]
        self.ctx.object_bags = {
            SessionScope.TopLevel: node.session_bag,
            SessionScope.Self: node.session_bag,
        }
        self.editor = TextEditor()

    def test_view_whole_file_numbers_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            target = base_dir / "file.txt"
            self.editor.call(
                self.ctx,
                command="create",
                path=str(target),
                file_text="line1\nline2\n",
            )

            output = self.editor.call(
                self.ctx,
                command="view",
                path=str(target),
            )

            self.assertIn("1|line1\n", output)
            self.assertIn("2|line2\n", output)
            self.assertNotIn("OUTPUT TRUNCATED", output)

    def test_str_replace_single_match_updates_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            target = base_dir / "file.txt"
            self.editor.call(
                self.ctx,
                command="create",
                path=str(target),
                file_text="a=1\nb=2\n",
            )

            result = self.editor.call(
                self.ctx,
                command="str_replace",
                path=str(target),
                old_str="b=2\n",
                new_str="b=3\n",
            )

            self.assertEqual("Replace successful.", result)
            self.assertEqual("a=1\nb=3\n", target.read_text(encoding="utf-8"))

    def test_str_replace_old_str_must_be_nonempty(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            target = base_dir / "file.txt"
            self.editor.call(
                self.ctx,
                command="create",
                path=str(target),
                file_text="content\n",
            )

            with self.assertRaises(TextEditorException) as cm:
                self.editor.call(
                    self.ctx,
                    command="str_replace",
                    path=str(target),
                    old_str="",
                    new_str="x",
                )

            msg = str(cm.exception)
            self.assertIn("`old_str` must be non-empty", msg)

    def test_insert_at_beginning_and_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            target = base_dir / "file.txt"
            self.editor.call(
                self.ctx,
                command="create",
                path=str(target),
                file_text="line1\nline2\n",
            )

            # Insert at the beginning (insert_line=0)
            self.editor.call(
                self.ctx,
                command="insert",
                path=str(target),
                insert_line=0,
                new_str="start\n",
            )
            self.assertEqual(
                "start\nline1\nline2\n",
                target.read_text(encoding="utf-8"),
            )

            # Insert at end (insert_line equals current line count)
            self.editor.call(
                self.ctx,
                command="insert",
                path=str(target),
                insert_line=3,
                new_str="end\n",
            )
            self.assertEqual(
                "start\nline1\nline2\nend\n",
                target.read_text(encoding="utf-8"),
            )

    def test_insert_line_out_of_range_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            target = base_dir / "file.txt"
            self.editor.call(
                self.ctx,
                command="create",
                path=str(target),
                file_text="line1\nline2\n",
            )

            with self.assertRaises(TextEditorException) as cm:
                self.editor.call(
                    self.ctx,
                    command="insert",
                    path=str(target),
                    insert_line=5,
                    new_str="oops\n",
                )

            msg = str(cm.exception)
            self.assertIn("insert_line 5 is out of range", msg)

    def test_empty_path_rejected(self) -> None:
        with self.assertRaises(TextEditorException) as cm:
            self.editor.call(
                self.ctx,
                command="view",
                path="   ",
            )

        msg = str(cm.exception)
        self.assertIn("Argument `path` is empty after stripping whitespace.", msg)

    def test_view_end_line_zero_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            target = base_dir / "file.txt"
            self.editor.call(
                self.ctx,
                command="create",
                path=str(target),
                file_text="line1\n",
            )

            with self.assertRaises(TextEditorException) as cm:
                self.editor.call(
                    self.ctx,
                    command="view",
                    path=str(target),
                    view_start_line=1,
                    view_end_line=0,
                )

        msg = str(cm.exception)
        self.assertIn("view_end_line cannot be 0", msg)

    def test_extraneous_argument_for_view(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            target = base_dir / "file.txt"
            self.editor.call(
                self.ctx,
                command="create",
                path=str(target),
                file_text="line1\n",
            )

            with self.assertRaises(TextEditorException) as cm:
                self.editor.call(
                    self.ctx,
                    command="view",
                    path=str(target),
                    file_text="ignored",
                )

            msg = str(cm.exception)
            self.assertIn("Extraneous arguments for command 'view'", msg)
            self.assertIn("file_text", msg)


class TestTextEditorCreateBehavior(unittest.TestCase):
    def setUp(self) -> None:
        node = _DummyNode()
        self.ctx = RunContext(runtime=None, node=node)  # type: ignore[arg-type]
        self.ctx.object_bags = {
            SessionScope.TopLevel: node.session_bag,
            SessionScope.Self: node.session_bag,
        }
        self.editor = TextEditor()

    def test_create_on_new_file_succeeds_without_alternate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            target = base_dir / "new_file.txt"

            before_paths = set(base_dir.iterdir())

            result = self.editor.call(
                self.ctx,
                command="create",
                path=str(target),
                file_text="hello\nworld\n",
            )

            after_paths = set(base_dir.iterdir())
            new_paths = after_paths - before_paths

            # Only the requested file should have been created.
            self.assertEqual({target}, new_paths)
            self.assertTrue(target.is_file())
            self.assertEqual(target.read_text(encoding="utf-8"), "hello\nworld\n")

            # Success message should include line count and path.
            self.assertIn("Successfully wrote 2 lines", result)
            self.assertIn(str(target), result)

    def test_create_on_existing_directory_does_not_attempt_alternate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            # Use a nested directory as the create target.
            target_dir = base_dir / "subdir"
            target_dir.mkdir()

            before_paths = set(base_dir.iterdir())

            with self.assertRaises(IsADirectoryError):
                self.editor.call(
                    self.ctx,
                    command="create",
                    path=str(target_dir),
                    file_text="contents",
                )

            after_paths = set(base_dir.iterdir())

            # No additional siblings (alternate files) should have been created.
            self.assertEqual(before_paths, after_paths)

    def test_create_on_existing_file_uses_expected_suffix_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)

            # Cover a few filename shapes to lock in suffix behavior.
            cases = [
                "plain",
                "file.txt",
                ".env",
                "archive.tar.gz",
            ]

            for name in cases:
                target = base_dir / name
                target.write_text("original", encoding="utf-8")

                # Provide deterministic suffix for reproducibility.
                def _fake_token_hex(n: int) -> str:
                    self.assertEqual(3, n)
                    return "deadbe"

                with patch("netflux.func_lib.text_editor.secrets.token_hex", _fake_token_hex):
                    before_paths = set(base_dir.iterdir())
                    with self.assertRaises(FileExistsError) as cm:
                        self.editor.call(
                            self.ctx,
                            command="create",
                            path=str(target),
                            file_text="new text",
                        )

                    msg = str(cm.exception)

                after_paths = set(base_dir.iterdir())
                new_paths = after_paths - before_paths

                # Exactly one new sibling should have been created.
                self.assertEqual(1, len(new_paths), (name, new_paths))
                alt_path = next(iter(new_paths))

                # Validate naming scheme based on original stem/suffix.
                stem = target.stem
                suffix = target.suffix
                expected_name = f"{stem}.deadbe{suffix}" if stem else f".deadbe{suffix}"
                self.assertEqual(expected_name, alt_path.name)

                # Error message should mention both original and alternate path.
                self.assertIn(str(target), msg)
                self.assertIn(str(alt_path), msg)

                # Alternate file should contain the new contents; original unchanged.
                self.assertTrue(alt_path.is_file())
                self.assertEqual("new text", alt_path.read_text(encoding="utf-8"))
                self.assertEqual("original", target.read_text(encoding="utf-8"))

    def test_create_on_existing_file_writes_to_alternate_and_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            target = base_dir / "existing.txt"
            target.write_text("original contents", encoding="utf-8")

            before_paths = set(base_dir.iterdir())

            with self.assertRaises(FileExistsError) as cm:
                self.editor.call(
                    self.ctx,
                    command="create",
                    path=str(target),
                    file_text="new contents",
                )

            msg = str(cm.exception)
            self.assertIn("File already exists:", msg)
            self.assertIn(str(target), msg)

            after_paths = set(base_dir.iterdir())
            new_paths = after_paths - before_paths

            # Expect exactly one new sibling file to have been created.
            self.assertEqual(len(new_paths), 1, new_paths)
            alt_path: Path = next(iter(new_paths))

            # The alternate path should be mentioned in the error message and contain the new contents.
            self.assertIn(str(alt_path), msg)
            self.assertTrue(alt_path.is_file())
            self.assertEqual(alt_path.read_text(encoding="utf-8"), "new contents")

            # Original file remains unchanged.
            self.assertEqual(target.read_text(encoding="utf-8"), "original contents")

    def test_create_on_existing_file_alternate_failure_mentions_both(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            target = base_dir / "existing.txt"
            target.write_text("original contents", encoding="utf-8")

            # Force a deterministic alternate filename and make it a directory,
            # so the attempt to open it as a file will fail.
            def _fake_token_hex(n: int) -> str:
                return "deadbe"

            stem = target.stem
            suffix = target.suffix
            alt_name = f"{stem}.deadbe{suffix}" if stem else f".deadbe{suffix}"
            alt_path = target.with_name(alt_name)
            alt_path.mkdir()

            with patch("netflux.func_lib.text_editor.secrets.token_hex", _fake_token_hex):
                with self.assertRaises(FileExistsError) as cm:
                    self.editor.call(
                        self.ctx,
                        command="create",
                        path=str(target),
                        file_text="new contents",
                    )

            msg = str(cm.exception)
            self.assertIn("File already exists:", msg)
            self.assertIn(str(target), msg)
            self.assertIn(str(alt_path), msg)
            self.assertIn("failed to write contents", msg)

            # Ensure the alternate path remains a directory and was not replaced.
            self.assertTrue(alt_path.is_dir())

    def test_raise_create_conflict_message_on_alternate_success(self) -> None:
        editor = TextEditor()
        p = Path("/original/path.txt")
        file_text = "line1\nline2\n"

        # Patch helper to simulate a successful alternate write without touching filesystem.
        with patch.object(
            editor,
            "_attempt_alternate_create",
            return_value=(Path("/alternate/path.txt"), None),
        ) as mock_alt:
            with self.assertRaises(FileExistsError) as cm:
                editor._raise_create_conflict(p, file_text)

        mock_alt.assert_called_once_with(p, file_text)
        msg = str(cm.exception)
        self.assertIn("File already exists:", msg)
        self.assertIn(str(p), msg)
        # Path rendering is platform-dependent (e.g., backslashes on Windows),
        # so match only the invariant parts of the message.
        self.assertIn("Successfully wrote 2 lines instead to alternate path", msg)
        self.assertIn("overwrite is not allowed for command=create", msg)

    def test_raise_create_conflict_message_on_alternate_error(self) -> None:
        editor = TextEditor()
        p = Path("/original/path.txt")
        file_text = "only one line\n"
        alt_exc = PermissionError("no write access")

        with patch.object(
            editor,
            "_attempt_alternate_create",
            return_value=(Path("/alternate/path.txt"), alt_exc),
        ) as mock_alt:
            with self.assertRaises(FileExistsError) as cm:
                editor._raise_create_conflict(p, file_text)

        mock_alt.assert_called_once_with(p, file_text)
        msg = str(cm.exception)
        self.assertIn("File already exists:", msg)
        self.assertIn(str(p), msg)
        self.assertIn("alternate path", msg)
        self.assertIn("Also failed to write contents", msg)
        self.assertIn("PermissionError", msg)
        self.assertIn("no write access", msg)


class TestTextEditorConcurrency(unittest.TestCase):
    def test_parallel_str_replace_same_file_is_serialized(self) -> None:
        editor = TextEditor()

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "file.txt"

            top_bag = SessionBag()
            node_a = _DummyNode()
            node_b = _DummyNode()
            ctx_a = RunContext(runtime=None, node=node_a)  # type: ignore[arg-type]
            ctx_b = RunContext(runtime=None, node=node_b)  # type: ignore[arg-type]
            ctx_a.object_bags = {SessionScope.TopLevel: top_bag, SessionScope.Self: node_a.session_bag}
            ctx_b.object_bags = {SessionScope.TopLevel: top_bag, SessionScope.Self: node_b.session_bag}

            # Ensure LF line endings (Path.write_text() may translate to CRLF on Windows).
            editor.call(ctx_a, command="create", path=str(target), file_text="a=1\nb=2\n")

            file_lock = editor._get_file_lock(ctx_a, target.resolve())
            orig_acquire = file_lock.acquire
            orig_release = file_lock.release

            b_waiting_on_lock = threading.Event()
            b_acquired_lock = threading.Event()

            def patched_acquire(*args, **kwargs):
                if threading.current_thread().name == "writer-b":
                    if not orig_acquire(False):
                        b_waiting_on_lock.set()
                    else:
                        orig_release()
                    out = orig_acquire(*args, **kwargs)
                    b_acquired_lock.set()
                    return out
                return orig_acquire(*args, **kwargs)

            file_lock.acquire = patched_acquire

            a_prewrite = threading.Event()
            allow_a_write = threading.Event()
            a_read_started = threading.Event()
            b_read_started = threading.Event()
            errors: list[Exception] = []

            original_atomic_write = editor._atomic_write_text
            original_read_text = editor._read_text_preserve_eols

            reads: dict[str, str] = {}
            writes: dict[str, str] = {}

            def patched_atomic_write(p: Path, data: str) -> None:
                writes[threading.current_thread().name] = data
                if threading.current_thread().name == "writer-a":
                    a_prewrite.set()
                    if not allow_a_write.wait(timeout=10):
                        raise AssertionError("Timed out waiting to allow writer-a to write")
                original_atomic_write(p, data)

            def patched_read_text(p: Path) -> str:
                content = original_read_text(p)
                reads[threading.current_thread().name] = content
                if threading.current_thread().name == "writer-a":
                    a_read_started.set()
                if threading.current_thread().name == "writer-b":
                    b_read_started.set()
                return content

            def worker_a() -> None:
                try:
                    editor.call(
                        ctx_a,
                        command="str_replace",
                        path=str(target),
                        old_str="a=1\n",
                        new_str="a=10\n",
                    )
                except Exception as exc:
                    errors.append(exc)

            def worker_b() -> None:
                try:
                    editor.call(
                        ctx_b,
                        command="str_replace",
                        path=str(target),
                        old_str="b=2\n",
                        new_str="b=20\n",
                    )
                except Exception as exc:
                    errors.append(exc)

            with (
                patch.object(editor, "_atomic_write_text", side_effect=patched_atomic_write),
                patch.object(editor, "_read_text_preserve_eols", side_effect=patched_read_text),
            ):
                thread_a = threading.Thread(target=worker_a, name="writer-a")
                thread_b = threading.Thread(target=worker_b, name="writer-b")

                thread_a.start()
                self.assertTrue(a_prewrite.wait(timeout=2))
                self.assertTrue(a_read_started.wait(timeout=2))

                # While writer-a is paused before writing, the on-disk file is still the original content.
                self.assertEqual("a=1\nb=2\n", target.read_text(encoding="utf-8"))

                thread_b.start()
                self.assertTrue(b_waiting_on_lock.wait(timeout=2))
                self.assertFalse(b_acquired_lock.is_set())
                self.assertFalse(b_read_started.is_set())

                allow_a_write.set()

                thread_a.join(timeout=2)
                thread_b.join(timeout=2)
                self.assertFalse(thread_a.is_alive())
                self.assertFalse(thread_b.is_alive())

            if errors:
                raise errors[0]

            self.assertTrue(b_read_started.is_set())
            self.assertEqual("a=1\nb=2\n", reads.get("writer-a"))
            self.assertEqual("a=10\nb=2\n", reads.get("writer-b"))
            self.assertEqual("a=10\nb=2\n", writes.get("writer-a"))
            self.assertEqual("a=10\nb=20\n", writes.get("writer-b"))
            self.assertEqual("a=10\nb=20\n", target.read_text(encoding="utf-8"))
