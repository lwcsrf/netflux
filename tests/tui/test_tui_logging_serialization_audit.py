from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from xml.etree import ElementTree as ET

from ...func_lib import ImageResult
from ...runtime import Runtime
from ...tui._logging import close_tui_logging, configure_tui_logging, log_tui_result
from ...tui.tui import TUI, _RunRecord
from .test_tui_controllers import _make_code_function, _make_output_view


class TestTUIResultSerializationAudit(unittest.TestCase):
    def setUp(self) -> None:
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        self.log_path = configure_tui_logging(Path(tmpdir.name) / "results.log")
        self.addCleanup(close_tui_logging, self.log_path)

    def _write(self, content: str, *, name: str = "session", stamp: float = 0.0) -> None:
        log_tui_result(
            self.log_path,
            created_at=stamp,
            finished_at=stamp + 1,
            name=name,
            content=content,
        )

    def _results(self) -> list[ET.Element]:
        # Read bytes so Python's universal-newline conversion cannot hide changes
        # made by the logging stream or the XML parser.
        text = self.log_path.read_bytes().decode("utf-8")
        return [
            ET.fromstring(fragment)
            for fragment in re.findall(r"<session_result>.*?</session_result>", text, re.DOTALL)
        ]

    def test_empty_and_whitespace_only_content_is_preserved(self) -> None:
        values = ["", " ", "\t", "\n", " \t\n  ", "\n\n"]
        for content in values:
            self._write(content)
        self.assertEqual([entry.findtext("content") for entry in self._results()], values)

    def test_multiline_indented_content_and_xml_metacharacters_are_preserved(self) -> None:
        content = (
            '  <session_result>\n    <content attr="x">&lt;answer&gt; & value</content>\n'
            "  </session_result>\n\t<!-- comment --> <![CDATA[body]]> ]]></content>\n"
        )
        name = 'A <name> & "quoted" session'
        self._write(content, name=name)
        entries = self._results()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].findtext("content"), content)
        self.assertEqual(entries[0].findtext("name"), name)
        self.assertEqual([child.tag for child in entries[0]], ["created_at", "finished_at", "name", "content"])

    def test_lone_carriage_returns_are_preserved(self) -> None:
        content = "progress 1\rprogress 2\rdone\r"
        self._write(content)
        self.assertEqual(self._results()[0].findtext("content"), content)

    def test_windows_crlf_content_is_preserved(self) -> None:
        content = "first\r\nsecond\r\n\r\nlast\r\n"
        self._write(content)
        self.assertEqual(self._results()[0].findtext("content"), content)

    def test_xml_illegal_characters_are_visible_in_name_and_content(self) -> None:
        points = [*range(0, 9), 11, 12, *range(14, 32), *range(0xD800, 0xE000), 0xFFFE, 0xFFFF]
        content = "|".join(chr(point) for point in points)
        expected = "|".join(f"\\u{point:04x}" for point in points)
        self._write(content, name=content)
        entry = self._results()[0]
        self.assertEqual(entry.findtext("name"), expected)
        self.assertEqual(entry.findtext("content"), expected)

    def test_all_other_unicode_codepoints_remain_valid_and_unchanged(self) -> None:
        # Every XML-permitted codepoint, including non-BMP text and noncharacters.
        # CR has separate tests because XML normally normalizes literal CR.
        content = "\t\n" + "".join(
            chr(point) for point in range(0x20, 0x110000)
            if not 0xD800 <= point <= 0xDFFF and point not in (0xFFFE, 0xFFFF)
        )
        self._write(content)
        restored = self._results()[0].findtext("content")
        self.assertEqual(len(restored or ""), len(content))
        self.assertTrue(restored == content, "An XML-permitted Unicode codepoint changed")

    def test_timestamps_roundtrip_across_epoch_fractional_dates_and_dst_boundaries(self) -> None:
        stamps = [
            0.0,
            1_700_000_000.125,
            datetime.fromisoformat("2026-03-08T09:59:59+00:00").timestamp(),
            datetime.fromisoformat("2026-03-08T10:00:00+00:00").timestamp(),
            datetime.fromisoformat("2026-11-01T08:59:59+00:00").timestamp(),
            datetime.fromisoformat("2026-11-01T09:00:00+00:00").timestamp(),
            2_147_483_648.5,
        ]
        for stamp in stamps:
            self._write("value", stamp=stamp)
        for stamp, entry in zip(stamps, self._results()):
            with self.subTest(stamp=stamp):
                for tag, expected in (("created_at", stamp), ("finished_at", stamp + 1)):
                    parsed = datetime.fromisoformat(entry.findtext(tag) or "")
                    self.assertIsNotNone(parsed.utcoffset())
                    self.assertEqual(parsed.timestamp(), expected)

    def test_large_escape_heavy_multiline_output_is_not_truncated(self) -> None:
        content = ('<answer> & "quoted"\n' * 100_000) + "final marker"
        self._write(content)
        restored = self._results()[0].findtext("content")
        self.assertEqual(len(restored or ""), len(content))
        self.assertTrue(restored == content, "Large session output was modified or truncated")

    def _log_outputs(self, values: list[object]) -> None:
        fn = _make_code_function("root")
        tui = TUI(Runtime([fn], client_factories={}), log_path=self.log_path)
        for index, output in enumerate(values):
            view = _make_output_view(fn, output=output, update_seqnum=index + 1)
            run = _RunRecord(
                name=f"run {index}",
                fn=fn,
                node=SimpleNamespace(id=view.id),
                renderer=Mock(),
                cancel_event=threading.Event(),
                latest_view=view,
                created_at=0.0,
            )
            tui._handle_terminal_run_if_needed(run)

    def test_common_non_string_outputs_use_their_complete_text_representation(self) -> None:
        recursive: list[object] = []
        recursive.append(recursive)
        values = [None, False, 0, 3.25, b"\x00binary\xff", {"answer": [1, 2, "<ok>&"]}, recursive]
        self._log_outputs(values)
        self.assertEqual([entry.findtext("content") for entry in self._results()], list(map(str, values)))

    def test_image_output_logs_status_without_accessing_media(self) -> None:
        class UnprintableMedia:
            def __str__(self) -> str:
                raise AssertionError("Image media must not be stringified")

            def __repr__(self) -> str:
                raise AssertionError("Image media must not be represented")

        status = "image/png, 100x100, 1000 bytes; unchanged."
        image = ImageResult(
            source_path="C:/image.png", mime_type="image/png", width=100,
            height=100, status=status, data=UnprintableMedia(), base64_data=UnprintableMedia(),
        )
        self._log_outputs([image])
        entries = self._results()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].findtext("content"), status)
