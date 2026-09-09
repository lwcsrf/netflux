from __future__ import annotations

from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import re
import tempfile
from xml.etree import ElementTree as ET


_TUI_FILE_HANDLER_NAME = "netflux.tui.file"
_XML_INVALID_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")


def configure_tui_logging(log_path: str | os.PathLike[str] | None = None) -> Path:
    path = _resolve_tui_log_path(log_path)
    logger = logging.getLogger("netflux")
    _remove_tui_file_handlers(logger)

    handler = logging.FileHandler(path, encoding="utf-8")
    handler.name = _TUI_FILE_HANDLER_NAME
    handler.setLevel(logging.ERROR)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s [%(threadName)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return path


def log_tui_result(
    log_path: str | os.PathLike[str],
    *,
    created_at: float,
    finished_at: float,
    name: str,
    content: str,
) -> None:
    """Write a top-level session's timestamps, name, and result as XML to its TUI log.
    Result entries bypass the file handler's normal ERROR threshold.
    """
    entry = ET.Element("session_result")
    for tag, value in (
        ("created_at", datetime.fromtimestamp(created_at, timezone.utc).astimezone().isoformat()),
        ("finished_at", datetime.fromtimestamp(finished_at, timezone.utc).astimezone().isoformat()),
        ("name", name),
        ("content", content),
    ):
        # Keep control characters visible without producing malformed XML.
        ET.SubElement(entry, tag).text = _XML_INVALID_CHARS.sub(
            lambda match: f"\\u{ord(match.group()):04x}", value
        )
    ET.indent(entry)
    # Character references preserve carriage returns through XML/newline normalization.
    xml_text = ET.tostring(entry, encoding="unicode").replace("\r", "&#13;")
    record = logging.LogRecord(
        name="netflux.tui.result",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg="\n" + xml_text,
        args=(),
        exc_info=None,
    )
    target_path = Path(log_path).expanduser().resolve()
    for handler in logging.getLogger("netflux").handlers:
        if (
            handler.name == _TUI_FILE_HANDLER_NAME
            and isinstance(handler, logging.FileHandler)
            and Path(handler.baseFilename).resolve() == target_path
        ):
            # Only explicit session results bypass the normal ERROR threshold.
            # Reuse the handler's lock, formatter, and flush without propagation.
            handler.handle(record)


def close_tui_logging(log_path: str | os.PathLike[str] | None = None) -> None:
    logger = logging.getLogger("netflux")
    target_path = None if log_path is None else Path(log_path).expanduser().resolve()
    _remove_tui_file_handlers(logger, target_path=target_path)


def default_tui_log_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d")
    tmp_dir = Path(tempfile.gettempdir())
    fd, raw_path = tempfile.mkstemp(
        dir=tmp_dir,
        prefix=f"netflux_tui_{stamp}_",
        suffix=".log",
    )
    os.close(fd)
    return Path(raw_path)


def _resolve_tui_log_path(log_path: str | os.PathLike[str] | None) -> Path:
    if log_path is None:
        return default_tui_log_path()

    path = Path(log_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    return path


def _remove_tui_file_handlers(
    logger: logging.Logger,
    *,
    target_path: Path | None = None,
) -> None:
    for handler in list(logger.handlers):
        if handler.name != _TUI_FILE_HANDLER_NAME:
            continue
        if target_path is not None:
            if not isinstance(handler, logging.FileHandler):
                continue
            if Path(handler.baseFilename).resolve() != target_path:
                continue
        logger.removeHandler(handler)
        try:
            handler.flush()
        finally:
            handler.close()
