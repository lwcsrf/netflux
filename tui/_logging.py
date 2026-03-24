from __future__ import annotations

from datetime import datetime
import logging
import os
from pathlib import Path
import tempfile


_TUI_FILE_HANDLER_NAME = "netflux.tui.file"


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
