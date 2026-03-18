from __future__ import annotations

import ctypes
import os
import shutil
import sys
import threading
from dataclasses import dataclass

if os.name != "nt":
    import select
else:  # pragma: no cover - runtime guarded use on non-POSIX terminals only
    select = None


@dataclass(frozen=True)
class MouseEvent:
    x: int
    y: int
    button: str


@dataclass(frozen=True)
class ResizeEvent:
    pass


@dataclass(frozen=True)
class InterruptEvent:
    pass


_console_lock = threading.Lock()
_console_ready = False
_POSIX_PENDING_ESCAPE_SEQ: str | None = None

_WIN_STD_OUTPUT_HANDLE = -11
_WIN_STD_INPUT_HANDLE = -10
_WIN_ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
_WIN_ENABLE_WINDOW_INPUT = 0x0008
_WIN_ENABLE_EXTENDED_FLAGS = 0x0080
_WIN_ENABLE_QUICK_EDIT_MODE = 0x0040
_WIN_ENABLE_MOUSE_INPUT = 0x0010
_WIN_SHIFT_PRESSED = 0x0010
_WIN_KEY_EVENT = 0x0001
_WIN_MOUSE_EVENT = 0x0002
_WIN_WINDOW_BUFFER_SIZE_EVENT = 0x0004
_WIN_DOUBLE_CLICK = 0x0002
_WIN_MOUSE_WHEELED = 0x0004
_WIN_FROM_LEFT_1ST_BUTTON_PRESSED = 0x0001
_WIN_RIGHTMOST_BUTTON_PRESSED = 0x0002
_WIN_FROM_LEFT_2ND_BUTTON_PRESSED = 0x0004


class _WinCoord(ctypes.Structure):
    _fields_ = [("X", ctypes.c_short), ("Y", ctypes.c_short)]


class _WinKeyEventRecord(ctypes.Structure):
    _fields_ = [
        ("bKeyDown", ctypes.c_int),
        ("wRepeatCount", ctypes.c_ushort),
        ("wVirtualKeyCode", ctypes.c_ushort),
        ("wVirtualScanCode", ctypes.c_ushort),
        ("uChar", ctypes.c_wchar),
        ("dwControlKeyState", ctypes.c_uint),
    ]


class _WinMouseEventRecord(ctypes.Structure):
    _fields_ = [
        ("dwMousePosition", _WinCoord),
        ("dwButtonState", ctypes.c_uint),
        ("dwControlKeyState", ctypes.c_uint),
        ("dwEventFlags", ctypes.c_uint),
    ]


class _WinWindowBufferSizeRecord(ctypes.Structure):
    _fields_ = [("dwSize", _WinCoord)]


class _WinEventUnion(ctypes.Union):
    _fields_ = [
        ("KeyEvent", _WinKeyEventRecord),
        ("MouseEvent", _WinMouseEventRecord),
        ("WindowBufferSizeEvent", _WinWindowBufferSizeRecord),
        ("_padding", ctypes.c_byte * 16),
    ]


class _WinInputRecord(ctypes.Structure):
    _fields_ = [("EventType", ctypes.c_ushort), ("Event", _WinEventUnion)]


def pre_console() -> None:
    if not sys.stdout.isatty():
        return
    global _console_ready, _POSIX_PENDING_ESCAPE_SEQ
    with _console_lock:
        if _console_ready:
            return
        _POSIX_PENDING_ESCAPE_SEQ = None
        _enable_vt_if_windows()
        _configure_windows_console_input(enable_mouse=True)
        sys.stdout.write(
            "\x1b[?1049h\x1b[?25l\x1b[?7l\x1b[3J"
            "\x1b[?1000h\x1b[?1006h"
        )
        sys.stdout.flush()
        _console_ready = True


def restore_console() -> None:
    if not sys.stdout.isatty():
        return
    global _console_ready, _POSIX_PENDING_ESCAPE_SEQ
    with _console_lock:
        if not _console_ready:
            return
        _POSIX_PENDING_ESCAPE_SEQ = None
        _configure_windows_console_input(enable_mouse=False)
        sys.stdout.write("\x1b[?1000l\x1b[?1006l\x1b[?25h\x1b[?7h\x1b[?1049l")
        sys.stdout.flush()
        _console_ready = False


def ui_driver(s: str) -> None:
    if not sys.stdout.isatty():
        return
    pre_console()
    sys.stdout.write("\x1b[H")
    sys.stdout.write(s)
    sys.stdout.flush()


def terminal_size_token() -> tuple[int, int]:
    sz = shutil.get_terminal_size(fallback=(80, 24))
    return (sz.columns, sz.lines)


def _enable_vt_if_windows() -> None:
    if os.name != "nt":
        return

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.GetStdHandle(_WIN_STD_OUTPUT_HANDLE)
    mode = ctypes.c_uint()
    if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        kernel32.SetConsoleMode(
            handle,
            mode.value | _WIN_ENABLE_VIRTUAL_TERMINAL_PROCESSING,
        )


_WINDOWS_INPUT_MODE_SAVED: int | None = None


def _configure_windows_console_input(*, enable_mouse: bool) -> None:
    global _WINDOWS_INPUT_MODE_SAVED

    if os.name != "nt":
        return

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.GetStdHandle(_WIN_STD_INPUT_HANDLE)
    mode = ctypes.c_uint()
    if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        return

    if enable_mouse:
        if _WINDOWS_INPUT_MODE_SAVED is None:
            _WINDOWS_INPUT_MODE_SAVED = mode.value
        # Window-buffer resize notifications are required for resize-only redraws on Windows.
        new_mode = (
            mode.value
            | _WIN_ENABLE_EXTENDED_FLAGS
            | _WIN_ENABLE_WINDOW_INPUT
            | _WIN_ENABLE_MOUSE_INPUT
        )
        new_mode &= ~_WIN_ENABLE_QUICK_EDIT_MODE
        kernel32.SetConsoleMode(handle, new_mode)
        return

    if _WINDOWS_INPUT_MODE_SAVED is not None:
        kernel32.SetConsoleMode(handle, _WINDOWS_INPUT_MODE_SAVED)
        _WINDOWS_INPUT_MODE_SAVED = None


def _decode_posix_escape_sequence(seq: str) -> str | MouseEvent | None:
    if seq == "[A":
        return "up"
    if seq == "[B":
        return "down"
    if seq == "[Z":
        return "shift_tab"
    if seq == "[5~":
        return "page_up"
    if seq == "[6~":
        return "page_down"
    if seq.startswith("[<") and seq[-1] in ("M", "m"):
        try:
            cb_s, cx_s, cy_s = seq[2:-1].split(";")
            cb = int(cb_s)
            cx = int(cx_s)
            cy = int(cy_s)
        except (TypeError, ValueError):
            return None
        x = max(0, cx - 1)
        y = max(0, cy - 1)
        if cb & 64:
            if cb & 1:
                return MouseEvent(x=x, y=y, button="wheel_down")
            return MouseEvent(x=x, y=y, button="wheel_up")
        if seq[-1] == "m" or (cb & 32):
            return None
        button_code = cb & 3
        if button_code == 0:
            return MouseEvent(x=x, y=y, button="left")
        if button_code == 1:
            return MouseEvent(x=x, y=y, button="middle")
        if button_code == 2:
            return MouseEvent(x=x, y=y, button="right")
    return None


def _read_posix_escape(fd: int, *, initial_seq: str = "") -> str | MouseEvent | None:
    global _POSIX_PENDING_ESCAPE_SEQ

    seq = initial_seq
    while True:
        try:
            ready, _, _ = select.select([fd], [], [], 0.05)  # type: ignore[union-attr]
        except (InterruptedError, OSError):
            if not seq:
                return "escape"
            _POSIX_PENDING_ESCAPE_SEQ = seq
            return None
        if not ready:
            if not seq:
                return "escape"
            _POSIX_PENDING_ESCAPE_SEQ = seq
            return None

        piece = os.read(fd, 1).decode("utf-8", errors="ignore")
        if not piece:
            if not seq:
                return "escape"
            _POSIX_PENDING_ESCAPE_SEQ = seq
            return None

        seq += piece
        if len(seq) > 64:
            _POSIX_PENDING_ESCAPE_SEQ = None
            return None
        if piece.isalpha() or piece in ("~", "m"):
            _POSIX_PENDING_ESCAPE_SEQ = None
            return _decode_posix_escape_sequence(seq)


def read_key(fd: int, timeout: float | None = 0.1) -> str | MouseEvent | None:
    global _POSIX_PENDING_ESCAPE_SEQ

    if select is None:
        return None

    if _POSIX_PENDING_ESCAPE_SEQ is not None:
        seq = _POSIX_PENDING_ESCAPE_SEQ
        _POSIX_PENDING_ESCAPE_SEQ = None
        return _read_posix_escape(fd, initial_seq=seq)

    try:
        ready, _, _ = select.select([fd], [], [], timeout)
    except (InterruptedError, OSError):
        return None

    if not ready:
        return None

    raw = os.read(fd, 1)
    if not raw:
        return None
    ch = raw.decode("utf-8", errors="ignore")

    if ch == "\x1b":
        return _read_posix_escape(fd)

    return ch


def _decode_windows_mouse_event(event: _WinMouseEventRecord) -> MouseEvent | None:
    x = max(0, int(event.dwMousePosition.X))
    y = max(0, int(event.dwMousePosition.Y))
    if event.dwEventFlags == _WIN_MOUSE_WHEELED:
        delta = ctypes.c_short((event.dwButtonState >> 16) & 0xFFFF).value
        return MouseEvent(
            x=x,
            y=y,
            button="wheel_up" if delta > 0 else "wheel_down",
        )
    if event.dwEventFlags not in (0, _WIN_DOUBLE_CLICK):
        return None
    if event.dwButtonState & _WIN_FROM_LEFT_1ST_BUTTON_PRESSED:
        return MouseEvent(x=x, y=y, button="left")
    if event.dwButtonState & _WIN_RIGHTMOST_BUTTON_PRESSED:
        return MouseEvent(x=x, y=y, button="right")
    if event.dwButtonState & _WIN_FROM_LEFT_2ND_BUTTON_PRESSED:
        return MouseEvent(x=x, y=y, button="middle")
    return None


def read_key_windows() -> str | MouseEvent | ResizeEvent | InterruptEvent | None:
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    handle = kernel32.GetStdHandle(_WIN_STD_INPUT_HANDLE)
    while True:
        preview = _WinInputRecord()
        count = ctypes.c_uint()
        if not kernel32.PeekConsoleInputW(handle, ctypes.byref(preview), 1, ctypes.byref(count)):
            return None
        if count.value == 0:
            return None

        record = _WinInputRecord()
        if not kernel32.ReadConsoleInputW(handle, ctypes.byref(record), 1, ctypes.byref(count)):
            return None
        if count.value == 0:
            return None
        if record.EventType == _WIN_KEY_EVENT:
            event = record.Event.KeyEvent
            if not event.bKeyDown:
                continue
            if event.uChar == "\x03":
                return InterruptEvent()
            vk = event.wVirtualKeyCode
            if vk == 0x26:
                return "up"
            if vk == 0x28:
                return "down"
            if vk == 0x21:
                return "page_up"
            if vk == 0x22:
                return "page_down"
            if vk == 0x1B:
                return "escape"
            if vk == 0x0D:
                return "\r"
            if vk == 0x09:
                if event.dwControlKeyState & _WIN_SHIFT_PRESSED:
                    return "shift_tab"
                return "\t"
            if event.uChar not in ("", "\x00"):
                return event.uChar
            continue
        if record.EventType == _WIN_MOUSE_EVENT:
            decoded = _decode_windows_mouse_event(record.Event.MouseEvent)
            if decoded is not None:
                return decoded
            continue
        if record.EventType == _WIN_WINDOW_BUFFER_SIZE_EVENT:
            return ResizeEvent()
