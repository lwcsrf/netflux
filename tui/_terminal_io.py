from __future__ import annotations

import ctypes
import os
import shutil
import sys
import threading
import time
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


class ConsoleBusyError(RuntimeError):
    """Console acquisition failed without changing the current owner's state."""


_console_lock = threading.Lock()
_console_ready = False
_console_restore_pending = False
_POSIX_PENDING_ESCAPE_SEQ: str | None = None
_POSIX_PENDING_ESCAPE_DEADLINE: float | None = None
_POSIX_PENDING_ESCAPE_DISCARD = False
_POSIX_ORPHAN_ESCAPE_DEADLINE: float | None = None
_POSIX_ESCAPE_INTERBYTE_TIMEOUT = 0.05
_POSIX_BARE_ESCAPE_GRACE = 0.05

_WIN_STD_OUTPUT_HANDLE = -11
_WIN_STD_INPUT_HANDLE = -10
_WIN_ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
_WIN_ENABLE_PROCESSED_INPUT = 0x0001
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


def _windows_kernel32() -> ctypes.CDLL:
    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    # ctypes otherwise defaults HANDLE arguments/results to 32-bit integers.
    signatures = (
        ("GetStdHandle", [ctypes.c_ulong], ctypes.c_void_p),
        ("GetConsoleMode", [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)], ctypes.c_int),
        ("SetConsoleMode", [ctypes.c_void_p, ctypes.c_uint], ctypes.c_int),
        ("FlushConsoleInputBuffer", [ctypes.c_void_p], ctypes.c_int),
        ("PeekConsoleInputW", [ctypes.c_void_p, ctypes.POINTER(_WinInputRecord), ctypes.c_uint,
                              ctypes.POINTER(ctypes.c_uint)], ctypes.c_int),
        ("ReadConsoleInputW", [ctypes.c_void_p, ctypes.POINTER(_WinInputRecord), ctypes.c_uint,
                              ctypes.POINTER(ctypes.c_uint)], ctypes.c_int),
    )
    for name, argtypes, restype in signatures:
        function = getattr(kernel32, name, None)
        if hasattr(function, "argtypes"):
            function.argtypes = argtypes
            function.restype = restype
    return kernel32


def prepare_windows_console_input() -> None:
    """Shield foreground work before controller startup, without changing screens."""
    if os.name != "nt":
        return
    with _console_lock:
        if _console_ready or _console_restore_pending or _WINDOWS_INPUT_MODE_SAVED is not None:
            raise ConsoleBusyError("Console cleanup is still pending; restore it before starting another session.")
        _configure_windows_console_input(enable_mouse=True)


def pre_console(*, windows_input_prepared: bool = False) -> None:
    if not sys.stdout.isatty():
        return
    global _console_ready, _console_restore_pending
    global _POSIX_PENDING_ESCAPE_SEQ, _POSIX_PENDING_ESCAPE_DEADLINE
    global _POSIX_PENDING_ESCAPE_DISCARD, _POSIX_ORPHAN_ESCAPE_DEADLINE
    with _console_lock:
        if _console_ready:
            return
        if _console_restore_pending or (_WINDOWS_INPUT_MODE_SAVED is not None and not windows_input_prepared):
            raise RuntimeError("Console cleanup is still pending; restore it before starting another session.")
        if windows_input_prepared and _WINDOWS_INPUT_MODE_SAVED is None:
            raise RuntimeError("Prepared Windows console input is no longer available.")
        _POSIX_PENDING_ESCAPE_SEQ = None
        _POSIX_PENDING_ESCAPE_DEADLINE = None
        _POSIX_PENDING_ESCAPE_DISCARD = False
        _POSIX_ORPHAN_ESCAPE_DEADLINE = None
        # Setup can fail after changing console modes or writing only part of
        # the escape sequence. Cleanup must still own that partial setup.
        _console_restore_pending = True
        _enable_vt_if_windows()
        if sys.stdin.isatty() and not windows_input_prepared:
            _configure_windows_console_input(enable_mouse=True)
        sys.stdout.write(
            "\x1b[?1049h\x1b[?25l\x1b[?7l\x1b[3J"
            "\x1b[?1000h\x1b[?1006h"
        )
        sys.stdout.flush()
        _console_ready = True


def restore_console() -> None:
    global _console_ready, _console_restore_pending
    global _POSIX_PENDING_ESCAPE_SEQ, _POSIX_PENDING_ESCAPE_DEADLINE
    global _POSIX_PENDING_ESCAPE_DISCARD, _POSIX_ORPHAN_ESCAPE_DEADLINE
    with _console_lock:
        if not (_console_ready or _console_restore_pending or _WINDOWS_INPUT_MODE_SAVED is not None):
            return
        _POSIX_PENDING_ESCAPE_SEQ = None
        _POSIX_PENDING_ESCAPE_DEADLINE = None
        _POSIX_PENDING_ESCAPE_DISCARD = False
        _POSIX_ORPHAN_ESCAPE_DEADLINE = None
        _console_ready = False
        error: BaseException | None = None
        try:
            # Input mode restoration is independent of stdout still being a TTY.
            _configure_windows_console_input(enable_mouse=False)
        except BaseException as exc:
            error = exc
        try:
            if _console_restore_pending and sys.stdout.isatty():
                sys.stdout.write("\x1b[?1000l\x1b[?1006l\x1b[?25h\x1b[?7h\x1b[?1049l")
                sys.stdout.flush()
                _console_restore_pending = False
        except BaseException as exc:
            if error is None:
                error = exc
        if error is not None:
            raise error


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

    kernel32 = _windows_kernel32()
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
    if not enable_mouse and _WINDOWS_INPUT_MODE_SAVED is None:
        return

    kernel32 = _windows_kernel32()
    handle = kernel32.GetStdHandle(_WIN_STD_INPUT_HANDLE)

    if enable_mouse:
        mode = ctypes.c_uint()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            raise OSError("GetConsoleMode failed for console input setup")
        if _WINDOWS_INPUT_MODE_SAVED is None:
            _WINDOWS_INPUT_MODE_SAVED = mode.value
        # Window-buffer resize notifications are required for resize-only redraws on Windows.
        new_mode = (
            mode.value
            | _WIN_ENABLE_EXTENDED_FLAGS
            | _WIN_ENABLE_WINDOW_INPUT
            | _WIN_ENABLE_MOUSE_INPUT
        )
        # Read Ctrl-C as an input record, so it cannot unwind an in-progress
        # controller callback or escape a cancellation confirmation screen.
        new_mode &= ~(_WIN_ENABLE_QUICK_EDIT_MODE | _WIN_ENABLE_PROCESSED_INPUT)
        if not kernel32.SetConsoleMode(handle, new_mode):
            raise OSError("SetConsoleMode failed for console input setup")
        return

    if _WINDOWS_INPUT_MODE_SAVED is not None:
        if not kernel32.SetConsoleMode(handle, _WINDOWS_INPUT_MODE_SAVED):
            raise OSError("SetConsoleMode failed for console input restoration")
        _WINDOWS_INPUT_MODE_SAVED = None


def enable_windows_terminal_interrupt() -> None:
    """Rearm physical Ctrl-C after confirmation without restoring UI modes."""
    kernel32 = _windows_kernel32()
    handle = kernel32.GetStdHandle(_WIN_STD_INPUT_HANDLE)
    mode = ctypes.c_uint()
    if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        raise OSError("GetConsoleMode failed while enabling force interrupt")
    if not kernel32.SetConsoleMode(handle, mode.value | _WIN_ENABLE_PROCESSED_INPUT):
        raise OSError("SetConsoleMode failed while enabling force interrupt")


def discard_pending_input(fd: int | None = None) -> None:
    """Drop input queued in the modal before accepting input in the next mode."""
    if os.name == "nt":
        kernel32 = _windows_kernel32()
        handle = kernel32.GetStdHandle(_WIN_STD_INPUT_HANDLE)
        if not kernel32.FlushConsoleInputBuffer(handle):
            raise OSError("FlushConsoleInputBuffer failed while leaving confirmation")
    elif fd is not None:
        import termios

        termios.tcflush(fd, termios.TCIFLUSH)
    _clear_posix_pending_escape()
    _clear_posix_orphan_escape()


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
    if seq.startswith("[M"):
        if len(seq) != 5:
            return None
        cb = ord(seq[2]) - 32
        cx = ord(seq[3]) - 33
        cy = ord(seq[4]) - 33
        if min(cb, cx, cy) < 0:
            return None
        x = max(0, cx)
        y = max(0, cy)
        if cb & 64:
            if cb & 1:
                return MouseEvent(x=x, y=y, button="wheel_down")
            return MouseEvent(x=x, y=y, button="wheel_up")
        if cb & 32:
            return None
        button_code = cb & 3
        if button_code == 3:
            return None
        if button_code == 0:
            return MouseEvent(x=x, y=y, button="left")
        if button_code == 1:
            return MouseEvent(x=x, y=y, button="middle")
        if button_code == 2:
            return MouseEvent(x=x, y=y, button="right")
    return None


def _clear_posix_pending_escape() -> None:
    global _POSIX_PENDING_ESCAPE_SEQ, _POSIX_PENDING_ESCAPE_DEADLINE
    global _POSIX_PENDING_ESCAPE_DISCARD
    _POSIX_PENDING_ESCAPE_SEQ = None
    _POSIX_PENDING_ESCAPE_DEADLINE = None
    _POSIX_PENDING_ESCAPE_DISCARD = False


def _clear_posix_orphan_escape() -> None:
    global _POSIX_ORPHAN_ESCAPE_DEADLINE
    _POSIX_ORPHAN_ESCAPE_DEADLINE = None


def _start_posix_orphan_escape() -> None:
    global _POSIX_ORPHAN_ESCAPE_DEADLINE
    _POSIX_ORPHAN_ESCAPE_DEADLINE = time.monotonic() + _POSIX_ESCAPE_INTERBYTE_TIMEOUT


def _posix_orphan_escape_active() -> bool:
    deadline = _POSIX_ORPHAN_ESCAPE_DEADLINE
    if deadline is None:
        return False
    if time.monotonic() >= deadline:
        _clear_posix_orphan_escape()
        return False
    return True


def _posix_escape_complete(seq: str) -> bool:
    if not seq:
        return False
    if seq.startswith("[M"):
        return len(seq) >= 5
    last = seq[-1]
    return last.isalpha() or last == "~"


def posix_pending_input_timeout() -> float | None:
    if select is None:
        return None
    if _POSIX_PENDING_ESCAPE_SEQ is None or _POSIX_PENDING_ESCAPE_DEADLINE is None:
        return None
    return max(0.0, _POSIX_PENDING_ESCAPE_DEADLINE - time.monotonic())


def _buffer_posix_escape(
    seq: str,
    *,
    deadline: float | None,
    discard: bool,
) -> str | MouseEvent | None:
    global _POSIX_PENDING_ESCAPE_SEQ, _POSIX_PENDING_ESCAPE_DEADLINE
    global _POSIX_PENDING_ESCAPE_DISCARD

    if seq:
        if deadline is not None and time.monotonic() >= deadline:
            _clear_posix_pending_escape()
            return None
        _POSIX_PENDING_ESCAPE_SEQ = seq
        _POSIX_PENDING_ESCAPE_DEADLINE = deadline
        _POSIX_PENDING_ESCAPE_DISCARD = discard
        return None

    if deadline is None:
        _POSIX_PENDING_ESCAPE_SEQ = ""
        _POSIX_PENDING_ESCAPE_DEADLINE = time.monotonic() + _POSIX_BARE_ESCAPE_GRACE
        _POSIX_PENDING_ESCAPE_DISCARD = False
        return None

    if time.monotonic() >= deadline:
        _clear_posix_pending_escape()
        _start_posix_orphan_escape()
        return "escape"

    _POSIX_PENDING_ESCAPE_SEQ = ""
    _POSIX_PENDING_ESCAPE_DEADLINE = deadline
    _POSIX_PENDING_ESCAPE_DISCARD = False
    return None


def _read_posix_escape(
    fd: int,
    *,
    initial_seq: str = "",
    deadline: float | None = None,
    discard: bool = False,
) -> str | MouseEvent | None:
    seq = initial_seq
    while True:
        timeout = _POSIX_ESCAPE_INTERBYTE_TIMEOUT if deadline is None else max(0.0, deadline - time.monotonic())
        try:
            ready, _, _ = select.select([fd], [], [], timeout)  # type: ignore[union-attr]
        except (InterruptedError, OSError):
            return _buffer_posix_escape(seq, deadline=deadline, discard=discard)
        if not ready:
            return _buffer_posix_escape(seq, deadline=deadline, discard=discard)

        piece = os.read(fd, 1).decode("utf-8", errors="ignore")
        if not piece:
            return _buffer_posix_escape(seq, deadline=deadline, discard=discard)
        if piece == "\x03":
            # Ctrl-C remains an interrupt even in a partial or discarded escape
            # sequence now that interactive terminals deliver it as input.
            _clear_posix_pending_escape()
            _clear_posix_orphan_escape()
            return piece

        seq += piece
        deadline = time.monotonic() + _POSIX_ESCAPE_INTERBYTE_TIMEOUT
        if len(seq) > 64:
            _clear_posix_pending_escape()
            return None
        if _posix_escape_complete(seq):
            _clear_posix_pending_escape()
            if discard:
                return None
            return _decode_posix_escape_sequence(seq)


def read_key(fd: int, timeout: float | None = 0.1) -> str | MouseEvent | None:
    global _POSIX_PENDING_ESCAPE_SEQ, _POSIX_PENDING_ESCAPE_DEADLINE
    global _POSIX_PENDING_ESCAPE_DISCARD

    if select is None:
        return None

    if _POSIX_PENDING_ESCAPE_SEQ is not None:
        seq = _POSIX_PENDING_ESCAPE_SEQ
        deadline = _POSIX_PENDING_ESCAPE_DEADLINE
        discard = _POSIX_PENDING_ESCAPE_DISCARD
        _clear_posix_pending_escape()
        return _read_posix_escape(
            fd,
            initial_seq=seq,
            deadline=deadline,
            discard=discard,
        )

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
        _clear_posix_orphan_escape()
        return _read_posix_escape(fd)

    if _posix_orphan_escape_active():
        if ch == "[":
            _clear_posix_orphan_escape()
            return _read_posix_escape(
                fd,
                initial_seq=ch,
                deadline=time.monotonic() + _POSIX_ESCAPE_INTERBYTE_TIMEOUT,
                discard=True,
            )
        _clear_posix_orphan_escape()

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
    kernel32 = _windows_kernel32()
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
