import os
from .viz import Render, start_view_loop
from .console import ConsoleRender

__all__ = [
    "Render",
    "ConsoleRender",
    "start_view_loop",
]

def enable_vt_if_windows() -> None:
    """Enable ANSI (VT) processing for classic Windows consoles.

    No-op on non-Windows. Uses Win32 console mode flag ENABLE_VIRTUAL_TERMINAL_PROCESSING.
    """
    if os.name != "nt":
        return
    import ctypes

    kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    STD_OUTPUT_HANDLE = -11
    ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
    handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
    mode = ctypes.c_uint()
    if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
        kernel32.SetConsoleMode(handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
