from __future__ import annotations

import ctypes
import os
import signal
import sys
import threading
import time

from ._contracts import SessionController, TerminalSize
from ._terminal_io import (
    InterruptEvent,
    ResizeEvent,
    pre_console,
    read_key,
    read_key_windows,
    restore_console,
    terminal_size_token,
    ui_driver,
)

if os.name != "nt":
    import select
    import termios
    import tty
else:  # pragma: no cover - runtime guarded use on non-POSIX terminals only
    select = None
    termios = None
    tty = None


_WIN_STD_INPUT_HANDLE = -10
_WIN_WAIT_OBJECT_0 = 0
_WIN_WAIT_TIMEOUT = 258
_WIN_INFINITE = 0xFFFFFFFF


class ConsoleSessionDriver:
    def __init__(self, *, spinner_hz: float = 10.0) -> None:
        self.spinner_hz = max(1.0, float(spinner_hz))
        self._t0 = time.monotonic()
        self._thread_wakeup = threading.Event()
        self._wake_pipe_read: int | None = None
        self._wake_pipe_write: int | None = None
        self._old_sigwinch = None
        self._win_kernel32 = None
        self._win_wake_event = None
        self._win_input_handle = None

    def _validate_interactive_startup(self, *, interactive: bool) -> None:
        if not interactive or os.name == "nt":
            return

        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError(
                "Interactive netflux TUI must be started from the main thread on POSIX "
                "because terminal resize handling uses SIGWINCH."
            )

        sigint_handler = signal.getsignal(signal.SIGINT)
        if sigint_handler is not signal.default_int_handler:
            raise RuntimeError(
                "Interactive netflux TUI requires the default Python SIGINT handler on "
                "POSIX. Detected a custom or unsupported SIGINT handler; remove it "
                "before starting the UI."
            )

    def _tick(self) -> int:
        return int((time.monotonic() - self._t0) * self.spinner_hz)

    def _seconds_until_next_tick(self, last_tick: int) -> float:
        next_tick = max(0, last_tick + 1)
        deadline = self._t0 + (next_tick / self.spinner_hz)
        return max(0.0, deadline - time.monotonic())

    def _request_wakeup(self) -> None:
        if self._wake_pipe_write is not None:
            try:
                os.write(self._wake_pipe_write, b"x")
            except OSError:
                pass
            return

        if self._win_kernel32 is not None and self._win_wake_event is not None:
            self._win_kernel32.SetEvent(self._win_wake_event)
            return

        self._thread_wakeup.set()

    def _setup_wakeup(self, *, interactive: bool) -> None:
        self._thread_wakeup.clear()
        if not interactive:
            return

        if os.name == "nt":
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            wake_event = kernel32.CreateEventW(None, True, False, None)
            if not wake_event:
                raise OSError("CreateEventW failed for UI wakeup")
            self._win_kernel32 = kernel32
            self._win_wake_event = wake_event
            self._win_input_handle = kernel32.GetStdHandle(_WIN_STD_INPUT_HANDLE)
            return

        read_fd, write_fd = os.pipe()
        os.set_blocking(read_fd, False)
        os.set_blocking(write_fd, False)
        self._wake_pipe_read = read_fd
        self._wake_pipe_write = write_fd

    def _cleanup_wakeup(self) -> None:
        if self._wake_pipe_read is not None:
            os.close(self._wake_pipe_read)
            self._wake_pipe_read = None
        if self._wake_pipe_write is not None:
            os.close(self._wake_pipe_write)
            self._wake_pipe_write = None
        if self._win_kernel32 is not None and self._win_wake_event is not None:
            self._win_kernel32.CloseHandle(self._win_wake_event)
        self._win_kernel32 = None
        self._win_wake_event = None
        self._win_input_handle = None
        self._thread_wakeup.clear()

    def _install_sigwinch(self) -> None:
        if os.name == "nt" or self._wake_pipe_write is None:
            return
        self._old_sigwinch = signal.getsignal(signal.SIGWINCH)

        def _handle_sigwinch(_signum, _frame) -> None:
            self._request_wakeup()

        signal.signal(signal.SIGWINCH, _handle_sigwinch)

    def _restore_sigwinch(self) -> None:
        if os.name == "nt" or self._old_sigwinch is None:
            return
        signal.signal(signal.SIGWINCH, self._old_sigwinch)
        self._old_sigwinch = None

    def _drain_posix_wakeup(self) -> None:
        if self._wake_pipe_read is None:
            return
        while True:
            try:
                if not os.read(self._wake_pipe_read, 4096):
                    return
            except BlockingIOError:
                return
            except OSError:
                return

    def _current_size(self) -> TerminalSize:
        cols, lines = terminal_size_token()
        return TerminalSize(columns=max(1, cols), lines=max(1, lines))

    def _render_if_needed(
        self,
        controller: SessionController,
        *,
        last_size: TerminalSize | None,
        last_tick: int,
        force_render: bool,
    ) -> tuple[TerminalSize, int, bool, bool]:
        changed = controller.pump_events()
        size = self._current_size()
        if size != last_size:
            changed = True

        wants_ticks = controller.wants_animation_ticks()
        tick = self._tick()
        if wants_ticks and tick != last_tick:
            changed = True

        if changed or force_render:
            frame = controller.render_frame(size, tick)
            ui_driver(frame)
            last_tick = tick
            force_render = False

        return size, last_tick, force_render, wants_ticks

    def run(self, controller: SessionController) -> None:
        stdout_tty = sys.stdout.isatty()
        interactive = sys.stdin.isatty() and stdout_tty
        start_attempted = False
        old_settings = None
        raised: BaseException | None = None

        try:
            self._validate_interactive_startup(interactive=interactive)
            self._setup_wakeup(interactive=interactive)
            controller.set_wakeup(self._request_wakeup)
            start_attempted = True
            controller.on_session_start(interactive=interactive)
            if controller.should_exit():
                return

            if interactive and os.name != "nt":
                if termios is None or tty is None:
                    interactive = False
                else:
                    fd = sys.stdin.fileno()
                    old_settings = termios.tcgetattr(fd)
                    tty.setcbreak(fd)
                    self._install_sigwinch()

            if interactive:
                pre_console()

            if not interactive:
                self._loop_noninteractive(controller)
            elif os.name == "nt":
                self._loop_windows(controller)
            else:
                self._loop_posix(controller, sys.stdin.fileno())
        except BaseException as exc:  # pragma: no cover - exercised through tests
            raised = exc
        finally:
            stop_error: BaseException | None = None
            if start_attempted:
                try:
                    controller.on_session_stop()
                except BaseException as exc:  # pragma: no cover - exercised through tests
                    stop_error = exc

            cleanup_error: BaseException | None = None
            try:
                self._restore_sigwinch()
                if old_settings is not None and termios is not None:
                    termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_settings)
                if stdout_tty:
                    restore_console()
            except BaseException as exc:  # pragma: no cover - exercised through tests
                cleanup_error = exc
            finally:
                self._cleanup_wakeup()

            if raised is not None:
                raise raised
            if stop_error is not None:
                raise stop_error
            if cleanup_error is not None:
                raise cleanup_error

    def _loop_noninteractive(self, controller: SessionController) -> None:
        last_size: TerminalSize | None = None
        last_tick = -1
        force_render = True

        while True:
            try:
                last_size, last_tick, force_render, wants_ticks = self._render_if_needed(
                    controller,
                    last_size=last_size,
                    last_tick=last_tick,
                    force_render=force_render,
                )
                if controller.should_exit():
                    break

                timeout = self._seconds_until_next_tick(last_tick) if wants_ticks else None
                self._thread_wakeup.wait(timeout=timeout)
                self._thread_wakeup.clear()
                force_render = True
            except KeyboardInterrupt:
                if controller.handle_interrupt():
                    break
                force_render = True

    def _loop_posix(self, controller: SessionController, fd: int) -> None:
        if select is None or self._wake_pipe_read is None:
            self._loop_noninteractive(controller)
            return

        last_size: TerminalSize | None = None
        last_tick = -1
        force_render = True

        while True:
            try:
                last_size, last_tick, force_render, wants_ticks = self._render_if_needed(
                    controller,
                    last_size=last_size,
                    last_tick=last_tick,
                    force_render=force_render,
                )
                if controller.should_exit():
                    break

                timeout = self._seconds_until_next_tick(last_tick) if wants_ticks else None
                ready, _, _ = select.select([fd, self._wake_pipe_read], [], [], timeout)
                if not ready:
                    force_render = True
                    continue

                if self._wake_pipe_read in ready:
                    self._drain_posix_wakeup()
                    force_render = True

                if fd in ready:
                    event = read_key(fd, timeout=0)
                    if event is None:
                        continue
                    if hasattr(event, "button"):
                        should_exit = controller.handle_mouse(event)
                    else:
                        should_exit = controller.handle_key(event)
                    if should_exit:
                        break
                    force_render = True
            except KeyboardInterrupt:
                if controller.handle_interrupt():
                    break
                force_render = True

    def _loop_windows(self, controller: SessionController) -> None:
        if self._win_kernel32 is None or self._win_wake_event is None or self._win_input_handle is None:
            self._loop_noninteractive(controller)
            return

        last_size: TerminalSize | None = None
        last_tick = -1
        force_render = True
        handles = (ctypes.c_void_p * 2)(self._win_input_handle, self._win_wake_event)

        while True:
            try:
                last_size, last_tick, force_render, wants_ticks = self._render_if_needed(
                    controller,
                    last_size=last_size,
                    last_tick=last_tick,
                    force_render=force_render,
                )
                if controller.should_exit():
                    break

                timeout_ms = _WIN_INFINITE
                if wants_ticks:
                    timeout_ms = max(0, int(self._seconds_until_next_tick(last_tick) * 1000))

                result = self._win_kernel32.WaitForMultipleObjects(
                    2,
                    handles,
                    False,
                    timeout_ms,
                )
                if result == _WIN_WAIT_TIMEOUT:
                    force_render = True
                    continue
                if result == _WIN_WAIT_OBJECT_0 + 1:
                    self._win_kernel32.ResetEvent(self._win_wake_event)
                    force_render = True
                    continue
                if result != _WIN_WAIT_OBJECT_0:
                    raise OSError(f"WaitForMultipleObjects failed: {result}")

                event = read_key_windows()
                if isinstance(event, InterruptEvent):
                    if controller.handle_interrupt():
                        break
                    force_render = True
                    continue
                if isinstance(event, ResizeEvent):
                    force_render = True
                    continue
                if event is None:
                    continue
                if hasattr(event, "button"):
                    should_exit = controller.handle_mouse(event)
                else:
                    should_exit = controller.handle_key(event)
                if should_exit:
                    break
                force_render = True
            except KeyboardInterrupt:
                if controller.handle_interrupt():
                    break
                force_render = True
