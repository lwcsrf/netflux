from __future__ import annotations

import ctypes
import io
import logging
import os
import signal
import sys
import threading
import time

from ._contracts import SessionController, TerminalSize
from ._terminal_io import (
    InterruptEvent,
    ResizeEvent,
    posix_pending_input_timeout,
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

logger = logging.getLogger(__name__)


class ConsoleSessionDriver:
    def __init__(self, *, spinner_hz: float = 10.0) -> None:
        self.spinner_hz = max(1.0, float(spinner_hz))
        self._t0 = time.monotonic()
        self._stage = "initialization"
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

    def _stdin_fileno(self) -> int | None:
        try:
            return sys.stdin.fileno()
        except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
            return None

    def _render_if_needed(
        self,
        controller: SessionController,
        *,
        last_size: TerminalSize | None,
        last_tick: int,
        force_render: bool,
    ) -> tuple[TerminalSize, int, bool, bool]:
        self._stage = f"{type(controller).__name__}.pump_events"
        changed = controller.pump_events()
        size = self._current_size()
        if size != last_size:
            changed = True

        self._stage = f"{type(controller).__name__}.wants_animation_ticks"
        wants_ticks = controller.wants_animation_ticks()
        tick = self._tick()
        if wants_ticks and tick != last_tick:
            changed = True

        if changed or force_render:
            self._stage = f"{type(controller).__name__}.render_frame"
            frame = controller.render_frame(size, tick)
            self._stage = "ui_driver"
            ui_driver(frame)
            last_tick = tick
            force_render = False

        return size, last_tick, force_render, wants_ticks

    def run(self, controller: SessionController) -> None:
        stdout_tty = sys.stdout.isatty()
        interactive = sys.stdin.isatty() and stdout_tty
        start_attempted = False
        stdin_fd: int | None = None
        old_settings = None
        raised: BaseException | None = None

        try:
            self._stage = "interactive startup detection"
            if interactive and os.name != "nt":
                if termios is None or tty is None:
                    interactive = False
                else:
                    stdin_fd = self._stdin_fileno()
                    if stdin_fd is None:
                        interactive = False
            self._stage = "interactive startup validation"
            self._validate_interactive_startup(interactive=interactive)
            self._stage = "wakeup setup"
            self._setup_wakeup(interactive=interactive)
            self._stage = f"{type(controller).__name__}.set_wakeup"
            controller.set_wakeup(self._request_wakeup)
            start_attempted = True
            self._stage = f"{type(controller).__name__}.on_session_start"
            controller.on_session_start(interactive=interactive)
            self._stage = f"{type(controller).__name__}.should_exit"
            if controller.should_exit():
                return

            if interactive and os.name != "nt":
                assert stdin_fd is not None
                self._stage = "terminal cbreak setup"
                old_settings = termios.tcgetattr(stdin_fd)
                tty.setcbreak(stdin_fd)
                self._stage = "SIGWINCH handler installation"
                self._install_sigwinch()

            if interactive:
                self._stage = "console initialization"
                pre_console()

            if not interactive:
                self._loop_noninteractive(controller)
            elif os.name == "nt":
                self._loop_windows(controller)
            else:
                assert stdin_fd is not None
                self._loop_posix(controller, stdin_fd)
        except BaseException as exc:  # pragma: no cover - exercised through tests
            raised = exc
            if not isinstance(exc, KeyboardInterrupt):
                logger.exception(
                    "Console session driver failed during %s.",
                    self._stage,
                )
        finally:
            stop_error: BaseException | None = None
            if start_attempted:
                try:
                    self._stage = f"{type(controller).__name__}.on_session_stop"
                    controller.on_session_stop()
                except BaseException as exc:  # pragma: no cover - exercised through tests
                    stop_error = exc
                    logger.exception(
                        "Console session driver failed during %s.",
                        self._stage,
                    )

            cleanup_error: BaseException | None = None
            try:
                self._stage = "SIGWINCH handler restoration"
                self._restore_sigwinch()
                if old_settings is not None and termios is not None and stdin_fd is not None:
                    self._stage = "terminal attribute restoration"
                    termios.tcsetattr(stdin_fd, termios.TCSADRAIN, old_settings)
                if stdout_tty:
                    self._stage = "console restoration"
                    restore_console()
            except BaseException as exc:  # pragma: no cover - exercised through tests
                cleanup_error = exc
                logger.exception(
                    "Console session driver failed during %s.",
                    self._stage,
                )
            finally:
                self._stage = "wakeup cleanup"
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
                self._stage = f"{type(controller).__name__}.should_exit"
                if controller.should_exit():
                    break

                timeout = self._seconds_until_next_tick(last_tick) if wants_ticks else None
                self._stage = "noninteractive wake wait"
                self._thread_wakeup.wait(timeout=timeout)
                self._thread_wakeup.clear()
                force_render = True
            except KeyboardInterrupt:
                self._stage = f"{type(controller).__name__}.handle_interrupt"
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
                self._stage = f"{type(controller).__name__}.should_exit"
                if controller.should_exit():
                    break

                timeout = self._seconds_until_next_tick(last_tick) if wants_ticks else None
                pending_timeout = posix_pending_input_timeout()
                if pending_timeout is not None:
                    timeout = pending_timeout if timeout is None else min(timeout, pending_timeout)
                self._stage = "posix input wait"
                ready, _, _ = select.select([fd, self._wake_pipe_read], [], [], timeout)
                if not ready:
                    pending_timeout = posix_pending_input_timeout()
                    if pending_timeout is not None and pending_timeout <= 0.0:
                        self._stage = "posix read_key"
                        event = read_key(fd, timeout=0)
                        if event is not None:
                            if hasattr(event, "button"):
                                self._stage = f"{type(controller).__name__}.handle_mouse"
                                should_exit = controller.handle_mouse(event)
                            else:
                                self._stage = f"{type(controller).__name__}.handle_key"
                                should_exit = controller.handle_key(event)
                            if should_exit:
                                break
                    force_render = True
                    continue

                if self._wake_pipe_read in ready:
                    self._stage = "posix wake drain"
                    self._drain_posix_wakeup()
                    force_render = True

                if fd in ready:
                    self._stage = "posix read_key"
                    event = read_key(fd, timeout=0)
                    if event is None:
                        continue
                    if hasattr(event, "button"):
                        self._stage = f"{type(controller).__name__}.handle_mouse"
                        should_exit = controller.handle_mouse(event)
                    else:
                        self._stage = f"{type(controller).__name__}.handle_key"
                        should_exit = controller.handle_key(event)
                    if should_exit:
                        break
                    force_render = True
            except KeyboardInterrupt:
                self._stage = f"{type(controller).__name__}.handle_interrupt"
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
                self._stage = f"{type(controller).__name__}.should_exit"
                if controller.should_exit():
                    break

                timeout_ms = _WIN_INFINITE
                if wants_ticks:
                    timeout_ms = max(0, int(self._seconds_until_next_tick(last_tick) * 1000))

                self._stage = "windows input wait"
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

                self._stage = "windows read_key"
                event = read_key_windows()
                if isinstance(event, InterruptEvent):
                    self._stage = f"{type(controller).__name__}.handle_interrupt"
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
                    self._stage = f"{type(controller).__name__}.handle_mouse"
                    should_exit = controller.handle_mouse(event)
                else:
                    self._stage = f"{type(controller).__name__}.handle_key"
                    should_exit = controller.handle_key(event)
                if should_exit:
                    break
                force_render = True
            except KeyboardInterrupt:
                self._stage = f"{type(controller).__name__}.handle_interrupt"
                if controller.handle_interrupt():
                    break
                force_render = True
