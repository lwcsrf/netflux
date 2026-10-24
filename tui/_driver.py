from __future__ import annotations

import ctypes
import io
import logging
import os
import signal
import sys
import threading
import time

from ._contracts import InterruptMode, SessionController, TerminalSize
from ._terminal_io import (
    ConsoleBusyError,
    InterruptEvent,
    ResizeEvent,
    discard_pending_input,
    enable_windows_terminal_interrupt,
    posix_pending_input_timeout,
    pre_console,
    prepare_windows_console_input,
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
        self._wakeup_lock = threading.Lock()
        self._wakeup_closed = False
        self._wake_pipe_read: int | None = None
        self._wake_pipe_write: int | None = None
        self._old_sigwinch = None
        self._old_sigint = None
        self._pending_interrupt = False
        self._interrupt_mode: InterruptMode = "defer"
        self._force_interrupt_enabled = False
        self._discard_modal_typeahead = False
        self._terminal_fd: int | None = None
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
        # Watchers can finish after on_session_stop; serialize their final wake
        # with handle disposal so a recycled descriptor can never be touched.
        with self._wakeup_lock:
            if self._wakeup_closed:
                return
            if not self._wake_os_handle():
                self._thread_wakeup.set()

    def _wake_os_handle(self) -> bool:
        # Signal handlers call this directly. Never acquire a Python lock here:
        # a signal may interrupt the main thread while it already holds one.
        if self._wakeup_closed:
            return False
        write_fd = self._wake_pipe_write
        if write_fd is not None:
            try:
                os.write(write_fd, b"x")
            except OSError:
                pass
            return True

        kernel32, wake_event = self._win_kernel32, self._win_wake_event
        if kernel32 is not None and wake_event is not None:
            kernel32.SetEvent(wake_event)
            return True
        return False

    @staticmethod
    def _configure_windows_wakeup_api(kernel32) -> None:
        # ctypes otherwise defaults to c_int, truncating HANDLEs on 64-bit Windows.
        signatures = {
            "CreateEventW": ([ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_wchar_p], ctypes.c_void_p),
            "GetStdHandle": ([ctypes.c_ulong], ctypes.c_void_p),
            "SetEvent": ([ctypes.c_void_p], ctypes.c_int),
            "ResetEvent": ([ctypes.c_void_p], ctypes.c_int),
            "CloseHandle": ([ctypes.c_void_p], ctypes.c_int),
            "WaitForMultipleObjects": (
                [ctypes.c_ulong, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, ctypes.c_ulong],
                ctypes.c_ulong,
            ),
        }
        for name, (argtypes, restype) in signatures.items():
            function = getattr(kernel32, name)
            function.argtypes = argtypes
            function.restype = restype

    def _setup_wakeup(self, *, interactive: bool) -> None:
        with self._wakeup_lock:
            self._wakeup_closed = False
            self._thread_wakeup.clear()
            if not interactive:
                return

            if os.name == "nt":
                kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
                self._configure_windows_wakeup_api(kernel32)
                wake_event = kernel32.CreateEventW(None, True, False, None)
                if not wake_event:
                    raise OSError("CreateEventW failed for UI wakeup")
                self._win_kernel32 = kernel32
                self._win_wake_event = wake_event
                self._win_input_handle = kernel32.GetStdHandle(_WIN_STD_INPUT_HANDLE)
                if self._win_input_handle in (None, 0, ctypes.c_void_p(-1).value):
                    raise OSError("GetStdHandle failed for UI input")
                return

            self._wake_pipe_read, self._wake_pipe_write = os.pipe()
            os.set_blocking(self._wake_pipe_read, False)
            os.set_blocking(self._wake_pipe_write, False)

    def _cleanup_wakeup(self) -> None:
        error: BaseException | None = None
        with self._wakeup_lock:
            # Signals are handled on this same thread. Once closed is set their
            # lock-free wake path stops touching resources, even during disposal.
            self._wakeup_closed = True
            read_fd, write_fd = self._wake_pipe_read, self._wake_pipe_write
            kernel32, wake_event = self._win_kernel32, self._win_wake_event
            self._wake_pipe_read = None
            self._wake_pipe_write = None
            self._win_kernel32 = None
            self._win_wake_event = None
            self._win_input_handle = None
            for fd in (read_fd, write_fd):
                if fd is not None:
                    try:
                        os.close(fd)
                    except BaseException as exc:
                        if error is None:
                            error = exc
            if kernel32 is not None and wake_event is not None:
                try:
                    kernel32.CloseHandle(wake_event)
                except BaseException as exc:
                    if error is None:
                        error = exc
            self._thread_wakeup.clear()
        if error is not None:
            raise error

    def _install_sigwinch(self) -> None:
        if os.name == "nt" or self._wake_pipe_write is None:
            return
        self._old_sigwinch = signal.getsignal(signal.SIGWINCH)

        def _handle_sigwinch(_signum, _frame) -> None:
            self._wake_os_handle()

        signal.signal(signal.SIGWINCH, _handle_sigwinch)

    def _restore_sigwinch(self) -> None:
        if os.name == "nt" or self._old_sigwinch is None:
            return
        signal.signal(signal.SIGWINCH, self._old_sigwinch)
        self._old_sigwinch = None

    def _install_sigint(self) -> None:
        # Defer interactive interrupts until cancellation is committed: raising
        # from a signal handler can unwind rendering or controller mutations.
        if threading.current_thread() is not threading.main_thread():
            return
        self._old_sigint = signal.getsignal(signal.SIGINT)

        def _handle_sigint(_signum, _frame) -> None:
            if self._interrupt_mode == "ignore":
                return
            if self._interrupt_mode == "force":
                # Latch off repeats while unwinding into protected cleanup.
                self._interrupt_mode = "ignore"
                raise KeyboardInterrupt
            self._pending_interrupt = True
            self._wake_os_handle()

        signal.signal(signal.SIGINT, _handle_sigint)

    def _restore_sigint(self) -> None:
        if self._old_sigint is not None:
            signal.signal(signal.SIGINT, self._old_sigint)
            self._old_sigint = None
        self._pending_interrupt = False
        self._interrupt_mode = "defer"
        self._force_interrupt_enabled = False
        self._discard_modal_typeahead = False

    def _set_interrupt_mode(self, mode: InterruptMode) -> None:
        # The controller only arms force after committing every cancel event.
        # Signals received in the modal must not carry into the resumed state.
        if mode == "ignore":
            self._interrupt_mode = mode
            self._pending_interrupt = False
            self._discard_modal_typeahead = True
            return
        if mode == "force":
            self._enable_terminal_interrupt()
            self._force_interrupt_enabled = True
        if self._interrupt_mode == "ignore":
            # A raw Ctrl-C pressed in the modal may still be in the OS input
            # queue behind Enter. Drop modal typeahead before interpreting it
            # in the resumed mode. Rearm physical signals first so no new raw
            # Ctrl-C can be queued between the discard and force enable.
            if self._win_input_handle is not None or self._terminal_fd is not None:
                discard_pending_input(self._terminal_fd)
        self._pending_interrupt = False
        self._interrupt_mode = mode
        self._discard_modal_typeahead = False

    def _enable_terminal_interrupt(self) -> None:
        # Windows supports UI threads other than main. Keep raw input there:
        # a console signal would instead interrupt the unrelated main thread.
        if self._old_sigint is None:
            return
        if os.name == "nt":
            enable_windows_terminal_interrupt()
        elif self._terminal_fd is not None and termios is not None:
            settings = termios.tcgetattr(self._terminal_fd)
            settings[3] |= termios.ISIG
            settings[6][termios.VINTR] = b"\x03"
            termios.tcsetattr(self._terminal_fd, termios.TCSANOW, settings)

    def _dispatch_pending_interrupt(self, controller: SessionController) -> bool | None:
        if not self._pending_interrupt:
            return None
        self._pending_interrupt = False
        self._stage = f"{type(controller).__name__}.handle_interrupt"
        try:
            return controller.handle_interrupt()
        except KeyboardInterrupt:
            self._interrupt_mode = "ignore"
            raise

    def _dispatch_input(
        self,
        controller: SessionController,
        event: object,
        *,
        rendered_size: TerminalSize | None = None,
    ) -> bool:
        if rendered_size is not None and self._current_size() != rendered_size:
            # A resize may arrive alongside a key or while decoding an escape
            # sequence. Refresh geometry before Enter or mouse hit testing.
            self._render_if_needed(
                controller, last_size=rendered_size, last_tick=-1, force_render=True,
            )
            self._stage = f"{type(controller).__name__}.should_exit"
            if controller.should_exit():
                return True
        if self._pending_interrupt:
            # Give an interrupt priority over input read during its delivery.
            return False
        if isinstance(event, InterruptEvent) or event == "\x03":
            if self._interrupt_mode != "ignore":
                self._pending_interrupt = True
            return False
        if hasattr(event, "button"):
            self._stage = f"{type(controller).__name__}.handle_mouse"
            return controller.handle_mouse(event)
        self._stage = f"{type(controller).__name__}.handle_key"
        return controller.handle_key(event)

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

    @staticmethod
    def _disable_terminal_interrupt(fd: int) -> None:
        assert termios is not None
        disabled = os.fpathconf(fd, "PC_VDISABLE")
        if not 0 <= disabled <= 255:
            raise RuntimeError("Terminal does not support disabling the interrupt character.")
        settings = termios.tcgetattr(fd)
        # Receive Ctrl-C as input rather than broadcasting SIGINT to foreground
        # subprocesses before the user confirms. Other signal keys stay intact.
        settings[6][termios.VINTR] = bytes([disabled])
        termios.tcsetattr(fd, termios.TCSANOW, settings)

    @staticmethod
    def _protect_terminal_output(fd: int) -> None:
        assert termios is not None
        settings = termios.tcgetattr(fd)
        # Keep consuming Ctrl-C as a signal, without the terminal driver
        # discarding screen restoration bytes while Python ignores SIGINT.
        settings[3] |= termios.NOFLSH
        termios.tcsetattr(fd, termios.TCSANOW, settings)

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
            if self._discard_modal_typeahead:
                # Only input generated after the first confirmation frame may
                # authorize cancellation. Flush after writing it to cover both
                # old typeahead and keys arriving while that write was blocked.
                self._stage = "confirmation input discard"
                if self._win_input_handle is not None or self._terminal_fd is not None:
                    discard_pending_input(self._terminal_fd)
                self._discard_modal_typeahead = False
            last_tick = tick
            force_render = False

        return size, last_tick, force_render, wants_ticks

    def run(self, controller: SessionController) -> None:
        stdout_tty = sys.stdout.isatty()
        restore_console_on_exit = False
        interactive = sys.stdin.isatty() and stdout_tty
        start_attempted = False
        stdin_fd: int | None = None
        old_settings = None
        windows_input_prepared = False
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
            if interactive:
                self._stage = "SIGINT handler installation"
                self._install_sigint()
            if interactive and os.name == "nt":
                self._stage = "console input initialization"
                restore_console_on_exit = True
                try:
                    prepare_windows_console_input()
                except ConsoleBusyError:
                    # Acquisition changed nothing; cleanup belongs to the
                    # session that already owns this process-wide console.
                    restore_console_on_exit = False
                    raise
                windows_input_prepared = True
            if interactive and os.name != "nt":
                assert stdin_fd is not None
                # Shield foreground work before a startup callback can block
                # or start watchers. The display is still untouched on veto.
                self._stage = "terminal cbreak setup"
                old_settings = termios.tcgetattr(stdin_fd)
                tty.setcbreak(stdin_fd)
                self._stage = "terminal interrupt character setup"
                self._terminal_fd = stdin_fd
                self._disable_terminal_interrupt(stdin_fd)
                self._stage = "SIGWINCH handler installation"
                self._install_sigwinch()
            self._stage = f"{type(controller).__name__}.set_wakeup"
            controller.set_wakeup(self._request_wakeup)
            if interactive:
                self._stage = f"{type(controller).__name__}.set_interrupt_mode_callback"
                controller.set_interrupt_mode_callback(self._set_interrupt_mode)
            start_attempted = True
            self._stage = f"{type(controller).__name__}.on_session_start"
            controller.on_session_start(interactive=interactive)
            self._stage = f"{type(controller).__name__}.should_exit"
            if controller.should_exit():
                return

            if interactive:
                self._stage = "console initialization"
                restore_console_on_exit = True
                if windows_input_prepared:
                    pre_console(windows_input_prepared=True)
                else:
                    pre_console()

            if not interactive:
                # Noninteractive controllers can still render to TTY stdout;
                # ui_driver acquires the console lazily on their first frame.
                restore_console_on_exit = stdout_tty
                self._loop_noninteractive(controller)
            elif os.name == "nt":
                self._loop_windows(controller)
            else:
                assert stdin_fd is not None
                self._loop_posix(controller, stdin_fd)
            self._interrupt_mode = "ignore"
        except BaseException as exc:  # pragma: no cover - exercised through tests
            self._interrupt_mode = "ignore"
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
                    try:
                        if self._force_interrupt_enabled:
                            self._interrupt_mode = "force"
                        controller.on_session_stop()
                    finally:
                        # Stop callbacks remain interruptible after confirmation;
                        # terminal/handle disposal must finish despite repeats.
                        self._interrupt_mode = "ignore"
                except BaseException as exc:  # pragma: no cover - exercised through tests
                    stop_error = exc
                    if not isinstance(exc, KeyboardInterrupt):
                        logger.exception(
                            "Console session driver failed during %s.",
                            self._stage,
                        )

            cleanup_error: BaseException | None = None
            cleanup_steps = []
            if (
                self._force_interrupt_enabled
                and old_settings is not None and termios is not None and stdin_fd is not None
            ):
                # Ignoring SIGINT in Python does not stop the terminal driver
                # from flushing output on Ctrl-C. Keep the current no-echo
                # settings and protect output until restoration is written.
                cleanup_steps.append(("terminal output protection", lambda: self._protect_terminal_output(
                    stdin_fd,
                )))
            if restore_console_on_exit:
                cleanup_steps.append(("console restoration", restore_console))
            cleanup_steps.append(("wakeup cleanup", self._cleanup_wakeup))
            if old_settings is not None and termios is not None and stdin_fd is not None:
                # Do not wait for a competing terminal writer to drain. The
                # screen restoration has already been flushed. Keep NOFLSH
                # until the final handoff to the caller's terminal and SIGINT.
                cleanup_steps.append(("terminal attribute restoration", lambda: termios.tcsetattr(
                    stdin_fd, termios.TCSANOW, old_settings,
                )))
            cleanup_steps.extend([
                ("SIGWINCH handler restoration", self._restore_sigwinch),
                ("SIGINT handler restoration", self._restore_sigint),
            ])
            for stage, cleanup in cleanup_steps:
                self._stage = stage
                try:
                    cleanup()
                except BaseException as exc:  # pragma: no cover - exercised through tests
                    if cleanup_error is None:
                        cleanup_error = exc
                    logger.exception("Console session driver failed during %s.", self._stage)
            self._terminal_fd = None

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
            # Deliberate KeyboardInterrupt from the controller's force-exit
            # path must escape rather than being dispatched a second time.
            interrupt_result = self._dispatch_pending_interrupt(controller)
            if interrupt_result is not None:
                force_render = True
            if interrupt_result:
                break
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

                if self._pending_interrupt:
                    force_render = True
                    continue

                timeout = self._seconds_until_next_tick(last_tick) if wants_ticks else None
                pending_timeout = posix_pending_input_timeout()
                if pending_timeout is not None:
                    timeout = pending_timeout if timeout is None else min(timeout, pending_timeout)
                self._stage = "posix input wait"
                ready, _, _ = select.select([fd, self._wake_pipe_read], [], [], timeout)
                if self._pending_interrupt:
                    force_render = True
                    continue
                if not ready:
                    pending_timeout = posix_pending_input_timeout()
                    if pending_timeout is not None and pending_timeout <= 0.0:
                        self._stage = "posix read_key"
                        event = read_key(fd, timeout=0)
                        if event is not None:
                            if self._dispatch_input(controller, event, rendered_size=last_size):
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
                    if self._dispatch_input(controller, event, rendered_size=last_size):
                        break
                    force_render = True
            except KeyboardInterrupt:
                if self._force_interrupt_enabled:
                    self._interrupt_mode = "ignore"
                    raise
                self._pending_interrupt = True
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
            interrupt_result = self._dispatch_pending_interrupt(controller)
            if interrupt_result is not None:
                force_render = True
            if interrupt_result:
                break
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

                if self._pending_interrupt:
                    force_render = True
                    continue

                timeout_ms = _WIN_INFINITE
                if wants_ticks:
                    timeout_ms = max(0, int(self._seconds_until_next_tick(last_tick) * 1000))
                if self._old_sigint is not None or self._force_interrupt_enabled:
                    # Python's Windows signal handler cannot run inside an
                    # infinite ctypes wait. External signals need delivery even
                    # before confirmation, when physical Ctrl-C is raw input.
                    timeout_ms = min(timeout_ms, 100)

                self._stage = "windows input wait"
                result = self._win_kernel32.WaitForMultipleObjects(
                    2,
                    handles,
                    False,
                    timeout_ms,
                )
                if self._pending_interrupt:
                    force_render = True
                    continue
                if result == _WIN_WAIT_TIMEOUT:
                    # Signal polling alone should not redraw an idle screen.
                    force_render = wants_ticks
                    continue
                if result == _WIN_WAIT_OBJECT_0 + 1:
                    self._win_kernel32.ResetEvent(self._win_wake_event)
                    force_render = True
                    continue
                if result != _WIN_WAIT_OBJECT_0:
                    raise OSError(f"WaitForMultipleObjects failed: {result}")

                self._stage = "windows read_key"
                event = read_key_windows()
                if isinstance(event, ResizeEvent):
                    force_render = True
                    continue
                if event is None:
                    continue
                if self._dispatch_input(controller, event, rendered_size=last_size):
                    break
                force_render = True
            except KeyboardInterrupt:
                if self._force_interrupt_enabled:
                    self._interrupt_mode = "ignore"
                    raise
                self._pending_interrupt = True
                force_render = True
