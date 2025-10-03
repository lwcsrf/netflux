import threading
import time
from abc import ABC, abstractmethod
from multiprocessing.synchronize import Event
from typing import Callable, Generic, Optional, TypeVar

from ..core import Node, NodeView, TerminalNodeStates

RenderType = TypeVar("RenderType")

class Render(ABC, Generic[RenderType]):
    """
    Abstraction of a renderer that translates `NodeView` and its subtree into a
    visualization of type `RenderType`.

    Consumers provide a concrete `Render` to `start_view_loop()` which will call
    `render(view)` whenever a new `NodeView` is available, and `render(None)`
    when no new update arrived but the UI tick (`update_interval`) elapsed.
    
    Subclasses should:
    - Treat `view != None` as "new immutable snapshot is available". They may cache it
      internally (e.g., as the current tree to render).
    - Treat `view == None` as "no change in ViewNode since last call". That allows
      renderers to animate purely as a function of time (spinners, progress tweens, etc.)
      even without new runtime updates.
    - Return any representation type `RenderType` (string, structured object, pixels, …)
      that the application's `ui_driver` callback accepts.

    Subclasses are called from a single background thread by default and need not be
    thread-safe unless shared.
    """
    @abstractmethod
    def render(self, view: Optional[NodeView]) -> RenderType:
        raise NotImplementedError


def start_view_loop(
    node: Node,
    *,
    render: Render[RenderType],
    ui_driver: Callable[[RenderType], None],
    update_interval: float = 0.1,
) -> threading.Thread:
    """
    Instead of consumers creating their own loop around `node.watch()`, they can use this
    method to get visualization updates about the task tree progression for UI purpose.
    The pattern is:

    - Consumer thread responds to signals or UI directives and can cancel the top-level task.
    - Consumer uses `node: Node = Runtime.invoke(func, cancel_event)` to launch the
      top-level task.
    - Consumer invokes this method with the `node`, plus:
        1. `render` is the pluggable way in which the NodeView and its subtree would get
           rendered. You instantiate the desired Render (e.g. TexRenderType) and give
           this instance to this method. Render[RenderType].render(NodeView) will return
           a representation of type `RenderType`., which is what the `ui_driver` expects
           as input.
        2. `ui_driver` is a callback that accepts the `RenderType` rendering and will
           store it somewhere for future visibility or update the UI presentation
           immediately (depending on the complexity of the UI) for the app user. The app
           author needs to decide how they use the rendering to update their UI and
           provide that logic in the callable.
        3. `update_interval`: seconds in float.
    - A thread is created to watch the Node and its subtree. Each time `node.watch()`
      returns a new NodeView or the watch expires, it will invoke
      `ui_driver(Render.render(node_view))`.
    - The thread is responsive to the top-level task reaching a terminal state (success,
      failure, or cancellation). In that case, the loop exits and the thread exits.
    - A reference to the `Thread` is returned so that the loop-invoking thread may join
      it for clean-up synchronization.

    Animation behavior:
    - The loop uses the timeout feature of `node.watch()` to wake up at `update_interval`
      boundaries when no new view is available. On these wake-ups, the loop will call
      `render(None)` and pass its result to `ui_driver`. This signals to the `Render`
      that the view state did not change but it may still choose to render differently
      (e.g., advance a spinner) based on current time.
    """
    def _loop() -> None:
        prev_seq = 0
        next_at = time.monotonic()
        interval = max(0.0, float(update_interval))
        last_view: Optional[NodeView] = None

        while True:
            now = time.monotonic()
            # ensure periodic wake-ups for animation by using watch timeout
            remaining = max(0.0, next_at - now)
            view = node.watch(as_of_seq=prev_seq, timeout=remaining)
            if view is not None:
                prev_seq = view.update_seqnum
                last_view = view

            payload = render.render(view)
            ui_driver(payload)

            # Exit condition: once the node reaches a terminal state, stop the loop,
            # regardless of cancellation. If cancellation is requested earlier,
            # continue rendering until terminal to visualize graceful shutdown.
            current = view if view is not None else last_view
            if current is not None and current.state in TerminalNodeStates:
                break

            next_at += interval

    t = threading.Thread(target=_loop, name="netflux-view-loop", daemon=True)
    t.start()
    return t
