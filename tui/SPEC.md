# `tui` UI Specification

Status: current implementation contract for the `tui` package in this branch.

Note:
- `tui` is an auxiliary convenience package rather than one of the project's golden packages.
- Unlike `core`, `func_lib`, and `providers`, it is not held to the same quality or maintainability bar. It was mostly auto-generated on top of the framework's clean abstractions. It has undergone the least review compared to the other packages. It is most like a "throwaway frontend" concept. That said, considerable iteration went into the current implementation.
- This spec is therefore primarily a practical contract for automatic code generation by agents and for keeping future `tui` changes aligned with the current implemented behavior.

Scope:
- This document describes the behavior implemented by `tui/console.py`, `tui/tui.py`, and the internal driver/controller helpers under `tui/`.
- It covers the two supported UI entrypoints: `ConsoleRender.run(node)` and `TUI(runtime).run()`.
- It describes current behavior and explicit v1 limitations. It is intended to replace the earlier branch-design draft now that the implemented behavior is captured here.

## 1. Public Surface

### 1.1 `ConsoleRender`

`ConsoleRender` is the stateful single-tree renderer/controller for one root execution tree.

It exposes:
- `run(node)` for the legacy standalone single-tree session.
- `assign_view(view)` to update the renderer with a newer `NodeView`.
- `render_body(width=..., height=..., ...)` to render only the tree body for a caller-supplied viewport.
- `apply_action(action)` for semantic right-pane actions.
- `handle_mouse_event(x, y, button=...)` for right-pane-local mouse input.
- `selected_tree_status()` for structured selected-tree status data.
- `right_pane_context()` for structured right-pane interaction context.

`ConsoleRender` renders only viewport-local tree content. It does not own:
- full-terminal frame composition,
- the bottom status bar,
- key decoding,
- terminal enter/restore operations,
- the shared event loop.

### 1.2 `TUI`

`TUI(runtime)` is the multi-root terminal UI.

It:
- discovers launchable functions from `runtime.invocable_functions`,
- owns one `ConsoleRender` per launched root,
- tracks only roots launched through that `TUI` instance,
- owns per-root cancel events for roots it launches,
- owns multi-pane layout, launch-form state, run selection, unread state, and the shared bottom bar for the multi-pane session.

### 1.3 Internal Architecture

The current implementation is split into:
- `ConsoleSessionDriver`: shared terminal/session driver.
- `SingleTreeConsoleController`: standalone controller used by `ConsoleRender.run(node)`.
- `ConsoleRender`: viewport-scoped single-tree renderer/controller.
- `TUI`: multi-pane controller.
- `_controller_helpers`: bottom-bar composition and layout helpers.
- `_contracts`: small controller/status protocols and dataclasses.
- `_terminal_io`: terminal setup/restore and low-level input decoding.

These internal modules are part of the implementation architecture, not a separate public API commitment.

## 2. Shared Terminal Session Rules

The shared terminal driver is wake-driven.

It redraws when:
- a controller reports queued runtime/view updates,
- the terminal size changes,
- an animation tick advances while the controller wants animation ticks,
- input handling changes controller state.

It does not use recurring polling to discover runtime updates or keyboard/mouse input.

Interactive terminal setup:
- enters the alternate screen,
- hides the cursor,
- disables line wrapping,
- clears scrollback,
- enables mouse reporting,
- restores terminal state on exit.

Cross-platform support:
- POSIX and Windows interactive paths are supported.
- On POSIX, interactive startup must happen on the process main thread.
- On POSIX, interactive startup requires the default Python `SIGINT` handler.
- On Windows, resize notifications and mouse input are enabled through the console input mode.

Non-interactive handling:
- `ConsoleRender.run(node)` supports non-interactive stdout/stdin combinations and avoids interactive terminal control behavior when stdout is not a TTY.
- `TUI(runtime).run()` currently treats non-interactive startup as a no-op session and exits immediately.

Threading model:
- terminal writes happen only on the shared driver thread,
- watcher threads never write to the terminal,
- watcher threads communicate root updates back to the controller through a thread-safe queue and a wakeup callback,
- watcher threads are daemon threads and are not joined during normal shutdown.

## 3. `ConsoleRender` Tree Rendering Contract

`ConsoleRender` consumes `NodeView` snapshots and renders a flattened, scrollable tree view.

### 3.1 Viewport Rendering

`render_body(width, height, ...)`:
- renders only the body area,
- respects the exact viewport width and height supplied by the caller,
- does not emit cursor-home, end-of-line clear, or full-frame terminal control sequences,
- returns a full-height string suitable for direct placement into a larger frame,
- shows a `(waiting for data...)` placeholder before the first assigned snapshot is rendered.

### 3.2 Navigation State

Per-renderer state includes:
- cursor position,
- scroll offset,
- follow mode,
- collapse overrides,
- selection-restoration anchors.

This state persists for the lifetime of the renderer and survives redraws and snapshot updates for the same root.

Manual navigation disables follow mode.
`go_bottom()` re-enables follow mode when the cursor reaches the final visible line.

If a different root id is assigned to the same renderer, cached per-tree state is reset. The intended usage is still one renderer per root.

### 3.3 Expand/Collapse Defaults

Default presentation:
- agent nodes start expanded,
- code-function nodes start collapsed,
- detail sections start collapsed.

User overrides persist for the lifetime of the renderer instance.

`expand_all` and `collapse_all` apply only to node-header collapse state. They do not expand or collapse every transcript/detail subsection.

### 3.4 Tree/Transcript Semantics

The rendered tree preserves:
- node hierarchy,
- transcript order,
- transcript-child correlation through `NodeView.transcript_child_map`,
- unmatched child nodes as visible orphan children after transcript-derived content.

Tool calls behave as follows:
- if a `ToolUsePart` maps to a real child node, that child subtree is rendered inline at the matching transcript position,
- if no child node exists, the tool call remains visible as a synthetic expandable function row.

### 3.5 Status and Interaction Context

`selected_tree_status()` exposes enough data for controller-owned status bars to show:
- cursor line and total lines,
- selected root state,
- cancel-pending indication,
- total tree token bill.

`right_pane_context()` exposes enough data for controller-owned shortcut formatting to decide whether the selected tree currently has:
- any lines,
- useful expand/collapse actions,
- useful agent-jump actions,
- follow mode enabled,
- a terminal root state.

## 4. Standalone `ConsoleRender.run(node)`

`ConsoleRender.run(node)` remains the supported public standalone single-tree entrypoint.

### 4.1 Preconditions

The root `node` must already have an invoke-time cooperative `cancel_event`.

If `node.cancel_event is None`, `run(node)` fails fast with a clear `ValueError`.

`ConsoleRender.run(node)` does not synthesize or replace the root cancel event after the node already exists.

### 4.2 Session Behavior

Standalone mode:
- displays exactly one root tree,
- uses the full body area for the tree viewport,
- renders live while the root is running,
- transitions into post-completion browse mode when an interactive session reaches a terminal root state,
- resets the tree into browse mode by disabling follow mode and moving the cursor to the top.

The standalone controller owns the full bottom bar and composes it from:
- standalone shortcut hints,
- selected-tree status data from the renderer.

### 4.3 Keyboard and Mouse

Standalone key bindings:
- `j` / Down: move down
- `k` / Up: move up
- Space / Enter: toggle
- `g` / `G`: go to enclosing node top/bottom
- Page Up / Page Down: page navigation
- `n` / `N`: next/previous visible agent
- `c`: collapse enclosing agent
- `E` / `C`: expand all / collapse all
- `q` / Escape: leave the post-completion browser only
- `Ctrl+C`: cooperative cancellation while live

Standalone mouse behavior:
- left-click on another visible row moves the cursor there,
- left-click on the current row triggers toggle behavior,
- left-click on the fold column toggles that row,
- wheel up/down scrolls by the fixed row increment,
- middle/right click have no effect.

### 4.4 Cancel and Exit Semantics

While live:
- the first `Ctrl+C` sets the root cancel event and keeps the session alive until a terminal update arrives,
- a later `Ctrl+C` takes the stronger exit path.

After a cooperative cancel request:
- the final terminal state is still rendered,
- the standalone session exits after that terminal render,
- it does not enter the normal post-completion browse loop.

### 4.5 Standalone Status Bar

The standalone bottom bar:
- reserves the bottom row,
- shows cursor position, selected-root state, and total tree token bill when available,
- preserves state/status data ahead of shortcut text under narrow widths,
- shows `^C:cancel` while live and cancelable,
- shows `q:quit` in terminal browse mode.

Token-bill formatting uses the compact per-provider format currently implemented by `ConsoleRender`.

### 4.6 Standalone Minimum Size

Standalone minimum size is:
- 40 columns,
- 6 total rows.

When smaller than that, the standalone controller renders a dedicated "terminal too small" frame and suspends normal browse interactions until the terminal is resized large enough again.

In standalone browse mode, `q` still exits from that too-small frame.

## 5. Multi-Pane `TUI(runtime)`

`TUI` is the multi-root interactive session controller.

### 5.1 Construction and Scope

`TUI`:
- accepts a `Runtime`,
- reads launchable functions from `runtime.invocable_functions`,
- preserves the runtime's function order exactly,
- raises `ValueError` if more than 10 invocable functions are present,
- tracks only runs launched through that `TUI` instance.

The v1 session retains all tracked runs and their per-run UI state for the life of the process. There is no pruning or removal UI.

### 5.2 Layout

Normal multi-pane layout has:
- a left pane,
- a one-column double-bar separator,
- a fixed two-column empty margin on the left edge of the right pane,
- a right pane rendered by the selected `ConsoleRender`,
- one full-width bottom bar.

Preferred left-pane width is 48 columns, clamped so the right pane retains at least 40 content columns.

Multi-pane minimum size is:
- 67 columns total,
- 15 total rows.

When smaller than that, the controller renders a dedicated "terminal too small" frame and suspends normal interactions until the terminal becomes large enough again.

### 5.3 Left Pane

The left pane is divided into:
- a top runs pane,
- a bottom functions pane.

Runs pane:
- shows only runs successfully launched and admitted into this `TUI`'s history,
- groups runs by top-level function name,
- orders groups by `runtime.invocable_functions`,
- keeps runs within a group in launch-history order,
- may include non-selectable group headers and blank separator rows,
- highlights the currently selected run row,
- uses the same state glyph semantics as `ConsoleRender`.

Function pane:
- stays anchored to the bottom of the left pane,
- shows `(index) function_name`,
- preserves `runtime.invocable_functions` order,
- clips vertically when there is not enough height,
- does not introduce pagination, filtering, or alternate layouts in v1.

### 5.4 Run Selection and Per-Run State

Each tracked run owns:
- its root node,
- its latest cached `NodeView`,
- its own `ConsoleRender`,
- its own cancel event,
- unread state,
- terminal-callback bookkeeping.

Only the selected run is rendered into the right pane.
Hidden runs:
- keep their renderer state,
- continue receiving cached snapshot updates,
- do not continuously render while hidden.

Switching selection restores the selected run's preserved renderer state rather than resetting cursor/follow/collapse state.

### 5.5 Multi-Pane Keyboard and Mouse

Normal non-modal multi-pane key bindings:
- Tab / Shift+Tab: select next/previous run row in grouped visible order
- `0`..`9`: open the launch form for that function index
- `C`: request cancellation for the selected run only
- `u`: toggle unread on the selected run
- `j` / `k` / Up / Down: selected-tree navigation
- Space / Enter: selected-tree toggle
- `g` / `G`: selected-tree top/bottom
- Page Up / Page Down: selected-tree page navigation
- `n` / `N`: selected-tree agent jumps
- `c`: collapse selected-tree enclosing agent
- `e` / `r`: selected-tree expand all / collapse all
- `Ctrl+C`: session-wide interrupt handling

There is currently no dedicated normal-keyboard quit path for the multi-pane TUI.

Mouse routing:
- the outer `TUI` owns absolute-screen hit testing,
- left-pane clicks select runs or open launch forms,
- clicks on the separator, bottom bar, or right-pane left margin are ignored,
- only clicks in the right-pane body are translated to right-pane-local coordinates and routed to the selected `ConsoleRender`,
- non-left clicks in the left pane are ignored.

### 5.6 Launch Form

Launching is modal.

While the launch form is open:
- it exclusively owns keyboard and mouse routing,
- normal tree browsing and run selection are suspended,
- the bottom bar switches to launch-form shortcut hints,
- selected-tree cursor position may be omitted from the bottom bar.

Form content:
- first field is a UI-only run name,
- remaining fields correspond to the selected function's declared `FunctionArg`s,
- below the function description, the header also shows each arg's declared type, description, and `[optional]` marker when applicable,
- `[Submit]` and `[Cancel]` appear immediately after the editable arg fields,
- below `[Submit]` / `[Cancel]`, the form may show up to 20 recent top-level runs of that same function from this `TUI` session, ordered newest first,
- each recent-run row shows the run name and a truncated inline args preview,
- function descriptions may span multiple wrapped lines,
- form layout, scrolling, and mouse hit-testing derive from the rendered header height rather than from a fixed header-row constant.

Current form editing behavior:
- Tab / Shift+Tab and Up / Down move the form cursor,
- Enter advances, applies a selected recent-run template, or submits depending on the current row,
- Escape cancels,
- Backspace deletes one character,
- printable characters are appended literally,
- there is no cursor-within-field editing model in v1.

Recent-run template behavior:
- selecting a recent-run row by keyboard or left-click does not launch immediately,
- instead it copies that run's args back into the editable arg fields,
- it also prepopulates the run-name field from that history entry, appending ` (1)` or incrementing an existing trailing ` (N)` suffix,
- optional args that were omitted or submitted as `None` repopulate as blank fields,
- after applying a recent-run template, the cursor returns to the first real arg field (or the run-name field when there are no args).

Submission/parsing rules:
- whitespace-only arg fields are treated as omitted,
- blank optional arg fields are submitted as explicit `None`,
- blank required arg fields remain missing and therefore fail top-level validation,
- `str` args preserve the raw entered string when non-blank,
- `int` args are parsed with `int(...)`,
- `float` args are parsed with `float(...)`,
- `bool` args accept only `true` or `false` (case-insensitive),
- enum-constrained args must match one of the declared values,
- top-level `Runtime.invoke(...)` remains the final authority on validation,
- top-level validation failures are shown inline in the form and do not create a root node.

Successful launch behavior:
- `TUI` creates a fresh per-root cancel event,
- top-level invoke is attempted directly,
- the run is admitted into history only after launch succeeds and watcher-thread startup succeeds,
- a blank run name falls back to `"{function_name} #{current_run_count}"`,
- the new run becomes selected immediately.

Exceptional post-launch setup failure:
- if setup fails after a real root has already been launched, the current v1 behavior is to request cancellation for that root, restore the console, and fail fast out of the process.

### 5.7 Unread Semantics

Unread state is per run and has two sources:
- automatic unread when a run reaches terminal state while not visible in the right pane,
- manual unread toggled by the user with `u`.

Current rules:
- a hidden terminal transition marks the run unread,
- selecting a different run clears unread on the destination run,
- navigating away from an unread run does not clear it,
- if the launch form or the dedicated too-small frame temporarily hides the selected run and that run reaches terminal state, the run becomes automatically unread,
- when that same selected run becomes visible again without a selection change, automatic unread clears,
- manual unread survives temporary hiding and re-showing without a selection change,
- if a manually unread run reaches terminal while still visible, it remains unread.

### 5.8 Session-Wide Interrupts and Per-Run Cancel

Selected-run cancel:
- `C` sets only the selected run's cancel event,
- the binding is omitted when the selected run is already terminal.

Session-wide interrupt handling:
- the first `Ctrl+C` sets the cancel event for every tracked run and keeps the session alive,
- once all tracked runs are terminal, the controller exits after rendering the final frame,
- a second `Ctrl+C` takes the stronger exit path immediately,
- before exiting on either path, the controller makes a best-effort attempt to deliver any pending terminal callbacks.

### 5.9 Terminal Callback

`TUI.register_terminal_callback(callback)` registers one optional callback for terminal top-level runs.

Behavior:
- the callback receives exactly one argument: the final total tree bill, as `Mapping[Provider, TokenBill]`,
- it is invoked only for terminal runs,
- it is attempted at most once per tracked run,
- registering the callback after some tracked runs are already terminal replays those unreported runs immediately,
- callback failures are logged and do not break the session.

## 6. Multi-Pane Bottom Bar

The multi-pane bottom bar is controller-owned and full-width.

It is composed from:
- global/left-pane shortcut segments,
- right-pane shortcut segments derived from the selected renderer's interaction context,
- selected-tree status segments derived from the selected renderer's structured status.

Packing policy:
- selected-tree status and token data have higher priority than shortcut text,
- within shortcut text, higher-priority shortcut variants are tried first,
- the interrupt hint is the highest-priority shortcut segment in the multi-pane shortcut area,
- lower-priority shortcut segments are dropped before truncating inside a kept segment.

In modal launch-form state and dedicated too-small state, the controller may omit current tree position from the selected-tree status contribution.

## 7. Watchers and Update Delivery

Standalone:
- one watcher thread watches the assigned root,
- it exits after enqueuing the terminal update.

Multi-pane:
- one daemon watcher thread is created per admitted run,
- watcher threads block on `node.watch(as_of_seq=prev_seq)`,
- each update enqueues a root-update event and wakes the controller thread,
- when a run reaches terminal state, its watcher may stop while the run remains in history.

Watcher threads do not call `ConsoleRender` methods directly.

## 8. Explicit v1 Limitations

Current v1 limitations include:
- multi-pane launch list hard-limited to 10 functions,
- no run pruning/removal UI,
- no pagination/filtering/search for functions or runs,
- no automatic discovery of external top-level roots,
- no normal-keyboard quit path in the multi-pane session,
- no alternate compact layout for narrow terminals; a dedicated too-small frame is shown instead,
- no advanced launch-form text editing beyond append/delete and row navigation.

## 9. Verification Surface

The behavior described here is exercised by the `tests/tui/` suite and is intended to stay aligned with:
- `tests/tui/test_console_render.py`
- `tests/tui/test_tui_controllers.py`

When implementation changes alter the behavior above, this file should be updated in the same change.
