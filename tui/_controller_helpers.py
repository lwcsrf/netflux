from __future__ import annotations

from typing import Sequence

from ..core import NodeState
from ._contracts import RightPaneInteractionContext, SelectedTreeStatus, TerminalSize
from .console import (
    BG_STATUS_BAR,
    BG_STATUS_STATE,
    BG_STATUS_TOKENS,
    BOLD,
    FG,
    RESET,
    ConsoleRender,
    _SPINNER_FRAMES,
    _color,
    _crop_line,
    _style_block,
    _visible_len,
)


_MULTI_PANE_LEFT_MIN = 24
_MULTI_PANE_LEFT_PREFERRED = 48
_MULTI_PANE_SEPARATOR_COLS = 1
_MULTI_PANE_RIGHT_MARGIN_COLS = 2
_MULTI_PANE_RIGHT_MIN = 40
_MULTI_PANE_MIN_WIDTH = (
    _MULTI_PANE_LEFT_MIN
    + _MULTI_PANE_SEPARATOR_COLS
    + _MULTI_PANE_RIGHT_MARGIN_COLS
    + _MULTI_PANE_RIGHT_MIN
)


def preferred_left_pane_width(columns: int) -> int:
    max_left = columns - (
        _MULTI_PANE_SEPARATOR_COLS
        + _MULTI_PANE_RIGHT_MARGIN_COLS
        + _MULTI_PANE_RIGHT_MIN
    )
    return max(_MULTI_PANE_LEFT_MIN, min(_MULTI_PANE_LEFT_PREFERRED, max_left))


def standalone_too_small(size: TerminalSize) -> bool:
    return size.columns < 40 or size.lines < 6


def multi_pane_too_small(size: TerminalSize) -> bool:
    return size.columns < _MULTI_PANE_MIN_WIDTH or size.lines < 15


def _format_state_text(status: SelectedTreeStatus, tick: int) -> str:
    state = status.state
    if state is None:
        return ""
    if state is NodeState.Running:
        frame = _SPINNER_FRAMES[tick % len(_SPINNER_FRAMES)]
        return _color(f"{frame} Running", fg="cyan")
    if state is NodeState.Success:
        return _color("✔ Complete", fg="green")
    if state is NodeState.Error:
        return _color("✖ Error", fg="red")
    if state is NodeState.Canceled:
        return _color("⏹ Canceled", fg="yellow")
    if state is NodeState.Waiting:
        return _color("… Waiting", fg="yellow")
    return ""


def format_selected_status_text(status: SelectedTreeStatus, tick: int) -> str:
    segments: list[str] = []
    if status.total_lines > 0:
        segments.append(f"{status.cursor_line}/{status.total_lines}")

    state_text = _format_state_text(status, tick)
    if state_text:
        segments.append(state_text)

    if status.cancel_pending and status.state not in {
        NodeState.Success,
        NodeState.Error,
        NodeState.Canceled,
    }:
        segments.append(f"{FG['magenta']}Cancel pending{RESET}")

    return "  ".join(segments)


def format_selected_token_text(status: SelectedTreeStatus) -> str:
    return ConsoleRender._format_total_token_bill(status.token_bill)


def _join_shortcuts(parts: Sequence[str]) -> str:
    return "  ".join(part for part in parts if part)


def _truncate_text(text: str, budget: int) -> str:
    if budget <= 0:
        return ""
    if _visible_len(text) <= budget:
        return text
    if budget == 1:
        return "…"
    cropped = _crop_line(text, budget - 1)
    if cropped.endswith(RESET):
        cropped = cropped[: -len(RESET)]
    return cropped + "…" + RESET


def _pack_segments(
    optional_segments: Sequence[str],
    *,
    mandatory_shortcuts: Sequence[str],
    budget: int,
) -> str:
    if budget <= 0:
        return ""

    kept: list[str] = []
    used = 0

    for segment in mandatory_shortcuts:
        separator = 2 if kept else 0
        segment_len = _visible_len(segment)
        if used + separator + segment_len <= budget:
            if separator:
                used += separator
            kept.append(segment)
            used += segment_len
            continue

        remaining = budget - used - separator
        if remaining <= 0:
            return _join_shortcuts(kept)
        if separator:
            used += separator
        kept.append(_truncate_text(segment, remaining))
        return _join_shortcuts(kept)

    for segment in optional_segments:
        separator = 2 if kept else 0
        segment_len = _visible_len(segment)
        if used + separator + segment_len > budget:
            break
        if separator:
            used += separator
        kept.append(segment)
        used += segment_len

    return _join_shortcuts(kept)


def _best_shortcut_text(
    variants: Sequence[Sequence[str]],
    *,
    mandatory_shortcuts: Sequence[str],
    budget: int,
) -> str:
    best = ""
    best_score = (-1, -1)
    for variant in variants:
        packed = _pack_segments(
            variant,
            mandatory_shortcuts=mandatory_shortcuts,
            budget=budget,
        )
        kept_segments = [part for part in packed.split("  ") if part]
        score = (len(kept_segments), _visible_len(packed))
        if score > best_score:
            best = packed
            best_score = score
    return best


def compose_bottom_bar(
    cols: int,
    *,
    shortcut_variants: Sequence[Sequence[str]],
    status: SelectedTreeStatus,
    tick: int,
    mandatory_shortcuts: Sequence[str] = (),
) -> str:
    status_text = format_selected_status_text(status, tick)
    token_text = format_selected_token_text(status)

    status_raw = f" {status_text} " if status_text else ""
    token_raw = f" {token_text} " if token_text else ""

    status_len = _visible_len(status_raw)
    token_len = _visible_len(token_raw)

    if status_len > cols:
        status_raw = _crop_line(status_raw, cols)
        token_raw = ""
    else:
        token_budget = max(0, cols - status_len)
        if token_len > token_budget:
            token_raw = _crop_line(token_raw, token_budget) if token_budget >= 8 else ""

    right_len = _visible_len(status_raw) + _visible_len(token_raw)
    left_budget = max(0, cols - right_len)
    left_text_budget = max(0, left_budget - 1)
    left = _best_shortcut_text(
        shortcut_variants,
        mandatory_shortcuts=mandatory_shortcuts,
        budget=left_text_budget,
    )

    left_raw = ""
    if left_budget > 0:
        left_raw = f" {left}" if left else ""
        vis_len = _visible_len(left_raw)
        if vis_len < left_budget:
            left_raw = left_raw + " " * (left_budget - vis_len)
        elif vis_len > left_budget:
            left_raw = _crop_line(left_raw, left_budget)

    segments: list[str] = []
    if left_raw:
        segments.append(_style_block(left_raw, FG["white"], BG_STATUS_BAR))
    if status_raw:
        segments.append(_style_block(status_raw, BOLD, FG["white"], BG_STATUS_STATE))
    if token_raw:
        segments.append(_style_block(token_raw, BOLD, FG["white"], BG_STATUS_TOKENS))
    return "".join(segments)


def render_too_small_frame(
    size: TerminalSize,
    *,
    message: str,
    hint: str,
    bottom_bar: str,
) -> str:
    rows = max(1, size.lines)
    cols = max(1, size.columns)
    body_rows = max(0, rows - 1)
    line = message[:cols]
    hint_line = hint[:cols]
    padding = [" " * cols for _ in range(body_rows)]
    if body_rows >= 1:
        padding[0] = line.ljust(cols)
    if body_rows >= 2:
        padding[1] = hint_line.ljust(cols)
    bar = bottom_bar or ""
    bar = _crop_line(bar, cols) if _visible_len(bar) > cols else bar
    if _visible_len(bar) < cols:
        bar = bar + " " * (cols - _visible_len(bar))
    if body_rows == 0:
        return bar
    padding.append(bar)
    return "\n".join(padding)


def _dedupe_variants(*variants: Sequence[str]) -> list[list[str]]:
    deduped: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for variant in variants:
        normalized = tuple(part for part in variant if part)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(list(normalized))
    return deduped or [[]]


def standalone_shortcut_variants(ctx: RightPaneInteractionContext) -> list[list[str]]:
    full: list[str] = []
    compact: list[str] = []
    terse: list[str] = []

    if ctx.has_lines:
        full.append("↑↓/jk:move")
        compact.append("↑↓/jk:move")
        terse.append("jk")

        if ctx.can_expand_collapse:
            full.append("␣:toggle")
            compact.append("␣:toggle")
            terse.append("␣")

        full.append("PgUp/PgDn:page")
        compact.append("Pg:page")
        terse.append("Pg")

        if ctx.can_jump_agents:
            full.append("n/N:next/prev")
            compact.append("n/N:next/prev")
            terse.append("n/N")

        if getattr(ctx, "can_copy_root_result", False):
            full.append("c:copy result")
            compact.append("c:copy")
            terse.append("c")

        if getattr(ctx, "can_focus_root_result", False):
            full.append("r:show result")
            compact.append("r:result")
            terse.append("r")

        if ctx.can_expand_collapse:
            full.append("a:agent")
            compact.append("a:agent")
            terse.append("a")

        full.append("g/G:top/btm")
        compact.append("g/G:top/btm")
        terse.append("g/G")

        if ctx.can_expand_collapse:
            full.append("e/E:all")
            compact.append("e/E:all")
            terse.append("e/E")

    return _dedupe_variants(full, compact, terse, [])


def multi_pane_shortcut_variants(
    ctx: RightPaneInteractionContext,
    *,
    can_cancel: bool = False,
    can_mark_unread: bool = False,
    interrupt_hint: str = "^C:cancel all",
) -> list[list[str]]:
    full = ["Tab/Shift+Tab:runs", "0-9:launch", interrupt_hint]
    compact = ["Tab/S-Tab:runs", "0-9:launch", interrupt_hint]
    terse = ["Tab", "0-9", interrupt_hint]

    if can_cancel:
        full.append("C:cancel tree")
        compact.append("C:cancel tree")
        terse.append("C")

    if can_mark_unread:
        full.append("u:unread")
        compact.append("u:unread")
        terse.append("u")

    if ctx.has_lines:
        full.append("↑↓/jk:tree")
        compact.append("↑↓/jk:tree")
        terse.append("jk")

        if ctx.can_expand_collapse:
            full.append("␣:toggle")
            compact.append("␣:toggle")
            terse.append("␣")

        full.append("Pg:page")
        compact.append("Pg:page")
        terse.append("Pg")

        if ctx.can_jump_agents:
            full.append("n/N:agent")
            compact.append("n/N:agent")
            terse.append("n/N")

        if getattr(ctx, "can_copy_root_result", False):
            full.append("c:copy result")
            compact.append("c:copy")
            terse.append("c")

        if getattr(ctx, "can_focus_root_result", False):
            full.append("r:show result")
            compact.append("r:result")
            terse.append("r")

        if ctx.can_expand_collapse:
            full.append("a:agent")
            compact.append("a:agent")
            terse.append("a")

        full.append("g/G:top/btm")
        compact.append("g/G:top/btm")
        terse.append("g/G")

        if ctx.can_expand_collapse:
            full.append("e/E:all")
            compact.append("e/E:all")
            terse.append("e/E")

    return _dedupe_variants(full, compact, terse)
