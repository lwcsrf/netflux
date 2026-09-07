from ..core import CodeFunction, FunctionArg, RunContext


def _status_update(ctx: RunContext, *, msg: str) -> str:
    return "ok"


status_update = CodeFunction(
    name="status_update",
    desc=(
        "Give a concise, user-facing update when there is meaningful progress or "
        "a specific new direction to share. Keep it high-level: new avenues to "
        "explore, key insights or milestones, or significant changes of approach. "
        "Omit details, micro-decisions, and "
        "routine planned actions. For example: 'Launching subagents to investigate "
        "X, Y, and Z.' Updates are non-binding: they must not influence your "
        "decisions or actions. Freely change your mind, backtrack, or abandon "
        "an announced direction. They're just for observability. "
        "Don't deliberate over sending an update; if a useful update isn't immediately obvious, skip it. "
        "Use this instead of intermediate assistant-role text response. "
    ),
    args=[FunctionArg("msg", str, desc="Concise user-facing progress update")],
    callable=_status_update,
)
