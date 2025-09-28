from typing import Optional

from ..core import CodeFunction, FunctionArg, RunContext


class Bash(CodeFunction):
    """
    Placeholder Bash tool for executing simple shell commands.

    Note: This is a scaffold. The real implementation will be provided later.
    For now, invocations will return a placeholder string so the ApplyDiffPatch
    agent can be wired with a `uses` reference.
    """

    def __init__(self) -> None:
        super().__init__(
            name="bash",
            desc=(
                "Execute a simple shell command and return stdout/stderr. "
                "This is a placeholder implementation."
            ),
            args=[
                FunctionArg(
                    "command",
                    str,
                    "The exact command line to run (as a single string).",
                ),
                FunctionArg(
                    "cwd",
                    str,
                    "Optional working directory for the command.",
                    optional=True,
                ),
                FunctionArg(
                    "timeout_sec",
                    float,
                    "Optional timeout in seconds.",
                    optional=True,
                ),
            ],
            callable=self._call,
        )

    def _call(
        self,
        ctx: RunContext,
        *,
        command: str,
        cwd: Optional[str] = None,
        timeout_sec: Optional[float] = None,
    ) -> str:
        # Placeholder behavior only.
        return (
            "[bash placeholder] Not implemented yet. "
            f"command={command!r}, cwd={cwd!r}, timeout_sec={timeout_sec!r}"
        )


# Built-in global singleton for author reference.
bash = Bash()