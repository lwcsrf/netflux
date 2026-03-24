import argparse
from typing import List, Optional, Sequence

from ..core import Function
from ..runtime import Runtime
from ..tui import TUI
from .apply_diff import apply_diff_patch
from .bash_stress import bash_stress_agent
from .client_factory import CLIENT_FACTORIES
from .perf_opt import perf_optimizer
from .puzzle import INTERLEAVE_AGENT


ROOT_FUNCTIONS: tuple[Function, ...] = (
    INTERLEAVE_AGENT,
    bash_stress_agent,
    perf_optimizer,
    apply_diff_patch,
)

class _DemoRuntime(Runtime):
    def __init__(
        self,
        launch_functions: Sequence[Function],
    ) -> None:
        self._launch_functions = tuple(launch_functions)
        super().__init__(
            specs=launch_functions,
            client_factories=CLIENT_FACTORIES,
        )

    @property
    def invocable_functions(self) -> tuple[Function, ...]:
        # Keep the launch pane focused on the four demo roots while the base Runtime
        # still registers their full transitive dependency graph for actual execution.
        return self._launch_functions


def build_runtime() -> Runtime:
    return _DemoRuntime(ROOT_FUNCTIONS)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the multi-root TUI demo over the four existing demo functions.",
    )
    parser.add_argument(
        "--spinner-hz",
        type=float,
        default=10.0,
        help="UI animation rate in hertz.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    runtime = build_runtime()
    tui = TUI(runtime, spinner_hz=args.spinner_hz)
    print(f"TUI log file: {tui.log_path}")
    tui.run()


if __name__ == "__main__":
    main()
