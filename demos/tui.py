import argparse
from collections import deque
from typing import Iterable, List, Optional, Sequence

from ..core import AgentFunction, Function, Provider
from ..runtime import Runtime
from ..viz import TUI
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


def _walk_functions(roots: Sequence[Function]) -> Iterable[Function]:
    queue = deque(roots)
    seen: set[int] = set()

    while queue:
        fn = queue.popleft()
        marker = id(fn)
        if marker in seen:
            continue
        seen.add(marker)
        yield fn
        queue.extend(fn.uses)


def _set_default_provider(roots: Sequence[Function], provider: Provider) -> None:
    for fn in _walk_functions(roots):
        if isinstance(fn, AgentFunction):
            fn.default_model = provider


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


def build_runtime(provider: Provider) -> Runtime:
    _set_default_provider(ROOT_FUNCTIONS, provider)
    return _DemoRuntime(ROOT_FUNCTIONS)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the multi-root TUI demo over the four existing demo functions.",
    )
    parser.add_argument(
        "--provider",
        choices=[p.value.lower() for p in Provider],
        required=True,
        help="Provider to assign as the default model for all agent-based demo roots.",
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
    provider_value = {p.value.lower(): p.value for p in Provider}[args.provider]
    provider = Provider(provider_value)
    runtime = build_runtime(provider)
    TUI(runtime, spinner_hz=args.spinner_hz).run()


if __name__ == "__main__":
    main()
