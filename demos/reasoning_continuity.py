"""Ask an agent to diagnose reasoning continuity in its current Netflux run (provider self-awareness)."""

import argparse
from datetime import datetime, timezone
import inspect
import os
from pathlib import Path
import sys
import tempfile
from threading import Event

from ..core import AgentFunction, Provider
from ..func_lib import bash, text_editor, status_update, raise_exception
from ..providers import get_AgentNode_impl
from ..runtime import Runtime
from ..tui import ConsoleRender
from .client_factory import CLIENT_FACTORIES


PROVIDERS = {provider.value.lower(): provider for provider in CLIENT_FACTORIES}


def build_agent(provider: Provider) -> AgentFunction:
    provider_impl = get_AgentNode_impl(provider)
    provider_path = Path(inspect.getfile(provider_impl)).resolve().as_posix()
    test_path = Path(__file__).resolve()
    run_started_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    user_prompt = (
        "You are running in the netflux harness right now using provider: "
        f"{provider.value}. The exact provider module running you is {provider_impl.__module__}, "
        f"whose source file is {provider_path}. The Netflux repo is at {test_path.parents[1].as_posix()}. "
        f"The test script running you is {test_path.as_posix()}. "
        "You are participating in a live execution of this exact test "
        "RIGHT NOW: its AgentFunction is you, and its runtime invocation launched "
        "your current run. You can inspect the source to see how you were started. "
        f"The repo's venv Python executable is {Path(sys.executable).as_posix()}; its environment "
        f"directory is {Path(sys.prefix).as_posix()}. This is the venv you may use for your experiments. "
        f"This run started at {run_started_at}.\n\n"
        "Your task is to report back whether you can, through your literal current "
        "experience, confirm that the provider has done proper reasoning continuity "
        "across function calls. Distinguish what you can experience and observe in this run from "
        "what you infer from the source code. If you cannot verify it, say so.\n"
        "Consider both a nonce test as well as incremental token accounting.\n\n"
    )
    system_prompt = (
        "You are investigating whether the netflux agent harness, which you are also running right now, "
        "has properly implemented reasoning continuity for your model. "
        "Use a tmp dir for any experiment files if needed.\n"
        "Give periodic status updates."
    )
    return AgentFunction(
        name="reasoning_continuity_experiment",
        desc="Inspect your current provider and report on continuity in this run.",
        args=[],
        system_prompt=system_prompt,
        user_prompt_template=user_prompt,
        uses=[bash, text_editor, raise_exception],
        default_model=provider,
    )


def run_continuity_tree(provider: Provider) -> None:
    agent = build_agent(provider)
    workspace = Path(tempfile.mkdtemp(prefix="netflux-continuity-")).resolve()
    print(f"Provider: {provider.value}", flush=True)
    print(f"Experiment directory: {workspace}", flush=True)

    runtime = Runtime(
        specs=[agent],
        client_factories=CLIENT_FACTORIES,
    )
    cancel_event = Event()
    original_cwd = Path.cwd()
    original_path = os.environ.get("PATH", "")
    node = None
    try:
        os.chdir(workspace)
        # Let the agent's shell find the same Python environment as this demo.
        os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{original_path}"
        node = runtime.get_ctx().invoke(
            agent,
            {},
            provider=provider,
            cancel_event=cancel_event,
        )
        ConsoleRender(spinner_hz=10.0).run(node)
        print(node.result())
    finally:
        cancel_event.set()
        if node is not None:
            node.wait()
        os.chdir(original_cwd)
        os.environ["PATH"] = original_path
        print(f"Experiment files kept at: {workspace}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        choices=PROVIDERS,
        required=True,
        help="provider to test for this run",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    run_continuity_tree(PROVIDERS[args.provider])


if __name__ == "__main__":
    main()
