"""Client factories used by the demos, which rely on simple api key."""

from pathlib import Path
from typing import Any, Callable, Dict

import anthropic
import google.genai as genai

from ..providers import Provider

DEMO_DIR = Path(__file__).resolve().parent


def _read_key(filename: str) -> str:
    path = DEMO_DIR / filename
    try:
        key = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Missing API key file '{filename}' in demos directory: {path}. "
            "Create the file and place your API key inside."
        ) from exc
    except Exception as exc:  # pragma: no cover - unexpected filesystem errors
        raise RuntimeError(
            f"Unable to read API key file '{filename}' from demos directory."
        ) from exc

    if not key:
        raise RuntimeError(
            f"API key file '{filename}' in demos directory is empty."
        )
    return key


def anthropic_client_factory() -> anthropic.Anthropic:
    key = _read_key("anthropic.key")
    return anthropic.Anthropic(api_key=key)


def gemini_client_factory() -> genai.Client:
    key = _read_key("gemini.key")
    return genai.Client(api_key=key)


CLIENT_FACTORIES: Dict[Provider, Callable[[], Any]] = {
    Provider.Anthropic: anthropic_client_factory,
    Provider.Gemini: gemini_client_factory,
}
