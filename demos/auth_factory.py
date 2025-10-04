"""Client factories used by the demos, which rely on simple api key."""

from pathlib import Path
from typing import Any, Callable, Dict

import anthropic
import httpx
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
    # Use Anthropic's DefaultHttpxClient to retain their socket keepalive tuning, and
    # right-size connection limits/timeouts for single-agent long reasoning streams.
    http_client = anthropic.DefaultHttpxClient(
        http2=True,
        limits=httpx.Limits(
            max_connections=4,
            max_keepalive_connections=2,
            keepalive_expiry=20.0,
        ),
        timeout=httpx.Timeout(
            connect=10.0,
            read=900.0,   # tolerate very long gaps between streamed chunks
            write=120.0,
            pool=10.0,
        ),
        # Optionally disable env proxies if they’re flaky:
        # trust_env=False,
    )
    return anthropic.Anthropic(api_key=key, http_client=http_client, max_retries=4)


def gemini_client_factory() -> genai.Client:
    key = _read_key("gemini.key")
    return genai.Client(api_key=key)


CLIENT_FACTORIES: Dict[Provider, Callable[[], Any]] = {
    Provider.Anthropic: anthropic_client_factory,
    Provider.Gemini: gemini_client_factory,
}
