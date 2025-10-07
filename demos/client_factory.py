"""Client factories used by the demos, which rely on simple api key."""

from pathlib import Path
from typing import Any, Callable, Dict

import httpx
import anthropic
import google.genai as genai
from google.genai import types

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
        # default in httpx; leave unless you need to disable env proxies:
        # trust_env=False,
    )
    # We have our own retry layer, but Anthropic may have different
    # or more informed retry criteria, so also use a small number of retries here.
    max_retries = 1

    return anthropic.Anthropic(api_key=key, http_client=http_client, max_retries=max_retries)


def gemini_client_factory() -> genai.Client:
    key = _read_key("gemini.key")

    # We have our own retry layer, but Gemini SDK may have different
    # or more informed retry criteria, so also use a small number of retries here.
    sdk_retry = types.HttpRetryOptions(
        attempts=2,             # total attempts = (1 original + 1 retry)
        initial_delay=1.0,      # seconds
        max_delay=5.0,          # seconds (our provider retry layer has longer max)
        exp_base=2.0,
        jitter=1.0,
    )
    httpx_limits = httpx.Limits(
        max_connections=4,
        max_keepalive_connections=2,
        keepalive_expiry=20.0,
    )
    httpx_timeout = httpx.Timeout(
        connect=10.0,
        read=900.0,  # tolerate very long gaps between streamed chunks
        write=120.0,
        pool=10.0,
    )
    http_options = types.HttpOptions(
        # Choose api_version if you want only GA endpoints; by default SDK uses v1beta for preview features.
        # api_version="v1",  # uncomment to pin to stable
        client_args={
            "http2": True,
            "limits": httpx_limits,
            "timeout": httpx_timeout,
            # default in httpx; leave unless you need to disable env proxies
            # "trust_env": True,
        },
        retry_options=sdk_retry,
        # Avoid setting HttpOptions.timeout here so we don't override the fine-grained HTTPX timeouts.
        # If you *do* set it, it will be used as the request timeout AND send X-Server-Timeout.
    )

    return genai.Client(api_key=key, http_options=http_options)


CLIENT_FACTORIES: Dict[Provider, Callable[[], Any]] = {
    Provider.Anthropic: anthropic_client_factory,
    Provider.Gemini: gemini_client_factory,
}
