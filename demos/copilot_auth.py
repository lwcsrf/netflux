"""GitHub Copilot authentication helper for the demos.

This reuses the *same* GitHub Copilot subscription entitlement you use in VS Code.
It obtains a GitHub OAuth token (via cached editor credentials or a one-time
device-login flow), then exchanges it for the short-lived Copilot API bearer
token that must accompany every request to `https://api.githubcopilot.com`.

Token lifecycle
---------------
* OAuth token: long-lived, cached on disk under the standard Copilot config dir
  (same file layout the CLI / Neovim plugin use). Obtained once via device flow
  if not already present.
* Copilot bearer token: short-lived (~30 min). Minted on demand from the OAuth
  token and refreshed automatically shortly before expiry.

Only the standard, documented device-login endpoints are used; no credentials
are transmitted anywhere except GitHub's own OAuth/token endpoints.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import httpx

# Well-known public OAuth client id used by GitHub Copilot editor integrations.
# It only grants Copilot entitlement checks; it is not a secret.
_CLIENT_ID = "Iv1.b507a08c87ecfe98"

_DEVICE_CODE_URL = "https://github.com/login/device/code"
_ACCESS_TOKEN_URL = "https://github.com/login/oauth/access_token"
_COPILOT_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"

_EDITOR_VERSION = "vscode/1.95.0"
_PLUGIN_VERSION = "copilot-chat/0.23.0"
_USER_AGENT = "GitHubCopilotChat/0.23.0"

# Refresh the bearer this many seconds before its stated expiry.
_REFRESH_SKEW = 120


def _config_dirs() -> list[Path]:
    dirs: list[Path] = []
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        dirs.append(Path(xdg) / "github-copilot")
    home = Path.home()
    dirs.append(home / ".config" / "github-copilot")
    local = os.environ.get("LOCALAPPDATA")
    if local:
        dirs.append(Path(local) / "github-copilot")
    appdata = os.environ.get("APPDATA")
    if appdata:
        dirs.append(Path(appdata) / "github-copilot")
    return dirs


def _read_cached_oauth_token() -> Optional[str]:
    # Environment override wins (useful for CI / headless).
    for env_name in ("GH_COPILOT_TOKEN", "GITHUB_COPILOT_TOKEN", "COPILOT_OAUTH_TOKEN"):
        val = os.environ.get(env_name)
        if val:
            return val.strip()

    for d in _config_dirs():
        for fname in ("apps.json", "hosts.json"):
            path = d / fname
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for key, entry in data.items():
                if not isinstance(entry, dict):
                    continue
                if "github.com" in key:
                    token = entry.get("oauth_token")
                    if token:
                        return str(token).strip()
    return None


def _write_cached_oauth_token(token: str, user: str = "") -> None:
    target_dir = _config_dirs()[1]  # ~/.config/github-copilot (portable across OSes)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "apps.json"
    data: Dict[str, dict] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data[f"github.com:{_CLIENT_ID}"] = {"user": user, "oauth_token": token}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _device_login() -> str:
    """Interactive one-time GitHub device-login flow. Returns an OAuth token."""
    headers = {
        "Accept": "application/json",
        "User-Agent": _USER_AGENT,
        "Editor-Version": _EDITOR_VERSION,
    }
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            _DEVICE_CODE_URL,
            headers=headers,
            data={"client_id": _CLIENT_ID, "scope": "read:user"},
        )
        resp.raise_for_status()
        dev = resp.json()

        user_code = dev["user_code"]
        verification_uri = dev["verification_uri"]
        device_code = dev["device_code"]
        interval = int(dev.get("interval", 5))

        print(
            "\n=== GitHub Copilot device login ===\n"
            f"1. Open: {verification_uri}\n"
            f"2. Enter code: {user_code}\n"
            "Waiting for authorization...",
            flush=True,
        )

        deadline = time.time() + int(dev.get("expires_in", 900))
        while time.time() < deadline:
            time.sleep(interval)
            tok = client.post(
                _ACCESS_TOKEN_URL,
                headers=headers,
                data={
                    "client_id": _CLIENT_ID,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
            )
            tok.raise_for_status()
            payload = tok.json()
            if "access_token" in payload:
                oauth_token = payload["access_token"]
                _write_cached_oauth_token(oauth_token)
                print("Authorization successful.\n", flush=True)
                return oauth_token
            err = payload.get("error")
            if err == "authorization_pending":
                continue
            if err == "slow_down":
                interval += int(payload.get("interval", 5))
                continue
            raise RuntimeError(f"Device login failed: {payload}")
    raise RuntimeError("Device login timed out before authorization.")


class CopilotTokenManager:
    """Thread-safe manager that yields fresh Copilot bearer tokens on demand."""

    def __init__(self, allow_device_login: bool = True):
        self._allow_device_login = allow_device_login
        self._lock = threading.Lock()
        self._oauth_token: Optional[str] = None
        self._bearer: Optional[str] = None
        self._bearer_expiry: float = 0.0

    def _ensure_oauth(self) -> str:
        if self._oauth_token:
            return self._oauth_token
        token = _read_cached_oauth_token()
        if not token:
            if not self._allow_device_login:
                raise RuntimeError(
                    "No cached GitHub Copilot OAuth token found and device login "
                    "is disabled. Set GH_COPILOT_TOKEN or run device login."
                )
            token = _device_login()
        self._oauth_token = token
        return token

    def _mint_bearer(self) -> None:
        oauth = self._ensure_oauth()
        headers = {
            "Authorization": f"token {oauth}",
            "Accept": "application/json",
            "User-Agent": _USER_AGENT,
            "Editor-Version": _EDITOR_VERSION,
            "Editor-Plugin-Version": _PLUGIN_VERSION,
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.get(_COPILOT_TOKEN_URL, headers=headers)
            if resp.status_code == 401:
                # Cached OAuth token is stale/revoked; force re-login next time.
                self._oauth_token = None
                if self._allow_device_login:
                    oauth = self._ensure_oauth()
                    headers["Authorization"] = f"token {oauth}"
                    resp = client.get(_COPILOT_TOKEN_URL, headers=headers)
            resp.raise_for_status()
            payload = resp.json()
        self._bearer = payload["token"]
        self._bearer_expiry = float(payload.get("expires_at", time.time() + 1500))

    def get_bearer(self) -> str:
        with self._lock:
            if not self._bearer or time.time() >= (self._bearer_expiry - _REFRESH_SKEW):
                self._mint_bearer()
            assert self._bearer is not None
            return self._bearer


# Shared default manager so repeated factory calls reuse cached tokens.
_DEFAULT_MANAGER = CopilotTokenManager()


def get_copilot_bearer() -> str:
    return _DEFAULT_MANAGER.get_bearer()


def copilot_default_headers() -> Dict[str, str]:
    return {
        "Editor-Version": _EDITOR_VERSION,
        "Editor-Plugin-Version": _PLUGIN_VERSION,
        "Copilot-Integration-Id": "vscode-chat",
        "User-Agent": _USER_AGENT,
    }
