"""One-shot helper: perform GitHub Copilot device login and cache the OAuth token.

Writes progress to a log file so the device code is reliably visible even when
a rich TUI or terminal buffering would otherwise hide stdout.
"""
import sys
import traceback
from pathlib import Path

LOG = Path(__file__).resolve().parent / "copilot_login.log"


def log(msg: str) -> None:
    with LOG.open("a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")
    print(msg, flush=True)


def main() -> int:
    LOG.write_text("", encoding="utf-8")
    try:
        from . import copilot_auth as ca

        # Monkeypatch print inside device login by calling the pieces directly so
        # we can log the user code to the file.
        import httpx

        headers = {
            "Accept": "application/json",
            "User-Agent": ca._USER_AGENT,
            "Editor-Version": ca._EDITOR_VERSION,
        }
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                ca._DEVICE_CODE_URL,
                headers=headers,
                data={"client_id": ca._CLIENT_ID, "scope": "read:user"},
            )
            resp.raise_for_status()
            dev = resp.json()
            log("=== GitHub Copilot device login ===")
            log(f"Open: {dev['verification_uri']}")
            log(f"Enter code: {dev['user_code']}")
            log("Waiting for authorization...")

            import time
            device_code = dev["device_code"]
            interval = int(dev.get("interval", 5))
            deadline = time.time() + int(dev.get("expires_in", 900))
            while time.time() < deadline:
                time.sleep(interval)
                tok = client.post(
                    ca._ACCESS_TOKEN_URL,
                    headers=headers,
                    data={
                        "client_id": ca._CLIENT_ID,
                        "device_code": device_code,
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    },
                )
                tok.raise_for_status()
                payload = tok.json()
                if "access_token" in payload:
                    ca._write_cached_oauth_token(payload["access_token"])
                    log("Authorization successful. OAuth token cached.")
                    # Now mint a bearer to validate entitlement.
                    mgr = ca.CopilotTokenManager()
                    bearer = mgr.get_bearer()
                    log(f"Copilot bearer minted OK (len={len(bearer)}).")
                    return 0
                err = payload.get("error")
                if err in ("authorization_pending", "slow_down"):
                    if err == "slow_down":
                        interval += 5
                    continue
                log(f"Device login error: {payload}")
                return 2
            log("Device login timed out.")
            return 3
    except Exception:
        log("EXCEPTION:\n" + traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
