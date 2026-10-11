"""Lets a hosted EvoLoRA website drive the local agent, only after the user approves.

Flow: the website opens ``http://127.0.0.1:<port>/pair.html?origin=<its origin>`` in a
popup; the user approves on that local page; the agent issues a token for that origin
and the page hands it back to the website with ``postMessage``. Every cross-origin API
call must then carry ``Authorization: Bearer <token>`` from the matching origin.

Browsers also require CORS and, for public sites reaching loopback, Private Network
Access preflights; both are answered here.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from pathlib import Path
from urllib.parse import urlparse

PUBLIC_PATHS = {"/api/agent/hello"}  # detection only; reveals nothing beyond "installed"
ALLOWED_METHODS = "GET, POST, DELETE, OPTIONS"
ALLOWED_HEADERS = "authorization, content-type"


def valid_origin(origin: str) -> bool:
    """https sites, or http on this machine (local development)."""
    parsed = urlparse(origin)
    if parsed.path not in ("", "/") or parsed.query or parsed.fragment or not parsed.hostname:
        return False
    if parsed.scheme == "https":
        return True
    return parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"}


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class PairingStore:
    """Approved origins and their token hashes, persisted in a small JSON file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def _load(self) -> dict[str, str]:
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self, data: dict[str, str]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def pair(self, origin: str) -> str:
        """Issue a new token for ``origin`` (replacing any earlier one)."""
        if not valid_origin(origin):
            raise ValueError(f"Not an allowed website origin: {origin!r}")
        token = secrets.token_urlsafe(32)
        data = self._load()
        data[origin] = _hash(token)
        self._save(data)
        return token

    def check(self, origin: str, token: str) -> bool:
        stored = self._load().get(origin)
        return bool(stored and token) and hmac.compare_digest(stored, _hash(token))

    def is_paired(self, origin: str) -> bool:
        return origin in self._load()

    def origins(self) -> list[str]:
        return sorted(self._load())

    def revoke(self, origin: str) -> bool:
        data = self._load()
        if data.pop(origin, None) is None:
            return False
        self._save(data)
        return True


class RemoteAccessMiddleware:
    """CORS + Private Network Access + per-origin tokens for cross-origin API calls.

    Same-origin requests (the agent's own pages) pass through untouched.
    """

    def __init__(self, app, store: PairingStore) -> None:
        self.app = app
        self.store = store

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
        origin = headers.get("origin", "")
        own = f"http://{headers.get('host', '')}"
        if not origin or origin == own or not scope["path"].startswith("/api/"):
            return await self.app(scope, receive, send)

        cors = [
            (b"access-control-allow-origin", origin.encode()),
            (b"vary", b"Origin"),
        ]
        if scope["method"] == "OPTIONS":
            if not valid_origin(origin):
                return await _respond(send, 403, b"origin not allowed", [])
            extra = [
                (b"access-control-allow-methods", ALLOWED_METHODS.encode()),
                (b"access-control-allow-headers", ALLOWED_HEADERS.encode()),
                (b"access-control-max-age", b"600"),
            ]
            if headers.get("access-control-request-private-network") == "true":
                extra.append((b"access-control-allow-private-network", b"true"))
            return await _respond(send, 204, b"", cors + extra)

        if scope["path"] not in PUBLIC_PATHS:
            auth = headers.get("authorization", "")
            token = auth[7:] if auth.lower().startswith("bearer ") else ""
            if not self.store.check(origin, token):
                return await _respond(
                    send, 401, b'{"detail":"Pair this website with EvoLoRA first"}',
                    cors + [(b"content-type", b"application/json")],
                )

        async def send_with_cors(message):
            if message["type"] == "http.response.start":
                message = {**message, "headers": [*message.get("headers", []), *cors]}
            await send(message)

        return await self.app(scope, receive, send_with_cors)


async def _respond(send, status: int, body: bytes, headers: list[tuple[bytes, bytes]]) -> None:
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})
