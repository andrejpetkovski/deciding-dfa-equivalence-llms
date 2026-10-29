"""HTTP client for the FINKI gateway (``docs/cluster.md``).

The transport contract is the one verified by ``scripts/probe_gateway.py`` on 2026-09-07 and
nothing more: a LiteLLM proxy at ``https://vllm.finki.ukim.mk/v1`` speaking the
OpenAI-compatible ``POST /chat/completions`` route, authenticated with a bearer virtual key.
Nothing about the serving engine, the weights, or an upstream release is inferred from that;
``x-litellm-model-api-base`` points at an internal ``opencode-adapter`` and ``/model/info``
is 403 for the issued key.

This class performs **one attempt** and returns a record. It never retries, never sleeps,
and never inspects the answer -- those belong to ``src/run_models.py``, where the attempt
ceiling lives and every attempt is logged.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from src.backends.base import (
    ChatRequest,
    RawResult,
    classify_failure,
    redact_headers,
    upstream_failure_code,
)

__all__ = ["GatewayHTTPBackend", "load_env_file", "resolve_credentials"]

#: How much of a non-JSON error page to keep. Enough to identify a proxy error, short of
#: pasting an HTML document into the append-only log.
_NON_JSON_LIMIT = 4000


def load_env_file(path: str | os.PathLike[str]) -> dict[str, str]:
    """Parse a ``KEY=VALUE`` ``.env`` file. A missing file yields an empty mapping."""
    file = Path(path)
    if not file.exists():
        return {}
    values: dict[str, str] = {}
    for raw in file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_credentials(
    env_path: str | os.PathLike[str] = ".env",
    base_url_var: str = "FINKI_API_BASE_URL",
    key_var: str = "FINKI_API_KEY",
) -> tuple[str, str]:
    """Return ``(base_url, api_key)`` from ``.env``, falling back to the environment.

    Raises:
        RuntimeError: naming the missing variables and where to set them. The message never
            echoes a value, and the caller must never ask for one in chat.
    """
    env = load_env_file(env_path)
    base_url = env.get(base_url_var) or os.environ.get(base_url_var, "")
    api_key = env.get(key_var) or os.environ.get(key_var, "")
    missing = [
        name
        for name, value in ((base_url_var, base_url), (key_var, api_key))
        if not value
    ]
    if missing:
        raise RuntimeError(
            f"missing {', '.join(missing)}; set them in {Path(env_path)} (see .env.example). "
            f"Never paste the key into a terminal or a chat message."
        )
    return base_url.rstrip("/"), api_key


class GatewayHTTPBackend:
    """One attempt per call against the gateway's chat-completions route."""

    name = "finki_gateway_http"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout_s: float = 900.0,
        chat_path: str = "/chat/completions",
        client: httpx.Client | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.chat_path = chat_path if chat_path.startswith("/") else f"/{chat_path}"
        self._timeout_s = timeout_s
        # Held privately and never stored on a record or printed. The only place it appears
        # is the Authorization header, which redact_headers rewrites before anything is
        # written.
        self._headers = {
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
            "accept": "application/json",
            **(extra_headers or {}),
        }
        self._client = client or httpx.Client(timeout=timeout_s)
        self._owns_client = client is None

    @classmethod
    def from_env(
        cls,
        env_path: str | os.PathLike[str] = ".env",
        *,
        timeout_s: float = 900.0,
        chat_path: str = "/chat/completions",
        base_url_var: str = "FINKI_API_BASE_URL",
        key_var: str = "FINKI_API_KEY",
        extra_headers: dict[str, str] | None = None,
    ) -> GatewayHTTPBackend:
        """Build a backend from named environment variables.

        The variable *names* are parameters because the execution environment is a recorded
        decision, not a constant: the config names which credentials a run uses, so switching
        providers is a config change with an audit trail rather than an edit here.
        """
        base_url, api_key = resolve_credentials(env_path, base_url_var, key_var)
        return cls(
            base_url,
            api_key,
            timeout_s=timeout_s,
            chat_path=chat_path,
            extra_headers=extra_headers,
        )

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}{self.chat_path}"

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> GatewayHTTPBackend:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def complete(self, request: ChatRequest, attempt: int = 1) -> RawResult:
        """Send one request. Returns a record for every outcome, including failures."""
        payload = request.payload()
        started = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        clock = time.perf_counter()

        try:
            response = self._client.post(
                self.endpoint, json=payload, headers=self._headers, timeout=self._timeout_s
            )
        except Exception as error:  # noqa: BLE001 -- every transport failure is data
            kind, retryable = classify_failure(None, error)
            return RawResult(
                ok=False,
                attempt=attempt,
                requested_model=request.model,
                request_sha256=request.sha256,
                started_at_utc=started,
                latency_ms=(time.perf_counter() - clock) * 1000.0,
                status=None,
                body=None,
                response_headers={},
                failure_kind=kind,
                failure_detail=f"{type(error).__name__}: {error}",
                retryable=retryable,
            )

        latency_ms = (time.perf_counter() - clock) * 1000.0
        body = self._decode(response)
        status = response.status_code
        kind, retryable = classify_failure(status, None)
        ok = kind == "ok"

        # A 2xx is not proof of a completion. OpenRouter returns 200 with an error envelope
        # when an upstream provider fails, and such a body carries no answer at all -- see
        # upstream_failure_code for why re-attempting it stays inside rule 4. The recovered
        # code is routed back through classify_failure so retryability is still decided in
        # exactly one place.
        if ok:
            hidden = upstream_failure_code(body)
            if hidden is not None:
                kind, retryable = classify_failure(hidden, None)
                ok = False

        return RawResult(
            ok=ok,
            attempt=attempt,
            requested_model=request.model,
            request_sha256=request.sha256,
            started_at_utc=started,
            latency_ms=latency_ms,
            status=status,
            body=body,
            response_headers=redact_headers(response.headers),
            failure_kind=None if ok else kind,
            failure_detail=None if ok else _error_message(body),
            retryable=retryable,
        )

    @staticmethod
    def _decode(response: httpx.Response) -> Any:
        """Decode the body, keeping non-JSON text rather than discarding it.

        A proxy error page is evidence about the run and belongs in the append-only log
        (rule 3), so a decode failure produces a record instead of an exception.
        """
        try:
            return response.json()
        except ValueError:
            text = response.text or ""
            return {
                "_non_json_text": text[:_NON_JSON_LIMIT],
                "_truncated": len(text) > _NON_JSON_LIMIT,
            }


def _error_message(body: Any) -> str | None:
    """Pull a human-readable reason out of an error body, if there is one."""
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if isinstance(body.get("detail"), str):
            return body["detail"]
        if isinstance(body.get("_non_json_text"), str) and body["_non_json_text"]:
            return body["_non_json_text"][:200]
    return None
