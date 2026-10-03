"""
Shared HTTP client for backend calls.

One `httpx.AsyncClient` per worker process — reused across calls so we get
connection pooling and HTTP/2 without paying setup cost per request.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import httpx

from ..settings import Settings, get_settings


class BackendClient:
    """Thin wrapper around httpx.AsyncClient with sensible per-worker defaults."""

    def __init__(self, base_url: str, default_timeout_ms: int) -> None:
        self._base_url = base_url.rstrip("/")
        self._default_timeout_s = default_timeout_ms / 1000.0
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._default_timeout_s),
            http2=True,
            headers={"User-Agent": "builderv2-worker/1.0"},
        )

    @property
    def base_url(self) -> str:
        return self._base_url

    async def get(
        self,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout_ms: int | None = None,
    ) -> httpx.Response:
        return await self._client.get(
            path,
            params=params,
            headers=headers,
            timeout=self._timeout(timeout_ms),
        )

    async def post_json(
        self,
        path: str,
        *,
        json_body: dict[str, Any],
        headers: dict[str, str] | None = None,
        timeout_ms: int | None = None,
    ) -> httpx.Response:
        return await self._client.post(
            path,
            json=json_body,
            headers=headers,
            timeout=self._timeout(timeout_ms),
        )

    def stream(
        self,
        method: str,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
        timeout_ms: int | None = None,
    ) -> httpx._client.AsyncClient.stream:  # type: ignore[valid-type]
        return self._client.stream(
            method,
            path,
            headers=headers,
            json=json_body,
            timeout=self._timeout(timeout_ms),
        )

    def _timeout(self, timeout_ms: int | None) -> httpx.Timeout:
        if timeout_ms is None:
            return httpx.Timeout(self._default_timeout_s)
        return httpx.Timeout(timeout_ms / 1000.0)

    async def aclose(self) -> None:
        await self._client.aclose()


@lru_cache(maxsize=1)
def get_backend_client() -> BackendClient:
    s: Settings = get_settings()
    return BackendClient(s.backend_url, s.runtime.backend.default_timeout_ms)
