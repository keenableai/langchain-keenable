"""Tools for the Keenable API: web search and page fetch."""

from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
from importlib import metadata
from typing import Any, Literal
from urllib.parse import urlsplit

import requests
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool, ToolException
from langchain_core.utils import secret_from_env
from pydantic import BaseModel, Field, SecretStr

try:
    _VERSION = metadata.version("langchain-keenable")
except metadata.PackageNotFoundError:  # pragma: no cover
    _VERSION = "unknown"

# Tagged User-Agent so Keenable can attribute traffic from this integration.
_USER_AGENT = f"keenable-langchain/{_VERSION}"

# Attribution header that the Keenable backend segments traffic by (the
# load-bearing signal for adoption dashboards; the User-Agent above is a
# secondary tag). Value is the platform display name.
_ATTRIBUTION_TITLE = "LangChain"

# Endpoint is read from the environment, never exposed as a public/LLM-settable
# param (an arbitrary base_url is an SSRF foothold).
_DEFAULT_BASE_URL = "https://api.keenable.ai"
_BASE_URL_ENV = "KEENABLE_API_URL"


def _candidate_ips(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """Every IP address ``host`` could denote, without doing DNS.

    Covers dotted/colon literals *and* the numeric IPv4 encodings that resolvers
    accept but :func:`ipaddress.ip_address` rejects as strings — decimal
    (``2130706433``), hex (``0x7f000001``), octal (``0177.0.0.1``) and short
    ``a.b``/``a.b.c`` forms — all of which ``socket.inet_aton`` canonicalizes to
    a real IPv4 so the private-range check sees the true address.
    """
    candidates: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    try:
        candidates.append(ipaddress.ip_address(host))
    except ValueError:
        pass
    try:
        packed = socket.inet_aton(host)
    except OSError:
        pass
    else:
        candidates.append(ipaddress.ip_address(socket.inet_ntoa(packed)))
    return candidates


def _redact(text: str, api_key: str | None) -> str:
    """Strip the API key from any text bound for an exception message or log.

    Server error bodies and transport-exception strings are attacker- or
    misconfiguration-influenced; a server that echoed the ``X-API-Key`` request
    header back in its response would otherwise leak the key into our
    ``ToolException`` text (and from there into the agent's ToolMessage / logs).
    """
    return text.replace(api_key, "***") if api_key else text


def _reject_private_fetch_target(url: str) -> None:
    """Refuse obviously private/internal fetch targets before sending (SSRF).

    The backend enforces private/internal-IP protection server-side too, but a
    client-side guard avoids leaking an internal hostname in a request and is
    required by the integration contract. Hostnames that are not IP literals
    (and not a numeric IPv4 form) pass through — the backend's SSRF guard is the
    backstop for those.
    """
    host = (urlsplit(url).hostname or "").strip().lower()
    # A trailing dot is the FQDN form of the same name (``localhost.`` ==
    # ``localhost``); strip it so it can't slip past the checks below.
    host = host.rstrip(".")
    if not host:
        msg = f"Refusing to fetch a URL with no host: {url!r}"
        raise ToolException(msg)
    if host in {"localhost", "metadata.google.internal"}:
        msg = f"Refusing to fetch a private/internal host: {host!r}"
        raise ToolException(msg)
    for ip in _candidate_ips(host):
        # ``is_reserved`` is intentionally omitted: it flags non-routable but
        # harmless ranges (e.g. the 2001:db8::/32 documentation prefix). The
        # checks below are the ones that matter for SSRF.
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_unspecified
        ):
            msg = f"Refusing to fetch a private/internal address: {host!r}"
            raise ToolException(msg)


def _resolve_base_url() -> str:
    """Resolve the API base URL from ``KEENABLE_API_URL`` and enforce HTTPS."""
    base = (os.environ.get(_BASE_URL_ENV) or _DEFAULT_BASE_URL).rstrip("/")
    parsed = urlsplit(base)
    host = (parsed.hostname or "").rstrip(".")
    # A usable absolute URL needs a host; bail out clearly on e.g. "https://"
    # instead of letting a malformed base produce a broken request URL later.
    if not host:
        msg = f"{_BASE_URL_ENV} must be an https:// URL with a host, got {base!r}"
        raise ToolException(msg)
    # Local-dev escape hatch: permit plain http only to an explicit loopback host.
    if parsed.scheme == "http" and host in {"localhost", "127.0.0.1", "::1"}:
        return base
    if parsed.scheme != "https":
        msg = f"{_BASE_URL_ENV} must be an https:// URL with a host, got {base!r}"
        raise ToolException(msg)
    # Over https, refuse a base URL pointing at a private/internal destination —
    # a misconfigured KEENABLE_API_URL must never ship API keys to an internal
    # host (the same SSRF set as _reject_private_fetch_target).
    if host == "metadata.google.internal" or any(
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_unspecified
        for ip in _candidate_ips(host)
    ):
        msg = (
            f"{_BASE_URL_ENV} must not point at a private/internal address, "
            f"got {base!r}"
        )
        raise ToolException(msg)
    return base


def _raise_for_keenable_status(
    response: requests.Response, api_key: str | None
) -> None:
    """Map a non-2xx Keenable response to a helpful ``ToolException``.

    The backend returns useful bodies (including upgrade/auth instructions); we
    surface them instead of swallowing them behind a generic error. Returning a
    ``ToolException`` (with ``handle_tool_error = True``) hands the message back
    to the agent as a ``ToolMessage`` so it can react, rather than crashing the
    agent loop. The body is attacker-/misconfiguration-influenced, so the key is
    redacted out of it before it reaches the exception text.
    """
    if response.ok:
        return

    detail = ""
    try:
        body = response.json()
        if isinstance(body, dict):
            detail = str(
                body.get("message") or body.get("error") or body.get("detail") or ""
            )
    except ValueError:
        detail = (response.text or "").strip()
    detail = _redact(detail[:200], api_key)

    label = {
        401: "Keenable authentication failed (401)",
        402: "Keenable: insufficient credits (402)",
        429: "Keenable rate limit exceeded (429)",
    }.get(response.status_code, f"Keenable API error ({response.status_code})")

    msg = f"{label}: {detail}" if detail else label
    raise ToolException(msg)


class _KeenableBaseTool(BaseTool):
    """Shared transport for the Keenable tools.

    Handles auth (keyed vs. keyless endpoint selection), the tagged User-Agent,
    HTTPS-only base URL resolution, and uniform error handling.
    """

    api_key: SecretStr | None = Field(
        default_factory=secret_from_env("KEENABLE_API_KEY", default=None),
        exclude=True,
        repr=False,
    )
    """Keenable API key. Falls back to the ``KEENABLE_API_KEY`` env var. When
    absent (or blank) the keyless public endpoint is used."""

    timeout: int = 30
    """Request timeout in seconds."""

    handle_tool_error: bool = True
    """Return API/transport errors to the agent as a ToolMessage instead of
    raising, so a rate limit or auth error doesn't crash the agent loop."""

    def _effective_key(self) -> str | None:
        """The non-blank API key, or ``None`` to use the free tier."""
        if self.api_key is None:
            return None
        key = self.api_key.get_secret_value().strip()
        return key or None

    def _url_and_headers(
        self, public_path: str, keyed_path: str
    ) -> tuple[str, dict[str, str]]:
        """Resolve the endpoint URL and headers, picking keyed vs. keyless."""
        key = self._effective_key()
        headers = {"User-Agent": _USER_AGENT, "X-Keenable-Title": _ATTRIBUTION_TITLE}
        if key is not None:
            headers["X-API-Key"] = key
            path = keyed_path
        else:
            path = public_path
        return f"{_resolve_base_url()}{path}", headers

    def _handle(self, response: requests.Response) -> dict[str, Any]:
        """Validate status and decode a JSON object body, or raise ToolException."""
        key = self._effective_key()
        _raise_for_keenable_status(response, key)
        try:
            data = response.json()
        except ValueError as e:
            snippet = _redact((response.text or "")[:200], key)
            msg = f"Keenable API returned a non-JSON response: {snippet!r}"
            raise ToolException(msg) from e

        if not isinstance(data, dict):
            msg = (
                "Unexpected response from the Keenable API: "
                f"{_redact(repr(data)[:200], key)}"
            )
            raise ToolException(msg)
        return data

    def _post(
        self, public_path: str, keyed_path: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """POST ``payload`` to the keyed or keyless endpoint and return the body."""
        url, headers = self._url_and_headers(public_path, keyed_path)
        headers["Content-Type"] = "application/json"
        try:
            response = requests.post(
                url, json=payload, headers=headers, timeout=self.timeout
            )
        except requests.RequestException as e:
            key = self._effective_key()
            msg = (
                f"Could not reach the Keenable API: "
                f"{type(e).__name__}: {_redact(str(e), key)}"
            )
            raise ToolException(msg) from e
        return self._handle(response)

    def _get(
        self, public_path: str, keyed_path: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        """GET the keyed or keyless endpoint with query ``params``; return the body."""
        url, headers = self._url_and_headers(public_path, keyed_path)
        try:
            response = requests.get(
                url, params=params, headers=headers, timeout=self.timeout
            )
        except requests.RequestException as e:
            key = self._effective_key()
            msg = (
                f"Could not reach the Keenable API: "
                f"{type(e).__name__}: {_redact(str(e), key)}"
            )
            raise ToolException(msg) from e
        return self._handle(response)


class KeenableSearchInput(BaseModel):
    """Input for the Keenable search tool. All filters are per-invocation so an
    agent can vary them per query."""

    query: str = Field(description="search query to look up")
    site: str | None = Field(
        default=None,
        description="restrict results to a single domain, e.g. 'techcrunch.com'",
    )
    published_after: str | None = Field(
        default=None,
        description="only pages published on or after this date (YYYY-MM-DD)",
    )
    published_before: str | None = Field(
        default=None,
        description="only pages published on or before this date (YYYY-MM-DD)",
    )
    acquired_after: str | None = Field(
        default=None,
        description="only pages indexed on or after this date (YYYY-MM-DD)",
    )
    acquired_before: str | None = Field(
        default=None,
        description="only pages indexed on or before this date (YYYY-MM-DD)",
    )
    mode: Literal["pro", "realtime"] | None = Field(
        default=None,
        description="search mode: 'pro' (default, deeper) or 'realtime' (low latency)",
    )


class KeenableSearch(_KeenableBaseTool):
    """Tool that queries the Keenable Search API and returns JSON results.

    Keenable is a web search API built for AI agents. The API key can be passed
    as the ``api_key`` argument or via the ``KEENABLE_API_KEY`` environment
    variable. When no key is configured, the tool falls back to the keyless
    public search endpoint (``/v1/search/public``).

    Setup:
        Install ``langchain-keenable``. Optionally set the environment variable
        ``KEENABLE_API_KEY`` (create a key at https://keenable.ai/console) to use
        the authenticated endpoint.

        .. code-block:: bash

            pip install -U langchain-keenable
            export KEENABLE_API_KEY="your-api-key"  # optional

    Instantiate:

        .. code-block:: python

            from langchain_keenable import KeenableSearch

            tool = KeenableSearch()

    Invoke directly with args:

        .. code-block:: python

            tool.invoke({"query": "typescript best practices", "site": "github.com"})

        .. code-block:: python

            [
                {
                    "title": "TypeScript Best Practices 2026",
                    "url": "https://example.com/ts-best-practices",
                    "description": "A comprehensive guide ...",
                    "published_at": "2026-01-15T10:30:00Z",
                    "acquired_at": "2026-01-16T08:12:34Z",
                },
            ]
    """

    name: str = "keenable_search"
    description: str = (
        "A web search engine built for AI agents, powered by Keenable. "
        "Useful for when you need to answer questions about current events or "
        "look up information on the web. Input should be a search query, "
        "optionally filtered by site or publication/index date. "
        "Output is a JSON list of the search results."
    )
    args_schema: type[BaseModel] = KeenableSearchInput

    mode: Literal["pro", "realtime"] = "pro"
    """Default search mode, overridable per invocation. ``"pro"`` (default) does
    deeper retrieval with higher result quality; use ``"realtime"`` for
    latency-sensitive cases such as voice agents. ``"realtime"`` requires an org
    key (it is not enabled on the keyless public endpoint)."""

    def _search(
        self,
        query: str,
        site: str | None,
        published_after: str | None,
        published_before: str | None,
        acquired_after: str | None,
        acquired_before: str | None,
        mode: str | None,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {"query": query, "mode": mode or self.mode}
        for field, value in (
            ("site", site),
            ("published_after", published_after),
            ("published_before", published_before),
            ("acquired_after", acquired_after),
            ("acquired_before", acquired_before),
        ):
            if value:
                payload[field] = value

        data = self._post("/v1/search/public", "/v1/search", payload)
        results = data.get("results")
        if not isinstance(results, list):
            msg = (
                "Unexpected response from the Keenable search API: "
                f"{_redact(repr(data)[:200], self._effective_key())}"
            )
            raise ToolException(msg)
        # The API returns a fixed-size result set; it is returned as-is (there is
        # no max_results parameter to honor).
        return results

    def _run(
        self,
        query: str,
        site: str | None = None,
        published_after: str | None = None,
        published_before: str | None = None,
        acquired_after: str | None = None,
        acquired_before: str | None = None,
        mode: str | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> list[dict[str, Any]]:
        """Query the Keenable search API."""
        return self._search(
            query,
            site,
            published_after,
            published_before,
            acquired_after,
            acquired_before,
            mode,
        )

    async def _arun(
        self,
        query: str,
        site: str | None = None,
        published_after: str | None = None,
        published_before: str | None = None,
        acquired_after: str | None = None,
        acquired_before: str | None = None,
        mode: str | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> list[dict[str, Any]]:
        """Async query (runs the sync request in a worker thread)."""
        return await asyncio.to_thread(
            self._search,
            query,
            site,
            published_after,
            published_before,
            acquired_after,
            acquired_before,
            mode,
        )


class KeenableFetchInput(BaseModel):
    """Input for the Keenable fetch tool."""

    url: str = Field(description="the URL of the page to fetch and extract as markdown")


class KeenableFetch(_KeenableBaseTool):
    """Tool that fetches a web page via Keenable and returns its content as markdown.

    Wraps the Keenable ``GET /v1/fetch?url=...`` endpoint: given a URL it returns
    ``{url, title, content, ...}`` (extra fields such as ``description``, ``author``
    and ``published_at`` appear when the page exposes them). Like search, it uses the
    keyless ``/v1/fetch/public`` endpoint when no key is configured. Pairs with
    :class:`KeenableSearch` — an agent discovers URLs via search, then reads the full
    page content with this tool.

    Instantiate:

        .. code-block:: python

            from langchain_keenable import KeenableFetch

            tool = KeenableFetch()
            tool.invoke({"url": "https://example.com/article"})
    """

    name: str = "keenable_fetch"
    description: str = (
        "Fetch a web page via Keenable and return its main content as markdown, "
        "along with the title, description, author, and publication date. "
        "Input should be a single URL. Use this to read a page found via search."
    )
    args_schema: type[BaseModel] = KeenableFetchInput

    def _fetch(self, url: str) -> dict[str, Any]:
        if not url.lower().startswith(("http://", "https://")):
            msg = f"Refusing to fetch a non-http(s) URL: {url!r}"
            raise ToolException(msg)
        _reject_private_fetch_target(url)
        # /v1/fetch is a GET with a ?url= query param (keyless /v1/fetch/public
        # works too). The backend also rejects private/internal IPs server-side.
        return self._get("/v1/fetch/public", "/v1/fetch", {"url": url})

    def _run(
        self,
        url: str,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> dict[str, Any]:
        """Fetch a page and return its extracted content."""
        return self._fetch(url)

    async def _arun(
        self,
        url: str,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> dict[str, Any]:
        """Async fetch (runs the sync request in a worker thread)."""
        return await asyncio.to_thread(self._fetch, url)
