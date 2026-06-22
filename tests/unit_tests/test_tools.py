"""Unit tests for the Keenable tools (mocked HTTP, no network)."""

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests
from langchain_core.tools import ToolException

from langchain_keenable import KeenableFetch, KeenableSearch, KeenableSearchInput
from langchain_keenable.tools import (
    _reject_private_fetch_target,
    _resolve_base_url,
)

_FAKE_RESULTS = [
    {
        "title": "TypeScript Best Practices 2026",
        "url": "https://example.com/ts-best-practices",
        "description": "A comprehensive guide to modern TypeScript.",
        "published_at": "2026-01-15T10:30:00Z",
        "acquired_at": "2026-01-16T08:12:34Z",
    },
    {
        "title": "Second result",
        "url": "https://example.com/second",
        "description": "Another result.",
        "published_at": "2026-01-10T00:00:00Z",
        "acquired_at": "2026-01-11T00:00:00Z",
    },
]

_FAKE_PAGE = {
    "url": "https://example.com/article",
    "title": "An Article",
    "content": "# An Article\n\nBody text.",
    "description": "An example article.",
    "author": "Jane Doe",
    "published_at": "2026-01-15T10:30:00Z",
}


def _ok_response(payload: Any) -> MagicMock:
    response = MagicMock(spec=requests.Response)
    response.ok = True
    response.status_code = 200
    response.json.return_value = payload
    return response


def _error_response(status: int, payload: Any = None, text: str = "") -> MagicMock:
    response = MagicMock(spec=requests.Response)
    response.ok = False
    response.status_code = status
    response.text = text
    if payload is None:
        response.json.side_effect = ValueError("no json")
    else:
        response.json.return_value = payload
    return response


@pytest.fixture
def mock_post() -> Generator[MagicMock, None, None]:
    with patch("langchain_keenable.tools.requests.post") as mock:
        mock.return_value = _ok_response({"results": _FAKE_RESULTS})
        yield mock


@pytest.fixture
def mock_get() -> Generator[MagicMock, None, None]:
    with patch("langchain_keenable.tools.requests.get") as mock:
        mock.return_value = _ok_response(_FAKE_PAGE)
        yield mock


# --- happy path & wire format -------------------------------------------------


def test_run_returns_results(mock_post: MagicMock) -> None:
    """The tool returns the parsed ``results`` list, unmodified."""
    tool = KeenableSearch(api_key="fake-key")  # type: ignore[arg-type]
    results = tool.invoke({"query": "typescript best practices"})

    assert results == _FAKE_RESULTS
    assert mock_post.call_args.kwargs["json"] == {
        "query": "typescript best practices",
        "mode": "pro",
    }


def test_run_sends_api_key_and_user_agent(mock_post: MagicMock) -> None:
    """The request carries the X-API-Key and a tagged User-Agent."""
    tool = KeenableSearch(api_key="fake-key")  # type: ignore[arg-type]
    tool.invoke({"query": "anything"})

    headers = mock_post.call_args.kwargs["headers"]
    assert headers["X-API-Key"] == "fake-key"
    assert headers["User-Agent"].startswith("keenable-langchain/")


def test_mode_defaults_to_pro(mock_post: MagicMock) -> None:
    tool = KeenableSearch(api_key="fake-key")  # type: ignore[arg-type]
    tool.invoke({"query": "anything"})
    assert mock_post.call_args.kwargs["json"]["mode"] == "pro"


def test_mode_class_default_override(mock_post: MagicMock) -> None:
    """A class-level ``mode`` default is used when no per-call mode is given."""
    tool = KeenableSearch(api_key="fake-key", mode="realtime")  # type: ignore[arg-type]
    tool.invoke({"query": "anything"})
    assert mock_post.call_args.kwargs["json"]["mode"] == "realtime"


def test_mode_per_invocation_override(mock_post: MagicMock) -> None:
    """A per-invocation ``mode`` wins over the class default."""
    tool = KeenableSearch(api_key="fake-key", mode="pro")  # type: ignore[arg-type]
    tool.invoke({"query": "anything", "mode": "realtime"})
    assert mock_post.call_args.kwargs["json"]["mode"] == "realtime"


def test_filters_are_per_invocation(mock_post: MagicMock) -> None:
    """site and date filters are sent from the invocation args, not config."""
    tool = KeenableSearch(api_key="fake-key")  # type: ignore[arg-type]
    tool.invoke(
        {
            "query": "anything",
            "site": "example.com",
            "published_after": "2026-01-01",
            "published_before": "2026-02-01",
            "acquired_after": "2026-01-05",
            "acquired_before": "2026-02-05",
        }
    )
    body = mock_post.call_args.kwargs["json"]
    assert body["site"] == "example.com"
    assert body["published_after"] == "2026-01-01"
    assert body["published_before"] == "2026-02-01"
    assert body["acquired_after"] == "2026-01-05"
    assert body["acquired_before"] == "2026-02-05"


def test_omitted_filters_not_in_payload(mock_post: MagicMock) -> None:
    """Unset filters are not sent at all."""
    tool = KeenableSearch(api_key="fake-key")  # type: ignore[arg-type]
    tool.invoke({"query": "anything"})
    body = mock_post.call_args.kwargs["json"]
    assert set(body) == {"query", "mode"}


def test_no_max_results_param() -> None:
    """The deprecated client-side ``max_results`` knob no longer exists."""
    assert "max_results" not in KeenableSearch.model_fields
    assert "max_results" not in KeenableSearchInput.model_fields


# --- endpoint / auth selection ------------------------------------------------


def test_api_key_from_env(
    mock_post: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KEENABLE_API_KEY", "env-key")
    tool = KeenableSearch()
    tool.invoke({"query": "anything"})
    assert mock_post.call_args.kwargs["headers"]["X-API-Key"] == "env-key"


def test_no_api_key_uses_public_endpoint(
    mock_post: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    tool = KeenableSearch()
    results = tool.invoke({"query": "anything"})

    assert results == _FAKE_RESULTS
    assert mock_post.call_args.args[0] == "https://api.keenable.ai/v1/search/public"
    assert "X-API-Key" not in mock_post.call_args.kwargs["headers"]


def test_api_key_uses_authenticated_endpoint(mock_post: MagicMock) -> None:
    tool = KeenableSearch(api_key="fake-key")  # type: ignore[arg-type]
    tool.invoke({"query": "anything"})
    assert mock_post.call_args.args[0] == "https://api.keenable.ai/v1/search"


def test_blank_api_key_falls_back_to_public(
    mock_post: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blank/whitespace key is treated as no key (free tier)."""
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    tool = KeenableSearch(api_key="   ")  # type: ignore[arg-type]
    tool.invoke({"query": "anything"})
    assert mock_post.call_args.args[0].endswith("/v1/search/public")
    assert "X-API-Key" not in mock_post.call_args.kwargs["headers"]


def test_base_url_from_env(
    mock_post: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The endpoint comes from KEENABLE_API_URL, not a public param."""
    monkeypatch.setenv("KEENABLE_API_URL", "https://staging.keenable.ai")
    tool = KeenableSearch(api_key="fake-key")  # type: ignore[arg-type]
    tool.invoke({"query": "anything"})
    assert mock_post.call_args.args[0] == "https://staging.keenable.ai/v1/search"


def test_base_url_is_not_a_public_param() -> None:
    """base_url must not be a settable field (SSRF foothold)."""
    assert "base_url" not in KeenableSearch.model_fields


def test_non_https_base_url_rejected(
    mock_post: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KEENABLE_API_URL", "http://evil.example.com")
    tool = KeenableSearch(api_key="fake-key")  # type: ignore[arg-type]
    out = tool.invoke({"query": "anything"})
    assert "https" in out.lower()
    mock_post.assert_not_called()


@pytest.mark.parametrize("bad_url", ["https://", "https:///path", "notaurl"])
def test_hostless_base_url_rejected(
    mock_post: MagicMock, monkeypatch: pytest.MonkeyPatch, bad_url: str
) -> None:
    """A malformed KEENABLE_API_URL with no host fails fast, not at request time."""
    monkeypatch.setenv("KEENABLE_API_URL", bad_url)
    tool = KeenableSearch(api_key="fake-key")  # type: ignore[arg-type]
    out = tool.invoke({"query": "anything"})
    assert isinstance(out, str)
    assert "host" in out.lower()
    mock_post.assert_not_called()


# --- error handling -----------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "needle"),
    [
        (401, "authentication failed (401)"),
        (402, "insufficient credits (402)"),
        (429, "rate limit exceeded (429)"),
        (500, "api error (500)"),
    ],
)
def test_http_errors_are_returned_not_raised(
    mock_post: MagicMock, status: int, needle: str
) -> None:
    """Errors come back to the agent as a string (handle_tool_error), surfacing
    the backend's message — they do not crash the agent loop."""
    mock_post.return_value = _error_response(
        status, {"message": "go upgrade at docs.keenable.ai"}
    )
    tool = KeenableSearch(api_key="fake-key")  # type: ignore[arg-type]

    out = tool.invoke({"query": "anything"})
    assert isinstance(out, str)
    assert needle in out.lower()
    assert "upgrade" in out


def test_realtime_feature_not_enabled_is_returned(mock_post: MagicMock) -> None:
    """A 403 'feature not enabled' (realtime on the keyless endpoint) is surfaced
    to the agent as a string, not raised."""
    mock_post.return_value = _error_response(
        403,
        {
            "error": "Feature not enabled",
            "message": "Realtime search mode is not enabled for your organization.",
        },
    )
    tool = KeenableSearch()
    out = tool.invoke({"query": "anything", "mode": "realtime"})

    assert isinstance(out, str)
    assert "403" in out
    assert "realtime" in out.lower()
    # the forwarded mode reached the request body
    assert mock_post.call_args.kwargs["json"]["mode"] == "realtime"


def test_network_timeout_is_returned(mock_post: MagicMock) -> None:
    mock_post.side_effect = requests.Timeout("timed out")
    tool = KeenableSearch(api_key="fake-key")  # type: ignore[arg-type]

    out = tool.invoke({"query": "anything"})
    assert isinstance(out, str)
    assert "could not reach" in out.lower()


def test_non_json_body_is_returned(mock_post: MagicMock) -> None:
    """A non-JSON body (e.g. an HTML 502) does not escape as a raw decode error."""
    bad = MagicMock(spec=requests.Response)
    bad.ok = True
    bad.status_code = 200
    bad.json.side_effect = ValueError("no json")
    bad.text = "<html>502 Bad Gateway</html>"
    mock_post.return_value = bad
    tool = KeenableSearch(api_key="fake-key")  # type: ignore[arg-type]

    out = tool.invoke({"query": "anything"})
    assert isinstance(out, str)
    assert "non-json" in out.lower()


def test_malformed_results_is_returned(mock_post: MagicMock) -> None:
    """A response whose ``results`` is missing/not-a-list is reported, not crashed."""
    mock_post.return_value = _ok_response({"error": "boom"})
    tool = KeenableSearch(api_key="fake-key")  # type: ignore[arg-type]

    out = tool.invoke({"query": "anything"})
    assert isinstance(out, str)
    assert "unexpected response" in out.lower()


# --- secrets hygiene ----------------------------------------------------------


def test_api_key_not_exposed_in_repr() -> None:
    tool = KeenableSearch(api_key="super-secret")  # type: ignore[arg-type]
    assert "super-secret" not in repr(tool)
    assert "super-secret" not in str(tool.model_dump())


# --- fetch tool ---------------------------------------------------------------


def test_fetch_returns_page(mock_get: MagicMock) -> None:
    """Fetch is a GET with a ?url= query param against /v1/fetch."""
    tool = KeenableFetch(api_key="fake-key")  # type: ignore[arg-type]

    out = tool.invoke({"url": "https://example.com/article"})
    assert out == _FAKE_PAGE
    assert mock_get.call_args.kwargs["params"] == {"url": "https://example.com/article"}
    assert mock_get.call_args.args[0] == "https://api.keenable.ai/v1/fetch"
    assert mock_get.call_args.kwargs["headers"]["X-API-Key"] == "fake-key"


def test_fetch_without_key_uses_public_path(
    mock_get: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("KEENABLE_API_KEY", raising=False)
    tool = KeenableFetch()
    tool.invoke({"url": "https://example.com/article"})
    assert mock_get.call_args.args[0] == "https://api.keenable.ai/v1/fetch/public"
    assert "X-API-Key" not in mock_get.call_args.kwargs["headers"]


def test_fetch_rejects_non_http_url(mock_get: MagicMock) -> None:
    """SSRF guard: non-http(s) schemes are refused before any request."""
    tool = KeenableFetch(api_key="fake-key")  # type: ignore[arg-type]
    out = tool.invoke({"url": "file:///etc/passwd"})
    assert isinstance(out, str)
    assert "non-http" in out.lower()
    mock_get.assert_not_called()


def test_fetch_user_agent(mock_get: MagicMock) -> None:
    tool = KeenableFetch(api_key="fake-key")  # type: ignore[arg-type]
    tool.invoke({"url": "https://example.com"})
    assert mock_get.call_args.kwargs["headers"]["User-Agent"].startswith(
        "keenable-langchain/"
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "https://localhost/secret",
        "http://[::1]/",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata
        "https://10.0.0.5/internal",
        "http://192.168.1.1/",
        "https://metadata.google.internal/",
    ],
)
def test_fetch_rejects_private_hosts(mock_get: MagicMock, url: str) -> None:
    """SSRF guard: loopback / private / link-local / metadata targets are refused
    before any request is sent (the backend also blocks them server-side)."""
    tool = KeenableFetch(api_key="fake-key")  # type: ignore[arg-type]
    out = tool.invoke({"url": url})
    assert isinstance(out, str)
    assert "private/internal" in out.lower()
    mock_get.assert_not_called()


@pytest.mark.parametrize(
    ("status", "needle"),
    [
        (401, "authentication failed (401)"),
        (402, "insufficient credits (402)"),
        (429, "rate limit exceeded (429)"),
        (500, "api error (500)"),
    ],
)
def test_fetch_http_errors_are_returned_not_raised(
    mock_get: MagicMock, status: int, needle: str
) -> None:
    """Fetch surfaces backend errors to the agent as a string, like search."""
    mock_get.return_value = _error_response(
        status, {"message": "go upgrade at docs.keenable.ai"}
    )
    tool = KeenableFetch(api_key="fake-key")  # type: ignore[arg-type]

    out = tool.invoke({"url": "https://example.com/article"})
    assert isinstance(out, str)
    assert needle in out.lower()
    assert "upgrade" in out


def test_fetch_non_json_body_is_returned(mock_get: MagicMock) -> None:
    """A non-JSON fetch body (e.g. an HTML 502) doesn't escape as a decode error."""
    bad = MagicMock(spec=requests.Response)
    bad.ok = True
    bad.status_code = 200
    bad.json.side_effect = ValueError("no json")
    bad.text = "<html>502 Bad Gateway</html>"
    mock_get.return_value = bad
    tool = KeenableFetch(api_key="fake-key")  # type: ignore[arg-type]

    out = tool.invoke({"url": "https://example.com/article"})
    assert isinstance(out, str)
    assert "non-json" in out.lower()


def test_fetch_network_timeout_is_returned(mock_get: MagicMock) -> None:
    mock_get.side_effect = requests.Timeout("timed out")
    tool = KeenableFetch(api_key="fake-key")  # type: ignore[arg-type]

    out = tool.invoke({"url": "https://example.com/article"})
    assert isinstance(out, str)
    assert "could not reach" in out.lower()


# --- attribution header -------------------------------------------------------


def test_search_sends_attribution_title(mock_post: MagicMock) -> None:
    """Search tags traffic with X-Keenable-Title for adoption attribution."""
    tool = KeenableSearch(api_key="fake-key")  # type: ignore[arg-type]
    tool.invoke({"query": "anything"})
    assert mock_post.call_args.kwargs["headers"]["X-Keenable-Title"] == "LangChain"


def test_fetch_sends_attribution_title(mock_get: MagicMock) -> None:
    tool = KeenableFetch(api_key="fake-key")  # type: ignore[arg-type]
    tool.invoke({"url": "https://example.com"})
    assert mock_get.call_args.kwargs["headers"]["X-Keenable-Title"] == "LangChain"


# --- SSRF guard: numeric-IP + trailing-dot bypasses ---------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://2130706433/secret",  # decimal form of 127.0.0.1
        "http://0x7f000001/secret",  # hex form of 127.0.0.1
        "http://0177.0.0.1/secret",  # octal-dotted form of 127.0.0.1
        "http://127.1/secret",  # short a.b form of 127.0.0.1
        "http://127.0.0.1./secret",  # trailing dot on IP literal
        "http://localhost./secret",  # trailing dot on hostname
        "http://LOCALHOST/secret",  # case
        "http://metadata.google.internal./secret",  # trailing dot on metadata host
    ],
)
def test_reject_ssrf_bypass_encodings(url: str) -> None:
    """Numeric/short/trailing-dot encodings of a private address are rejected."""
    with pytest.raises(ToolException):
        _reject_private_fetch_target(url)


def test_reject_ssrf_bypass_encodings_via_tool(mock_get: MagicMock) -> None:
    """The fetch tool surfaces the rejection as a string without sending."""
    tool = KeenableFetch(api_key="fake-key")  # type: ignore[arg-type]
    out = tool.invoke({"url": "http://2130706433/secret"})
    assert isinstance(out, str)
    assert "private/internal" in out.lower()
    mock_get.assert_not_called()


def test_public_numeric_ip_allowed(mock_get: MagicMock) -> None:
    """A globally routable numeric IP must still pass the guard."""
    # Direct helper: no exception.
    _reject_private_fetch_target("https://8.8.8.8/x")
    # And through the tool it reaches the request.
    tool = KeenableFetch(api_key="fake-key")  # type: ignore[arg-type]
    tool.invoke({"url": "https://8.8.8.8/x"})
    assert mock_get.call_args.kwargs["params"] == {"url": "https://8.8.8.8/x"}


# --- base URL rejects private/internal https hosts ----------------------------


@pytest.mark.parametrize(
    "bad_base",
    [
        "https://169.254.169.254",  # AWS/GCP metadata, link-local
        "https://metadata.google.internal",  # GCP metadata host
        "https://[fe80::1]",  # IPv6 link-local
        "https://10.0.0.1",  # RFC1918 private
        "https://192.168.1.1",  # RFC1918 private
        "https://172.16.0.1",  # RFC1918 private
        "https://127.0.0.1",  # loopback over https
        "https://2130706433",  # decimal-encoded loopback
        "https://0x7f000001",  # hex-encoded loopback
    ],
)
def test_base_url_rejects_private_and_metadata(
    monkeypatch: pytest.MonkeyPatch, bad_base: str
) -> None:
    """A KEENABLE_API_URL pointing at an internal host must not ship the key."""
    monkeypatch.setenv("KEENABLE_API_URL", bad_base)
    with pytest.raises(ToolException):
        _resolve_base_url()


def test_base_url_private_rejected_via_tool(
    mock_post: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A private base URL is surfaced as a string and no request is sent."""
    monkeypatch.setenv("KEENABLE_API_URL", "https://169.254.169.254")
    tool = KeenableSearch(api_key="fake-key")  # type: ignore[arg-type]
    out = tool.invoke({"query": "anything"})
    assert isinstance(out, str)
    assert "private/internal" in out.lower()
    mock_post.assert_not_called()


def test_base_url_http_loopback_still_allowed(
    mock_post: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The local-dev http loopback escape hatch is preserved."""
    monkeypatch.setenv("KEENABLE_API_URL", "http://localhost:8000")
    assert _resolve_base_url() == "http://localhost:8000"


# --- error/key-leak redaction -------------------------------------------------


@pytest.mark.parametrize("status", [401, 402, 429, 500])
def test_error_body_redacts_echoed_key(mock_post: MagicMock, status: int) -> None:
    """A server that echoes the X-API-Key in its error body must not leak it into
    the surfaced error message — redacted for every status, not just 401."""
    key = "sk-leak-123"
    mock_post.return_value = _error_response(status, {"message": f"got key={key}"})
    tool = KeenableSearch(api_key=key)  # type: ignore[arg-type]
    out = tool.invoke({"query": "anything"})
    assert isinstance(out, str)
    assert key not in out
    assert "***" in out


def test_transport_error_redacts_key_and_no_repr(mock_post: MagicMock) -> None:
    """A transport exception surfaces type + redacted message, not the raw repr."""
    key = "sk-transport-secret-XYZ"

    class _Boom(requests.RequestException):
        def __repr__(self) -> str:
            return "Boom(secret-in-repr)"

    mock_post.side_effect = _Boom(f"failed with header key={key}")
    tool = KeenableSearch(api_key=key)  # type: ignore[arg-type]
    out = tool.invoke({"query": "anything"})
    assert isinstance(out, str)
    assert key not in out
    assert "secret-in-repr" not in out
    assert "***" in out
    assert "_Boom" in out


def test_non_json_body_redacts_key(mock_post: MagicMock) -> None:
    """A non-JSON body that echoes the key must not leak it into the message."""
    key = "sk-in-html-body"
    bad = MagicMock(spec=requests.Response)
    bad.ok = True
    bad.status_code = 200
    bad.json.side_effect = ValueError("no json")
    bad.text = f"<html>error key={key}</html>"
    mock_post.return_value = bad
    tool = KeenableSearch(api_key=key)  # type: ignore[arg-type]
    out = tool.invoke({"query": "anything"})
    assert isinstance(out, str)
    assert key not in out
    assert "***" in out


def test_search_bad_payload_redacts_key(mock_post: MagicMock) -> None:
    """A 200 body lacking a valid `results` but echoing the key must not leak it."""
    key = "sk-echo-in-200-body"
    mock_post.return_value = _ok_response({"results": None, "debug": f"key={key}"})
    tool = KeenableSearch(api_key=key)  # type: ignore[arg-type]
    out = tool.invoke({"query": "anything"})
    assert isinstance(out, str)
    assert "unexpected response" in out.lower()
    assert key not in out
    assert "***" in out
