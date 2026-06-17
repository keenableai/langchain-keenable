# langchain-keenable

This package contains the LangChain integration with [Keenable](https://keenable.ai), a web search
and page-fetch API built for AI agents.

## Installation

```bash
pip install -U langchain-keenable
```

Optionally set a `KEENABLE_API_KEY` environment variable to use the authenticated endpoints. Without a
key, both search and fetch transparently fall back to their keyless public endpoints.

```bash
export KEENABLE_API_KEY="your-api-key"  # optional; create one at https://keenable.ai/console
```

The API endpoint defaults to `https://api.keenable.ai` and can be overridden (e.g. for staging) with the
`KEENABLE_API_URL` environment variable. It must be an `https://` URL.

## Tools

### `KeenableSearch`

Queries the Keenable search API and returns a list of result dictionaries. All filters are
**per-invocation**, so an agent can vary them per query:

```python
from langchain_keenable import KeenableSearch

# Works with no key (keyless public endpoint) or with KEENABLE_API_KEY set.
tool = KeenableSearch()

results = tool.invoke({
    "query": "typescript best practices",
    "site": "github.com",          # optional: restrict to a domain
    "published_after": "2026-01-01",  # optional: YYYY-MM-DD date filters
    # "published_before" / "acquired_after" / "acquired_before" also supported
    # "mode": "realtime",          # optional per-call override (needs an org key)
})
for result in results:
    print(result["title"], result["url"])
```

`mode` defaults to `"pro"` (deeper retrieval); use `"realtime"` for latency-sensitive cases such as
voice agents. It can be set as a class default and overridden per call. `realtime` requires an org key.

### `KeenableFetch`

Fetches a page via Keenable and returns its main content as markdown — pair it with `KeenableSearch` so
an agent can read the pages it discovers:

```python
from langchain_keenable import KeenableFetch

tool = KeenableFetch()
page = tool.invoke({"url": "https://example.com/article"})
print(page["title"], page["content"])
```

## Error handling

Both tools set `handle_tool_error = True`: rate limits (429), auth (401) and credit (402) errors, network
timeouts and malformed responses are surfaced to the agent as an error **string** (carrying the backend's
message) rather than raising and crashing the agent loop.

## Async

Both tools implement `_arun`, so `await tool.ainvoke({...})` works (the request runs in a worker thread).

The tools can be bound to any LangChain chat model that supports tool calling and used within an agent.
