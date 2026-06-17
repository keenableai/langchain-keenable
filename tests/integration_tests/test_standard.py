"""Standard LangChain integration tests for the Keenable tools.

These make live calls. Both tools run against the keyless public endpoints, so no
API key is required; set ``KEENABLE_API_KEY`` to exercise the authenticated ones.
"""

from typing import Any

from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_keenable import KeenableFetch, KeenableSearch


class TestKeenableSearchIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[KeenableSearch]:
        return KeenableSearch

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        # Empty -> keyless public endpoint (falls back automatically).
        return {}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "what is the weather in SF"}


class TestKeenableFetchIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[KeenableFetch]:
        return KeenableFetch

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"url": "https://example.com"}
