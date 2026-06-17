"""Standard LangChain unit tests for the Keenable tools."""

from typing import Any

from langchain_tests.unit_tests import ToolsUnitTests

from langchain_keenable import KeenableFetch, KeenableSearch


class TestKeenableSearchUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[KeenableSearch]:
        return KeenableSearch

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"api_key": "test-key"}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"query": "what is the weather in SF"}


class TestKeenableFetchUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[KeenableFetch]:
        return KeenableFetch

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"api_key": "test-key"}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {"url": "https://example.com"}
