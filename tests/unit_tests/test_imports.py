"""Test that the public API of langchain_keenable is stable."""

from langchain_keenable import __all__

EXPECTED_ALL = [
    "KeenableFetch",
    "KeenableFetchInput",
    "KeenableSearch",
    "KeenableSearchInput",
]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)
