from __future__ import annotations

from inference_optimizer.benchmark.token_counter import (
    FixedTokenCounter,
    HFTokenizerCounter,
    WhitespaceApproxCounter,
)


class TestFixedTokenCounter:
    def test_returns_predetermined_counts_in_order(self) -> None:
        counter = FixedTokenCounter([3, 7, 1])
        assert counter.count("ignored") == 3
        assert counter.count("also ignored") == 7
        assert counter.count("still ignored") == 1

    def test_returns_none_once_exhausted(self) -> None:
        counter = FixedTokenCounter([5])
        assert counter.count("x") == 5
        assert counter.count("x") is None

    def test_method_is_fixed(self) -> None:
        assert FixedTokenCounter([1]).method == "fixed"


class TestWhitespaceApproxCounter:
    def test_counts_words(self) -> None:
        counter = WhitespaceApproxCounter()
        assert counter.count("one two three") == 3

    def test_empty_text_is_zero(self) -> None:
        assert WhitespaceApproxCounter().count("") == 0

    def test_method_is_approximate(self) -> None:
        assert WhitespaceApproxCounter().method == "approximate"


class TestHFTokenizerCounter:
    def test_degrades_to_unavailable_when_load_fails(self) -> None:
        # A bogus repo id can never resolve -- exercises the graceful-degradation
        # path (missing dep or bad repo) without needing network access to succeed.
        counter = HFTokenizerCounter("this/repo-does-not-exist-12345")
        assert counter.count("hello world") is None
        assert counter.method == "unavailable"

    def test_repeated_failed_loads_do_not_retry_every_call(self) -> None:
        counter = HFTokenizerCounter("this/repo-does-not-exist-12345")
        counter.count("first call triggers the failed load")
        assert counter._load_failed is True  # noqa: SLF001 - verifying the cache behavior
        # Second call must short-circuit via _load_failed, not attempt to load again.
        assert counter.count("second call") is None
