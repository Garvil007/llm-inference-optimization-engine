"""Deterministic benchmark-client tests. No real sleeps: timing is driven entirely by
an injected fake clock (a predetermined sequence of perf_counter-like values), and
streaming "delay" is simulated by the fact that MockTransport's response body is
consumed instantly while the fake clock supplies whatever timestamps the test wants --
decoupling the test from real wall-clock time entirely.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Iterator

import httpx
import pytest

from inference_optimizer.benchmark.client import BenchmarkClient, _consume_sse_stream
from inference_optimizer.benchmark.token_counter import FixedTokenCounter


def _fake_clock(values: list[float]) -> Callable[[], float]:
    iterator: Iterator[float] = iter(values)
    return lambda: next(iterator)


async def _lines(items: list[str]) -> AsyncIterator[str]:
    for item in items:
        yield item


class TestConsumeSseStream:
    @pytest.mark.asyncio
    async def test_normal_stream_captures_three_clock_samples(self) -> None:
        clock = _fake_clock([0.0, 0.05, 0.2])
        lines = _lines(
            ['data: {"text": "Hello"}', 'data: {"text": " world"}', "data: [DONE]"]
        )

        text, start, first_token, end, error = await _consume_sse_stream(lines, clock)

        assert text == "Hello world"
        assert start == 0.0
        assert first_token == 0.05
        assert end == 0.2
        assert error is None

    @pytest.mark.asyncio
    async def test_empty_stream_never_samples_first_token_clock(self) -> None:
        clock = _fake_clock([0.0, 0.1])  # only start + end -- no first-token call
        lines = _lines(["data: [DONE]"])

        text, start, first_token, end, error = await _consume_sse_stream(lines, clock)

        assert text == ""
        assert first_token is None
        assert error is None

    @pytest.mark.asyncio
    async def test_error_event_mid_stream_preserves_partial_text(self) -> None:
        clock = _fake_clock([0.0, 0.05, 0.1])
        lines = _lines(['data: {"text": "partial"}', 'data: {"error": "boom"}'])

        text, start, first_token, end, error = await _consume_sse_stream(lines, clock)

        assert text == "partial"
        assert first_token == 0.05
        assert error == "boom"

    @pytest.mark.asyncio
    async def test_non_data_lines_are_ignored(self) -> None:
        clock = _fake_clock([0.0, 0.01, 0.02])
        lines = _lines([": comment", 'data: {"text": "hi"}', "data: [DONE]"])

        text, *_rest = await _consume_sse_stream(lines, clock)

        assert text == "hi"


class TestBenchmarkClientRunOne:
    @pytest.mark.asyncio
    async def test_success_computes_ttft_latency_tpot_throughput(self) -> None:
        sse_body = (
            'data: {"text": "Hello"}\n\n'
            'data: {"text": " world"}\n\n'
            "data: [DONE]\n\n"
        )

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=sse_body)

        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://test"
        )
        client = BenchmarkClient(
            base_url="http://test",
            token_counter=FixedTokenCounter([4]),
            http_client=http_client,
            clock=_fake_clock([0.0, 0.05, 0.2]),
        )

        observation = await client.run_one(0, "hi")

        assert observation.success is True
        assert observation.ttft_ms == pytest.approx(50.0)
        assert observation.latency_ms == pytest.approx(200.0)
        assert observation.output_token_count == 4
        assert observation.token_count_method == "fixed"
        # tpot = (latency - ttft) / (tokens - 1) = (200 - 50) / 3
        assert observation.tpot_ms == pytest.approx(50.0)
        # throughput = tokens / (latency_ms / 1000) = 4 / 0.2
        assert observation.output_tokens_per_second == pytest.approx(20.0)

    @pytest.mark.asyncio
    async def test_http_error_status_produces_failed_observation(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, content=b"unavailable")

        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://test"
        )
        client = BenchmarkClient(
            base_url="http://test",
            token_counter=FixedTokenCounter([4]),
            http_client=http_client,
            clock=_fake_clock([0.0, 0.0]),
        )

        observation = await client.run_one(0, "hi")

        assert observation.success is False
        assert observation.error is not None
        assert "HTTP error" in observation.error
        assert observation.output_token_count is None

    @pytest.mark.asyncio
    async def test_connection_error_produces_failed_observation(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://test"
        )
        client = BenchmarkClient(
            base_url="http://test",
            token_counter=FixedTokenCounter([4]),
            http_client=http_client,
            clock=_fake_clock([0.0, 0.0]),
        )

        observation = await client.run_one(0, "hi")

        assert observation.success is False
        assert "refused" in (observation.error or "")

    @pytest.mark.asyncio
    async def test_stream_level_error_event_produces_failed_observation(self) -> None:
        sse_body = 'data: {"text": "partial"}\n\ndata: {"error": "backend down"}\n\n'

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=sse_body)

        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://test"
        )
        client = BenchmarkClient(
            base_url="http://test",
            token_counter=FixedTokenCounter([4]),
            http_client=http_client,
            clock=_fake_clock([0.0, 0.05, 0.1]),
        )

        observation = await client.run_one(0, "hi")

        assert observation.success is False
        assert observation.error == "backend down"
        # Tokens are never counted for a failed request, by policy.
        assert observation.output_token_count is None


class TestBenchmarkClientRunMany:
    @pytest.mark.asyncio
    async def test_tags_leading_requests_as_warmup_and_aggregates(self) -> None:
        sse_body = 'data: {"text": "hi"}\n\ndata: [DONE]\n\n'

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=sse_body)

        http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(handler), base_url="http://test"
        )
        # 3 requests * 3 clock samples each (start, first-token, end -- "hi" is
        # non-empty so there IS a first-token sample).
        client = BenchmarkClient(
            base_url="http://test",
            token_counter=FixedTokenCounter([2, 2, 2]),
            http_client=http_client,
            clock=_fake_clock([0.0, 0.01, 0.02] * 3),
        )

        result = await client.run_many(["a", "b", "c"], warmup_count=1)

        assert [o.is_warmup for o in result.observations] == [True, False, False]
        assert result.summary.total_requests == 3
        assert result.summary.warmup_excluded == 1
        assert result.summary.measured_requests == 2
        assert result.config["warmup_count"] == 1
        assert result.config["prompt_count"] == 3
