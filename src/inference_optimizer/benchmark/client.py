"""Benchmark client: issues sequential requests against the API's /generate/stream
endpoint and records a RequestObservation per request. Only three clock samples are
taken per request -- start, first non-empty content, completion -- never one per SSE
chunk, since a chunk is not a token and timing every chunk would imply a granularity
the data doesn't have.

The SSE-parsing core (`_consume_sse_stream`) is clock-injectable and takes a plain
async iterator of lines, independent of httpx, specifically so tests can drive it with
a fake clock and a canned line sequence for fully deterministic timing assertions
(see tests/unit/test_benchmark_client.py) instead of relying on real sleeps.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime

import httpx

from inference_optimizer.benchmark.aggregate import summarize
from inference_optimizer.benchmark.models import BenchmarkResult, RequestObservation
from inference_optimizer.benchmark.token_counter import TokenCounter
from inference_optimizer.logging import get_logger

logger = get_logger(__name__)

Clock = Callable[[], float]


async def _consume_sse_stream(
    lines: AsyncIterator[str], clock: Clock
) -> tuple[str, float, float | None, float, str | None]:
    """Returns (output_text, start_perf, first_token_perf, end_perf, error)."""
    start_perf = clock()
    first_token_perf: float | None = None
    chunks: list[str] = []
    error: str | None = None

    async for line in lines:
        if not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ").strip()
        if data == "[DONE]":
            break
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            error = f"Malformed SSE payload: {exc}"
            break
        if "error" in payload:
            error = str(payload["error"])
            break
        text = payload.get("text", "")
        if text:
            if first_token_perf is None:
                first_token_perf = clock()
            chunks.append(text)

    end_perf = clock()
    return "".join(chunks), start_perf, first_token_perf, end_perf, error


def _build_observation(
    index: int,
    is_warmup: bool,
    started_at: datetime,
    output_text: str,
    start_perf: float,
    first_token_perf: float | None,
    end_perf: float,
    stream_error: str | None,
    token_counter: TokenCounter | None,
) -> RequestObservation:
    latency_ms = (end_perf - start_perf) * 1000
    ttft_ms = (first_token_perf - start_perf) * 1000 if first_token_perf is not None else None
    success = stream_error is None

    token_count: int | None = None
    token_count_method = "unavailable"
    tpot_ms: float | None = None
    throughput: float | None = None

    if success and token_counter is not None:
        token_count = token_counter.count(output_text)
        token_count_method = token_counter.method
        if token_count is not None:
            if token_count >= 2 and ttft_ms is not None:
                tpot_ms = (latency_ms - ttft_ms) / (token_count - 1)
            if latency_ms > 0:
                throughput = token_count / (latency_ms / 1000)

    return RequestObservation(
        request_index=index,
        is_warmup=is_warmup,
        success=success,
        error=stream_error,
        started_at=started_at,
        start_perf=start_perf,
        first_token_perf=first_token_perf,
        end_perf=end_perf,
        ttft_ms=ttft_ms,
        latency_ms=latency_ms,
        output_text_length=len(output_text),
        output_token_count=token_count,
        token_count_method=token_count_method,  # type: ignore[arg-type]
        tpot_ms=tpot_ms,
        output_tokens_per_second=throughput,
    )


class BenchmarkClient:
    def __init__(
        self,
        base_url: str,
        token_counter: TokenCounter | None,
        http_client: httpx.AsyncClient | None = None,
        request_timeout_seconds: float = 120.0,
        clock: Clock = time.perf_counter,
    ) -> None:
        self._token_counter = token_counter
        self._client = http_client or httpx.AsyncClient(
            base_url=base_url, timeout=request_timeout_seconds
        )
        self._owns_client = http_client is None
        self._clock = clock

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def run_one(
        self, index: int, prompt: str, is_warmup: bool = False, **generate_params: object
    ) -> RequestObservation:
        started_at = datetime.now(UTC)
        payload: dict[str, object] = {"prompt": prompt, **generate_params}

        try:
            async with self._client.stream("POST", "/generate/stream", json=payload) as response:
                response.raise_for_status()
                output_text, start_perf, first_token_perf, end_perf, error = (
                    await _consume_sse_stream(response.aiter_lines(), self._clock)
                )
        except httpx.HTTPError as exc:
            now = self._clock()
            return _build_observation(
                index=index,
                is_warmup=is_warmup,
                started_at=started_at,
                output_text="",
                start_perf=now,
                first_token_perf=None,
                end_perf=now,
                stream_error=f"HTTP error: {exc}",
                token_counter=self._token_counter,
            )

        if error:
            logger.warning("benchmark_request_failed", index=index, error=error)

        return _build_observation(
            index=index,
            is_warmup=is_warmup,
            started_at=started_at,
            output_text=output_text,
            start_perf=start_perf,
            first_token_perf=first_token_perf,
            end_perf=end_perf,
            stream_error=error,
            token_counter=self._token_counter,
        )

    async def run_many(
        self, prompts: list[str], warmup_count: int = 0, **generate_params: object
    ) -> BenchmarkResult:
        """Runs `prompts` sequentially (no concurrency -- that's a later day's
        concern); the first `warmup_count` requests are tagged `is_warmup=True`.
        """
        observations: list[RequestObservation] = []
        for index, prompt in enumerate(prompts):
            observation = await self.run_one(
                index, prompt, is_warmup=index < warmup_count, **generate_params
            )
            observations.append(observation)

        return BenchmarkResult(
            observations=observations,
            summary=summarize(observations),
            config={
                "base_url": str(self._client.base_url),
                "prompt_count": len(prompts),
                "warmup_count": warmup_count,
                "generate_params": generate_params,
                "token_count_method_requested": (
                    self._token_counter.method if self._token_counter else "unavailable"
                ),
            },
        )
