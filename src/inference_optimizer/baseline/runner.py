"""Day 6 baseline runner: verifies backend readiness, warms up, runs a fixed
workload sequentially (concurrency = 1) at least `repeats` times, and persists
raw observations (JSONL) plus aggregate metrics (JSON + CSV) alongside full
run provenance (git commit, software versions, hardware, model/backend config,
process + GPU memory). Drives the InferenceBackend directly (not through the
Day 3 HTTP API), so a baseline run needs nothing already running -- only a
model file and, for llama.cpp, a CPU.

Never fabricates: if the backend can't start (missing model file, unsupported
hardware), the run stops before any observation/aggregate file is written and
the returned BaselineReport carries status="unavailable" with the real
exception message as unavailable_reason.
"""

from __future__ import annotations

import csv
import json
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from inference_optimizer.backends import (
    BackendConfig,
    ModelConfig,
    create_backend,
    load_model_config,
)
from inference_optimizer.backends.base import InferenceBackend
from inference_optimizer.baseline.config import BaselineConfig
from inference_optimizer.baseline.metadata import collect_run_metadata
from inference_optimizer.baseline.models import BaselineReport, RepeatResult
from inference_optimizer.baseline.resources import sample_memory
from inference_optimizer.benchmark import HFTokenizerCounter, TokenCounter, summarize
from inference_optimizer.benchmark.models import RequestObservation
from inference_optimizer.exceptions import InferenceOptimizerError
from inference_optimizer.logging import get_logger
from inference_optimizer.workload import load_dataset, sample_records

logger = get_logger(__name__)

_CSV_FIELDNAMES = [
    "repeat_index",
    "wall_time_seconds",
    "requests_per_second",
    "measured_requests",
    "successful_requests",
    "failed_requests",
    "failure_rate",
    "ttft_ms_p50",
    "ttft_ms_p95",
    "ttft_ms_p99",
    "ttft_ms_mean",
    "latency_ms_p50",
    "latency_ms_p95",
    "latency_ms_p99",
    "latency_ms_mean",
    "tpot_ms_p50",
    "tpot_ms_p95",
    "tpot_ms_p99",
    "tpot_ms_mean",
    "throughput_tps_p50",
    "throughput_tps_p95",
    "throughput_tps_p99",
    "throughput_tps_mean",
]


async def _run_one_against_backend(
    backend: InferenceBackend,
    token_counter: TokenCounter | None,
    index: int,
    is_warmup: bool,
    prompt: str,
    **params: object,
) -> RequestObservation:
    """Drives the backend directly (no HTTP/SSE layer) -- generate_stream() already
    yields decoded text chunks, so this only needs the same three-clock-sample
    policy as benchmark/client.py (start, first non-empty chunk, completion), not
    SSE parsing.
    """
    started_at = datetime.now(UTC)
    start_perf = time.perf_counter()
    first_token_perf: float | None = None
    chunks: list[str] = []
    error: str | None = None

    try:
        async for chunk in backend.generate_stream(prompt, **params):
            if chunk:
                if first_token_perf is None:
                    first_token_perf = time.perf_counter()
                chunks.append(chunk)
    except InferenceOptimizerError as exc:
        error = str(exc)

    end_perf = time.perf_counter()
    output_text = "".join(chunks)
    latency_ms = (end_perf - start_perf) * 1000
    ttft_ms = (first_token_perf - start_perf) * 1000 if first_token_perf is not None else None
    success = error is None

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
        error=error,
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


def _append_observations_jsonl(
    path: Path, repeat_index: int, observations: list[RequestObservation]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for observation in observations:
            record = observation.model_dump(mode="json")
            record["repeat_index"] = repeat_index
            handle.write(json.dumps(record))
            handle.write("\n")


def _write_summary_csv(path: Path, repeats: list[RepeatResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        for repeat in repeats:
            summary = repeat.summary
            writer.writerow(
                {
                    "repeat_index": repeat.repeat_index,
                    "wall_time_seconds": repeat.wall_time_seconds,
                    "requests_per_second": repeat.requests_per_second,
                    "measured_requests": summary.measured_requests,
                    "successful_requests": summary.successful_requests,
                    "failed_requests": summary.failed_requests,
                    "failure_rate": summary.failure_rate,
                    "ttft_ms_p50": summary.ttft_ms.p50,
                    "ttft_ms_p95": summary.ttft_ms.p95,
                    "ttft_ms_p99": summary.ttft_ms.p99,
                    "ttft_ms_mean": summary.ttft_ms.mean,
                    "latency_ms_p50": summary.latency_ms.p50,
                    "latency_ms_p95": summary.latency_ms.p95,
                    "latency_ms_p99": summary.latency_ms.p99,
                    "latency_ms_mean": summary.latency_ms.mean,
                    "tpot_ms_p50": summary.tpot_ms.p50,
                    "tpot_ms_p95": summary.tpot_ms.p95,
                    "tpot_ms_p99": summary.tpot_ms.p99,
                    "tpot_ms_mean": summary.tpot_ms.mean,
                    "throughput_tps_p50": summary.throughput_tokens_per_second.p50,
                    "throughput_tps_p95": summary.throughput_tokens_per_second.p95,
                    "throughput_tps_p99": summary.throughput_tokens_per_second.p99,
                    "throughput_tps_mean": summary.throughput_tokens_per_second.mean,
                }
            )


async def run_baseline(
    config: BaselineConfig,
    repo_root: Path | None = None,
    token_counter: TokenCounter | None = None,
    backend_factory: Callable[[str, ModelConfig, BackendConfig], InferenceBackend]
    | None = None,
) -> BaselineReport:
    """`token_counter` and `backend_factory` default to the real HF tokenizer and
    the real backend registry respectively; tests inject fakes for both so no
    network call or subprocess is needed to exercise the orchestration logic.
    """
    model_config = load_model_config(config.model_config_path)
    backend_config = BackendConfig(backend_type=config.backend_type)
    run_metadata = collect_run_metadata(model_config, backend_config, repo_root)

    records = load_dataset(config.workload_path)
    sampled = sample_records(
        records,
        count=config.sample_count,
        seed=config.sample_seed,
        categories=config.categories,
        input_length_targets=config.input_length_targets,
    )
    prompts = [record.prompt for record in sampled]
    config_snapshot = config.model_dump(mode="json")

    repo_id = config.tokenizer_repo or model_config.tokenizer
    resolved_token_counter = token_counter or HFTokenizerCounter(repo_id)
    params: dict[str, object] = (
        {"max_tokens": config.max_tokens_override} if config.max_tokens_override else {}
    )

    resolved_backend_factory = backend_factory or create_backend
    backend = resolved_backend_factory(config.backend_type, model_config, backend_config)
    try:
        await backend.start()
        health = await backend.health_check()
        if not health.healthy:
            raise InferenceOptimizerError(f"Backend reported unhealthy: {health.detail}")
    except InferenceOptimizerError as exc:
        logger.error("baseline_backend_unavailable", error=str(exc))
        return BaselineReport(
            run_metadata=run_metadata,
            config=config_snapshot,
            workload_record_ids=[record.id for record in sampled],
            repeats=[],
            status="unavailable",
            unavailable_reason=str(exc),
        )

    try:
        for warmup_index in range(config.warmup_count):
            if not prompts:
                break
            warmup_prompt = prompts[warmup_index % len(prompts)]
            await _run_one_against_backend(
                backend, resolved_token_counter, warmup_index, True, warmup_prompt, **params
            )

        output_dir = config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        jsonl_path = output_dir / f"baseline_{timestamp}_observations.jsonl"

        repeat_results: list[RepeatResult] = []
        for repeat_index in range(config.repeats):
            memory_before = sample_memory(f"repeat_{repeat_index}_before")
            wall_start = time.perf_counter()
            observations: list[RequestObservation] = []
            for request_index, prompt in enumerate(prompts):
                observation = await _run_one_against_backend(
                    backend, resolved_token_counter, request_index, False, prompt, **params
                )
                observations.append(observation)
            wall_elapsed = time.perf_counter() - wall_start
            memory_after = sample_memory(f"repeat_{repeat_index}_after")

            summary = summarize(observations)
            requests_per_second = (
                summary.measured_requests / wall_elapsed if wall_elapsed > 0 else None
            )
            _append_observations_jsonl(jsonl_path, repeat_index, observations)
            repeat_results.append(
                RepeatResult(
                    repeat_index=repeat_index,
                    wall_time_seconds=wall_elapsed,
                    requests_per_second=requests_per_second,
                    memory_before=memory_before,
                    memory_after=memory_after,
                    summary=summary,
                )
            )

        report = BaselineReport(
            run_metadata=run_metadata,
            config=config_snapshot,
            workload_record_ids=[record.id for record in sampled],
            repeats=repeat_results,
            status="completed",
        )

        json_path = output_dir / f"baseline_{timestamp}_report.json"
        json_path.write_text(report.model_dump_json(indent=2))
        csv_path = output_dir / f"baseline_{timestamp}_summary.csv"
        _write_summary_csv(csv_path, repeat_results)

        return report
    finally:
        await backend.stop()
