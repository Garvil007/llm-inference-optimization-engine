"""Baseline orchestration tests against a fake InferenceBackend -- no real
subprocess/model/tokenizer download. Covers: unavailable-backend short circuit
(no files written, no fabricated numbers), a completed run's repeat count,
fixed-workload reuse across repeats, and the JSONL/JSON/CSV outputs.
"""

from __future__ import annotations

import csv
import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from inference_optimizer.backends.base import InferenceBackend
from inference_optimizer.backends.models import BackendConfig, HealthStatus, ModelConfig
from inference_optimizer.baseline.config import BaselineConfig
from inference_optimizer.baseline.runner import run_baseline
from inference_optimizer.benchmark.token_counter import FixedTokenCounter
from inference_optimizer.exceptions import BackendStartupError
from inference_optimizer.workload.models import PromptRecord


def _model_config() -> ModelConfig:
    return ModelConfig(
        name="fake-model",
        tokenizer="fake/tokenizer",
        revision="deadbeef",
        context_limit=2048,
        quantization="Q4_K_M",
        license="apache-2.0",
    )


class FakeBackend(InferenceBackend):
    def __init__(
        self,
        chunks: list[str] | None = None,
        healthy: bool = True,
        start_error: Exception | None = None,
    ) -> None:
        super().__init__(_model_config(), BackendConfig(backend_type="llama-cpp"))
        self._chunks = chunks if chunks is not None else ["Hello", " world"]
        self._healthy = healthy
        self._start_error = start_error
        self.started = False

    async def start(self) -> None:
        if self._start_error is not None:
            raise self._start_error
        self.started = True

    async def stop(self) -> None:
        self.started = False

    async def health_check(self) -> HealthStatus:
        return HealthStatus(
            healthy=self._healthy,
            backend_type="llama-cpp",
            detail="ok" if self._healthy else "model not loaded",
            model_loaded=self._healthy,
        )

    async def generate_stream(self, prompt: str, **params: object) -> AsyncIterator[str]:
        for chunk in self._chunks:
            yield chunk


def _write_dataset(path: Path, count: int) -> None:
    records = [
        PromptRecord(
            id=f"short_qa-128-{i}",
            category="short_qa",
            input_length_target=128,
            output_target_tokens=64,
            prompt=f"Question {i}: what is {i} + {i}?",
            input_tokens_measured=10,
            length_calculation_method="approximate",
        )
        for i in range(count)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")


def _config(tmp_path: Path, repeats: int = 2, warmup_count: int = 1) -> BaselineConfig:
    model_config_path = tmp_path / "model.yaml"
    model_config_path.write_text(
        "name: fake-model\ntokenizer: fake/tokenizer\nrevision: deadbeef\n"
        "context_limit: 2048\nquantization: Q4_K_M\nlicense: apache-2.0\n",
        encoding="utf-8",
    )
    workload_path = tmp_path / "prompts.jsonl"
    _write_dataset(workload_path, count=3)
    return BaselineConfig(
        model_config_path=model_config_path,
        workload_path=workload_path,
        sample_count=2,
        sample_seed=0,
        warmup_count=warmup_count,
        repeats=repeats,
        output_dir=tmp_path / "out",
    )


class TestRunBaselineUnavailable:
    @pytest.mark.asyncio
    async def test_backend_startup_failure_marks_unavailable_and_writes_nothing(
        self, tmp_path: Path
    ) -> None:
        config = _config(tmp_path)
        backend = FakeBackend(start_error=BackendStartupError("model file missing"))

        report = await run_baseline(
            config,
            token_counter=FixedTokenCounter([5, 5, 5, 5, 5, 5]),
            backend_factory=lambda *_args: backend,
        )

        assert report.status == "unavailable"
        assert report.unavailable_reason == "model file missing"
        assert report.repeats == []
        assert not config.output_dir.exists()

    @pytest.mark.asyncio
    async def test_unhealthy_backend_marks_unavailable(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        backend = FakeBackend(healthy=False)

        report = await run_baseline(
            config,
            token_counter=FixedTokenCounter([5, 5, 5, 5, 5, 5]),
            backend_factory=lambda *_args: backend,
        )

        assert report.status == "unavailable"
        assert "unhealthy" in (report.unavailable_reason or "")
        assert report.repeats == []


class TestRunBaselineCompleted:
    @pytest.mark.asyncio
    async def test_runs_configured_repeats_with_fixed_sampled_workload(
        self, tmp_path: Path
    ) -> None:
        config = _config(tmp_path, repeats=3, warmup_count=1)
        backend = FakeBackend(chunks=["Hello", " world"])

        report = await run_baseline(
            config,
            token_counter=FixedTokenCounter([5] * 20),
            backend_factory=lambda *_args: backend,
        )

        assert report.status == "completed"
        assert len(report.repeats) == 3
        assert report.workload_record_ids
        # Same sampled ids every repeat -- "fixed workload", not re-sampled.
        assert len(report.workload_record_ids) == config.sample_count
        for repeat in report.repeats:
            assert repeat.summary.measured_requests == config.sample_count
            assert repeat.summary.failed_requests == 0
            assert repeat.memory_before.label == f"repeat_{repeat.repeat_index}_before"
            assert repeat.memory_after.label == f"repeat_{repeat.repeat_index}_after"

    @pytest.mark.asyncio
    async def test_writes_jsonl_json_and_csv_outputs(self, tmp_path: Path) -> None:
        config = _config(tmp_path, repeats=2, warmup_count=0)
        backend = FakeBackend(chunks=["Hello", " world"])

        report = await run_baseline(
            config,
            token_counter=FixedTokenCounter([5] * 20),
            backend_factory=lambda *_args: backend,
        )

        jsonl_files = list(config.output_dir.glob("*_observations.jsonl"))
        json_files = list(config.output_dir.glob("*_report.json"))
        csv_files = list(config.output_dir.glob("*_summary.csv"))
        assert len(jsonl_files) == 1
        assert len(json_files) == 1
        assert len(csv_files) == 1

        lines = jsonl_files[0].read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == config.sample_count * config.repeats
        parsed = json.loads(lines[0])
        assert "repeat_index" in parsed
        assert "latency_ms" in parsed

        report_from_disk = json.loads(json_files[0].read_text(encoding="utf-8"))
        assert report_from_disk["status"] == "completed"
        assert report_from_disk["run_metadata"]["run_id"] == report.run_metadata.run_id

        with csv_files[0].open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == config.repeats
        assert "latency_ms_p50" in rows[0]
        assert "throughput_tps_p50" in rows[0]

    @pytest.mark.asyncio
    async def test_failed_requests_are_recorded_not_silently_dropped(
        self, tmp_path: Path
    ) -> None:
        config = _config(tmp_path, repeats=1, warmup_count=0)

        class _FailingBackend(FakeBackend):
            async def generate_stream(self, prompt: str, **params: object) -> AsyncIterator[str]:
                raise BackendStartupError("generation failed mid-stream")
                yield ""  # pragma: no cover - unreachable, keeps this an async generator

        backend = _FailingBackend()
        report = await run_baseline(
            config,
            token_counter=FixedTokenCounter([5] * 20),
            backend_factory=lambda *_args: backend,
        )

        assert report.status == "completed"
        assert report.repeats[0].summary.failed_requests == config.sample_count
        assert report.repeats[0].summary.failure_rate == 1.0
