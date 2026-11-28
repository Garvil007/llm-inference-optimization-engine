"""Week 1 integration test: workload generation -> fixed-workload baseline run
-> report generation, end to end. Uses a FakeBackend (no real model/subprocess,
mirroring tests/unit/test_baseline_runner.py's pattern) and FixedTokenCounter
(no network/HF download) so this runs offline and fast, while still exercising
every real code path: generate_dataset -> sample_records -> run_baseline ->
generate_report -> real Markdown/HTML/plot files on disk.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from inference_optimizer.backends.base import InferenceBackend
from inference_optimizer.backends.models import BackendConfig, HealthStatus, ModelConfig
from inference_optimizer.baseline.config import BaselineConfig
from inference_optimizer.baseline.runner import run_baseline
from inference_optimizer.benchmark.token_counter import FixedTokenCounter, WhitespaceApproxCounter
from inference_optimizer.exceptions import BackendStartupError
from inference_optimizer.report import generate_report
from inference_optimizer.workload import generate_dataset, save_dataset


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
    def __init__(self, chunks: list[str] | None = None, fail: bool = False) -> None:
        super().__init__(_model_config(), BackendConfig(backend_type="llama-cpp"))
        self._chunks = chunks if chunks is not None else ["Hello", " world", " again"]
        self._fail = fail

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def health_check(self) -> HealthStatus:
        return HealthStatus(
            healthy=True, backend_type="llama-cpp", detail="ok", model_loaded=True
        )

    async def generate_stream(self, prompt: str, **params: object) -> AsyncIterator[str]:
        if self._fail:
            raise BackendStartupError("generation failed mid-stream")
            yield ""  # pragma: no cover - unreachable, keeps this an async generator
        for chunk in self._chunks:
            yield chunk


def _write_baseline_config(tmp_path: Path, repeats: int) -> BaselineConfig:
    model_config_path = tmp_path / "model.yaml"
    model_config_path.write_text(
        "name: fake-model\ntokenizer: fake/tokenizer\nrevision: deadbeef\n"
        "context_limit: 2048\nquantization: Q4_K_M\nlicense: apache-2.0\n",
        encoding="utf-8",
    )

    workload_path = tmp_path / "prompts.jsonl"
    records = generate_dataset(WhitespaceApproxCounter(), context_limit=2048)
    save_dataset(workload_path, records)

    return BaselineConfig(
        model_config_path=model_config_path,
        workload_path=workload_path,
        sample_count=3,
        sample_seed=0,
        warmup_count=1,
        repeats=repeats,
        output_dir=tmp_path / "baseline_out",
    )


@pytest.mark.asyncio
async def test_full_pipeline_workload_to_report_with_three_repeats(tmp_path: Path) -> None:
    config = _write_baseline_config(tmp_path, repeats=3)
    backend = FakeBackend()

    baseline_report = await run_baseline(
        config,
        token_counter=FixedTokenCounter([5] * 50),
        backend_factory=lambda *_args: backend,
    )
    assert baseline_report.status == "completed"

    report_json = next(config.output_dir.glob("*_report.json"))
    observations_jsonl = next(config.output_dir.glob("*_observations.jsonl"))
    report_output_dir = tmp_path / "report_out"

    report_data = generate_report(report_json, observations_jsonl, report_output_dir)

    assert (report_output_dir / "report.md").exists()
    assert (report_output_dir / "report.html").exists()
    # 3 repeats of a FakeBackend meet the "complete" threshold -- no incomplete flag.
    assert not any(f.code == "incomplete_repeats" for f in report_data.flags)
    # Aggregates recomputed from the raw JSONL must match the stored summary exactly,
    # since both come from the same summarize() function on the same data.
    assert all(result.matches for result in report_data.validation)
    markdown = (report_output_dir / "report.md").read_text(encoding="utf-8")
    assert "fake-model" in markdown
    assert baseline_report.run_metadata.run_id in markdown


@pytest.mark.asyncio
async def test_pipeline_flags_incomplete_and_failed_runs(tmp_path: Path) -> None:
    config = _write_baseline_config(tmp_path, repeats=1)
    backend = FakeBackend(fail=True)

    baseline_report = await run_baseline(
        config,
        token_counter=FixedTokenCounter([5] * 50),
        backend_factory=lambda *_args: backend,
    )
    assert baseline_report.status == "completed"  # the run completes; requests fail

    report_json = next(config.output_dir.glob("*_report.json"))
    observations_jsonl = next(config.output_dir.glob("*_observations.jsonl"))
    report_output_dir = tmp_path / "report_out"

    report_data = generate_report(report_json, observations_jsonl, report_output_dir)

    flag_codes = {f.code for f in report_data.flags}
    assert "incomplete_repeats" in flag_codes  # only 1 repeat, target is >= 3
    assert "requests_failed" in flag_codes  # every request failed in the single repeat


@pytest.mark.asyncio
async def test_pipeline_handles_an_unavailable_backend_without_fabricating_a_report(
    tmp_path: Path,
) -> None:
    config = _write_baseline_config(tmp_path, repeats=3)

    class _UnavailableBackend(FakeBackend):
        async def start(self) -> None:
            raise BackendStartupError("model file missing")

    baseline_report = await run_baseline(
        config,
        token_counter=FixedTokenCounter([5] * 50),
        backend_factory=lambda *_args: _UnavailableBackend(),
    )
    assert baseline_report.status == "unavailable"
    assert not config.output_dir.exists()

    # Write the unavailable report manually (the runner never persists one)
    # to confirm the report pipeline still surfaces it correctly, with no
    # observations file required.
    report_json = tmp_path / "unavailable_report.json"
    report_json.write_text(baseline_report.model_dump_json())
    report_output_dir = tmp_path / "report_out_unavailable"

    report_data = generate_report(report_json, None, report_output_dir)

    assert report_data.flags[0].code == "run_unavailable"
    markdown = (report_output_dir / "report.md").read_text(encoding="utf-8")
    assert "model file missing" in markdown
