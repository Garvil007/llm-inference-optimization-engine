"""CLI entry point.

Usage:
    python -m inference_optimizer hardware-info [--pretty]
"""

import asyncio
import json
from pathlib import Path

import typer

from inference_optimizer.backends import create_backend, load_model_config
from inference_optimizer.config import get_settings
from inference_optimizer.exceptions import InferenceOptimizerError
from inference_optimizer.hardware import detect_hardware
from inference_optimizer.logging import configure_logging, get_logger

_DEFAULT_MODEL_CONFIG = Path("configs/models/qwen2.5-0.5b-instruct.yaml")
_DEFAULT_WORKLOAD_PATH = Path("data/workloads/prompts_v1.jsonl")

app = typer.Typer(
    name="inference-optimizer",
    help="LLM Inference Optimization Engine CLI.",
    add_completion=False,
)
logger = get_logger(__name__)


@app.callback()
def _bootstrap() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


@app.command("hardware-info")
def hardware_info(
    pretty: bool = typer.Option(
        True,
        "--pretty/--no-pretty",
        help="Pretty-print JSON with indentation (set --no-pretty for compact output).",
    ),
) -> None:
    """Detect and print this machine's hardware/software capabilities as JSON."""
    try:
        report = detect_hardware()
    except InferenceOptimizerError as exc:
        logger.error("hardware_detection_failed", error=str(exc))
        typer.echo(f"Hardware detection failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    payload = report.model_dump()
    indent = 2 if pretty else None
    typer.echo(json.dumps(payload, indent=indent))


@app.command("backend-status")
def backend_status(
    backend: str = typer.Option("llama-cpp", help="Backend to check: llama-cpp or vllm."),
    config: Path = typer.Option(  # noqa: B008 - typer's standard idiom
        _DEFAULT_MODEL_CONFIG, help="Model config YAML path."
    ),
) -> None:
    """Start a backend, run a health check against it, then shut it down."""
    asyncio.run(_run_backend_status(backend, config))


async def _run_backend_status(backend_type: str, config_path: Path) -> None:
    model_config = load_model_config(config_path)
    inference_backend = create_backend(backend_type, model_config)
    try:
        await inference_backend.start()
        status = await inference_backend.health_check()
    except InferenceOptimizerError as exc:
        logger.error("backend_status_failed", backend=backend_type, error=str(exc))
        typer.echo(f"Backend check failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        await inference_backend.stop()
    typer.echo(json.dumps(status.model_dump(), indent=2))


@app.command("generate")
def generate(
    prompt: str = typer.Argument(..., help="User prompt to send to the model."),
    backend: str = typer.Option("llama-cpp", help="Backend to use: llama-cpp or vllm."),
    config: Path = typer.Option(  # noqa: B008 - typer's standard idiom
        _DEFAULT_MODEL_CONFIG, help="Model config YAML path."
    ),
    stream: bool = typer.Option(
        True, "--stream/--no-stream", help="Stream tokens as they arrive."
    ),
    max_tokens: int = typer.Option(0, help="Override max_tokens (0 = use model config default)."),
    temperature: float = typer.Option(
        -1.0, help="Override temperature (negative = use model config default)."
    ),
) -> None:
    """Run one prompt-response cycle against a backend and print the result."""
    overrides: dict[str, object] = {}
    if max_tokens > 0:
        overrides["max_tokens"] = max_tokens
    if temperature >= 0:
        overrides["temperature"] = temperature
    asyncio.run(_run_generate(prompt, backend, config, stream, overrides))


async def _run_generate(
    prompt: str,
    backend_type: str,
    config_path: Path,
    stream: bool,
    overrides: dict[str, object],
) -> None:
    model_config = load_model_config(config_path)
    inference_backend = create_backend(backend_type, model_config)
    try:
        await inference_backend.start()
        if stream:
            async for chunk in inference_backend.generate_stream(prompt, **overrides):
                typer.echo(chunk, nl=False)
            typer.echo()
        else:
            result = await inference_backend.generate(prompt, **overrides)
            typer.echo(result.text)
    except InferenceOptimizerError as exc:
        logger.error("generate_failed", backend=backend_type, error=str(exc))
        typer.echo(f"Generation failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        await inference_backend.stop()


@app.command("benchmark")
def benchmark(
    prompt: str = typer.Argument(..., help="Prompt to send repeatedly."),
    base_url: str = typer.Option("http://127.0.0.1:8000", help="Running API server base URL."),
    iterations: int = typer.Option(10, min=1, help="Total requests to send."),
    warmup: int = typer.Option(2, min=0, help="Leading requests excluded from aggregates."),
    tokenizer_repo: str | None = typer.Option(
        None,
        help="HF repo id for real token counts (defaults to the model config's tokenizer).",
    ),
    config: Path = typer.Option(  # noqa: B008 - typer's standard idiom
        _DEFAULT_MODEL_CONFIG, help="Model config YAML path (for the tokenizer default)."
    ),
    max_tokens: int = typer.Option(0, help="Override max_tokens (0 = model config default)."),
) -> None:
    """Run a sequential benchmark against a running API server and save raw
    observations + aggregate metrics to results/.
    """
    asyncio.run(
        _run_benchmark(prompt, base_url, iterations, warmup, tokenizer_repo, config, max_tokens)
    )


async def _run_benchmark(
    prompt: str,
    base_url: str,
    iterations: int,
    warmup: int,
    tokenizer_repo: str | None,
    config_path: Path,
    max_tokens: int,
) -> None:
    from inference_optimizer.benchmark import BenchmarkClient, HFTokenizerCounter
    from inference_optimizer.config import get_settings

    repo_id = tokenizer_repo or load_model_config(config_path).tokenizer
    params: dict[str, object] = {"max_tokens": max_tokens} if max_tokens > 0 else {}

    client = BenchmarkClient(base_url=base_url, token_counter=HFTokenizerCounter(repo_id))
    try:
        result = await client.run_many([prompt] * iterations, warmup_count=warmup, **params)
    except InferenceOptimizerError as exc:
        logger.error("benchmark_failed", error=str(exc))
        typer.echo(f"Benchmark failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    finally:
        await client.aclose()

    settings = get_settings()
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    from datetime import UTC, datetime

    out_path = settings.results_dir / f"benchmark_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json"
    out_path.write_text(result.model_dump_json(indent=2))

    typer.echo(json.dumps(result.summary.model_dump(), indent=2))
    typer.echo(f"Raw observations + summary written to {out_path}")


@app.command("workload-generate")
def workload_generate(
    path: Path = typer.Option(_DEFAULT_WORKLOAD_PATH, help="Output JSONL dataset path."),  # noqa: B008
    tokenizer_repo: str | None = typer.Option(
        None, help="HF repo id for tokenizer-aware lengths (defaults to model config's tokenizer)."
    ),
    config: Path = typer.Option(  # noqa: B008 - typer's standard idiom
        _DEFAULT_MODEL_CONFIG, help="Model config YAML path (tokenizer + context_limit source)."
    ),
    dataset_version: str = typer.Option("v1", help="Version label recorded in the manifest."),
    seed: int = typer.Option(0, help="Recorded in the manifest as the default sampling seed."),
) -> None:
    """Generate the versioned prompt dataset (JSONL) plus its provenance manifest."""
    from inference_optimizer.benchmark import HFTokenizerCounter
    from inference_optimizer.workload import (
        build_manifest,
        generate_dataset,
        save_dataset,
        save_manifest,
        validate_dataset,
    )
    from inference_optimizer.workload.generator import (
        DEFAULT_SAFETY_MARGIN,
        DEFAULT_TIERS,
        GENERATOR_SOURCE_PATH,
    )

    model_config = load_model_config(config)
    repo_id = tokenizer_repo or model_config.tokenizer
    tokenizer = HFTokenizerCounter(repo_id)
    context_limit = model_config.context_limit

    records = generate_dataset(tokenizer, context_limit=context_limit)
    errors, warnings = validate_dataset(records, context_limit=context_limit)
    for warning in warnings:
        typer.echo(f"warning: {warning}", err=True)
    if errors:
        for error in errors:
            typer.echo(f"error: {error}", err=True)
        raise typer.Exit(code=1)

    save_dataset(path, records)
    manifest = build_manifest(
        dataset_version=dataset_version,
        dataset_path=path,
        tokenizer_repo_id=repo_id,
        tokenizer_method=tokenizer.method,
        seed=seed,
        record_count=len(records),
        generation_settings={
            "tiers": DEFAULT_TIERS,
            "context_limit": context_limit,
            "safety_margin": DEFAULT_SAFETY_MARGIN,
            "categories": sorted({record.category for record in records}),
            "variants_per_tier": 2,
        },
        generator_source_path=GENERATOR_SOURCE_PATH,
    )
    manifest_path = path.with_suffix(".manifest.json")
    save_manifest(manifest_path, manifest)

    typer.echo(f"Wrote {len(records)} records to {path}")
    typer.echo(f"Wrote manifest to {manifest_path}")


@app.command("workload-validate")
def workload_validate(
    path: Path = typer.Option(_DEFAULT_WORKLOAD_PATH, help="JSONL dataset path to validate."),  # noqa: B008
) -> None:
    """Validate a prompt dataset (built-in or user-provided)."""
    from inference_optimizer.workload import load_dataset, validate_dataset

    try:
        records = load_dataset(path)
    except InferenceOptimizerError as exc:
        typer.echo(f"Failed to load dataset: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    errors, warnings = validate_dataset(records)
    for warning in warnings:
        typer.echo(f"warning: {warning}")
    for error in errors:
        typer.echo(f"error: {error}", err=True)

    if errors:
        typer.echo(
            f"{len(errors)} error(s), {len(warnings)} warning(s) across {len(records)} records."
        )
        raise typer.Exit(code=1)
    typer.echo(f"OK: {len(records)} records valid, {len(warnings)} warning(s).")


@app.command("workload-sample")
def workload_sample(
    path: Path = typer.Option(  # noqa: B008
        _DEFAULT_WORKLOAD_PATH, help="JSONL dataset path (built-in or user-provided)."
    ),
    count: int = typer.Option(..., help="Number of records to sample."),
    seed: int = typer.Option(  # noqa: B008
        ..., help="Random seed; same seed + inputs always yields the same sample."
    ),
    category: list[str] = typer.Option(  # noqa: B008
        [], help="Restrict to these categories (repeatable)."
    ),
    input_length: list[int] = typer.Option(  # noqa: B008
        [], help="Restrict to these input-length tiers (repeatable)."
    ),
) -> None:
    """Reproducibly sample records from a dataset and print them as JSONL."""
    from inference_optimizer.workload import load_dataset, sample_records

    try:
        records = load_dataset(path)
    except InferenceOptimizerError as exc:
        typer.echo(f"Failed to load dataset: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    sampled = sample_records(
        records,
        count=count,
        seed=seed,
        categories=category or None,  # type: ignore[arg-type]
        input_length_targets=input_length or None,  # type: ignore[arg-type]
    )
    for record in sampled:
        typer.echo(record.model_dump_json())


@app.command("baseline")
def baseline(
    config: Path = typer.Option(Path("configs/baseline.yaml"), help="Baseline config YAML."),  # noqa: B008
) -> None:
    """Run the Day 6 baseline benchmark end-to-end (readiness check, warmup,
    fixed workload at concurrency=1, repeated), writing raw JSONL + aggregate
    JSON/CSV to the config's output_dir. See docs/baseline.md.
    """
    asyncio.run(_run_baseline_command(config))


async def _run_baseline_command(config_path: Path) -> None:
    from inference_optimizer.baseline import load_baseline_config, run_baseline

    try:
        baseline_config = load_baseline_config(config_path)
    except InferenceOptimizerError as exc:
        typer.echo(f"Failed to load baseline config: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    report = await run_baseline(baseline_config)

    if report.status == "unavailable":
        typer.echo(f"Baseline run unavailable: {report.unavailable_reason}", err=True)
        raise typer.Exit(code=1)

    typer.echo(
        f"Baseline run {report.run_metadata.run_id} completed "
        f"({len(report.repeats)} repeat(s)); results in {baseline_config.output_dir}"
    )
    for repeat in report.repeats:
        summary = repeat.summary
        typer.echo(
            f"repeat {repeat.repeat_index}: latency p50={summary.latency_ms.p50}ms "
            f"p95={summary.latency_ms.p95}ms p99={summary.latency_ms.p99}ms "
            f"ttft p50={summary.ttft_ms.p50}ms tpot p50={summary.tpot_ms.p50}ms "
            f"throughput p50={summary.throughput_tokens_per_second.p50}tok/s "
            f"req/s={repeat.requests_per_second} failure_rate={summary.failure_rate}"
        )


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", help="Host to bind the API server to."),
    port: int = typer.Option(8000, help="Port to bind the API server to."),
) -> None:
    """Run the FastAPI inference service (uvicorn), blocking until interrupted."""
    import uvicorn

    from inference_optimizer.api import create_app

    uvicorn.run(create_app(), host=host, port=port)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
