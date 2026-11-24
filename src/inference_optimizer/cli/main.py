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
