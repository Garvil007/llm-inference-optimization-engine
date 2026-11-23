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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
