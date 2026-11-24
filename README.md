# LLM Inference Optimization Engine

End-to-end system for deploying open-weight LLMs, benchmarking inference
performance, profiling CPU/GPU resources, comparing quantization and batching
strategies, and recommending configurations that improve throughput and
latency while tracking output quality.

## Status

Day 4 of 28: project scaffold, hardware detection, typed config, structured
logging, a working llama.cpp inference backend (Qwen2.5-0.5B-Instruct) and a
hardware-gated vLLM backend scaffold, a production-style FastAPI service
(`/health`, `/models`, `/generate`, `/generate/stream`, `/metrics`) wrapping
the backend over HTTP, and a benchmark client that measures TTFT/latency/TPOT/
throughput against the streaming API (see [docs/benchmarking.md](docs/benchmarking.md)
for metric definitions and aggregation policies). See [PROGRESS.md](PROGRESS.md)
for detailed status and [ARCHITECTURE.md](ARCHITECTURE.md) for design decisions.

## Requirements

- Python 3.11+
- Optional: NVIDIA GPU + driver for the `gpu` extra (vLLM/torch); the project
  runs CPU-only without it. vLLM additionally requires Linux (native or
  WSL2) — it has no official native Windows build.
- Optional: Docker, for containerized serving in later milestones.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # Linux/macOS

pip install -e ".[dev]"

# To actually run the llama.cpp backend, also install the `serving` extra.
# On Windows/macOS without a C/C++ toolchain, llama-cpp-python must come
# from the prebuilt wheel index, not the default PyPI sdist:
pip install -e ".[serving]"
pip install llama-cpp-python --prefer-binary \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Download the model (Apache-2.0, ~491 MB):
mkdir -p .cache/models
curl -L -o .cache/models/qwen2.5-0.5b-instruct-q4_k_m.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf
```

## Usage

```bash
# Detect and print this machine's hardware/software capabilities as JSON
python -m inference_optimizer hardware-info

# Compact (non-pretty) JSON
python -m inference_optimizer hardware-info --no-pretty

# Start the llama.cpp backend, health-check it, then shut it down
python -m inference_optimizer backend-status --backend llama-cpp

# One real prompt-response cycle, streamed to stdout as it generates
python -m inference_optimizer generate "What is the capital of France?"

# Non-streaming, with overrides
python -m inference_optimizer generate "Name one primary color." \
  --no-stream --max-tokens 20

# vLLM is a scaffold on this machine (Windows / no confirmed CUDA host) —
# this is expected to fail fast with a clear "unsupported hardware" error:
python -m inference_optimizer backend-status --backend vllm

# Start the FastAPI service (loads the real llama.cpp backend on startup).
# Use a port different from the backend's internal port (default 8731) --
# they would otherwise collide on the same machine.
python -m inference_optimizer serve --port 8000
```

### API examples

With the server from `serve --port 8000` running:

```bash
# Health and model info
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/models

# Non-streaming generation
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is the capital of France?", "max_tokens": 40}'

# Streaming generation (Server-Sent Events)
curl -N -X POST http://127.0.0.1:8000/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Name one primary color.", "max_tokens": 20}'

# Prometheus-text metrics
curl http://127.0.0.1:8000/metrics
```

`GenerateRequest` fields (all but `prompt` optional, override the model
config defaults for that one request): `prompt`, `max_tokens`, `temperature`,
`top_p`, `seed` (passed through to the backend when it supports it --
llama.cpp's OpenAI-compatible server does), `timeout_seconds` (per-request
override of `INFERENCE_OPTIMIZER_API_DEFAULT_REQUEST_TIMEOUT_SECONDS`).

Every response carries an `X-Request-ID` header (echoed back if the caller
sends one); it is also bound into structured log lines for that request and
included in the JSON body of `/generate` responses and error payloads.

**Deployment security assumption:** the API has no authentication and is
meant for local/dev use only, directly on `127.0.0.1` or inside a trusted
network boundary. Do not expose it on a public interface without adding an
auth layer (API key, mTLS, or a reverse proxy that handles auth) in front of
it first — nothing here checks who is calling.

### Benchmark

With the API server running (as above), measure real performance against it:

```bash
# Optional: real per-model token counts instead of the approximate fallback
pip install -e ".[benchmark]"

python -m inference_optimizer benchmark "Name one primary color." \
  --base-url http://127.0.0.1:8000 --iterations 10 --warmup 2 --max-tokens 20
```

Prints the aggregate summary (p50/p95/p99/mean/stddev for TTFT, end-to-end
latency, TPOT, and throughput, plus failure rate) and writes both the raw
per-request observations and that summary to `results/benchmark_<timestamp>.json`.
See [docs/benchmarking.md](docs/benchmarking.md) for exactly what each metric
means and the policies for warmup exclusion, failed requests, and missing
token counts.

## Development

Cross-platform task runner (works without `make` on Windows):

```bash
python tasks.py setup    # pip install -e ".[dev]"
python tasks.py lint     # ruff + mypy
python tasks.py format   # ruff format + autofix
python tasks.py test     # pytest with coverage
```

Equivalent `make` targets exist in the `Makefile` for Unix/WSL users.

## Project Layout

```
src/inference_optimizer/   application code (hardware, config, logging, backends, api, cli)
tests/unit/                 fast, mocked unit tests
tests/integration/           real-subprocess/CLI tests, incl. opt-in model smoke test
configs/models/               model config YAML (name, tokenizer, revision, quant, params)
scripts/                     one-off operational scripts
benchmarks/                   benchmark suite (added later)
results/                     persisted experiment output (gitignored)
dashboard/                   Streamlit dashboard (added later)
docs/                        design docs and reports
```

## License

MIT — see model-specific licenses in `docs/` once a model is selected.
