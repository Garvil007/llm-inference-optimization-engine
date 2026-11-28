# LLM Inference Optimization Engine

End-to-end system for deploying open-weight LLMs, benchmarking inference
performance, profiling CPU/GPU resources, comparing quantization and batching
strategies, and recommending configurations that improve throughput and
latency while tracking output quality.

## Status

Day 7 of 28: project scaffold, hardware detection, typed config, structured
logging, a working llama.cpp inference backend (Qwen2.5-0.5B-Instruct) and a
hardware-gated vLLM backend scaffold, a production-style FastAPI service
(`/health`, `/models`, `/generate`, `/generate/stream`, `/metrics`) wrapping
the backend over HTTP, a benchmark client that measures TTFT/latency/TPOT/
throughput against the streaming API (see [docs/benchmarking.md](docs/benchmarking.md)),
a reproducible workload dataset generator (5 categories x 3 length tiers,
tokenizer-aware, seeded sampling -- see [docs/workload.md](docs/workload.md)),
a config-driven baseline benchmark runner that drives the backend
directly end-to-end (readiness check, warmup, fixed workload, repeated runs,
full provenance) -- see [docs/baseline.md](docs/baseline.md)), and a report
pipeline that validates/flags/plots those baseline results into Markdown +
HTML -- see [docs/reporting.md](docs/reporting.md) and the
[Week 1 technical report](docs/week1_report.md).
See [PROGRESS.md](PROGRESS.md) for detailed status and
[ARCHITECTURE.md](ARCHITECTURE.md) for design decisions.

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

### Workload dataset

A versioned, reproducible prompt dataset lives at `data/workloads/prompts_v1.jsonl`
(5 categories x 3 input-length tiers x 2 variants = 30 records), with its
provenance manifest at `data/workloads/prompts_v1.manifest.json`.

```bash
# Regenerate it (deterministic given the same tokenizer version):
python -m inference_optimizer workload-generate --path data/workloads/prompts_v1.jsonl

# Validate it (or any JSONL dataset of your own):
python -m inference_optimizer workload-validate --path data/workloads/prompts_v1.jsonl

# Reproducibly sample from it:
python -m inference_optimizer workload-sample \
  --path data/workloads/prompts_v1.jsonl --count 10 --seed 42
```

See [docs/workload.md](docs/workload.md) for category/tier definitions, the
model-context-limit clamping policy for the 2048-token tier, and exact
reproduction steps.

### Baseline benchmark

A config-driven end-to-end baseline run (no API server required -- drives the
backend directly):

```bash
python -m inference_optimizer baseline --config configs/baseline.yaml
```

Verifies backend readiness, warms up, runs a fixed seeded workload sample
sequentially (concurrency=1) across several repeats, and writes raw
observations (JSONL) plus aggregate metrics (JSON + CSV) with full run
provenance (git commit, software versions, hardware, model/backend config,
process + GPU memory) to `results/baseline/`. See
[docs/baseline.md](docs/baseline.md) for the metric definitions, the
GPU-memory-is-whole-device caveat, and a real run's numbers from this
machine.

### Report generation

A Day 7 pipeline that parses a baseline run's real artifacts, validates
aggregates against raw observations, flags incomplete/failed/non-comparable
runs, and renders a Markdown + HTML report with real-measurement
distribution plots:

```bash
pip install -e ".[report]"   # matplotlib
python -m inference_optimizer report \
  --baseline-report results/baseline/baseline_<timestamp>_report.json \
  --observations results/baseline/baseline_<timestamp>_observations.jsonl \
  --output-dir reports/baseline
```

See [docs/reporting.md](docs/reporting.md) for flag/metric definitions and
measurement limitations, and [docs/week1_report.md](docs/week1_report.md)
for the Week 1 technical report generated from this machine's real Day 6
baseline run.

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
src/inference_optimizer/   application code (hardware, config, logging, backends, api, benchmark, workload, baseline, report, cli)
tests/unit/                 fast, mocked unit tests
tests/integration/           real-subprocess/CLI tests, incl. opt-in model smoke test, Week 1 pipeline test
configs/models/               model config YAML (name, tokenizer, revision, quant, params)
data/workloads/                versioned prompt dataset (JSONL) + manifest, version-controlled
scripts/                     one-off operational scripts
benchmarks/                   benchmark suite (added later)
results/                     persisted experiment output (gitignored)
reports/                     generated Markdown/HTML reports + plots (e.g. reports/week1/)
dashboard/                   Streamlit dashboard (added later)
docs/                        design docs and reports (benchmarking.md, workload.md, baseline.md, reporting.md, week1_report.md)
```

## License

MIT — see model-specific licenses in `docs/` once a model is selected.
