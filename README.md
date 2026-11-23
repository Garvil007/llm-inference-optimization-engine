# LLM Inference Optimization Engine

End-to-end system for deploying open-weight LLMs, benchmarking inference
performance, profiling CPU/GPU resources, comparing quantization and batching
strategies, and recommending configurations that improve throughput and
latency while tracking output quality.

## Status

Day 2 of 28: project scaffold, hardware detection, typed config, structured
logging, plus a working llama.cpp inference backend (Qwen2.5-0.5B-Instruct)
and a hardware-gated vLLM backend scaffold. See [PROGRESS.md](PROGRESS.md)
for detailed status and [ARCHITECTURE.md](ARCHITECTURE.md) for design
decisions.

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
```

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
src/inference_optimizer/   application code (hardware, config, logging, backends, cli)
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
