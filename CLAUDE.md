# Project: LLM Inference Optimization Engine

28-day build. One task per day from the user; implement only that day's scope,
never jump ahead. Full requirements are in the original project brief (see
PROGRESS.md Day 0 entry for the verbatim spec) — the rules below are the ones
that matter for every session.

## Hardware ground truth (this dev machine)

- Windows 11, Python 3.12.0, AMD64, 16 logical cores, ~16.5 GB RAM.
- GPU: NVIDIA RTX 3050 Laptop, **4096 MiB VRAM**, driver 566.36, CUDA 12.7,
  compute capability 8.6. 4 GB VRAM is the binding constraint — only small
  quantized models (≈3-7B at 4-bit) fit. Do not assume a bigger GPU.
- Docker 27.4.0 available. Disk: ~57 GB free (tight — avoid caching multiple
  large model checkpoints at once).
- Always re-run `python -m inference_optimizer hardware-info` rather than
  trusting this snapshot if hardware-dependent decisions matter — it can go
  stale.

## Hard rules

- Never fabricate benchmark numbers, optimization percentages, quality scores,
  or utilization data. If something can't be measured here, say so and give
  an executable alternative.
- Distinguish measured vs. estimated vs. unavailable metrics explicitly.
- Don't install GPU-only packages (the `gpu` extra) unless CUDA is confirmed
  working in the target environment.
- No speculative abstraction — build only what the current day's task needs.
- At the end of each session: summarize files touched, explain design
  decisions, report exact test outcomes, list unresolved issues, update
  PROGRESS.md and ARCHITECTURE.md, suggest a commit message.

## Current architecture (Day 6)

`src/inference_optimizer/{hardware,config,logging,backends,api,benchmark,workload,baseline,cli}`
— see ARCHITECTURE.md for the full module map and rationale. Active model:
Qwen2.5-0.5B-Instruct GGUF (Q4_K_M), config at
`configs/models/qwen2.5-0.5b-instruct.yaml`. llama.cpp backend works here;
vLLM is a hardware-gated scaffold only (no Linux+CUDA host available).
FastAPI service (`python -m inference_optimizer serve`) wraps the backend
over HTTP: `/health`, `/models`, `/generate`, `/generate/stream`, `/metrics`.
No auth — local/dev only. `llama-cpp-python` is pinned exactly to `0.3.2`
(newer prebuilt wheels crash with an illegal-instruction error on this CPU
— see PROGRESS.md Day 3). Benchmark client (`python -m inference_optimizer
benchmark`) measures TTFT/latency/TPOT/throughput against the streaming
API — see docs/benchmarking.md for metric definitions/policies. Installing
the `benchmark` extra (`tokenizers`) can upgrade `click` past typer's pin;
re-pin `click<8.2,>=8.1` afterward if `hardware-info`/other flag-using
commands start crashing. Versioned workload dataset at
`data/workloads/prompts_v1.jsonl` (+ `.manifest.json`), built via
`workload-generate`/validated via `workload-validate`/sampled via
`workload-sample` — see docs/workload.md. Day 6 adds a config-driven
end-to-end baseline runner (`python -m inference_optimizer baseline
--config configs/baseline.yaml`) that drives the backend directly (no
`serve` needed), with full provenance (git commit, package versions,
hardware) and raw JSONL + aggregate JSON/CSV output — see docs/baseline.md,
including the GPU-memory-is-whole-device caveat. Not yet wired into the
Day 4 `benchmark` CLI command (still takes one repeated prompt against a
running server) — kept as a separate command, see docs/baseline.md's
naming note.
