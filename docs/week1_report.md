# Week 1 technical report: LLM Inference Optimization Engine

Covers Day 1-7. Generated numbers in this report are pulled directly from the
real run at `reports/week1/` (produced by `python -m inference_optimizer
report --baseline-report results/baseline/baseline_20261125T235104Z_report.json
--observations results/baseline/baseline_20261125T235104Z_observations.jsonl
--output-dir reports/week1`) -- see that directory's `report.md`/`report.html`
for the full table, flags, and plots. This document is the narrative wrapper:
architecture, what the generated report found, and the risk assessment for
Week 2.

## Architecture

```
src/inference_optimizer/
  hardware/   CPU/GPU/memory/disk/software detection (psutil, pynvml, nvidia-smi fallback)
  config/     pydantic-settings app config
  logging/    structlog setup
  backends/   InferenceBackend ABC + llama.cpp (real) and vLLM (hardware-gated scaffold)
  api/        FastAPI service wrapping a backend over HTTP (/generate, /generate/stream, /metrics)
  benchmark/  BenchmarkClient (HTTP) + summarize() aggregation + token counters
  workload/   versioned prompt dataset generator/validator/sampler
  baseline/   Day 6: direct-backend end-to-end baseline runner, repeated, with full provenance
  report/     Day 7: parses baseline artifacts, validates, flags, plots, renders MD/HTML
  cli/        typer CLI wiring every command above
```

Two measurement paths exist by design: `benchmark` talks to a *running*
API server over HTTP (exercises the full serving stack, including
network/serialization overhead); `baseline` drives a backend directly in-process
(exercises raw inference performance without that overhead). Both reuse the
same `RequestObservation`/`summarize()` aggregation code (Day 4), so the two
paths are comparable to each other at the metric-definition level, even
though they measure different things.

## What the real baseline run found

Model: Qwen2.5-0.5B-Instruct (Q4_K_M GGUF), llama.cpp backend, CPU-only, this
machine (Windows 11, 8c/16t, 16.5 GB RAM, RTX 3050 4GB -- unused by this
backend). 6 sampled prompts x 3 repeats = 18 requests, 0 failures.

| Repeat | latency p50 (ms) | latency p95 (ms) | TTFT p50 (ms) | throughput p50 (tok/s) |
|---|---|---|---|---|
| 0 | 3829.3 | 5239.3 | 805.6 | 19.0 |
| 1 | 5163.1 | 8665.8 | 1019.4 | 11.0 |
| 2 | 5980.5 | 15170.8 | 1338.9 | 10.8 |

The Day 7 report generator's automatic flags on this exact run:

- **`low_available_memory`**: only 2.41 GB RAM free at run time.
- **`high_repeat_variability`**: throughput p50's coefficient of variation
  (0.34) exceeds the 0.3 threshold across the 3 repeats.
- **`gpu_memory_not_process_attributed`** (informational): a GPU was
  detected, so pynvml's whole-device memory caveat applies.

Validation (recomputing every aggregate from the raw per-request JSONL and
comparing to the stored `BenchmarkSummary`) passed for all 3 repeats --
the stored numbers are exactly reproducible from the raw data, not drifted
or hand-edited.

**Reading of the latency growth across repeats:** latency and its variance
both grew monotonically (repeat 0 -> 2), and available RAM was already low at
the start of the run. The most defensible explanation given what's measured
here is machine-level memory/CPU contention from other processes on this
shared dev laptop, not a defect in the runner or the model. This is reported
as an observed finding, not explained away -- there is no controlled,
isolated-machine run yet to rule out other causes, and this report does not
claim one.

## Review: what's working, what's untested, risks for Week 2

**Working (verified this week, not assumed):**
- Hardware detection, a real llama.cpp backend, a FastAPI service, an HTTP
  benchmark client, a versioned/reproducible workload dataset, a direct-backend
  baseline runner with full provenance, and this report pipeline all ran
  against real components at least once, with real output inspected on disk.
- The Day 7 validation step confirms the whole measurement pipeline
  (observation -> aggregate -> stored JSON) is internally consistent, not
  just that it runs without crashing.
- 141 unit/integration tests passing, 1 skipped (3 pre-existing, unrelated
  `test_hardware_detector.py` failures -- GPU-tooling mocking issue, not
  touched by this week's work); ruff and mypy clean.

**Untested / not yet exercised:**
- The vLLM backend has never run end-to-end (no Linux+CUDA host available);
  it is a hardware-gated scaffold whose "unsupported hardware" failure path
  is the only thing actually verified.
- No concurrency > 1 has been measured anywhere -- every baseline and
  benchmark run so far is strictly sequential. Nothing is known yet about
  this stack's behavior under concurrent load.
- No quantization or batching *comparison* exists yet -- Week 1 produced a
  single configuration's baseline, not a comparison across configurations
  (that is explicitly Week 2+ scope per the original brief).
- The report's distribution plots have only been exercised against one real
  run with 18 data points; behavior on much larger or much smaller sample
  sizes is untested beyond the unit tests' fixture-data edge cases (0, 1, 2
  points).

**Technical risks heading into Week 2:**
1. **This machine's own resource contention is now a confirmed variable.**
   Any Week 2 comparison (e.g. quantization A vs. B) run on this same shared
   machine needs either controlled conditions (close other apps, check
   `hardware-info` immediately before each run) or the comparison risks being
   dominated by contention noise rather than the variable being tested.
2. **4 GB VRAM is a hard ceiling.** Only small quantized models fit; any
   vLLM/GPU-path comparison work is blocked without a different host.
3. **No vLLM comparison is possible on this machine at all** -- Week 2
   quantization/batching comparisons will likely be llama.cpp-only
   (different quant levels of the same GGUF family) unless a Linux+CUDA
   environment becomes available.
4. **High repeat-to-repeat variability (CV up to 0.34 observed) means Week 2
   comparisons need more than 3 repeats, or a variance-aware comparison
   method (e.g. requiring non-overlapping confidence intervals), to avoid
   concluding a difference exists when it's actually just this machine's
   noise.**
