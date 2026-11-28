# Architecture

## Module map (Day 1)

```
src/inference_optimizer/
├── __main__.py          python -m inference_optimizer entry shim
├── exceptions.py         InferenceOptimizerError hierarchy
├── hardware/
│   ├── models.py         pydantic models: CPUInfo, MemoryInfo, GPUInfo, DiskInfo,
│   │                      SoftwareInfo, HardwareReport
│   └── detector.py       probing functions (detect_cpu/memory/gpu/disk/docker)
├── config/
│   └── settings.py       pydantic-settings Settings + get_settings()
├── logging/
│   └── setup.py          structlog configuration (console or JSON renderer)
└── cli/
    └── main.py            typer app; `hardware-info` command
```

## Key decisions

**pydantic-settings for config.** The project already needs pydantic for the
hardware report's structured JSON output, so pydantic-settings adds typed env
parsing (required-field errors, type coercion, `.env` support) without a new
dependency family. `get_settings()` is not cached — settings are cheap to
build and tests mutate env vars per-case, so a cached singleton would make
tests order-dependent.

**GPU detection: pynvml first, nvidia-smi fallback.** `pynvml` gives structured
data (no text parsing) and lives behind the optional `gpu` extra, so most dev
environments won't have it installed. `detect_gpu()` tries it first and falls
back to parsing `nvidia-smi --format=csv`, which only requires the NVIDIA
driver (not the full CUDA toolkit) — this is why the dev machine, which has
the driver but not `pynvml` installed, is correctly detected via the fallback
path. If neither path finds a GPU, `GPUInfo.available=False` is returned, never
an exception — hardware-info must work on CPU-only machines.

**Every hardware probe degrades, never raises.** Missing binaries, timeouts,
and permission errors are caught per-probe and logged as warnings; the field
is left `None`/`False`. `detect_hardware()` only raises
`HardwareDetectionError` for a genuine programming bug — environmental
absence is expected and must not crash the CLI.

**structlog over stdlib logging directly.** Gives structured key-value
logging with a one-line switch between human-readable console output (dev)
and JSON (CI/log collectors), via `INFERENCE_OPTIMIZER_LOG_JSON`.

**typer over argparse for the CLI.** Declarative options/help with minimal
boilerplate; the project will likely grow more subcommands (benchmark, serve,
report) over the next 27 days, and typer scales to that without rewriting the
argument-parsing layer.

**`click` pinned to `<8.2`.** `typer==0.12.5` crashes on boolean
`--flag/--no-flag` options under `click>=8.2` (`TypeError: Secondary flag is
not valid for non-boolean flag`) — a known upstream incompatibility, found by
actually running the CLI, not by unit tests. Pinned in `pyproject.toml` rather
than worked around in application code.

**Windows subprocess calls use `shell=True` conditionally.** `docker` and
`nvidia-smi` resolved via `shutil.which` can point at a `.cmd`/`.bat` shim on
Windows, which `CreateProcess` can't exec directly (`WinError 193`).
`detect_docker`/`_detect_gpu_via_nvidia_smi` pass `shell=True` only when
`sys.platform == "win32"`; POSIX systems are unaffected.

**Package named `logging/`, not `log/`.** Python 3 uses absolute imports by
default, so `import logging` inside `inference_optimizer/logging/setup.py`
correctly resolves to the stdlib module, not itself — verified by running the
test suite and CLI rather than assumed.

## Module map (Day 2 additions)

```
src/inference_optimizer/backends/
├── models.py              ModelConfig, BackendConfig, HealthStatus, GenerationResult,
│                            GenerationParams, LlamaCppModelSource, VLLMModelSource
├── base.py                 InferenceBackend ABC (model_info, health_check,
│                            generate_stream, generate, start/stop, async context manager)
├── llama_cpp_backend.py    LlamaCppBackend: subprocess-manages llama_cpp.server
├── vllm_backend.py         VLLMBackend: hardware-gated scaffold, same HTTP contract
├── registry.py             create_backend(backend_type, model_config, backend_config)
└── config_loader.py        load_model_config(path) -> ModelConfig from YAML

configs/models/qwen2.5-0.5b-instruct.yaml   model identity + both backends' sources
```

## Key decisions (Day 2)

**Every backend is an OpenAI-compatible HTTP client, not an in-process library
call.** `LlamaCppBackend` launches `python -m llama_cpp.server` as a subprocess
and talks to its `/v1/chat/completions` over `httpx`; `VLLMBackend` does the
same against `vllm.entrypoints.openai.api_server`. Both backends are therefore
interchangeable from the caller's point of view (same request/response shape),
which is what later benchmarking days need — swapping backends means swapping
one factory call, not rewriting the client. The cost is one extra process
hop vs. an in-process `Llama()` call; acceptable since benchmarking day will
measure realistic server-serving numbers anyway, not raw library overhead.

**Model chosen: Qwen2.5-0.5B-Instruct.** Apache-2.0, tokenizer/base repo
`Qwen/Qwen2.5-0.5B-Instruct`, GGUF from `Qwen/Qwen2.5-0.5B-Instruct-GGUF`
@ `9217f5db79a29953eb74d5343926648285ec7e67` (quant Q4_K_M, ~491 MB,
downloaded and verified this session). Small enough to run on CPU or the
4 GB VRAM GPU detected Day 1; real generations were produced and checked
for factual correctness (see PROGRESS.md).

**Real bug found and fixed: subprocess stdout pipe deadlock.** The first
`backend-status` run against the real server hung for the full 60s startup
timeout and failed. Root cause: `llama_cpp.server` prints a large chat-template
dump to stdout at startup; nothing was draining the `asyncio.subprocess.PIPE`
concurrently, so the OS pipe buffer filled and blocked the subprocess before
it ever reached "Uvicorn running". Fixed by spawning a background
`_pump_output()` task immediately after `create_subprocess_exec` in both
`LlamaCppBackend` and `VLLMBackend`, which continuously reads stdout into a
bounded tail buffer (used for error messages on crash). This was only
findable by actually running the backend against a real server — the mocked
unit tests never spawn a subprocess and would never have caught it.

**vLLM backend is gated, not simulated.** `VLLMBackend._check_hardware_support()`
reuses Day 1's `detect_hardware()` to fail fast with a specific message
(wrong OS, no GPU, or package not installed) before ever attempting a
subprocess launch. On this Windows/no-confirmed-CUDA machine this always
raises `UnsupportedHardwareError`, which the CLI surfaces as a clean exit 1 —
verified this session (`backend-status --backend vllm`), and this is the
*correct* outcome per the Day 2 brief ("enabling actual execution only when
hardware and dependencies support it"), not a defect.

**llama-cpp-python installed from a prebuilt wheel index, not PyPI sdist.**
`pip install llama-cpp-python` tries to compile from source via CMake, which
fails on this machine (no C/C++ compiler installed). Installed instead from
`https://abetlen.github.io/llama-cpp-python/whl/cpu`, which ships a prebuilt
CPU wheel for this Python/platform combination. Documented in `pyproject.toml`
and README so a fresh clone doesn't hit the same build failure blind.

**Streaming tested with `httpx.MockTransport`, not a fake server process.**
Unit tests inject a mocked `httpx.AsyncClient` directly into `backend._client`
and assert against synthetic SSE bodies shaped like the real server's output
(captured from the actual running server this session). This avoids spawning
real subprocesses in the unit suite while still exercising the exact SSE
parsing path (`data: {...}` chunks, `[DONE]` sentinel) that real traffic uses.
The one real-subprocess path is `tests/integration/test_llama_cpp_smoke.py`,
skipped by default and opt-in via `RUN_REAL_BACKEND_TESTS=1`.

## Module map (Day 3 additions)

```
src/inference_optimizer/api/
├── schemas.py       GenerateRequest, GenerateResponse, ModelInfo, HealthResponse, ErrorResponse
├── config.py          ApiSettings (env prefix INFERENCE_OPTIMIZER_API_), get_api_settings()
├── concurrency.py     ConcurrencyLimiter / AdmissionRejected -- bounded admission, no queueing
├── metrics.py          Metrics -- hand-rolled Prometheus text exposition
└── app.py               create_app(settings, backend) -- FastAPI factory, routes, middleware
```

## Key decisions (Day 3)

**`create_app(settings, backend)` keeps the backend out of FastAPI's hands.**
Production (`serve` CLI command) passes `backend=None` and the `lifespan`
builds a real one via the Day 2 `create_backend`/`load_model_config`
functions -- unchanged. Tests pass a fake `InferenceBackend` directly, so
the API test suite (`tests/unit/test_api.py`, 15 tests) never spawns a real
subprocess. `api/app.py` has zero `import` of `llama_cpp_backend` or
`vllm_backend` by name; it only knows the `InferenceBackend` ABC.

**Bounded concurrency admission, not a queue.** `ConcurrencyLimiter` is a
plain counter guarded by `asyncio.Lock`, not `asyncio.Semaphore` -- asyncio's
`Semaphore` has no non-blocking try-acquire, and a queued semaphore would let
request latency grow unbounded under load instead of giving callers a fast,
explicit 429. For the streaming route specifically, admission is acquired
*before* `StreamingResponse` is constructed (not inside the generator):
once ASGI streaming starts, status 200 is already committed, so an
`AdmissionRejected` raised mid-generator could never surface as a 429.

**Streaming timeout is per-chunk, not one `wait_for` around the whole
generator.** `asyncio.wait_for(stream.__anext__(), timeout=remaining)` is
called on every chunk against a shared wall-clock deadline, so a stream that
keeps producing tokens past its total time budget is still cut off -- a
single outer `wait_for` around the entire async-for loop would not time out
a generator that's still actively yielding. Since HTTP status can't change
after streaming starts, a timeout or backend error mid-stream is surfaced as
one `data: {"error": ...}` SSE event, not an HTTP error code.

**Metrics are hand-rolled, not `prometheus_client`.** Three counters/gauges
(`requests_total`, `request_duration_seconds_{sum,count}`,
`requests_in_flight`) don't justify a new dependency; `Metrics.render_prometheus()`
emits the standard text exposition format directly so `/metrics` is
Prometheus-scrapeable without the library.

**Real bug found and fixed: llama-cpp-python's newer prebuilt wheel crashes
on this CPU.** Verifying end-to-end against the real server (not just the
mocked API tests) surfaced `OSError: [WinError -1073741795]`
(`STATUS_ILLEGAL_INSTRUCTION`) inside `llama_backend_init()` with
`llama-cpp-python` 0.3.38 (the version `>=0.3,<0.4` resolved to today) on
this AMD Ryzen 7 5800H. `llama-cpp-python==0.3.2` from the same wheel index
runs cleanly and served real completions and streams this session. Pinned
exactly in `pyproject.toml`'s `serving` extra with a comment explaining why
-- this is exactly the kind of thing that's only findable by actually
running inference, not by any mocked test, per the project's "never claim
inference works unless a real response was generated" rule.

## Module map (Day 4 additions)

```
src/inference_optimizer/benchmark/
├── models.py          RequestObservation, MetricStats, BenchmarkSummary, BenchmarkResult
├── token_counter.py     TokenCounter protocol, HFTokenizerCounter, FixedTokenCounter,
│                          WhitespaceApproxCounter
├── aggregate.py          summarize() -- percentile/stddev math + aggregation policies
└── client.py              BenchmarkClient, clock-injectable _consume_sse_stream

docs/benchmarking.md   metric definitions (TTFT/TPOT/inter-token latency/throughput/
                         end-to-end latency) and aggregation policies
```

## Key decisions (Day 4)

**Exactly 3 clock samples per request, never one per SSE chunk.** A
streaming chunk is not a token -- llama.cpp/vLLM's deltas are arbitrary text
fragments. `_consume_sse_stream` samples `time.perf_counter()` only at
request start, first non-empty content, and stream end. TPOT is derived
arithmetically from those 3 samples plus a token count
(`(latency_ms - ttft_ms) / (token_count - 1)`), not from per-chunk
timestamps -- this system explicitly does not measure true inter-token
latency (the per-token-gap distribution), and docs/benchmarking.md says so
rather than letting TPOT be mistaken for it.

**Token counting prefers the real model tokenizer over chunk/word
counting.** `HFTokenizerCounter` loads the actual model's tokenizer (HF
`tokenizers` library, lazy, degrades to `method="unavailable"` on any
failure rather than raising) and counts real token IDs on the concatenated
output text. `FixedTokenCounter`/`WhitespaceApproxCounter` exist for tests
and as an explicit fallback, respectively -- every `RequestObservation`
carries which method actually ran, so an estimate is never silently
presented as a measurement.

**The SSE-parsing core is clock-injectable and httpx-independent.**
`_consume_sse_stream(lines: AsyncIterator[str], clock: Callable[[], float])`
takes a plain async iterator, not an httpx response -- `BenchmarkClient`
wires in `response.aiter_lines()` and `time.perf_counter` for real runs, and
tests inject a fake clock (a predetermined sequence of values) plus a
canned line sequence, making every timing assertion in
`tests/unit/test_benchmark_client.py` exact and deterministic with zero
real `asyncio.sleep` calls anywhere in the suite.

**Aggregation policies are explicit, not implicit in the math.**
`aggregate.py`'s module docstring states the warmup-exclusion, failed-
request, missing-token-count, percentile (linear interpolation, numpy's
default method), and stddev (sample, ddof=1) policies in full, mirrored in
docs/benchmarking.md -- per the brief's explicit "define clear policies"
requirement.

**Real bug found and fixed: installing `tokenizers` reopened the Day 1
click/typer bug.** `pip install tokenizers` pulled in `huggingface-hub`,
which requires `click>=8.4.2`, silently upgrading past the `<8.2` pin typer
needs (same `TypeError: Secondary flag is not valid for non-boolean flag`
failure mode as Day 1). Fixed by re-pinning `click<8.2,>=8.1` immediately
after installing the `benchmark` extra; this is a real, currently
unresolved tension between the `benchmark` extra's dependency tree and the
CLI's click pin, noted in PROGRESS.md rather than hidden.

## Module map (Day 5 additions)

```
src/inference_optimizer/workload/
├── models.py       PromptRecord, WorkloadManifest
├── dataset.py         JSONL load/save
├── generator.py         generate_dataset() -- categories x tiers x variants,
│                          tokenizer-aware padding
├── validator.py          validate_dataset() -- errors vs. warnings
├── sampler.py             sample_records() -- seeded, reproducible
└── manifest.py             build_manifest()/save_manifest() -- sha256 provenance

data/workloads/prompts_v1.jsonl + .manifest.json   version-controlled dataset deliverable
docs/workload.md   categories/tiers, reproduction steps, clamping policy
```

## Key decisions (Day 5)

**Padding never touches the footer.** Every prompt is `header + padding +
footer`; padding only adds/removes whole filler units (a coarse paragraph,
then a fine sentence for precise top-up). The footer -- the actual
question/instruction/record -- is built once and never trimmed, so no
amount of length-targeting can corrupt the task itself.

**The 2048-token input tier is clamped to the model's real context budget,
not generated at the raw label.** `effective_target = min(input_label,
context_limit - output_target - safety_margin)`, which is 1760 for this
model's 2048-token context with a 256-token output target. Generating a
literal 2048-token prompt would leave zero room for any output. `models.py`
keeps `input_length_target` (the categorical label from the brief) and
`input_tokens_measured` (the real tokenizer count) as two separate fields
specifically so this clamping is never hidden -- the same
target-vs-observed discipline Day 4 applies to output tokens.

**Token counting is reused from Day 4, not duplicated.** `generator.py`
takes any `inference_optimizer.benchmark.token_counter.TokenCounter` (the
same protocol the benchmark client uses) -- `HFTokenizerCounter` for real
generation, `WhitespaceApproxCounter` for offline/deterministic tests. One
counting abstraction serves both benchmarking and workload generation.

**Validation separates errors from warnings.** Structural/safety problems
(duplicate ids, empty prompts, zero token counts, exceeding the model's
context budget) are errors; a measured length far from its target is a
warning -- because a user-provided dataset (the brief's "option to use
user-provided datasets") may legitimately not hit exact target lengths
without being unusable for benchmarking.

**Sampling uses a private seeded `random.Random`, never the global
`random` module.** `sample_records(records, count, seed, ...)` is a pure
function of its inputs -- the same call always returns the same sequence,
independent of what else in the process has consumed randomness, verified
this session by running the real CLI command twice with the same seed and
getting identical output.

**All prompt content is original.** Headers, filler paragraphs, and
footers (questions/instructions/records) were written for this project --
no existing dataset, benchmark suite, or licensed corpus is used anywhere
in `generator.py`, per the brief's explicit instruction to avoid
copyrighted or restricted datasets.

## Module map (Day 6 additions)

`src/inference_optimizer/baseline/`:
- `models.py` — `RunMetadata`, `SoftwareVersions`, `MemorySample`,
  `RepeatResult`, `BaselineReport`.
- `config.py` — `BaselineConfig` (pydantic, YAML-loaded via
  `load_baseline_config`), the schema behind `configs/baseline.yaml`.
- `metadata.py` — `collect_run_metadata`: git commit hash, installed
  package versions, full `HardwareReport`, model/backend config snapshots.
- `resources.py` — `sample_memory`: process RSS (`psutil`) + GPU memory
  used (`pynvml`), both degrading to `None`/`"unavailable"` on failure.
- `runner.py` — `run_baseline`: the orchestration itself (readiness,
  warmup, fixed-workload repeats, JSONL/JSON/CSV output).

## Key decisions (Day 6)

**Drives the `InferenceBackend` directly, not through the Day 3 HTTP API.**
A baseline run's job is to answer "does this actually work end-to-end and
how fast," so it shouldn't need a second process (`serve`) already running.
`_run_one_against_backend` in `runner.py` reimplements the same
three-clock-sample policy as `benchmark/client.py`'s `_consume_sse_stream`
(start / first non-empty chunk / completion) but over the backend's own
`AsyncIterator[str]` chunks directly -- genuinely simpler than Day 4's
version (no SSE line/JSON parsing needed), so this is deliberately not
sharing code with `_consume_sse_stream` rather than forcing a coupling
between the two contexts.

**"Fixed workload" = sample once, reuse every repeat.** `sample_records()`
is called a single time before the repeat loop; the same prompts, in the
same order, feed every one of the `repeats` passes. This isolates what
varies between repeats to the backend's own run-to-run variance, not the
input -- the thing a baseline is supposed to characterize.

**Unavailable backend writes nothing.** If `backend.start()` raises or
`health_check()` reports unhealthy, `run_baseline` returns immediately with
`BaselineReport.status="unavailable"` and the real exception text as
`unavailable_reason` -- before `output_dir` is even created. No
observation/aggregate file exists for a run that never actually measured
anything, so a later reader can't mistake "the run failed" for "the run
measured near-zero latency."

**Reuses Day 4's `summarize()`/`RequestObservation`/`BenchmarkSummary`
unmodified.** The per-request measurement contract and aggregation policies
(warmup exclusion, failure handling, missing-token-count handling,
percentile method, stdev) don't change just because requests now come from
a direct backend call instead of an HTTP stream -- same functions, called
with a different data source.

**GPU memory is a whole-device reading, documented as such.** `pynvml`'s
`nvmlDeviceGetMemoryInfo` reports total device memory in use, not this
process's share. Confirmed on the real run (docs/baseline.md): since the
llama.cpp backend here is CPU-only, the sampled GPU numbers reflect
unrelated GPU activity on the machine, not the backend's own consumption --
reported as "GPU memory in use on this machine" rather than mislabeled as
the backend's footprint.

**`token_counter`/`backend_factory` are optional constructor parameters on
`run_baseline`**, mirroring `BenchmarkClient`'s injectable `http_client` --
lets tests exercise the full orchestration (readiness/warmup/repeats/file
writing) against a `FakeBackend` and `FixedTokenCounter`, with no real
subprocess, model, or tokenizer download.

## Module map (Day 7 additions)

`src/inference_optimizer/report/`:
- `models.py` — `RawObservationRecord`, `ValidationResult`,
  `VariabilityStat`, `RunFlag`, `ComparisonNote`, `ReportData`.
- `parser.py` — `load_baseline_report`/`load_raw_observations`: reads real
  Day 6 artifacts back into typed objects.
- `validate.py` — `validate_aggregates`: recomputes `BenchmarkSummary` from
  raw observations via Day 4's `summarize()` and diffs against the stored
  aggregate.
- `variability.py` — `compute_variability`: repeat-to-repeat mean/stddev/
  coefficient-of-variation per metric.
- `flags.py` — `compute_flags`: incomplete/failed/mismatched/contended/
  non-comparable quality flags.
- `comparison.py` — `build_comparisons`: internal-consistency checks only
  (no external SLA exists yet).
- `plots.py` — `generate_distribution_plots`: matplotlib Agg histograms.
- `content.py` — shared table-row builders for both renderers.
- `markdown_renderer.py` / `html_renderer.py` — `render_markdown`/
  `render_html`.
- `generator.py` — `generate_report`: orchestrates parse → validate →
  variability → flags → comparisons → plots → render.

## Key decisions (Day 7)

**Validation reuses Day 4's `summarize()` unmodified**, the same reuse
discipline as Day 6 — a report that recomputed aggregates with different
logic than the runner that produced them would validate against its own
assumptions, not the actual pipeline.

**Flags annotate, never mutate.** `compute_flags` never changes a stored
number; a run with `requests_failed` or `aggregate_mismatch` still renders a
full report with that flag attached, so the reader sees both the data and
the reason to distrust it.

**"Comparison against expectations" is internal-consistency only.** There is
no documented SLA or vendor-published expected-performance number for this
model+hardware combination — this week produced the *first* baseline, so
`comparison.py` checks only that measured numbers are mutually consistent
(TTFT p50 <= latency p50) and explicitly notes `no_external_baseline` rather
than fabricating a target to compare against.

**Markdown/HTML renderers share table-building logic (`content.py`)** so the
two output formats can't drift apart in content, only in layout — adding a
field means editing one function, not two renderers independently.

**Plots are skipped, not degraded, below 2 data points.** A single-point
"distribution" would misrepresent spread that doesn't exist; `plots.py`
simply omits that metric from `plot_paths` rather than drawing a misleading
chart.

**Fixture data for plot tests is explicitly labeled.** `test_report_plots.py`
names its synthetic values `_FIXTURE_*` and documents them as fixture data
in the module docstring — the only place in this codebase a plot is ever
drawn from invented numbers, and it's never confused with a real
measurement in any generated report.

## Not yet built

Profiling, quantization experiments, evaluation harness, dashboard,
Docker/CI, concurrent/load benchmarking, config recommendation, and wiring
the workload dataset into the Day 4 benchmark CLI (which still takes a
single repeated prompt) — scoped to later days per the 28-day plan.
