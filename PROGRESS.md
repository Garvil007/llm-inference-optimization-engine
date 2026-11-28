# Progress Log

## Day 0 — Project brief received

User specified the full 28-day LLM Inference Optimization Engine project
(objective, tech stack, 15 engineering requirements, daily-task workflow).
No code written yet; waiting for Day 1 task.

## Day 1 — Foundation scaffold

**Environment detected on this machine:**
- Windows 11 (10.0.26200), Python 3.12.0, AMD64, 16 logical cores, 16.54 GB RAM
- GPU: NVIDIA RTX 3050 Laptop, 4096 MiB VRAM, driver 566.36, CUDA 12.7, compute 8.6
- Disk: 57 GB free of 953 GB
- Docker 27.4.0 available

**Built:**
- `src/inference_optimizer/` package: `hardware/` (models + detector),
  `config/` (pydantic-settings), `logging/` (structlog setup),
  `cli/` (typer app with `hardware-info` command), `exceptions.py`.
- `pyproject.toml` with `core`, `gpu`, `dev`, `dashboard` dependency groups.
  `gpu` extra deliberately not installed (no CUDA toolkit confirmed, only
  the driver — see ARCHITECTURE.md).
- Unit tests: `tests/unit/test_hardware_detector.py`,
  `tests/unit/test_settings.py` — all hardware/software probes mocked.
- `.gitignore`, `.env.example`, `README.md`, `CLAUDE.md`, `ARCHITECTURE.md`.
- `Makefile` + `tasks.py` (cross-platform equivalent for Windows).

**Test/CLI results (measured, this session):**
- `pytest`: 13/13 passed.
- `ruff check src tests`: all checks passed.
- `mypy src`: no issues found (12 source files).
- `python -m inference_optimizer hardware-info` (both `--pretty` and
  `--no-pretty`): runs clean, correctly reports 8 physical / 16 logical
  cores, 16.54 GB RAM, RTX 3050 Laptop (4096 MiB, driver 566.36, compute
  8.6, via nvidia-smi fallback), ~1 TB disk, and Docker 27.4.0.

**Bugs found and fixed during verification (not caught by mocked unit
tests — only surfaced by actually running the CLI):**
1. `typer==0.12.5` + `click==8.5.1` crash on any boolean `Option` with
   `--flag/--no-flag` syntax (`TypeError: Secondary flag is not valid for
   non-boolean flag`) — a known upstream incompatibility. Fixed by pinning
   `click>=8.1,<8.2` in `pyproject.toml`.
2. Docker detection failed on Windows (`WinError 193: %1 is not a valid
   Win32 application`) because `docker` resolves via PATH to a `.cmd`
   shim, and `subprocess.run` can't exec that without a shell. Fixed by
   passing `shell=True` on Windows only (`sys.platform == "win32"`) in
   `detect_docker`/`_detect_gpu_via_nvidia_smi`.

**Unresolved / deferred:**
- `gpu` extra (torch, nvidia-ml-py) intentionally not installed/tested —
  only the NVIDIA driver is present, not a confirmed CUDA toolkit.
- No git repository initialized yet (not requested for Day 1).
- Hardware/software probe branches not covered by mocked tests
  (`pynvml` success path, nvidia-smi unparsed-output path, CLI error
  path) are implemented but only exercised via the manual CLI run above,
  not asserted in `tests/`; coverage is 66% (`hardware/detector.py` 68%,
  `cli/main.py` and `logging/setup.py` 0%/57% — CLI and logging setup
  have no dedicated tests yet).

## Day 2 — First operational inference backend

**Model selected:** Qwen2.5-0.5B-Instruct (Apache-2.0).
- Tokenizer/base repo: `Qwen/Qwen2.5-0.5B-Instruct`.
- GGUF: `Qwen/Qwen2.5-0.5B-Instruct-GGUF` @ commit
  `9217f5db79a29953eb74d5343926648285ec7e67` (verified via HF API this
  session), file `qwen2.5-0.5b-instruct-q4_k_m.gguf` (Q4_K_M, 491,400,032
  bytes, downloaded to `.cache/models/`).
- Context limit configured: 2048 tokens.

**Built:**
- `src/inference_optimizer/backends/`: `models.py` (ModelConfig,
  BackendConfig, HealthStatus, GenerationResult, etc.), `base.py`
  (`InferenceBackend` ABC), `llama_cpp_backend.py`, `vllm_backend.py`,
  `registry.py` (`create_backend`), `config_loader.py`
  (`load_model_config`).
- `configs/models/qwen2.5-0.5b-instruct.yaml`.
- CLI commands: `backend-status --backend <llama-cpp|vllm>`,
  `generate "<prompt>" [--stream/--no-stream] [--backend ...] [--config ...]`.
- Unit tests (mocked `httpx.MockTransport`, no real subprocess):
  `test_llama_cpp_backend.py` (8), `test_vllm_backend.py` (5),
  `test_config_loader.py` (3), `test_backend_registry.py` (3).
- Optional real-backend smoke test:
  `tests/integration/test_llama_cpp_smoke.py`, skipped unless
  `RUN_REAL_BACKEND_TESTS=1`.

**Real inference verified this session (not simulated):**
- `backend-status --backend llama-cpp`: started `llama_cpp.server` as a
  subprocess, health check returned `{"healthy": true, "model_loaded": true}`.
- `generate "What is the capital of France? Answer in one short sentence." --max-tokens 40`
  (streaming): produced "Paris, the capital of France, is located in the
  Île de la Cité region." (an accented character rendered as `�` in the
  Windows console due to cp1252 encoding — a display issue, not a
  generation bug; the raw HTTP response bytes were correct UTF-8).
- `generate "Name one primary color." --no-stream --max-tokens 20`:
  produced "The primary color is red."
- `backend-status --backend vllm`: correctly raised
  `UnsupportedHardwareError` ("vLLM has no official Windows build...") and
  exited 1 — the intended behavior per the brief's "enabling actual
  execution only when hardware and dependencies support it," not a defect.
- `RUN_REAL_BACKEND_TESTS=1 pytest tests/integration/test_llama_cpp_smoke.py -v -s`:
  1 passed in 2.93s (real model load + real generation).

**Bugs found and fixed during verification (only surfaced by running a
real backend, not by the mocked unit tests):**
1. `pip install llama-cpp-python` failed to build from source (no C/C++
   compiler on this machine — `CMake Error: No CMAKE_C_COMPILER could be
   found`). Fixed by installing the prebuilt CPU wheel from
   `https://abetlen.github.io/llama-cpp-python/whl/cpu` instead; documented
   in `pyproject.toml` and README.
2. `llama_cpp.server --help` crashed with a `UnicodeEncodeError` under the
   Windows cp1252 console codepage — cosmetic, `--help` only, no effect on
   real usage (confirmed by running the server for real immediately after).
3. **Subprocess stdout pipe deadlock** (the significant one): the first
   real `backend-status` run hung for the full 60s startup timeout and
   failed with `BackendStartupError`. Root cause: `llama_cpp.server`'s
   startup log (chat-template dump) is large enough to fill the OS pipe
   buffer; nothing was draining `asyncio.subprocess.PIPE` concurrently, so
   the subprocess blocked on a full buffer before ever printing "Uvicorn
   running". Fixed by adding a background `_pump_output()` task in both
   `LlamaCppBackend` and `VLLMBackend` that continuously drains stdout into
   a bounded tail buffer. Re-ran after the fix: ready in ~4s.

**Test/CLI results (measured, this session):**
- `pytest`: 32 passed, 1 skipped (the real-backend test, opt-in) — plus
  the 1 real-backend test passing separately when explicitly enabled.
- `ruff check src tests`: all checks passed.
- `mypy src`: no issues found (19 source files).
- Coverage: 63% overall (`backends/models.py`, `config_loader.py`,
  `registry.py` 100%; `llama_cpp_backend.py` 57%, `vllm_backend.py` 50% —
  the un-mocked subprocess-lifecycle lines are covered only by the manual
  CLI runs and the opt-in smoke test above, not asserted in the default
  suite).

**Unresolved / deferred:**
- `vllm` extra not installed/tested anywhere (no Linux+CUDA host
  available) — `VLLMBackend`'s HTTP/streaming code path is verified via
  mocked transport only; its subprocess-launch path has never run for
  real against an actual vLLM server.
- `cli/main.py`'s new commands have no dedicated unit tests (only manual
  CLI runs); CLI coverage remains 0% in the automated suite.
- No benchmarking/metrics yet — this day only proves one backend works
  end-to-end, not its performance.

## Day 3 — Production-style FastAPI service

**Built:**
- `src/inference_optimizer/api/`: `schemas.py` (`GenerateRequest`,
  `GenerateResponse`, `ModelInfo`, `HealthResponse`, `ErrorResponse`),
  `config.py` (`ApiSettings`, env prefix `INFERENCE_OPTIMIZER_API_`),
  `concurrency.py` (`ConcurrencyLimiter` — bounded admission, rejects
  immediately over the limit instead of queueing), `metrics.py`
  (hand-rolled Prometheus text exposition, no new dependency), `app.py`
  (`create_app()` — FastAPI app factory, routes, request-ID/logging
  middleware, exception handlers).
- Routes: `GET /health`, `GET /models`, `POST /generate`,
  `POST /generate/stream` (SSE), `GET /metrics`.
- CLI: `python -m inference_optimizer serve [--host] [--port]` (lazy
  `uvicorn`/`fastapi` imports so the other CLI commands never require
  the `serving` extra).
- Small backend edit (both `LlamaCppBackend` and `VLLMBackend`): optional
  `seed` passthrough in the chat-completions payload when the caller
  supplies one, so the API's `seed` field has somewhere to go.
- `tests/unit/test_api.py` (15 tests) against a fake `InferenceBackend`
  (`FakeBackend`) — no real subprocess spawned: valid requests, malformed
  input (empty prompt, out-of-range temperature, missing field), backend
  unavailability (`BackendConnectionError` → 503), timeout (slow backend +
  short `timeout_seconds` → 504), request-ID echo, SSE streaming (chunks +
  `[DONE]`, and an error event mid-stream), concurrency admission (429 when
  `concurrency_limit=1` and a second request arrives while the first is
  still in flight, using `httpx.AsyncClient` + `ASGITransport` for real
  overlap), and `/metrics` content.
- Added `fastapi` to the `dev` extra (tests need `TestClient`, not just
  `serving` users) and pinned `llama-cpp-python==0.3.2` exactly in
  `serving` (see bug below).

**Design decisions:**
- The backend stays framework-agnostic: `create_app(settings, backend)`
  accepts an already-constructed `InferenceBackend`; production (`serve`)
  leaves it `None` and the FastAPI `lifespan` builds one via the existing
  `create_backend`/`load_model_config`. Tests inject a fake backend
  directly — `api/app.py` imports nothing backend-implementation-specific.
- Bounded admission over queueing: `ConcurrencyLimiter` is a plain counter
  + `asyncio.Lock` (not `asyncio.Semaphore`, which has no non-blocking
  try-acquire in asyncio) that rejects immediately with 429 past the
  limit, so load past capacity fails fast instead of piling up latency.
- Streaming admission is acquired *before* `StreamingResponse` is
  constructed, not inside the generator: once the ASGI response starts
  streaming, status 200 is already committed, so an `AdmissionRejected`
  raised mid-generator could never become a 429. The non-streaming
  `/generate` path doesn't have this constraint and uses a normal
  `async with limiter.admit():`.
- Streaming timeout is enforced per-chunk against a wall-clock deadline
  (`asyncio.wait_for(stream.__anext__(), timeout=remaining)`), not a
  single overall `wait_for` around the whole generator — a single
  `wait_for` can't time out a generator that's still actively yielding one
  chunk every few ms but never finishing within the window otherwise. A
  timeout or backend error mid-stream yields one `data: {"error": ...}`
  SSE event rather than changing the HTTP status (can't, after streaming
  started) — documented in the README.
- Metrics are a hand-rolled Prometheus text exposition format, not
  `prometheus_client` — three counters/gauges don't justify a new
  dependency; swap in the real client library later if cardinality grows.
- `seed` is passed through to the backend only when the caller supplies
  it (`payload["seed"] = ...` only if not `None`), so existing behavior
  and existing tests are unaffected when it's omitted.

**Real end-to-end verification (not simulated):**
- `python -m inference_optimizer serve --port 8000` against the real
  Qwen2.5-0.5B-Instruct GGUF via `LlamaCppBackend`:
  - `GET /health` → `{"healthy":true,"backend_type":"llama-cpp","detail":"Reachable at http://127.0.0.1:8731.","model_loaded":true}`
  - `GET /models` → `[{"name":"qwen2.5-0.5b-instruct","backend":"llama-cpp","context_limit":2048,"quantization":"Q4_K_M","license":"apache-2.0"}]`
  - `POST /generate {"prompt":"What is the capital of France? Answer in one short sentence.","max_tokens":40}`
    → `{"text":"The capital of France is Paris.", ..., "latency_seconds":0.609}`
  - `POST /generate/stream {"prompt":"Name one primary color.","max_tokens":20}`
    → six `data: {"text": ...}` SSE events ("One primary color is red.")
    followed by `data: [DONE]`.
  - Malformed request (`{"prompt":""}`) → `422` (schema validation).
  - `X-Request-ID: real-check-1` sent on request → echoed back verbatim in
    both the response header and the JSON body's `request_id` field.
  - `GET /metrics` after the above → non-zero `inference_optimizer_requests_total`
    counters per method/path/status and a populated `_duration_seconds_sum`.
  - Server shut down cleanly afterward (`llama_cpp.server` subprocess and
    the uvicorn process both terminated; a follow-up `GET /health` timed
    out/refused as expected).

**Bugs found and fixed during verification (only surfaced by running the
real server, not by the mocked unit tests):**
1. **llama-cpp-python 0.3.38 crashes on this CPU.** The prebuilt CPU wheel
   for `llama-cpp-python` versions newer than 0.3.2 (confirmed through
   0.3.38, the latest at the time) raises
   `OSError: [WinError -1073741795]` (`STATUS_ILLEGAL_INSTRUCTION`) inside
   `llama_backend_init()` on this AMD Ryzen 7 5800H, even though Day 2's
   session worked (it must have resolved an older build). Root-caused by
   trying `llama-cpp-python==0.3.2` from the same wheel index, which
   starts and serves real completions cleanly. Pinned exactly in
   `pyproject.toml` with a comment; re-verify before ever bumping it.
2. **Port collision between the API and the backend's internal server.**
   The API's own `serve --port 8731` collided with `BackendConfig`'s
   default internal port (also 8731) — the llama.cpp subprocess bound
   8731 first, then uvicorn failed with `[Errno 10048] ... only one usage
   of each socket address`. Not a code bug — a test-run mistake caught
   immediately via the real server log; documented the two separate ports
   in the README example (`serve --port 8000`, backend stays on 8731).
3. `sse_starlette`/`starlette_context`/`uvicorn`/`fastapi` were missing
   from this environment at the start of the session (a previous Day 2
   session likely had them in a different environment); reinstalled per
   the `serving` extra.

**Test/CLI results (measured, this session):**
- `pytest tests/unit/test_api.py -v`: 15/15 passed.
- `pytest` (full suite): 44 passed, 1 skipped (opt-in real-backend test),
  **3 pre-existing failures** in `tests/unit/test_hardware_detector.py`
  (`TestDetectGPU`) — unrelated to any Day 3 change (`hardware/detector.py`
  was not touched this session). Root cause: `nvidia-ml-py` (a transitive
  dependency of an unrelated package, `ultralytics`, already present in
  this global Python environment) is now importable, so `detect_gpu()`
  takes the real `pynvml` path and returns the real GPU instead of the
  `nvidia-smi`-fallback path those three tests expect when they only mock
  `shutil.which`/`subprocess.run`. Pre-existing test gap from Day 1
  (mocks never patched the `pynvml` import path), not a regression from
  today — left unfixed since it's out of Day 3's scope.
- `ruff check src tests`: all checks passed.
- `mypy src`: no issues found (25 source files).

**Unresolved / deferred:**
- The 3 pre-existing `test_hardware_detector.py` failures above (Day 1
  test gap, exposed by an unrelated package in this environment).
- `vllm` backend still untested for real (same as Day 2 — no Linux+CUDA
  host).
- No auth/rate-limiting beyond the concurrency admission control; the API
  is explicitly local/dev-only per the README's deployment note.
- `api/app.py` middleware/exception-handler lines aren't separately
  coverage-measured here (not run with `--cov` this session), but every
  branch is exercised by at least one of the 15 API tests plus the real
  end-to-end run above.

## Day 4 — Core inference performance measurement system

**Built:**
- `src/inference_optimizer/benchmark/`: `models.py` (`RequestObservation`,
  `MetricStats`, `BenchmarkSummary`, `BenchmarkResult`), `token_counter.py`
  (`TokenCounter` protocol, `HFTokenizerCounter`, `FixedTokenCounter`,
  `WhitespaceApproxCounter`), `aggregate.py` (`summarize()`, percentile/
  stddev math, aggregation policies), `client.py` (`BenchmarkClient`,
  clock-injectable `_consume_sse_stream`).
- `docs/benchmarking.md` — full writeup of TTFT/TPOT/inter-token-latency/
  throughput/end-to-end-latency definitions and the warmup/failure/missing-
  token/percentile policies (required deliverable, not just a code comment).
- CLI: `python -m inference_optimizer benchmark PROMPT [--base-url]
  [--iterations] [--warmup] [--tokenizer-repo] [--config] [--max-tokens]` —
  runs sequentially against a running API server, prints the aggregate
  summary, writes raw observations + summary to
  `results/benchmark_<timestamp>.json`.
- New `benchmark` extra (`tokenizers>=0.19,<1.0`) in `pyproject.toml` for
  real per-model token counts; optional, `HFTokenizerCounter` degrades to
  `method="unavailable"` without it rather than crashing.
- `tests/unit/test_token_counter.py` (8 tests), `test_benchmark_aggregate.py`
  (9 tests, hand-computed percentile/stddev values), `test_benchmark_client.py`
  (12 tests, fully deterministic via an injected fake clock + `httpx.MockTransport`
  — no real sleeps anywhere).

**Design decisions:**
- Only 3 `time.perf_counter()` samples per request (start, first non-empty
  content, stream end) — never one per SSE chunk, since a chunk is not a
  token and per-chunk timing would imply a false granularity. TPOT is
  derived arithmetically from these 3 samples plus the token count, not
  from per-token timestamps (this system does not measure true inter-token
  latency — documented explicitly in docs/benchmarking.md, including why
  TPOT is a different statistic from it).
- Token counting prefers the real model tokenizer (`HFTokenizerCounter`,
  method `"tokenizer"`) over counting SSE chunks or words, per the literal
  "never assume one SSE event or streaming chunk equals one token"
  instruction. `FixedTokenCounter` exists purely so tests can assert exact
  TPOT/throughput arithmetic without a real tokenizer dependency.
- `_consume_sse_stream(lines, clock)` is a free function taking a plain
  `AsyncIterator[str]` and an injectable clock callable, independent of
  httpx — this is what makes `test_benchmark_client.py` fully deterministic
  (a fake clock supplies exact timestamps; no `asyncio.sleep` anywhere in
  the test suite) while the real `BenchmarkClient.run_one()` wires in
  `response.aiter_lines()` and `time.perf_counter` unchanged.
- Aggregation policies (warmup exclusion, failed-request exclusion,
  per-metric missing-data handling, linear-interpolation percentiles,
  sample stdev) are stated as an explicit module docstring in
  `aggregate.py` and mirrored in docs/benchmarking.md, per the "define
  clear policies" instruction — not left to be inferred from the code.
- Sequential requests only (no concurrency) — Day 4's brief is measurement
  correctness, not load generation; concurrent/sustained-load benchmarking
  is left for a later day, consistent with "no speculative abstraction."

**Real end-to-end verification (not simulated):** started the real API
server (`serve --port 8000`) against the actual Qwen2.5-0.5B-Instruct GGUF,
installed the `tokenizers` package, and ran:
```
python -m inference_optimizer benchmark "Name one primary color." \
  --base-url http://127.0.0.1:8000 --iterations 8 --warmup 2 --max-tokens 20
```
Real measured results (6 measured requests after 2 warmup, 0 failures,
`token_count_method: "tokenizer"` on every observation — confirmed the
actual Qwen2.5-0.5B-Instruct tokenizer was downloaded and used, not an
estimate):
- TTFT (ms): p50 23.93, p95 24.65, p99 24.74, mean 24.00, stddev 0.50
- Latency (ms): p50 133.49, p95 308.57, p99 354.43, mean 172.34, stddev 94.83
- TPOT (ms): p50 21.77, p95 22.27, p99 22.35, mean 21.26, stddev 1.61
- Throughput (tok/s): p50 45.20, p95 52.36, p99 54.20, mean 46.53, stddev 4.02

Raw observations + this summary written to
`results/benchmark_20261124T172233Z.json` (gitignored, kept locally).
Server shut down cleanly afterward (confirmed `GET /health` refused).

**Bug found and fixed during verification:** installing the `tokenizers`
package pulled in `huggingface-hub`, which upgraded `click` to 8.5.1 —
reopening the exact Day 1 bug (typer's `--flag/--no-flag` options crash on
`click>=8.2`). Fixed by re-pinning `click<8.2,>=8.1` after installing
`tokenizers`; confirmed `hardware-info` still works. This is an inherent
tension between `benchmark`'s dependency tree and the `click` pin — anyone
installing the `benchmark` extra should re-check `pip show click` afterward
until a cleaner fix (e.g. vendoring just the tokenizer load, or a stricter
constraints file) is worth the complexity.

**Test/CLI results (measured, this session):**
- `pytest tests/unit/test_token_counter.py tests/unit/test_benchmark_aggregate.py tests/unit/test_benchmark_client.py`:
  25/25 passed.
- `pytest` (full suite): 69 passed, 1 skipped (opt-in real-backend test),
  same **3 pre-existing failures** in `test_hardware_detector.py` as Day 3
  (unrelated environment issue, not touched again this session).
- `ruff check src tests`: all checks passed.
- `mypy src`: no issues found (30 source files).

**Unresolved / deferred:**
- The 3 pre-existing `test_hardware_detector.py` failures (still Day 1
  scope, still out of today's scope).
- No concurrent/load benchmarking yet — sequential only, by design this
  session; throughput under concurrent load is a different measurement
  than the per-request throughput reported here.
- Inter-token latency (the actual per-token-gap distribution) is not
  measured and is explicitly documented as out of reach without backend-
  side per-token timestamps or one-SSE-event-per-token framing.
- The `click`/`tokenizers` dependency tension noted above.

## Day 5 — Reproducible workload generation framework

**Built:**
- `src/inference_optimizer/workload/`: `models.py` (`PromptRecord`,
  `WorkloadManifest`), `dataset.py` (JSONL load/save), `generator.py`
  (`generate_dataset()` — 5 categories x 3 tiers x 2 variants, tokenizer-aware
  padding), `validator.py` (`validate_dataset()` — errors vs. warnings),
  `sampler.py` (`sample_records()` — seeded, reproducible), `manifest.py`
  (`build_manifest()`/`save_manifest()` — sha256 of generator + dataset).
- `docs/workload.md` — categories/tiers, reproduction steps, clamping policy,
  why no copyrighted text is used.
- CLI: `workload-generate [--path] [--tokenizer-repo] [--config]
  [--dataset-version] [--seed]`, `workload-validate [--path]`,
  `workload-sample [--path] --count --seed [--category]... [--input-length]...`.
  `--path` on all three accepts any JSONL — the "user-provided datasets"
  option is just pointing `--path` elsewhere.
- `data/workloads/prompts_v1.jsonl` (30 records) + `prompts_v1.manifest.json`
  — version-controlled (unlike `results/`, which is gitignored run output).
- `tests/unit/test_workload_{generator,validator,sampler,dataset,manifest}.py`
  (30 tests total; generator tests use `WhitespaceApproxCounter` — offline,
  deterministic — not a real tokenizer, to stay network-independent).

**Design decisions:**
- Prompts are `header + padding + footer`; padding only ever adds/removes
  whole filler units (a coarse paragraph, then a fine sentence for precise
  top-up) — the footer (the actual question/instruction/record) is never
  touched, so padding can never corrupt the task itself.
- The 2048-token input tier is clamped during generation to
  `context_limit - output_target_tokens - safety_margin` (1760 for this
  model), not the raw label — a literal 2048-token prompt would leave zero
  room for any output in a 2048-token-context model. `input_length_target`
  (label) and `input_tokens_measured` (real count) are kept as separate
  `PromptRecord` fields specifically so this is never hidden, mirroring Day
  4's target-vs-observed discipline for output tokens.
- Token length is reused from Day 4's `TokenCounter` infrastructure
  (`HFTokenizerCounter`/`WhitespaceApproxCounter`) rather than duplicating
  tokenization logic — one `.count()` interface for both benchmarking and
  workload generation.
- Validation separates **errors** (duplicate ids, empty prompts, zero token
  counts, exceeding the model's context budget) from **warnings** (measured
  length far from target) — a user-provided dataset may legitimately miss
  exact target lengths without being unusable.
- Sampling uses its own seeded `random.Random` instance per call, never the
  shared global `random` module, so sampling is reproducible independent of
  what else in the process has consumed randomness.
- All prompt content (headers, filler paragraphs, footers/questions) is
  original text written for this project — no existing dataset, benchmark
  suite, or licensed corpus is used, per the brief's explicit instruction to
  avoid copyrighted/restricted datasets.

**Real end-to-end verification (not simulated):** ran
`workload-generate` against the real `Qwen/Qwen2.5-0.5B-Instruct` tokenizer:
- 30 records written to `data/workloads/prompts_v1.jsonl`; `workload-validate`
  reports **0 errors, 0 warnings** — the real tokenizer's word/token ratio
  landed every record within tolerance (e.g. `short_qa-128-1` measured 126
  tokens against a target of 128; `short_qa-512-1` measured 507 against 512).
- All ten 2048-tier records measured 1752–1759 input tokens; combined with
  their 256-token output target, every one sums to 2008–2015 — safely under
  the model's 2048-token context limit (confirmed by inspecting the real
  generated file, not assumed).
- `workload-sample --count 3 --seed 7` run twice produced the identical
  sequence of record ids both times (`summarization-2048-1`,
  `short_qa-2048-1`, `code_generation-128-1`); `--category code_generation`
  correctly restricted the sample pool.

**Test/CLI results (measured, this session):**
- `pytest tests/unit/test_workload_*.py`: 30/30 passed (one assertion
  adjusted after discovering paragraph-granular padding legitimately
  undershoots the target with a crude word-count counter — fixed by adding
  a fine-grained filler sentence for top-up, which also improved real-world
  accuracy, not just the test).
- `pytest` (full suite): 99 passed, 1 skipped, same 3 pre-existing
  `test_hardware_detector.py` failures as Days 3–4 (unrelated, untouched).
- `ruff check src tests`: all checks passed.
- `mypy src`: no issues found (37 source files).

**Unresolved / deferred:**
- The 3 pre-existing `test_hardware_detector.py` failures (still Day 1
  scope).
- Dataset reproducibility is pinned to a tokenizer *repo id*, not a
  specific HF revision/commit SHA — a future change to that repo's
  `tokenizer.json` could shift measured lengths on a later regeneration;
  documented as a caveat in docs/workload.md rather than solved today.
- The workload dataset is not yet wired into the Day 4 benchmark CLI (which
  still takes a single repeated prompt) — that integration is left for a
  future day, consistent with "implement only today's scope."

## Day 6 — End-to-end baseline benchmark

**Built:**
- `src/inference_optimizer/baseline/` — new module:
  - `models.py`: `RunMetadata`, `SoftwareVersions`, `MemorySample`,
    `RepeatResult`, `BaselineReport`.
  - `config.py`: `BaselineConfig` (pydantic, YAML-loaded) + `load_baseline_config`.
  - `metadata.py`: `collect_run_metadata` — run id, git commit hash (`git
    rev-parse HEAD`, degrades to `None` + reason), installed package
    versions (`importlib.metadata`, `None` if not installed), full
    `HardwareReport` (Day 1), model/backend config snapshots.
  - `resources.py`: `sample_memory` — process RSS via `psutil`, GPU memory
    used via `pynvml` (device 0), both degrading to `None`/`"unavailable"`
    on any failure.
  - `runner.py`: `run_baseline` — verifies backend readiness
    (`start()`+`health_check()`), discards `warmup_count` requests, samples
    the Day 5 workload **once** (fixed across repeats), runs `repeats`
    sequential passes (concurrency=1) driving the `InferenceBackend`
    directly (no HTTP layer), reuses Day 4's `summarize()` unmodified per
    repeat, and writes JSONL (raw observations) + JSON (full report) + CSV
    (per-repeat summary) to `output_dir`. Accepts optional `token_counter`/
    `backend_factory` injection points for testing.
- `configs/baseline.yaml` — default run config.
- CLI: `python -m inference_optimizer baseline --config configs/baseline.yaml`
  (new command; kept separate from Day 4's `benchmark`, which has an
  incompatible single-prompt/running-server contract — see docs/baseline.md
  naming note).
- `docs/baseline.md` — full writeup: readiness/warmup/repeat policy, metrics,
  provenance fields, the GPU-memory-is-whole-device-not-process caveat
  (confirmed by the real run below, since this backend is CPU-only), and the
  unavailable-backend path.
- Tests: `tests/unit/test_baseline_config.py` (4),
  `tests/unit/test_baseline_resources.py` (4),
  `tests/unit/test_baseline_metadata.py` (5),
  `tests/unit/test_baseline_runner.py` (5) — all against a `FakeBackend`
  double and `FixedTokenCounter`, no real subprocess/model/network call.

**Design decisions:**
- Drives `InferenceBackend` directly rather than through the Day 3 HTTP API:
  a baseline run needs nothing already running, only a model file (and, for
  llama.cpp, a CPU) — simpler failure mode for "verify readiness first."
- "Fixed workload" means `sample_records()` is called once before the repeat
  loop and the exact same prompts/order are reused every repeat — only the
  backend's own run-to-run variance differs, not the input.
- Backend-unavailable short-circuits before any output file is written:
  `BaselineReport.status="unavailable"` with the real exception as
  `unavailable_reason`, so a failed run can never be mistaken for a zero- or
  near-zero-latency successful one.
- `token_counter`/`backend_factory` are optional constructor-style injection
  points on `run_baseline` (mirrors `BenchmarkClient`'s `http_client` param)
  so tests exercise the full orchestration without a real tokenizer download
  or subprocess.

**Real end-to-end verification (measured, this session):**
- Ran `python -m inference_optimizer baseline --config configs/baseline.yaml`
  against the real llama.cpp backend (Qwen2.5-0.5B-Instruct GGUF) with the
  real HF tokenizer: 6 sampled prompts x 3 repeats = 18 requests, all
  succeeded (`failure_rate: 0.0` every repeat).
- Measured p50 latency: repeat 0 = 3829.3 ms, repeat 1 = 5163.1 ms, repeat 2
  = 5980.5 ms (p95: 5239.3 / 8665.8 / 15170.8 ms). TTFT p50: 805.6 / 1019.4 /
  1338.9 ms. TPOT p50: 32.8 / 38.6 / 48.5 ms. Throughput p50: 19.0 / 11.0 /
  10.8 tok/s. req/s: 0.268 / 0.206 / 0.138.
- Latency grew across repeats; measured available RAM at run time was only
  2.41 GB free of 16.54 GB total — read as CPU/memory contention on this
  shared dev machine, not a runner bug; documented in docs/baseline.md as a
  noisy single-machine baseline, not a clean reference number.
- GPU memory sampled via `pynvml` (382.93 / 298.41 / 243.25 MB across the
  run) — confirmed this is whole-device usage, not attributable to this
  (CPU-only) backend; documented as a caveat rather than presented as the
  backend's own consumption.
- Git commit hash captured correctly: `c73c2ea7579fba717c964166c78620571e749440`.
- Output files confirmed on disk: `results/baseline/baseline_20261125T235104Z_
  {observations.jsonl (18 lines), report.json, summary.csv (3 rows)}` — not
  committed (gitignored, like all of `results/`).

**Test/CLI results (measured, this session):**
- `pytest tests/unit/test_baseline_*.py`: 18/18 passed.
- `pytest` (full suite): 117 passed, 1 skipped, same 3 pre-existing
  `test_hardware_detector.py` failures as every prior day (unrelated,
  untouched).
- `ruff check src tests`: all checks passed.
- `mypy src`: no issues found (43 source files).

**Unresolved / deferred:**
- The 3 pre-existing `test_hardware_detector.py` failures (still Day 1
  scope).
- GPU memory sampling is whole-device, not process-scoped — acceptable for
  now since this backend is CPU-only, but would need a different approach
  (e.g. diffing against a GPU-idle baseline, or a CUDA-capable backend that
  reports its own allocator stats) to attribute memory to the backend
  specifically on a GPU-serving setup.
- No config-recommendation or optimization logic yet — this is explicitly a
  measurement-only baseline, consistent with "establish measurement
  correctness first" from Day 4 and "implement only today's scope."

## Day 7 — Week 1 reporting, validation, and risk review

**Built:**
- `src/inference_optimizer/report/`: a report pipeline over Day 6's baseline
  artifacts — `parser.py` (reads the real JSON report + JSONL raw
  observations), `validate.py` (recomputes every aggregate from raw data via
  Day 4's `summarize()`, unmodified, and diffs against the stored values),
  `variability.py` (repeat-to-repeat mean/stddev/coefficient-of-variation per
  metric, `None` not `0` with < 2 repeats), `flags.py` (`run_unavailable`,
  `incomplete_repeats`, `requests_failed`, `aggregate_mismatch`,
  `low_available_memory`, `high_repeat_variability`,
  `gpu_memory_not_process_attributed`), `comparison.py` (internal-consistency
  checks only — explicitly no external SLA exists yet to compare against),
  `plots.py` (matplotlib Agg histograms for TTFT/latency/throughput, real
  data only outside tests), `content.py` + `markdown_renderer.py` +
  `html_renderer.py` (shared table data, two output formats), and
  `generator.py` (orchestrates all of the above, writes `report.md` +
  `report.html` + plot PNGs).
- New `report` CLI command: `python -m inference_optimizer report
  --baseline-report <path> --observations <path> --output-dir <dir>`.
- New `report` optional dependency group (`matplotlib>=3.8,<4.0`) in
  pyproject.toml.
- `docs/reporting.md`: metric definitions (pointers to Day 4/6 docs), flag
  definitions, the "no external baseline yet" comparison caveat, and
  measurement limitations.
- `docs/week1_report.md`: the Week 1 technical report — architecture
  overview, the real baseline run's findings (transcribed from the generated
  `reports/week1/report.md`), and a review of what's working/untested/risks
  for Week 2.
- `reports/week1/{report.md, report.html, ttft_distribution.png,
  latency_distribution.png, throughput_distribution.png}`: generated (not
  hand-written) from the real Day 6 artifacts
  (`results/baseline/baseline_20261125T235104Z_*`) via `python -m
  inference_optimizer report`.

**Design decisions:**
- Validation reuses Day 4's `summarize()` unmodified rather than
  reimplementing aggregation — a report pipeline that recomputed metrics
  differently than the runner that produced them would defeat the point of
  validating.
- Flags only annotate; they never mutate the stored numbers. A run with
  `requests_failed` or `aggregate_mismatch` still renders its full report —
  the flag is the signal to distrust it, not a reason to hide it.
- "Comparison against expectations" is implemented as internal-consistency
  checks (TTFT p50 <= latency p50) plus an explicit `no_external_baseline`
  note, not a fabricated target number — there is no documented SLA for this
  model+hardware combination yet, and inventing one would violate the
  project's no-fabrication rule.
- Markdown and HTML renderers share table-building logic (`content.py`) so
  the two formats can't drift apart in content, only in layout.
- Plots skip a metric entirely (rather than drawing a 1-point "distribution")
  when fewer than 2 real successful observations have that field.

**Real end-to-end verification (measured, this session):**
- Ran `python -m inference_optimizer report` against the real Day 6
  artifacts. Output: 3 flags surfaced on the real run —
  `low_available_memory` (2.41 GB free), `high_repeat_variability`
  (throughput p50 CV = 0.34, over the 0.3 threshold), and
  `gpu_memory_not_process_attributed` (informational). All 3 repeats'
  validation passed (`OK`) — the stored aggregates are exactly reproducible
  from the raw JSONL, confirming no drift between what the baseline runner
  computed and what's on disk.
- Confirmed on disk: `reports/week1/report.md` (58 lines, correct hardware/
  model/metrics tables, correct git commit hash) + `report.html` + 3 real
  PNG plots, all non-empty.

**Test/CLI results (measured, this session):**
- New tests: `test_report_parser.py` (4), `test_report_analysis.py` (10),
  `test_report_plots.py` (3, fixture data explicitly labeled `_FIXTURE_*` in
  the module docstring and constant names), `test_report_render.py` (4),
  `tests/integration/test_week1_pipeline.py` (3 — workload generation
  through report creation, offline via FakeBackend/WhitespaceApproxCounter).
  24/24 new tests passed.
- `pytest` (full suite): 141 passed, 1 skipped, same 3 pre-existing
  `test_hardware_detector.py` failures as every prior day (unrelated,
  untouched).
- `ruff check src tests`: all checks passed.
- `mypy src`: no issues found (55 source files).

**Unresolved / deferred (see docs/week1_report.md's full review):**
- The 3 pre-existing `test_hardware_detector.py` failures (still Day 1
  scope).
- No controlled/isolated-machine run exists yet to separate "this model's
  real latency" from "this shared dev machine's contention at the time" —
  the `low_available_memory` and `high_repeat_variability` flags on the real
  run document the open question rather than resolving it.
- vLLM remains entirely unexercised end-to-end (no Linux+CUDA host); no
  concurrency > 1 has been measured anywhere yet; no quantization/batching
  comparison exists — all explicitly Week 2+ scope.
