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
