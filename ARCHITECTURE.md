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

## Not yet built

Benchmarking, profiling, quantization experiments, evaluation harness,
dashboard, Docker/CI — scoped to later days per the 28-day plan.
