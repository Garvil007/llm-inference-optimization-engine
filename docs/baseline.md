# Day 6: the baseline benchmark runner

`python -m inference_optimizer baseline --config configs/baseline.yaml` is the
first end-to-end baseline run: it drives a real `InferenceBackend` directly
(no separately-running API server needed, unlike Day 4's `benchmark`
command), verifies it's actually healthy, warms it up, then runs a fixed,
seeded sample of the Day 5 workload dataset sequentially (concurrency = 1),
repeated several times, recording everything needed to explain the numbers
later.

Naming note: the brief's example was `inference_optimizer benchmark --config
configs/baseline.yaml`. This project already has a `benchmark` command (Day
4) with an incompatible contract (a required positional `prompt`, hitting a
*running* API server over HTTP). Reusing that name for a config-driven,
backend-direct runner would have silently broken Day 4's documented usage, so
this is a separate `baseline` command instead -- same spirit, no breaking
rename.

## Running it

```bash
python -m inference_optimizer baseline --config configs/baseline.yaml
```

Requires the `serving` extra and the model file (see README Setup). Writes
three files per invocation to the config's `output_dir` (default
`results/baseline/`, gitignored like the rest of `results/`):

- `baseline_<timestamp>_observations.jsonl` -- every `RequestObservation`
  (one line per request, across all repeats, each tagged with `repeat_index`).
- `baseline_<timestamp>_report.json` -- the full `BaselineReport`: run
  metadata, config, sampled record ids, and every repeat's `BenchmarkSummary`
  (p50/p95/p99/mean TTFT, latency, TPOT, throughput; measured/failed counts;
  failure rate) plus wall time, requests/sec, and memory samples.
- `baseline_<timestamp>_summary.csv` -- one row per repeat, flattening the
  same aggregate numbers for spreadsheet/pandas use.

## What "fixed workload" means here

`sample_records()` (Day 5) is called **once**, before the repeat loop, with
the config's `sample_seed` -- the same prompts, in the same order, are reused
for every repeat. Nothing is re-sampled between repeats; only the model's own
run-to-run variance differs.

## Readiness, warmup, repeats

1. `backend.start()` then `backend.health_check()`. If start raises
   (`ModelLoadError`, `BackendStartupError`, `UnsupportedHardwareError`, ...)
   or health reports unhealthy, the run stops immediately: `BaselineReport.status
   = "unavailable"`, `unavailable_reason` holds the real exception text, and
   **no JSONL/JSON/CSV file is written** -- nothing fabricated, nothing
   partially written that could be mistaken for a real run.
2. `warmup_count` requests (default 2) are sent and discarded (not even
   passed into `summarize()`), since they exist only to let the backend
   finish any first-request setup cost before measurement starts.
3. `repeats` (default 3, config `repeats` key -- "at least three... when
   resources permit": on constrained hardware, lower this rather than lying
   about having run three) full passes over the fixed workload, sequential
   (`await` one request at a time -- concurrency=1 per the brief).

## Metrics reported

Per repeat: `measured_requests`, `successful_requests`, `failed_requests`,
`failure_rate`, and `MetricStats` (count/p50/p95/p99/mean/stddev) for TTFT,
end-to-end latency, TPOT, and output-tokens/sec -- identical definitions and
aggregation policies to Day 4 (see [docs/benchmarking.md](benchmarking.md));
this runner calls the same `summarize()` function, unmodified. Additionally:
`wall_time_seconds` (the repeat's total sequential wall-clock time) and
`requests_per_second = measured_requests / wall_time_seconds`.

## Provenance captured

`RunMetadata` (`src/inference_optimizer/baseline/metadata.py`): a fresh
`uuid4` `run_id`, ISO-8601 `started_at`, `git_commit_hash` (via `git
rev-parse HEAD`, `None` with a reason string if git/`.git` isn't available --
never guessed), the full `HardwareReport` (Day 1), installed versions of
`pydantic`/`httpx`/`typer`/`fastapi`/`uvicorn`/`llama-cpp-python`/
`tokenizers`/`psutil` (`None` for anything not installed, via
`importlib.metadata`), and snapshots of the exact `ModelConfig` and
`BackendConfig` used.

## Memory sampling and its real caveat

`sample_memory()` (`src/inference_optimizer/baseline/resources.py`) takes a
process RSS reading (`psutil`) and a GPU-memory-used reading (`pynvml`,
device 0) before and after each repeat. **Caveat confirmed by the real run
below:** `pynvml`'s `nvmlDeviceGetMemoryInfo` reports the *whole GPU
device's* memory usage, not this process's attribution. On this machine the
`llama-cpp-python` backend runs CPU-only (see README/CLAUDE.md on the
Windows wheel pin), so the GPU memory numbers in a real run reflect whatever
else happens to be using the GPU at that moment -- not this benchmark's own
usage. Read it as "GPU memory in use on this machine during the run," not
"GPU memory this benchmark consumed." Both readings degrade to
`(None, "unavailable")`, never a guess, when `pynvml`/the GPU/the driver
aren't available.

## Real run on this machine

Ran `python -m inference_optimizer baseline --config configs/baseline.yaml`
(6 sampled prompts, 2 warmup, 3 repeats) against the real llama.cpp backend
with the real Qwen2.5-0.5B-Instruct GGUF and the real HF tokenizer for token
counts. All 18 requests (6 x 3 repeats) succeeded (`failure_rate: 0.0` every
repeat). Measured (not estimated) results, written to
`results/baseline/baseline_20261125T235104Z_{observations.jsonl,report.json,summary.csv}`
(gitignored, not committed -- these are this machine's run output, not
checked-in reference numbers):

| Repeat | latency p50 (ms) | latency p95 (ms) | TTFT p50 (ms) | TPOT p50 (ms) | throughput p50 (tok/s) | req/s |
|---|---|---|---|---|---|---|
| 0 | 3829.3 | 5239.3 | 805.6 | 32.8 | 19.0 | 0.268 |
| 1 | 5163.1 | 8665.8 | 1019.4 | 38.6 | 11.0 | 0.206 |
| 2 | 5980.5 | 15170.8 | 1338.9 | 48.5 | 10.8 | 0.138 |

Git commit at run time: `c73c2ea7579fba717c964166c78620571e749440`.

Latency grew noticeably across repeats on this run (p50 3.8s -> 5.2s -> 6.0s,
p95 even more so). This machine's available RAM was measured at only 2.41 GB
free at run time (16.5 GB total) -- consistent with memory/CPU contention
from other running processes on this shared dev machine, not a bug in the
runner. This is reported as observed, not smoothed over or discarded; later
days' optimization work should treat this run as a noisy single-machine
baseline, not a clean reference number.

## If the backend is unavailable

If the model file is missing or the backend can't start, the command exits
non-zero, prints the real exception, and (per the above) writes no result
files. To see this path without the real model, delete/rename the GGUF file
and rerun -- `ModelLoadError` surfaces as `status: "unavailable"` with the
exact missing-file message, not a fabricated result.

## Reproducing

```bash
pip install -e ".[serving,benchmark]"
python -m inference_optimizer baseline --config configs/baseline.yaml
```

Edit `configs/baseline.yaml` (`sample_count`, `sample_seed`, `warmup_count`,
`repeats`, `categories`, `input_length_targets`, `tokenizer_repo`,
`max_tokens_override`, `output_dir`) to change the run; every field maps
1:1 to `BaselineConfig` in `src/inference_optimizer/baseline/config.py`.
