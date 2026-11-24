# Benchmarking: metric definitions and policies

This system measures, it does not optimize (that's a later day). Everything below
describes exactly what is measured and how, so numbers from `python -m
inference_optimizer benchmark` can be trusted and reproduced.

## What is actually measured

Per request, the client (`inference_optimizer.benchmark.client.BenchmarkClient`)
takes exactly three `time.perf_counter()` samples while consuming the
`/generate/stream` SSE response:

1. **start** -- immediately before the request is sent.
2. **first token** -- the instant the first non-empty `"text"` SSE chunk arrives.
3. **end** -- when the stream closes (`[DONE]` or an error event).

No other timestamps are taken. In particular, **individual SSE chunks are not
timed**, because a chunk is not a token -- llama.cpp/vLLM's streaming deltas are
arbitrary text fragments that can contain zero, one, or several tokens. Timing every
chunk and calling the gaps "per-token latency" would be measuring something that
doesn't correspond to actual token boundaries.

## Metric definitions

| Metric | Formula | Meaning |
|---|---|---|
| **TTFT** (time to first token) | `(first_token - start) * 1000` ms | How long until the model starts responding at all. Dominated by prompt processing / prefill. |
| **End-to-end latency** | `(end - start) * 1000` ms | Total wall-clock time for the whole request, including TTFT. What a user actually waits. |
| **TPOT** (time per output token) | `(latency_ms - ttft_ms) / (token_count - 1)` | The *average* per-token cost of the generation (decode) phase, excluding the TTFT/prefill cost. Requires >= 2 output tokens -- with exactly 1 token there is no "after the first token" interval to average over, so TPOT is undefined (`None`), not zero. |
| **Throughput** | `token_count / (latency_ms / 1000)` tokens/sec | Output tokens per second over the *whole* request (TTFT included). This is the common vendor-benchmark definition; it is lower than a generation-phase-only throughput would be, because it amortizes the prefill cost into the rate. |
| **Inter-token latency** | *(not measured by this system)* | The distribution of gaps between individual consecutive output tokens. This would require timing every token boundary, which SSE chunk boundaries do not reliably provide (see above) -- measuring it correctly needs either a backend that emits one event per token or token-level timestamps from the backend itself. TPOT is an *average* stand-in; it is not the same statistic and does not have its own distribution (no p95 inter-token gap, for example). |

**Why TPOT and throughput are two different numbers, not duplicates:** TPOT excludes
TTFT (it's "cost of each token after the first"), while throughput divides by the
*entire* request duration including TTFT. A request with a long TTFT but fast
per-token decoding will show a low TPOT but a throughput number dragged down by the
prefill time. Both are reported because they answer different questions: TPOT is
"how fast does it decode once it starts," throughput is "what rate did I actually
get for this whole request."

## Token counting

Never inferred from the number of SSE events. In priority order:

1. **Real model tokenizer** (`HFTokenizerCounter`, method `"tokenizer"`) -- loads
   the actual model's tokenizer (Hugging Face `tokenizers` library, from the repo id
   in the model config) and counts the actual token IDs of the concatenated output
   text. This is the authoritative count when available.
2. **Predetermined** (`FixedTokenCounter`, method `"fixed"`) -- test-only; returns
   counts supplied in advance, ignoring the text.
3. **Approximate** (`WhitespaceApproxCounter`, method `"approximate"`) -- a
   whitespace word count, clearly *not* a token count (most tokenizers produce more
   tokens than words). Only used as an explicit fallback.
4. **Unavailable** (method `"unavailable"`) -- the tokenizer failed to load (no
   network, missing `tokenizers` install, bad repo id) or counting wasn't attempted
   (e.g. the request failed). `output_token_count` is `None`; `tpot_ms` and
   `output_tokens_per_second` are also `None` for that request (see policy below).

Every `RequestObservation.token_count_method` records which of these actually
happened for that request -- a benchmark result never silently presents an estimate
as a measurement.

## Aggregation policies

Implemented in `inference_optimizer.benchmark.aggregate`:

1. **Warmup exclusion.** The first `warmup_count` requests of a run are tagged
   `is_warmup=True` and excluded from *every* aggregate (they measure cold-start
   effects: model/connection setup, OS/file-cache warmup -- not steady-state
   performance). They remain in the raw observation list for inspection.
2. **Failed requests.** Excluded from every per-metric aggregate, but counted in
   `failure_rate = failed_requests / measured_requests` (computed over non-warmup
   requests). A failed request's partial data (e.g. a TTFT recorded before a
   mid-stream error) is never used in an aggregate -- only fully successful
   requests contribute measurements, even if some per-field data exists.
3. **Missing token counts.** Handled per-metric, not per-request: if
   `output_token_count`, `tpot_ms`, or `output_tokens_per_second` is `None` for a
   given observation, that specific aggregate's sample list skips it -- the same
   observation's `ttft_ms`/`latency_ms` still contribute to those aggregates if
   present. No request is dropped wholesale because one field is missing.
4. **Percentiles.** Linear interpolation between the two closest ranks over the
   sorted sample -- the same method as `numpy.percentile(..., method="linear")`
   (the default). A sample of size 1 reports that single value for every
   percentile. An empty sample reports `count=0` and every statistic as `None`
   (not `0` -- a `0` would imply a measured zero-duration request that never
   happened).
5. **Standard deviation.** Sample stdev (`ddof=1`), which needs >= 2 data points.
   With exactly 1 data point, stddev is `0.0` (no evidence of variance from a
   single observation -- a meaningfully different state from "no data," hence
   `None` only when `count == 0`).

## Running it

See the README's "Benchmark" section for the CLI command and example output. Raw
observations and the aggregate summary are both written to `results/` as JSON
(`BenchmarkResult`), so a run can be re-aggregated or audited later without
re-executing requests against the model.
