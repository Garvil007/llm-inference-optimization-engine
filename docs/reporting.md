# Day 7: the report pipeline

`python -m inference_optimizer report --baseline-report <path> --observations
<path> --output-dir reports/<name>` parses a Day 6 baseline run's real
artifacts, validates its stored aggregates against the raw per-request data,
computes repeat-to-repeat variability, flags runs that shouldn't be read as a
clean/comparable baseline, builds internal-consistency comparisons, and
writes a Markdown + HTML report with distribution plots to the output
directory.

## Running it

```bash
pip install -e ".[report]"   # matplotlib, for the distribution plots
python -m inference_optimizer report \
  --baseline-report results/baseline/baseline_<timestamp>_report.json \
  --observations results/baseline/baseline_<timestamp>_observations.jsonl \
  --output-dir reports/baseline
```

`--observations` can be omitted only when the baseline report's `status` is
`"unavailable"` (there's nothing to validate or plot). Writes `report.md`,
`report.html`, and up to three `*_distribution.png` files (one per metric
with >= 2 real data points) to the output directory.

## What each piece does

- **Parser** (`report/parser.py`): reads the JSON `BaselineReport` and the
  JSONL raw observations back into typed objects, restoring the
  `repeat_index` tag the baseline runner stamped onto each JSONL line. Raises
  `ConfigurationError` (never guesses) if a file is missing.
- **Validation** (`report/validate.py`): recomputes `BenchmarkSummary` from
  the raw observations using Day 4's `summarize()`, unmodified, and compares
  every stored aggregate field against the recomputed one (tolerance
  `1e-6`). A mismatch means the stored aggregate doesn't match its own raw
  data -- surfaced as an `aggregate_mismatch` flag, never silently trusted.
- **Variability** (`report/variability.py`): the repeat-to-repeat spread
  (mean, stddev, coefficient of variation) of each metric's p50 across
  repeats. This is **not** the same as the within-repeat `stddev` already in
  each repeat's `MetricStats` -- that measures spread *within* one repeat's
  requests; this measures spread *between* repeats. Needs >= 2 repeats; with
  exactly 1, every field is `None` (not fabricated as `0`).
- **Flags** (`report/flags.py`): automatic quality flags --
  `run_unavailable` (backend never started), `incomplete_repeats` (< 3
  repeats), `requests_failed` (any non-zero failure rate), `aggregate_mismatch`
  (validation failed), `low_available_memory` (< 3 GB free RAM at run time --
  a proxy for machine contention), `high_repeat_variability` (coefficient of
  variation > 0.3 for any metric -- repeats aren't directly comparable), and
  `gpu_memory_not_process_attributed` (informational, whenever a GPU was
  detected -- see docs/baseline.md's pynvml whole-device caveat).
- **Comparisons** (`report/comparison.py`): **internal consistency checks
  only.** There is no documented SLA or vendor-published expected-performance
  number for this model+hardware combination -- Day 6/7 produced the first
  baseline, so there is nothing external yet to compare against. The one
  check implemented (TTFT p50 <= latency p50) verifies the measured numbers
  are mutually consistent, not that they hit some target. A
  `no_external_baseline` note makes this limitation explicit in every
  generated report rather than letting a reader assume otherwise.
- **Plots** (`report/plots.py`): TTFT, end-to-end latency, and output
  tokens/sec distribution histograms via matplotlib's non-interactive `Agg`
  backend (headless-safe). A metric is skipped (not plotted with 1 point
  pretending to be a distribution) when fewer than 2 real successful
  observations have that field. **Only ever called with real measured
  observations from a parsed baseline run in production code paths.** The
  unit test `tests/unit/test_report_plots.py` is the one place synthetic
  numbers are used, and they are explicitly named `_FIXTURE_*` and documented
  as fixture data in that file's module docstring -- never claimed as a real
  measurement anywhere else.
- **Renderers** (`report/markdown_renderer.py`, `report/html_renderer.py`):
  share table-building logic (`report/content.py`) so the Markdown and HTML
  outputs can't drift apart in content, only in format.

## Metric definitions

Identical to Day 4/6 -- see [docs/benchmarking.md](benchmarking.md) and
[docs/baseline.md](baseline.md) for TTFT, end-to-end latency, TPOT, and
throughput definitions, warmup exclusion, and the percentile/stddev
aggregation policy. This report never recomputes a metric differently than
`benchmark/aggregate.py` already does.

## Measurement limitations (read before trusting a number)

1. **A single machine, single run.** All numbers come from one dev laptop at
   one point in time. They say nothing about a different machine, a
   different model, or a different hour on this same machine.
2. **No external performance target exists yet.** The "Comparisons" section
   checks internal consistency, not whether the numbers are "good" --
   there's no documented baseline to be good or bad relative to.
3. **GPU memory samples are whole-device, not process-attributed** (see
   docs/baseline.md) -- on a CPU-only backend they reflect unrelated system
   GPU activity, not this run's own usage.
4. **High variance across repeats is itself a measured, reported finding**,
   not noise to be averaged away. The `high_repeat_variability` flag and the
   variability table exist specifically so a reader can see this rather than
   trusting a single percentile as "the" latency.
5. **Low available RAM at run time** (flagged via `low_available_memory`)
   means later-day optimizations on this same machine should expect
   measurement noise from whatever else is running, not treat this report's
   numbers as a clean, controlled baseline.
