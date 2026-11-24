# Workload dataset: reproduction and design

## What's checked in

- `data/workloads/prompts_v1.jsonl` -- 30 records (5 categories x 3 input-length
  tiers x 2 variants), generated this session against the real
  `Qwen/Qwen2.5-0.5B-Instruct` tokenizer.
- `data/workloads/prompts_v1.manifest.json` -- its provenance manifest.

Both are version-controlled (unlike `results/`, which holds gitignored run
output) -- the dataset is source content, not an experiment artifact.

## Reproducing it exactly

```bash
pip install -e ".[benchmark]"   # tokenizers, for real token counts

python -m inference_optimizer workload-generate \
  --path data/workloads/prompts_v1.jsonl \
  --dataset-version v1 \
  --seed 0
```

This is deterministic: the prompt templates, filler text, and padding algorithm
in `inference_optimizer/workload/generator.py` have no randomness, so the same
tokenizer version will always produce the same JSONL content byte-for-byte. The
manifest's `source_file_hashes.generator.py` lets you confirm your local
`generator.py` matches what produced a given dataset file, and
`source_file_hashes.prompts_v1.jsonl` lets you confirm the dataset file itself
hasn't changed since.

**Caveat on exact reproducibility:** `HFTokenizerCounter` loads the tokenizer by
repo id (`Qwen/Qwen2.5-0.5B-Instruct`), not a pinned commit SHA, so a future
change to that repo's `tokenizer.json` could change the measured token counts
on a re-run. The manifest's `tokenizer_repo_id` records what was used; pin to a
specific HF revision instead if byte-for-byte reproducibility across time (not
just across machines today) becomes a requirement.

## Validating a dataset (built-in or your own)

```bash
python -m inference_optimizer workload-validate --path data/workloads/prompts_v1.jsonl

# Point it at your own dataset instead:
python -m inference_optimizer workload-validate --path path/to/your-dataset.jsonl
```

Checks (see `inference_optimizer/workload/validator.py` for the full policy):
non-empty, unique ids, non-empty prompts, every `input_tokens_measured` > 0 and
within the model's context budget (`context_limit - output_target_tokens -
safety_margin`, an **error** if violated), and a **warning** (not an error) if
a record's measured length is far from its target -- a warning rather than an
error because a legitimate user-provided dataset may not hit the target
lengths exactly.

## Sampling reproducibly

```bash
python -m inference_optimizer workload-sample \
  --path data/workloads/prompts_v1.jsonl \
  --count 10 --seed 42
```

The same `(path, count, seed, category filters, input-length filters)` always
produces the same sequence of records -- `inference_optimizer/workload/sampler.py`
uses a seeded `random.Random` instance, never the shared global `random` module.
Add `--category short_qa --category code_generation` and/or
`--input-length 128 --input-length 512` to restrict the pool before sampling.
If `--count` exceeds the number of matching records, sampling falls back to
with-replacement (logged as a warning) so you still get exactly `count` records.

## Categories and length tiers

| Category | What it tests |
|---|---|
| `short_qa` | Direct factual/arithmetic questions |
| `summarization` | Condensing a (synthetic) status report |
| `code_generation` | Implementing a small function from a spec |
| `structured_extraction` | Pulling fields from a record into JSON |
| `long_context_qa` | Answering a question that requires reading padded context |

| Input-length tier (label) | Output target | Effective input budget used during generation |
|---|---|---|
| 128 | 64 | 128 (well within context) |
| 512 | 128 | 512 (well within context) |
| 2048 | 256 | `min(2048, context_limit - 256 - safety_margin)` = 1760 for this model's 2048-token context |

The 2048 tier's **label** stays `2048` (it's the categorical bucket from the
brief), but its **measured** length is intentionally lower -- a 2048-token
prompt would leave zero budget for any output in a 2048-token-context model.
`PromptRecord.input_length_target` (the label) and
`PromptRecord.input_tokens_measured` (the real tokenizer count) are kept as
separate fields specifically so this distinction is never lost, mirroring Day
4's target-vs-observed discipline for output tokens.

## How prompts are built (and why no copyrighted text is used)

Every prompt is `header + padding + footer`, where `header` and `footer` are
original text written for this project, and `padding` is one of two original
filler passages (a longer paragraph for coarse padding, a short sentence for
fine-grained top-up) repeated until the measured token count is as close as
possible to the target *without exceeding it*. The footer -- the actual
question/instruction/record to extract -- is never touched by padding, so it's
never truncated or corrupted. No prompt in this dataset is sourced from an
existing public dataset, benchmark suite, or any text requiring a license or
credentials to use.
