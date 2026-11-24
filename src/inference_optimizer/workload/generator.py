"""Builds the default prompt dataset: 5 categories x 3 input-length tiers x 2
variants = 30 records. All prompt text below is original (written for this
project), not sourced from any existing dataset -- avoiding the "copyrighted or
restricted datasets" the brief warns against.

Length control: each record is `header + padding-of-filler-paragraphs + footer`.
The footer (the actual question/instruction) is never truncated or touched by
padding -- only whole filler paragraphs are added/removed, so padding can never
corrupt the task itself. Length is checked with the real `TokenCounter` from Day 4
(`inference_optimizer.benchmark.token_counter`), reused rather than duplicated.

The 2048-token input tier is clamped to `context_limit - output_target_tokens -
safety_margin` (not the raw label) so a generated record actually fits in this
model's context window with room left for its output -- see models.py for why
`input_length_target` (the label) and `input_tokens_measured` (the real count)
are kept as two separate fields.
"""

from __future__ import annotations

from pathlib import Path

from inference_optimizer.benchmark.token_counter import TokenCounter
from inference_optimizer.workload.models import (
    Category,
    InputLengthTarget,
    OutputTargetTokens,
    PromptRecord,
)

DEFAULT_DATASET_VERSION = "v1"
DEFAULT_CONTEXT_LIMIT = 2048
DEFAULT_SAFETY_MARGIN = 32
DEFAULT_TIERS: list[tuple[InputLengthTarget, OutputTargetTokens]] = [
    (128, 64),
    (512, 128),
    (2048, 256),
]

GENERATOR_SOURCE_PATH = Path(__file__)

_CATEGORY_SPECS: dict[Category, dict[str, object]] = {
    "short_qa": {
        "header": "Answer the following question concisely and directly.",
        "filler_unit": (
            "Background context: the fictional Meridian Library Network maintains a "
            "shared catalog of books, periodicals, and digital media across its member "
            "branches. Items are tagged with a unique accession number, a branch code, "
            "and a status flag (available, checked out, or in transit). The catalog "
            "system synchronizes branch inventories once per hour and flags discrepancies "
            "for manual review by branch staff."
        ),
        "filler_fine": "Routine catalog audits occur weekly.",
        "footers": [
            "Question: What is the capital of France?\nAnswer:",
            "Question: What is 15 multiplied by 6?\nAnswer:",
        ],
    },
    "summarization": {
        "header": "Summarize the following status report in two or three sentences.",
        "filler_unit": (
            "Quarterly engineering status (fictional): the platform team shipped 14 minor "
            "releases and 2 major releases this quarter. Median API latency held steady at "
            "around 80 milliseconds, with a brief regression during a database migration "
            "that was resolved within six hours. Three incidents were logged, all rated low "
            "severity, and all were traced to misconfigured retry logic in a downstream "
            "service rather than the core platform. Test coverage increased from 71% to 76% "
            "across the affected modules."
        ),
        "filler_fine": "No further incidents were reported this period.",
        "footers": [
            "Summary:",
            "Provide a concise summary of the report above:",
        ],
    },
    "code_generation": {
        "header": (
            "You are given background context for a coding task. After reading it, "
            "implement the requested function."
        ),
        "filler_unit": (
            "Background (fictional): the codebase maintains an in-memory inventory tracker "
            "used by internal tooling. Functions in this module are expected to be pure "
            "(no hidden state, no I/O) and to include type hints, since the project's CI "
            "pipeline runs mypy in strict mode on every pull request. Existing utilities in "
            "this module favor small, composable functions over large multi-purpose ones."
        ),
        "filler_fine": "No additional constraints apply beyond standard style guidelines.",
        "footers": [
            "Task: Write a Python function `is_palindrome(s: str) -> bool` that returns "
            "True if `s` is a palindrome, ignoring case and non-alphanumeric characters.\n\n"
            "```python\n",
            "Task: Write a Python function `flatten(nested: list) -> list` that flattens an "
            "arbitrarily nested list of integers into a single flat list.\n\n```python\n",
        ],
    },
    "structured_extraction": {
        "header": (
            "Extract the following fields from the record below and return them as JSON: "
            "customer, date, amount, status."
        ),
        "filler_unit": (
            "Additional context (fictional, for padding only -- not part of the record to "
            "extract): the order-processing system logs ancillary metadata alongside each "
            "transaction, including the originating sales channel, the warehouse that "
            "fulfilled the order, and a free-text notes field used by support staff. This "
            "metadata is not required for the extraction task below."
        ),
        "filler_fine": "No further notes were attached to this record.",
        "footers": [
            "Record:\nCustomer: Jordan Vance\nDate: 2024-03-14\nAmount: $482.10\n"
            "Status: Shipped\n\nJSON:",
            "Record:\nCustomer: Priya Natarajan\nDate: 2024-07-02\nAmount: $129.99\n"
            "Status: Pending\n\nJSON:",
        ],
    },
    "long_context_qa": {
        "header": "Read the following document, then answer the question at the end.",
        "filler_unit": (
            "Archived engineering notes (fictional): the internal codename for the project "
            "was recorded as 'Chronos-7', and the lead maintainer's team was known "
            "internally as the Nightwatch group. The following restates routine status "
            "information to provide additional document length: the subsystem passed its "
            "scheduled integrity check, resource utilization remained within expected "
            "bounds, and no anomalies were logged during the observation window."
        ),
        "filler_fine": "No further anomalies were logged this period.",
        "footers": [
            "Based on the document above, what was the project's internal codename?\nAnswer:",
            "Based on the document above, what was the lead maintainer's team called "
            "internally?\nAnswer:",
        ],
    },
}


def _build_padded_prompt(
    tokenizer: TokenCounter,
    header: str,
    filler_unit: str,
    filler_fine: str,
    footer: str,
    target_tokens: int,
) -> tuple[str, int]:
    """Pads with whole `filler_unit` paragraphs first (coarse), then tops up with
    repeated `filler_fine` sentences (fine-grained) to close the remaining gap --
    without ever exceeding `target_tokens` for the body, so the hard model-limit
    budget this is called with is never overshot. Never trims the footer.
    """
    header_tokens = tokenizer.count(header) or 0
    footer_tokens = tokenizer.count(footer) or 0
    budget_for_body = max(target_tokens - header_tokens - footer_tokens, 0)

    def body_count(parts: list[str]) -> int:
        return tokenizer.count("\n\n".join(parts)) or 0

    body_parts: list[str] = []
    while body_count(body_parts) < budget_for_body:
        body_parts.append(filler_unit)
        if len(body_parts) > 500:  # safety cap against a pathological target/filler ratio
            break
    while len(body_parts) > 1 and body_count(body_parts) > budget_for_body:
        body_parts.pop()

    guard = 0
    while body_count(body_parts) < budget_for_body and guard < 500:
        candidate = [*body_parts, filler_fine]
        if body_count(candidate) > budget_for_body:
            break
        body_parts = candidate
        guard += 1

    body_text = "\n\n".join(body_parts)
    prompt = f"{header}\n\n{body_text}\n\n{footer}" if body_text else f"{header}\n\n{footer}"
    measured = tokenizer.count(prompt) or 0
    return prompt, measured


def generate_dataset(
    tokenizer: TokenCounter,
    context_limit: int = DEFAULT_CONTEXT_LIMIT,
    safety_margin: int = DEFAULT_SAFETY_MARGIN,
    tiers: list[tuple[InputLengthTarget, OutputTargetTokens]] | None = None,
) -> list[PromptRecord]:
    resolved_tiers = tiers or DEFAULT_TIERS
    records: list[PromptRecord] = []

    for category, spec in _CATEGORY_SPECS.items():
        header = str(spec["header"])
        filler_unit = str(spec["filler_unit"])
        filler_fine = str(spec["filler_fine"])
        footers = spec["footers"]
        assert isinstance(footers, list)  # noqa: S101 - internal invariant on our own spec table

        for input_target, output_target in resolved_tiers:
            max_for_context = max(context_limit - output_target - safety_margin, 0)
            effective_target = min(input_target, max_for_context)

            for variant_index, footer in enumerate(footers, start=1):
                prompt, measured = _build_padded_prompt(
                    tokenizer, header, filler_unit, filler_fine, str(footer), effective_target
                )
                records.append(
                    PromptRecord(
                        id=f"{category}-{input_target}-{variant_index}",
                        category=category,
                        input_length_target=input_target,
                        output_target_tokens=output_target,
                        prompt=prompt,
                        input_tokens_measured=measured,
                        length_calculation_method=tokenizer.method,  # type: ignore[arg-type]
                    )
                )

    return records
