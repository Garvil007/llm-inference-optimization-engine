"""Reproducible sampling: a given (dataset, filters, seed, count) always produces
the same sequence of records, by using a seeded `random.Random` instance that is
never shared with any other randomness in the process (no reliance on the global
`random` module, which other code could perturb).

Policy: if `count` is <= the number of records matching the filters, sampling is
without replacement (`Random.sample`) -- each record appears at most once. If
`count` exceeds the number of matching records, sampling falls back to with-
replacement (`Random.choices`) so the caller still gets exactly `count` records,
and a warning is logged since this means some records necessarily repeat.
"""

from __future__ import annotations

import random

from inference_optimizer.logging import get_logger
from inference_optimizer.workload.models import Category, InputLengthTarget, PromptRecord

logger = get_logger(__name__)


def sample_records(
    records: list[PromptRecord],
    count: int,
    seed: int,
    categories: list[Category] | None = None,
    input_length_targets: list[InputLengthTarget] | None = None,
) -> list[PromptRecord]:
    if count < 0:
        raise ValueError(f"count must be >= 0, got {count}")

    filtered = [
        record
        for record in records
        if (categories is None or record.category in categories)
        and (input_length_targets is None or record.input_length_target in input_length_targets)
    ]

    rng = random.Random(seed)

    if count == 0 or not filtered:
        return []
    if count <= len(filtered):
        return rng.sample(filtered, k=count)

    logger.warning(
        "workload_sample_with_replacement",
        requested=count,
        available=len(filtered),
    )
    return rng.choices(filtered, k=count)
