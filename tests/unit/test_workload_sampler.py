from __future__ import annotations

import pytest

from inference_optimizer.workload.models import PromptRecord
from inference_optimizer.workload.sampler import sample_records


def _records() -> list[PromptRecord]:
    records = []
    for category in ["short_qa", "summarization"]:
        for input_length in [128, 512]:
            records.append(
                PromptRecord(
                    id=f"{category}-{input_length}",
                    category=category,  # type: ignore[arg-type]
                    input_length_target=input_length,  # type: ignore[arg-type]
                    output_target_tokens=64,
                    prompt="x",
                    input_tokens_measured=input_length,
                    length_calculation_method="tokenizer",
                )
            )
    return records


class TestSampleRecords:
    def test_same_seed_produces_same_sample(self) -> None:
        records = _records()
        first = sample_records(records, count=2, seed=42)
        second = sample_records(records, count=2, seed=42)
        assert [r.id for r in first] == [r.id for r in second]

    def test_different_seeds_can_produce_different_samples(self) -> None:
        records = _records()
        samples = {tuple(r.id for r in sample_records(records, count=2, seed=s)) for s in range(10)}
        assert len(samples) > 1

    def test_count_zero_returns_empty(self) -> None:
        assert sample_records(_records(), count=0, seed=1) == []

    def test_sampling_without_replacement_never_repeats_when_count_fits(self) -> None:
        records = _records()
        sampled = sample_records(records, count=len(records), seed=1)
        ids = [r.id for r in sampled]
        assert len(ids) == len(set(ids))

    def test_count_exceeding_available_uses_replacement(self) -> None:
        records = _records()
        sampled = sample_records(records, count=len(records) * 3, seed=1)
        assert len(sampled) == len(records) * 3

    def test_category_filter_restricts_the_pool(self) -> None:
        records = _records()
        sampled = sample_records(records, count=2, seed=1, categories=["short_qa"])
        assert all(r.category == "short_qa" for r in sampled)

    def test_input_length_filter_restricts_the_pool(self) -> None:
        records = _records()
        sampled = sample_records(records, count=2, seed=1, input_length_targets=[128])
        assert all(r.input_length_target == 128 for r in sampled)

    def test_negative_count_raises(self) -> None:
        with pytest.raises(ValueError, match="count must be >= 0"):
            sample_records(_records(), count=-1, seed=1)
