"""Generator tests use WhitespaceApproxCounter (word-count based, offline,
deterministic) instead of a real HF tokenizer, so these tests never touch the
network and give exact, reproducible token counts for assertions.
"""

from __future__ import annotations

from inference_optimizer.benchmark.token_counter import WhitespaceApproxCounter
from inference_optimizer.workload.generator import generate_dataset
from inference_optimizer.workload.validator import validate_dataset


class TestGenerateDataset:
    def test_produces_five_categories_three_tiers_two_variants_each(self) -> None:
        records = generate_dataset(WhitespaceApproxCounter())

        assert len(records) == 5 * 3 * 2
        categories = {record.category for record in records}
        assert categories == {
            "short_qa",
            "summarization",
            "code_generation",
            "structured_extraction",
            "long_context_qa",
        }
        assert {record.input_length_target for record in records} == {128, 512, 2048}
        assert {record.output_target_tokens for record in records} == {64, 128, 256}

    def test_all_ids_are_unique(self) -> None:
        records = generate_dataset(WhitespaceApproxCounter())
        ids = [record.id for record in records]
        assert len(ids) == len(set(ids))

    def test_footer_instruction_always_present_verbatim(self) -> None:
        # Padding must never corrupt the actual task instruction.
        records = generate_dataset(WhitespaceApproxCounter())
        short_qa_128 = [
            r for r in records if r.category == "short_qa" and r.input_length_target == 128
        ]
        assert any("capital of France" in r.prompt for r in short_qa_128)
        assert any("multiplied by 6" in r.prompt for r in short_qa_128)

    def test_measured_length_approaches_target_for_small_tiers(self) -> None:
        records = generate_dataset(WhitespaceApproxCounter())
        tier_128 = [r for r in records if r.input_length_target == 128]
        for record in tier_128:
            # Fine-grained filler closes most of the gap; policy is "never exceed,
            # approximately reach" -- so measured must be close to but not over target.
            assert 100 <= record.input_tokens_measured <= 128

    def test_2048_tier_is_clamped_to_fit_context_limit_with_output_budget(self) -> None:
        records = generate_dataset(WhitespaceApproxCounter(), context_limit=2048, safety_margin=32)
        tier_2048 = [r for r in records if r.input_length_target == 2048]
        for record in tier_2048:
            # Labeled 2048, but the model only has 2048 total context and this
            # tier's output_target is 256 -- so measured must stay well under 2048.
            assert record.input_tokens_measured < 2048 - record.output_target_tokens

    def test_generated_dataset_passes_its_own_validator(self) -> None:
        records = generate_dataset(WhitespaceApproxCounter())
        errors, _warnings = validate_dataset(records)
        assert errors == []

    def test_length_calculation_method_is_recorded(self) -> None:
        records = generate_dataset(WhitespaceApproxCounter())
        assert all(record.length_calculation_method == "approximate" for record in records)
