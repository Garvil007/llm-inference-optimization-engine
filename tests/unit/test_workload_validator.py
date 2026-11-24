from __future__ import annotations

from inference_optimizer.workload.models import PromptRecord
from inference_optimizer.workload.validator import validate_dataset


def _record(
    id_: str = "short_qa-128-1",
    category: str = "short_qa",
    input_length_target: int = 128,
    output_target_tokens: int = 64,
    prompt: str = "Question: x?\nAnswer:",
    input_tokens_measured: int = 128,
    method: str = "tokenizer",
) -> PromptRecord:
    return PromptRecord(
        id=id_,
        category=category,  # type: ignore[arg-type]
        input_length_target=input_length_target,  # type: ignore[arg-type]
        output_target_tokens=output_target_tokens,  # type: ignore[arg-type]
        prompt=prompt,
        input_tokens_measured=input_tokens_measured,
        length_calculation_method=method,  # type: ignore[arg-type]
    )


class TestValidateDataset:
    def test_empty_dataset_is_an_error(self) -> None:
        errors, warnings = validate_dataset([])
        assert errors == ["Dataset is empty."]
        assert warnings == []

    def test_valid_record_within_tolerance_has_no_errors_or_warnings(self) -> None:
        errors, warnings = validate_dataset([_record(input_tokens_measured=128)])
        assert errors == []
        assert warnings == []

    def test_duplicate_ids_are_an_error(self) -> None:
        errors, _warnings = validate_dataset([_record(id_="dup"), _record(id_="dup")])
        assert any("Duplicate id" in error for error in errors)

    def test_empty_prompt_is_an_error(self) -> None:
        errors, _warnings = validate_dataset([_record(prompt="   ")])
        assert any("prompt is empty" in error for error in errors)

    def test_zero_measured_tokens_is_an_error(self) -> None:
        errors, _warnings = validate_dataset([_record(input_tokens_measured=0)])
        assert any("must be > 0" in error for error in errors)

    def test_exceeding_model_limit_budget_is_an_error(self) -> None:
        # context_limit=2048, output_target=256, safety_margin=32 -> max allowed 1760
        record = _record(
            input_length_target=2048, output_target_tokens=256, input_tokens_measured=2000
        )
        errors, _warnings = validate_dataset([record], context_limit=2048, safety_margin=32)
        assert any("exceeds the model-limit budget" in error for error in errors)

    def test_length_far_from_target_is_a_warning_not_an_error(self) -> None:
        record = _record(input_length_target=128, output_target_tokens=64, input_tokens_measured=10)
        errors, warnings = validate_dataset([record])
        assert errors == []
        assert any("outside +/-" in warning for warning in warnings)
