from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.exceptions import ConfigurationError
from inference_optimizer.workload.dataset import load_dataset, save_dataset
from inference_optimizer.workload.models import PromptRecord


def _record(id_: str) -> PromptRecord:
    return PromptRecord(
        id=id_,
        category="short_qa",
        input_length_target=128,
        output_target_tokens=64,
        prompt="Question: x?\nAnswer:",
        input_tokens_measured=128,
        length_calculation_method="tokenizer",
    )


class TestSaveAndLoadDataset:
    def test_round_trips_records(self, tmp_path: Path) -> None:
        path = tmp_path / "dataset.jsonl"
        records = [_record("a"), _record("b")]

        save_dataset(path, records)
        loaded = load_dataset(path)

        assert loaded == records

    def test_one_record_per_line(self, tmp_path: Path) -> None:
        path = tmp_path / "dataset.jsonl"
        save_dataset(path, [_record("a"), _record("b")])

        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 2

    def test_blank_lines_are_skipped_on_load(self, tmp_path: Path) -> None:
        path = tmp_path / "dataset.jsonl"
        save_dataset(path, [_record("a")])
        path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")

        loaded = load_dataset(path)
        assert len(loaded) == 1

    def test_missing_file_raises_configuration_error(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigurationError, match="not found"):
            load_dataset(tmp_path / "missing.jsonl")

    def test_malformed_line_raises_configuration_error(self, tmp_path: Path) -> None:
        path = tmp_path / "dataset.jsonl"
        path.write_text("not valid json\n", encoding="utf-8")
        with pytest.raises(ConfigurationError, match="invalid PromptRecord"):
            load_dataset(path)
