"""JSONL read/write for the prompt dataset -- one PromptRecord per line, UTF-8,
no wrapping array (standard JSONL so it's diffable and streamable line-by-line).
"""

from __future__ import annotations

import json
from pathlib import Path

from inference_optimizer.exceptions import ConfigurationError
from inference_optimizer.workload.models import PromptRecord


def load_dataset(path: Path) -> list[PromptRecord]:
    if not path.exists():
        raise ConfigurationError(f"Workload dataset not found at {path}.")
    records: list[PromptRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(PromptRecord.model_validate_json(stripped))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ConfigurationError(
                    f"{path}:{line_number}: invalid PromptRecord JSON: {exc}"
                ) from exc
    return records


def save_dataset(path: Path, records: list[PromptRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(record.model_dump_json())
            handle.write("\n")
