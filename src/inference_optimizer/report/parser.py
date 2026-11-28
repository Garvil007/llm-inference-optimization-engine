"""Reads real baseline artifacts written by Day 6's runner (baseline/runner.py)
-- the JSON aggregate report and the JSONL raw observations -- back into typed
objects. Never synthesizes a value that isn't present in the file.
"""

from __future__ import annotations

import json
from pathlib import Path

from inference_optimizer.baseline.models import BaselineReport
from inference_optimizer.benchmark.models import RequestObservation
from inference_optimizer.exceptions import ConfigurationError
from inference_optimizer.report.models import RawObservationRecord


def load_baseline_report(path: Path) -> BaselineReport:
    if not path.exists():
        raise ConfigurationError(f"Baseline report file not found: {path}")
    return BaselineReport.model_validate_json(path.read_text(encoding="utf-8"))


def load_raw_observations(path: Path) -> list[RawObservationRecord]:
    if not path.exists():
        raise ConfigurationError(f"Raw observations file not found: {path}")

    records: list[RawObservationRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        repeat_index = payload.pop("repeat_index")
        records.append(
            RawObservationRecord(
                repeat_index=repeat_index,
                observation=RequestObservation.model_validate(payload),
            )
        )
    return records
