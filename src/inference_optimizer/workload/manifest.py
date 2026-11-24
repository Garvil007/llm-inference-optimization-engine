"""Builds and persists the WorkloadManifest: dataset version, tokenizer identity,
generation settings, and sha256 hashes of the generator source and the produced
dataset file -- so anyone can later confirm a dataset on disk matches what its
manifest claims generated it, without re-running generation.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

from inference_optimizer.workload.models import WorkloadManifest


def _sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def build_manifest(
    dataset_version: str,
    dataset_path: Path,
    tokenizer_repo_id: str,
    tokenizer_method: str,
    seed: int,
    record_count: int,
    generation_settings: dict[str, object],
    generator_source_path: Path,
) -> WorkloadManifest:
    return WorkloadManifest(
        dataset_version=dataset_version,
        dataset_path=str(dataset_path),
        tokenizer_repo_id=tokenizer_repo_id,
        tokenizer_method=tokenizer_method,  # type: ignore[arg-type]
        generated_at=datetime.now(UTC),
        seed=seed,
        record_count=record_count,
        generation_settings=generation_settings,
        source_file_hashes={
            "generator.py": _sha256_of_file(generator_source_path),
            dataset_path.name: _sha256_of_file(dataset_path),
        },
    )


def save_manifest(path: Path, manifest: WorkloadManifest) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
