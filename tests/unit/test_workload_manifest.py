from __future__ import annotations

import hashlib
from pathlib import Path

from inference_optimizer.workload.manifest import build_manifest, save_manifest


class TestBuildManifest:
    def test_records_requested_fields(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.jsonl"
        dataset_path.write_text('{"a": 1}\n', encoding="utf-8")
        generator_source = tmp_path / "generator.py"
        generator_source.write_text("# fake generator source\n", encoding="utf-8")

        manifest = build_manifest(
            dataset_version="v1",
            dataset_path=dataset_path,
            tokenizer_repo_id="Qwen/Qwen2.5-0.5B-Instruct",
            tokenizer_method="tokenizer",
            seed=42,
            record_count=1,
            generation_settings={"tiers": [[128, 64]]},
            generator_source_path=generator_source,
        )

        assert manifest.dataset_version == "v1"
        assert manifest.tokenizer_repo_id == "Qwen/Qwen2.5-0.5B-Instruct"
        assert manifest.seed == 42
        assert manifest.record_count == 1
        assert manifest.generation_settings == {"tiers": [[128, 64]]}
        assert "generator.py" in manifest.source_file_hashes
        assert "dataset.jsonl" in manifest.source_file_hashes

    def test_hash_matches_real_sha256_of_file_contents(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.jsonl"
        content = b'{"a": 1}\n'
        dataset_path.write_bytes(content)
        generator_source = tmp_path / "generator.py"
        generator_source.write_text("# fake\n", encoding="utf-8")

        manifest = build_manifest(
            dataset_version="v1",
            dataset_path=dataset_path,
            tokenizer_repo_id="repo",
            tokenizer_method="tokenizer",
            seed=0,
            record_count=0,
            generation_settings={},
            generator_source_path=generator_source,
        )

        expected = hashlib.sha256(content).hexdigest()
        assert manifest.source_file_hashes["dataset.jsonl"] == expected

    def test_save_manifest_writes_readable_json(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.jsonl"
        dataset_path.write_text("{}\n", encoding="utf-8")
        generator_source = tmp_path / "generator.py"
        generator_source.write_text("# fake\n", encoding="utf-8")

        manifest = build_manifest(
            dataset_version="v1",
            dataset_path=dataset_path,
            tokenizer_repo_id="repo",
            tokenizer_method="tokenizer",
            seed=0,
            record_count=0,
            generation_settings={},
            generator_source_path=generator_source,
        )
        out_path = tmp_path / "manifest.json"
        save_manifest(out_path, manifest)

        assert out_path.exists()
        assert "dataset_version" in out_path.read_text(encoding="utf-8")
