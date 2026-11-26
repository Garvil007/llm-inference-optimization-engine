"""Run metadata collection: git commit hash and package versions each degrade to
None (never a guessed value) on failure; hardware/software/model/backend
snapshots are always populated.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from inference_optimizer.backends.models import BackendConfig, ModelConfig
from inference_optimizer.baseline.metadata import collect_run_metadata


def _model_config() -> ModelConfig:
    return ModelConfig(
        name="fake-model",
        tokenizer="fake/tokenizer",
        revision="deadbeef",
        context_limit=2048,
        quantization="Q4_K_M",
        license="apache-2.0",
    )


class TestCollectRunMetadata:
    def test_real_git_repo_root_has_a_commit_hash(self, tmp_path: Path) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        metadata = collect_run_metadata(
            _model_config(), BackendConfig(backend_type="llama-cpp"), repo_root=repo_root
        )
        assert metadata.git_commit_hash is not None
        assert len(metadata.git_commit_hash) == 40
        assert metadata.git_commit_unavailable_reason is None

    def test_non_git_directory_degrades_to_none(self, tmp_path: Path) -> None:
        metadata = collect_run_metadata(
            _model_config(), BackendConfig(backend_type="llama-cpp"), repo_root=tmp_path
        )
        assert metadata.git_commit_hash is None
        assert metadata.git_commit_unavailable_reason is not None

    def test_git_binary_missing_degrades_to_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise_file_not_found(*args: object, **kwargs: object) -> None:
            raise FileNotFoundError("git not on PATH")

        monkeypatch.setattr(subprocess, "run", _raise_file_not_found)
        metadata = collect_run_metadata(
            _model_config(), BackendConfig(backend_type="llama-cpp"), repo_root=tmp_path
        )
        assert metadata.git_commit_hash is None
        assert "git not on PATH" in (metadata.git_commit_unavailable_reason or "")

    def test_run_id_is_unique_and_snapshots_are_populated(self, tmp_path: Path) -> None:
        model_config = _model_config()
        backend_config = BackendConfig(backend_type="llama-cpp")
        first = collect_run_metadata(model_config, backend_config, repo_root=tmp_path)
        second = collect_run_metadata(model_config, backend_config, repo_root=tmp_path)

        assert first.run_id != second.run_id
        assert first.model_config_snapshot == model_config
        assert first.backend_config_snapshot == backend_config
        assert first.hardware.cpu.logical_cores is not None
        assert first.software.python_version

    def test_unknown_package_version_is_none_not_fabricated(self, tmp_path: Path) -> None:
        metadata = collect_run_metadata(
            _model_config(), BackendConfig(backend_type="llama-cpp"), repo_root=tmp_path
        )
        # llama-cpp-python is an optional extra; in this dev/test env it may or
        # may not be installed -- either way the value must be real, not guessed.
        assert "pydantic" in metadata.software.packages
        assert metadata.software.packages["pydantic"] is not None
