"""Run provenance: a unique run id, wall-clock start time, the git commit this
code was run at, installed package versions, and a full hardware snapshot --
everything needed to later explain *why* a baseline run produced the numbers
it did. Every field degrades to None (never a guessed value) when it can't be
read in this environment.
"""

from __future__ import annotations

import subprocess
import sys
import uuid
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path

from inference_optimizer.backends.models import BackendConfig, ModelConfig
from inference_optimizer.baseline.models import RunMetadata, SoftwareVersions
from inference_optimizer.hardware import detect_hardware
from inference_optimizer.logging import get_logger

logger = get_logger(__name__)

_GIT_TIMEOUT_SECONDS = 5
_TRACKED_PACKAGES = [
    "pydantic",
    "httpx",
    "typer",
    "fastapi",
    "uvicorn",
    "llama-cpp-python",
    "tokenizers",
    "psutil",
]


def _git_commit_hash(repo_root: Path) -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(  # noqa: S603 - fixed args, no user input
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=True,
        )
        return result.stdout.strip(), None
    except (subprocess.SubprocessError, OSError) as exc:
        logger.warning("git_commit_hash_unavailable", error=str(exc))
        return None, str(exc)


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in _TRACKED_PACKAGES:
        try:
            versions[package] = pkg_version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def collect_run_metadata(
    model_config: ModelConfig,
    backend_config: BackendConfig,
    repo_root: Path | None = None,
) -> RunMetadata:
    git_commit_hash, git_unavailable_reason = _git_commit_hash(repo_root or Path.cwd())
    return RunMetadata(
        run_id=str(uuid.uuid4()),
        started_at=datetime.now(UTC).isoformat(),
        git_commit_hash=git_commit_hash,
        git_commit_unavailable_reason=git_unavailable_reason,
        hardware=detect_hardware(),
        software=SoftwareVersions(
            python_version=sys.version.split()[0],
            packages=_package_versions(),
        ),
        model_config_snapshot=model_config,
        backend_config_snapshot=backend_config,
    )
