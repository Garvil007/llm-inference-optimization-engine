"""Cross-platform task runner (Windows-friendly alternative to Makefile).

Usage: python tasks.py <setup|lint|format|test|hardware-info>
"""

from __future__ import annotations

import subprocess
import sys

_TASKS: dict[str, list[list[str]]] = {
    "setup": [[sys.executable, "-m", "pip", "install", "-e", ".[dev]"]],
    "lint": [
        [sys.executable, "-m", "ruff", "check", "src", "tests"],
        [sys.executable, "-m", "mypy", "src"],
    ],
    "format": [
        [sys.executable, "-m", "ruff", "format", "src", "tests"],
        [sys.executable, "-m", "ruff", "check", "--fix", "src", "tests"],
    ],
    "test": [
        [sys.executable, "-m", "pytest", "--cov=src", "--cov-report=term-missing"],
    ],
    "hardware-info": [
        [sys.executable, "-m", "inference_optimizer", "hardware-info"],
    ],
}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in _TASKS:
        task_names = ", ".join(sorted(_TASKS))
        print(f"Usage: python tasks.py <{task_names}>", file=sys.stderr)
        return 1

    for command in _TASKS[sys.argv[1]]:
        result = subprocess.run(command)  # noqa: S603 - fixed, hardcoded commands
        if result.returncode != 0:
            return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
