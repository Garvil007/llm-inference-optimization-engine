"""Memory sampling: real process RSS is always readable in-test; GPU memory
degrades to (None, "unavailable") when pynvml can't be imported, without
raising -- mirrors hardware/detector.py's GPU-probe degradation tests.
"""

from __future__ import annotations

import builtins

import pytest

from inference_optimizer.baseline import resources
from inference_optimizer.baseline.models import MemorySample
from inference_optimizer.baseline.resources import sample_memory


class TestSampleMemory:
    def test_label_is_echoed_back(self) -> None:
        sample = sample_memory("before_warmup")
        assert isinstance(sample, MemorySample)
        assert sample.label == "before_warmup"

    def test_process_rss_is_a_positive_measurement(self) -> None:
        sample = sample_memory("check")
        assert sample.process_rss_mb is not None
        assert sample.process_rss_mb > 0

    def test_gpu_memory_degrades_when_pynvml_unimportable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_import = builtins.__import__

        def _blocked_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "pynvml":
                raise ImportError("no pynvml in this environment")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(builtins, "__import__", _blocked_import)

        used_mb, source = resources._gpu_memory_used_mb()
        assert used_mb is None
        assert source == "unavailable"

    def test_gpu_memory_degrades_on_nvml_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _FakePynvml:
            @staticmethod
            def nvmlInit() -> None:
                raise RuntimeError("driver not loaded")

        monkeypatch.setitem(__import__("sys").modules, "pynvml", _FakePynvml())

        used_mb, source = resources._gpu_memory_used_mb()
        assert used_mb is None
        assert source == "unavailable"
