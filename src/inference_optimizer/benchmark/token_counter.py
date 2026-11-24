"""Output-token counting for benchmark measurements.

A streaming chunk is NOT a token -- llama.cpp/vLLM's SSE deltas are arbitrary text
fragments (sometimes multiple tokens, sometimes a partial token's UTF-8 bytes), so
counting SSE events would silently fabricate a token count. Every counter here either
produces a real count from the actual model's tokenizer, a clearly-labeled estimate,
or None -- never a guess presented as a measurement. `.method` on every counter says
which one was used, and that label is carried into every RequestObservation.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from inference_optimizer.logging import get_logger

logger = get_logger(__name__)


class TokenCounter(Protocol):
    """`.count()` may return None (counting unavailable); `.method` must always
    reflect what actually happened on the most recent call, so callers never
    mistake an estimate for a measurement.
    """

    @property
    def method(self) -> str: ...

    def count(self, text: str) -> int | None: ...


class FixedTokenCounter:
    """Returns predetermined counts in order, ignoring the text entirely.

    For deterministic tests only: lets a test assert exact TPOT/throughput math
    without depending on a real tokenizer or on text length.
    """

    method = "fixed"

    def __init__(self, counts: Sequence[int]) -> None:
        self._counts = iter(counts)

    def count(self, text: str) -> int | None:
        try:
            return next(self._counts)
        except StopIteration:
            return None


class WhitespaceApproxCounter:
    """Crude fallback: whitespace-split word count. Not a token count -- most
    tokenizers produce more tokens than words (subword splitting, punctuation).
    Only used when no real tokenizer is available; always labeled "approximate"
    so aggregates never conflate it with a measured count.
    """

    method = "approximate"

    def count(self, text: str) -> int | None:
        if not text:
            return 0
        return len(text.split())


class HFTokenizerCounter:
    """Counts using the actual model's tokenizer (Hugging Face `tokenizers`
    library, loaded from the model's HF repo id). This is the "actual model
    tokenization" path the benchmark should prefer when available.

    Loading is lazy and best-effort: a missing `tokenizers` install, no network
    access, or a bad repo id degrades to `.method == "unavailable"` and
    `.count()` returning None, rather than raising and aborting a benchmark run.
    """

    def __init__(self, repo_id: str) -> None:
        self._repo_id = repo_id
        self._tokenizer: object | None = None
        self._load_failed = False

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None or self._load_failed:
            return
        try:
            from tokenizers import Tokenizer  # type: ignore[import-not-found]

            self._tokenizer = Tokenizer.from_pretrained(self._repo_id)
        except Exception as exc:  # noqa: BLE001 - any failure here must degrade, not crash
            logger.warning("tokenizer_load_failed", repo_id=self._repo_id, error=str(exc))
            self._load_failed = True

    def count(self, text: str) -> int | None:
        self._ensure_loaded()
        if self._tokenizer is None:
            return None
        encoding = self._tokenizer.encode(text)  # type: ignore[attr-defined]
        return len(encoding.ids)

    @property
    def method(self) -> str:
        return "unavailable" if self._tokenizer is None else "tokenizer"
