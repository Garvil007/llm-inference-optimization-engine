"""Bounded concurrency admission: reject immediately over the limit rather than queue
(queueing would let latency grow unbounded under load; a hard cap gives a server-level
circuit breaker instead). Uses a plain counter + asyncio.Lock rather than asyncio.Semaphore
because Semaphore has no non-blocking try-acquire in asyncio -- we need to fail fast, not wait.
"""

from __future__ import annotations

import asyncio
from types import TracebackType


class AdmissionRejected(Exception):
    """Raised when the concurrency limit is already saturated."""


class ConcurrencyLimiter:
    def __init__(self, limit: int) -> None:
        if limit < 1:
            raise ValueError(f"concurrency limit must be >= 1, got {limit}")
        self._limit = limit
        self._in_flight = 0
        self._lock = asyncio.Lock()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def in_flight(self) -> int:
        return self._in_flight

    async def _try_acquire(self) -> bool:
        async with self._lock:
            if self._in_flight >= self._limit:
                return False
            self._in_flight += 1
            return True

    async def _release(self) -> None:
        async with self._lock:
            self._in_flight -= 1

    def admit(self) -> _Admission:
        """Usage: `async with limiter.admit():` -- raises AdmissionRejected if full."""
        return _Admission(self)


class _Admission:
    def __init__(self, limiter: ConcurrencyLimiter) -> None:
        self._limiter = limiter
        self._acquired = False

    async def __aenter__(self) -> None:
        self._acquired = await self._limiter._try_acquire()  # noqa: SLF001 - cooperating class
        if not self._acquired:
            raise AdmissionRejected(
                f"Concurrency limit ({self._limiter.limit}) reached; try again shortly."
            )

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._acquired:
            await self._limiter._release()  # noqa: SLF001 - cooperating class
