"""Hand-rolled Prometheus text-exposition metrics -- no new dependency (prometheus_client)
for three counters/gauges; swap in a real client library later if cardinality grows.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


@dataclass
class Metrics:
    requests_total: dict[tuple[str, str, int], int] = field(default_factory=dict)
    request_duration_seconds_sum: dict[tuple[str, str], float] = field(default_factory=dict)
    request_duration_seconds_count: dict[tuple[str, str], int] = field(default_factory=dict)
    in_flight: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def observe_request(
        self, method: str, path: str, status_code: int, duration_seconds: float
    ) -> None:
        async with self._lock:
            key = (method, path, status_code)
            self.requests_total[key] = self.requests_total.get(key, 0) + 1
            duration_key = (method, path)
            self.request_duration_seconds_sum[duration_key] = (
                self.request_duration_seconds_sum.get(duration_key, 0.0) + duration_seconds
            )
            self.request_duration_seconds_count[duration_key] = (
                self.request_duration_seconds_count.get(duration_key, 0) + 1
            )

    async def inc_in_flight(self) -> None:
        async with self._lock:
            self.in_flight += 1

    async def dec_in_flight(self) -> None:
        async with self._lock:
            self.in_flight -= 1

    def render_prometheus(self) -> str:
        lines = [
            "# HELP inference_optimizer_requests_total Total HTTP requests by method/path/status.",
            "# TYPE inference_optimizer_requests_total counter",
        ]
        for (method, path, status_code), count in sorted(self.requests_total.items()):
            lines.append(
                f'inference_optimizer_requests_total{{method="{method}",path="{path}",'
                f'status="{status_code}"}} {count}'
            )

        lines += [
            "# HELP inference_optimizer_request_duration_seconds_sum Total request latency.",
            "# TYPE inference_optimizer_request_duration_seconds_sum counter",
        ]
        for (method, path), total in sorted(self.request_duration_seconds_sum.items()):
            lines.append(
                f'inference_optimizer_request_duration_seconds_sum{{method="{method}",'
                f'path="{path}"}} {total}'
            )

        lines += [
            "# HELP inference_optimizer_request_duration_seconds_count Request count for latency avg.",  # noqa: E501
            "# TYPE inference_optimizer_request_duration_seconds_count counter",
        ]
        for (method, path), count in sorted(self.request_duration_seconds_count.items()):
            lines.append(
                f'inference_optimizer_request_duration_seconds_count{{method="{method}",'
                f'path="{path}"}} {count}'
            )

        lines += [
            "# HELP inference_optimizer_requests_in_flight Requests currently being processed.",
            "# TYPE inference_optimizer_requests_in_flight gauge",
            f"inference_optimizer_requests_in_flight {self.in_flight}",
        ]
        return "\n".join(lines) + "\n"
