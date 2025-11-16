from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Any


@dataclass
class EndpointTelemetry:
    """Track endpoint performance and neuro-impact."""

    score: float = 1.0
    failures: int = 0
    latency_sum: float = 0.0
    calls: int = 0


class NeuroAPIRouter:
    """Route and audit external API calls with self-healing."""

    def __init__(
        self,
        endpoints: Dict[str, Callable[..., Any]],
        *,
        gpt4_cost_hook: Callable[[], float] | None = None,
        cost_threshold: float = 0.02,
        local_model: Callable[[str], str] | None = None,
        gpt4_model: Callable[[str], str] | None = None,
    ) -> None:
        self.endpoints = endpoints
        self.telemetry: Dict[str, EndpointTelemetry] = {
            name: EndpointTelemetry() for name in endpoints
        }
        self.gpt4_cost_hook = gpt4_cost_hook or (lambda: 0.0)
        self.cost_threshold = cost_threshold
        self.local_model = local_model
        self.gpt4_model = gpt4_model

    # ------------------------------------------------------------------
    def fetch(self, name: str, *args: Any, **kwargs: Any) -> Any:
        start = time.time()
        try:
            result = self.endpoints[name](*args, **kwargs)
            latency = time.time() - start
            tele = self.telemetry[name]
            tele.latency_sum += latency
            tele.calls += 1
            if latency > 1.0:
                tele.score *= 0.9
            return result
        except Exception:
            tele = self.telemetry[name]
            tele.failures += 1
            tele.score *= 0.5
            raise

    # ------------------------------------------------------------------
    def route_model(self, prompt: str) -> str:
        """Switch to local model if GPT-4 price spikes."""
        if self.gpt4_cost_hook() > self.cost_threshold and self.local_model:
            return self.local_model(prompt)
        if self.gpt4_model:
            return self.gpt4_model(prompt)
        return ""

    # ------------------------------------------------------------------
    def audit(self) -> None:
        """Demote poor endpoints and promote healthy ones."""
        for tele in self.telemetry.values():
            avg_latency = tele.latency_sum / tele.calls if tele.calls else 0.0
            if tele.failures > 2 or avg_latency > 2.0:
                tele.score = max(0.1, tele.score * 0.5)
            else:
                tele.score = min(2.0, tele.score + 0.1)
            tele.failures = 0
            tele.latency_sum = 0.0
            tele.calls = 0


__all__ = ["NeuroAPIRouter", "EndpointTelemetry"]
