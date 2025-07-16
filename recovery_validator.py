"""Replay failed workflows to validate recovery."""
from __future__ import annotations

from typing import Callable, Iterable


class ReplayValidator:
    def __init__(self, replay_fn: Callable[[str], None]):
        self.replay_fn = replay_fn

    def validate(self, workflows: Iterable[str]) -> dict[str, bool]:
        results: dict[str, bool] = {}
        for wf in workflows:
            try:
                self.replay_fn(wf)
                results[wf] = True
            except Exception:
                results[wf] = False
        return results


__all__ = ["ReplayValidator"]
