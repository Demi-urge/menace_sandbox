"""Central optional-dependency fallback shim definitions.

These shims are safe to import and instantiate in production environments where
optional dependencies are unavailable.
"""

from __future__ import annotations

from typing import Any


class OptionalDependencyShim:
    """optional-dependency fallback shim with deterministic no-op behavior.

    The shim accepts any constructor signature and exposes common runtime method
    names used by optional integrations. Return values are intentionally stable
    so call paths remain predictable in degraded environments.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"ok": False, "shim": True, "args": list(args), "kwargs": kwargs}

    def fit(self, *args: Any, **kwargs: Any) -> "OptionalDependencyShim":
        return self

    def predict(self, *args: Any, **kwargs: Any) -> list[float]:
        return [0.0]

    def search(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    def track(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"tracked": False, "shim": True}

    def add(self, *args: Any, **kwargs: Any) -> bool:
        return False

    def record(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"recorded": False, "shim": True}

    def run(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "shim", "ok": False}

    def build(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {}

    def load(self, *args: Any, **kwargs: Any) -> "OptionalDependencyShim":
        return self


class HumanAlignmentAgentFallback(OptionalDependencyShim):
    """optional-dependency fallback shim for ``HumanAlignmentAgent``.

    ``evaluate_changes`` returns an empty warning map so patch execution can
    proceed without alignment scoring when the optional dependency is missing.
    """

    def evaluate_changes(
        self,
        workflow_changes: list[dict[str, Any]] | None,
        *_args: Any,
        **_kwargs: Any,
    ) -> dict[str, bool]:
        if not workflow_changes:
            return {}
        return {
            str(change.get("file", f"change_{index}")): False
            for index, change in enumerate(workflow_changes)
        }
