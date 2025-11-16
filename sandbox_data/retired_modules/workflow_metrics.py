"""Convenience wrappers exposing common workflow metric helpers."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Dict, Iterable

from .workflow_scorer_core import (
    compute_workflow_synergy,
    compute_bottleneck_index,
    compute_patchability,
)


def compute_workflow_entropy(spec: Dict[str, Any] | Iterable[Any]) -> float:
    """Return the Shannon entropy of modules used in ``spec``.

    Parameters
    ----------
    spec:
        Workflow specification.  Either a mapping containing a ``steps``
        sequence or a bare sequence of step mappings/strings.  Each step is
        expected to expose a ``module`` name; strings are treated directly as
        module names.

    Returns
    -------
    float
        The Shannon entropy (base 2) of the module frequency distribution.
    """

    if isinstance(spec, dict):
        steps = spec.get("steps", [])
    else:  # Accept raw sequence of steps
        steps = list(spec) if spec is not None else []

    modules: list[str] = []
    for step in steps:
        mod: str | None
        if isinstance(step, str):
            mod = step
        elif isinstance(step, dict):
            mod = step.get("module")
        else:
            mod = None
        if mod:
            modules.append(mod)

    total = len(modules)
    if not total:
        return 0.0

    counts = Counter(modules)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


__all__ = [
    "compute_workflow_synergy",
    "compute_bottleneck_index",
    "compute_patchability",
    "compute_workflow_entropy",
]
