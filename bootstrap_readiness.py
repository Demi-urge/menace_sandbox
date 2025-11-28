from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping

import os

CORE_COMPONENTS: set[str] = {"vector_seeding", "retriever_hydration", "db_index_load"}
OPTIONAL_COMPONENTS: set[str] = {"orchestrator_state", "background_loops"}


@dataclass(frozen=True)
class ReadinessStage:
    name: str
    steps: tuple[str, ...]
    optional: bool = False


READINESS_STAGES: tuple[ReadinessStage, ...] = (
    ReadinessStage("db_index_load", ("context_builder", "bot_registry"), optional=False),
    ReadinessStage("retriever_hydration", ("data_bot",), optional=False),
    ReadinessStage(
        "vector_seeding",
        (
            "embedder_preload",
            "prepare_pipeline",
            "seed_final_context",
            "push_final_context",
        ),
        optional=False,
    ),
    ReadinessStage("orchestrator_state", ("promote_pipeline",), optional=True),
    ReadinessStage("background_loops", ("bootstrap_complete",), optional=True),
)

_STEP_TO_STAGE: dict[str, str] = {}
for stage in READINESS_STAGES:
    for step in stage.steps:
        _STEP_TO_STAGE[step] = stage.name


def stage_for_step(step: str) -> str | None:
    return _STEP_TO_STAGE.get(step)


_STAGE_BUDGET_ALIASES: Mapping[str, tuple[str, ...]] = {
    "db_index_load": ("db_index_load", "db_indexes", "db_index"),
    "retriever_hydration": ("retriever_hydration", "retrievers", "retriever"),
    "vector_seeding": ("vector_seeding", "vectorizers", "vector"),
    "orchestrator_state": ("orchestrator_state", "orchestrator"),
    "background_loops": ("background_loops", "background"),
}


def _resolve_stage_budget(stage: str, budgets: Mapping[str, float] | None) -> float | None:
    if not budgets:
        return None
    if stage in budgets:
        return float(budgets[stage])
    for alias in _STAGE_BUDGET_ALIASES.get(stage, ()):  # pragma: no branch - small tuples
        if alias in budgets:
            return float(budgets[alias])
    return None


def build_stage_deadlines(
    baseline_timeout: float,
    *,
    heavy_detected: bool = False,
    soft_deadline: bool = False,
    heavy_scale: float = 1.5,
    component_budgets: Mapping[str, float] | None = None,
    component_floors: Mapping[str, float] | None = None,
) -> dict[str, dict[str, object]]:
    """Construct stage-aware deadlines for bootstrap orchestration."""
    scale = heavy_scale if heavy_detected and not soft_deadline else 1.0
    stage_deadlines: dict[str, dict[str, object]] = {}
    for stage in READINESS_STAGES:
        enforced = not stage.optional and not soft_deadline
        stage_budget = _resolve_stage_budget(stage.name, component_budgets)
        resolved_budget = stage_budget if stage_budget is not None else baseline_timeout
        scaled_budget = resolved_budget * scale if resolved_budget is not None else None
        stage_floor = _resolve_stage_budget(stage.name, component_floors)
        if stage.optional and scaled_budget is not None:
            scaled_budget *= 0.8

        deadline = scaled_budget
        if stage_floor is not None and deadline is not None:
            deadline = max(deadline, stage_floor)

        deadline = None if soft_deadline and not stage.optional else deadline
        stage_deadlines[stage.name] = {
            "deadline": deadline,
            "optional": stage.optional,
            "enforced": enforced,
            "floor": stage_floor,
            "budget": resolved_budget,
            "scaled_budget": scaled_budget,
            "soft_budget": stage_budget,
            "scale": scale,
            "soft_degrade": stage.optional,
        }
    return stage_deadlines


def _degraded_quorum(online_state: Mapping[str, object] | None = None) -> int:
    """Return the minimum number of degraded components required for quorum."""

    env_override = os.getenv("MENACE_DEGRADED_CORE_QUORUM")
    online_state = online_state or {}
    for candidate in (env_override, online_state.get("degraded_quorum")):
        try:
            if candidate is not None:
                parsed = int(candidate)
                if parsed > 0:
                    return parsed
        except (TypeError, ValueError):
            continue
    return max(1, len(CORE_COMPONENTS) - 1)


def minimal_online(
    online_state: Mapping[str, object]
) -> tuple[bool, set[str], set[str], bool]:
    components = online_state.get("components", {}) if isinstance(online_state, Mapping) else {}
    lagging: set[str] = set()
    degraded: set[str] = set()
    readyish: set[str] = set()
    for component in CORE_COMPONENTS:
        status = str(components.get(component, "pending"))
        if status == "ready":
            readyish.add(component)
            continue
        if status in {"partial", "warming", "degraded"}:
            degraded.add(component)
            readyish.add(component)
            continue
        lagging.add(component)
    quorum = _degraded_quorum(online_state)
    degraded_online = len(degraded) >= quorum and len(lagging) > 0
    return len(lagging) == 0, lagging, degraded, degraded_online


def lagging_optional_components(online_state: Mapping[str, object]) -> set[str]:
    components = online_state.get("components", {}) if isinstance(online_state, Mapping) else {}
    lagging: set[str] = set()
    for name in OPTIONAL_COMPONENTS:
        status = str(components.get(name, "pending"))
        if status != "ready":
            lagging.add(name)
    return lagging


__all__ = [
    "CORE_COMPONENTS",
    "OPTIONAL_COMPONENTS",
    "READINESS_STAGES",
    "build_stage_deadlines",
    "lagging_optional_components",
    "minimal_online",
    "stage_for_step",
]
