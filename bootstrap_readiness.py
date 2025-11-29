from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping

import os

from bootstrap_timeout_policy import (
    _COMPONENT_TIMEOUT_MINIMUMS,
    _DEFERRED_COMPONENT_TIMEOUT_MINIMUMS,
)

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


_COMPONENT_BASELINES: Mapping[str, float] = {
    **_COMPONENT_TIMEOUT_MINIMUMS,
    **_DEFERRED_COMPONENT_TIMEOUT_MINIMUMS,
}


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


def _baseline_for_stage(
    stage: str,
    *,
    component_budgets: Mapping[str, float] | None,
    component_floors: Mapping[str, float] | None,
    fallback: float,
) -> tuple[float | None, float | None]:
    """Return the target budget and floor for a readiness stage."""

    stage_budget = _resolve_stage_budget(stage, component_budgets)
    stage_floor = _resolve_stage_budget(stage, component_floors)

    if stage_floor is None:
        for alias in _STAGE_BUDGET_ALIASES.get(stage, (stage,)):
            if alias in _COMPONENT_BASELINES:
                stage_floor = float(_COMPONENT_BASELINES[alias])
                break

    resolved_budget = stage_budget
    if resolved_budget is None:
        resolved_budget = stage_floor if stage_floor is not None else fallback

    return resolved_budget, stage_floor


def build_stage_deadlines(
    baseline_timeout: float,
    *,
    heavy_detected: bool = False,
    soft_deadline: bool = False,
    heavy_scale: float = 1.5,
    component_budgets: Mapping[str, float] | None = None,
    component_floors: Mapping[str, float] | None = None,
    adaptive_window: float | None = None,
) -> dict[str, dict[str, object]]:
    """Construct stage-aware deadlines for bootstrap orchestration.

    Core stages (``db_index_load``, ``retriever_hydration``, ``vector_seeding``)
    return hard deadlines when ``soft_deadline`` is ``False``. Optional stages
    (``orchestrator_state`` and ``background_loops``) only publish "soft"
    budgets so they can warm in the background after core readiness without
    tripping fatal watchdogs.
    """

    window_scale = 1.0
    if adaptive_window is not None and baseline_timeout:
        window_scale = max(adaptive_window / baseline_timeout, 1.0)
    scale = (heavy_scale if heavy_detected and not soft_deadline else 1.0) * window_scale
    stage_deadlines: dict[str, dict[str, object]] = {}
    for stage in READINESS_STAGES:
        resolved_budget, stage_floor = _baseline_for_stage(
            stage.name,
            component_budgets=component_budgets,
            component_floors=component_floors,
            fallback=baseline_timeout,
        )
        scaled_budget = resolved_budget * scale if resolved_budget is not None else None
        if stage.optional and scaled_budget is not None:
            # Give optional background phases slightly more time so they can
            # converge without tripping fatal watchdogs while the system is
            # already serving traffic in a degraded state.
            scaled_budget *= 1.25

        if stage_floor is not None and scaled_budget is not None:
            scaled_budget = max(scaled_budget, stage_floor)

        enforced = not stage.optional and not soft_deadline
        hard_deadline = None if stage.optional else scaled_budget
        if soft_deadline:
            hard_deadline = None

        # Core stages are now degradable: the initial deadline is treated as a
        # soft budget that triggers degraded readiness instead of aborting the
        # bootstrap loop. Optional stages retain soft budgets but already warm
        # in the background.
        soft_degrade = True if stage.name in CORE_COMPONENTS else stage.optional

        stage_deadlines[stage.name] = {
            "deadline": hard_deadline,
            "soft_budget": scaled_budget,
            "optional": stage.optional,
            "enforced": enforced,
            "floor": stage_floor,
            "budget": resolved_budget,
            "scaled_budget": scaled_budget,
            "soft_degrade": soft_degrade,
            "scale": scale,
            "window_scale": window_scale,
            "core_gate": not stage.optional,
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
    """Evaluate readiness using only core components.

    Optional bootstrap stages are treated as post-ready warmups, so their
    status is excluded from the readiness calculation. Returns
    ``(ready, lagging_core, degraded_core, degraded_online)`` where ``ready``
    reflects core quorum, not background orchestration work.
    """
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
    readyish_online = len(readyish) >= quorum
    degraded_online = readyish_online and len(lagging) > 0
    fully_ready = len(lagging) == 0
    return fully_ready or readyish_online, lagging, degraded, degraded_online


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
