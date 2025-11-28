from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping

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


def build_stage_deadlines(
    baseline_timeout: float,
    *,
    heavy_detected: bool = False,
    soft_deadline: bool = False,
    heavy_scale: float = 1.5,
) -> dict[str, dict[str, object]]:
    """Construct stage-aware deadlines for bootstrap orchestration."""

    base_timeout = baseline_timeout
    if heavy_detected and not soft_deadline:
        base_timeout *= heavy_scale

    stage_deadlines: dict[str, dict[str, object]] = {}
    for stage in READINESS_STAGES:
        enforced = not stage.optional and not soft_deadline
        deadline = None if soft_deadline and not stage.optional else base_timeout
        if stage.optional:
            deadline = base_timeout * 0.8
        stage_deadlines[stage.name] = {
            "deadline": deadline,
            "optional": stage.optional,
            "enforced": enforced,
        }
    return stage_deadlines


def minimal_online(online_state: Mapping[str, object]) -> bool:
    components = online_state.get("components", {}) if isinstance(online_state, Mapping) else {}
    return all(str(components.get(component, "pending")) == "ready" for component in CORE_COMPONENTS)


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
