from __future__ import annotations

from menace_sandbox.intended_production_bots import INTENDED_PRODUCTION_BOTS
from menace_sandbox.runnable_bots_registry import RUNNABLE_BOT_REGISTRY


def test_intended_production_bots_are_in_runnable_registry():
    intended = set(INTENDED_PRODUCTION_BOTS)
    registry_names = {entry.name for entry in RUNNABLE_BOT_REGISTRY}

    missing = sorted(intended - registry_names)
    assert not missing, (
        "Intended production bots missing from RUNNABLE_BOT_REGISTRY: "
        + ", ".join(missing)
    )
