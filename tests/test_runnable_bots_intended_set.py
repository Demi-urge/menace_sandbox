from __future__ import annotations

from menace_sandbox.intended_production_bots import INTENDED_PRODUCTION_BOTS
from menace_sandbox.production_bot_manifest import PRODUCTION_BOT_MANIFEST
from menace_sandbox.runnable_bot_discovery import discover_runnable_startup_callables
from menace_sandbox.runnable_bots_registry import RUNNABLE_BOT_REGISTRY


def test_production_manifest_entries_are_in_runnable_registry():
    intended = {
        entry.name for entry in PRODUCTION_BOT_MANIFEST if entry.intended_for_production
    }
    registry_names = {entry.name for entry in RUNNABLE_BOT_REGISTRY}

    missing = sorted(intended - registry_names)
    assert not missing, (
        "Production-intended bots missing from RUNNABLE_BOT_REGISTRY: "
        + ", ".join(missing)
    )


def test_intended_production_tuple_is_manifest_derived():
    discovered = tuple(
        entry.name for entry in PRODUCTION_BOT_MANIFEST if entry.intended_for_production
    )
    assert INTENDED_PRODUCTION_BOTS == discovered


def test_runnable_registry_tracks_manifest_entries_exactly():
    manifest_names = tuple(entry.name for entry in PRODUCTION_BOT_MANIFEST)
    registry_names = tuple(entry.name for entry in RUNNABLE_BOT_REGISTRY)
    assert registry_names == manifest_names

    discovered_workers = set(discover_runnable_startup_callables())
    manifest_workers = {entry.startup_callable for entry in PRODUCTION_BOT_MANIFEST}
    assert discovered_workers <= manifest_workers


def test_discovered_runnable_workers_are_covered_by_manifest() -> None:
    discovered_workers = set(discover_runnable_startup_callables())
    manifest_workers = {entry.startup_callable for entry in PRODUCTION_BOT_MANIFEST}

    missing = sorted(discovered_workers - manifest_workers)
    assert not missing, (
        "Discovered runnable workers missing from PRODUCTION_BOT_MANIFEST: "
        + ", ".join(missing)
    )
