from __future__ import annotations

import logging
from pathlib import Path

from self_improvement.component_workflow_synthesis import discover_component_workflows


def _write_module(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def test_component_workflow_synthesis_emits_specs(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_module(repo / "allocator_service.py", '"""Allocator service."""\n')
    _write_module(
        repo / "systems" / "context_manager.py",
        '"""Context manager."""\n',
    )

    specs = discover_component_workflows(base_path=repo, logger=logging.getLogger("test"))
    ids = {spec["workflow_id"] for spec in specs}
    assert "component::allocator_service" in ids
    assert "component::systems.context_manager" in ids

    alloc = next(spec for spec in specs if spec["workflow_id"] == "component::allocator_service")
    assert alloc["workflow"] == ["allocator_service"]
    assert alloc["task_sequence"] == ["allocator_service"]
    metadata = alloc.get("metadata", {})
    assert metadata.get("component_type") == "service"
    assert metadata.get("capability") == "allocator_service"
    assert metadata.get("description") == "Allocator service."


def test_component_workflow_synthesis_excludes_directories(tmp_path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_module(repo / "tests" / "ignored_service.py", '"""Ignored."""\n')
    _write_module(repo / "observability" / "alert_router.py", '"""Alert router."""\n')

    specs = discover_component_workflows(base_path=repo)
    ids = {spec["workflow_id"] for spec in specs}
    assert "component::observability.alert_router" in ids
    assert all("ignored_service" not in workflow_id for workflow_id in ids)
