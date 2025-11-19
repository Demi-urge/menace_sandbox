from __future__ import annotations

import ast
import importlib
import logging
import sys
from pathlib import Path
from typing import Any, Callable, Mapping

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _load_discover_repo_workflows() -> tuple[Callable[..., Any], list[Path]]:
    source_path = _repo_root() / "start_autonomous_sandbox.py"
    source_text = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source_text, filename=str(source_path))

    func_source: str | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_discover_repo_workflows":
            func_source = ast.get_source_segment(source_text, node)
            break
    if func_source is None:  # pragma: no cover - guard against refactors
        raise AssertionError("_discover_repo_workflows definition missing")

    called_with: list[Path] = []

    def _fake_discover_workflow_specs(*, base_path: Path, logger: logging.Logger):
        called_with.append(Path(base_path))
        return []

    module_globals: dict[str, Any] = {
        "__name__": "start_autonomous_sandbox_for_test",
        "__file__": str(source_path),
        "Path": Path,
        "Mapping": Mapping,
        "Any": Any,
        "log_record": lambda **_: {},
        "discover_workflow_specs": _fake_discover_workflow_specs,
    }
    exec(func_source, module_globals)
    return module_globals["_discover_repo_workflows"], called_with


def test_bot_discovery_flat_import(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = _repo_root()

    for name in list(sys.modules):
        if name == "menace_sandbox" or name.startswith("menace_sandbox."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delitem(sys.modules, "bot_discovery", raising=False)

    monkeypatch.setattr(sys, "path", [str(repo_root)])

    module = importlib.import_module("bot_discovery")
    iterator = module._iter_bot_modules(repo_root)
    first = next(iterator, None)
    assert first is not None
    assert first.name.endswith("_bot.py")


def test_discover_repo_workflows_emits_bot_entries() -> None:
    discover_repo_workflows, discover_calls = _load_discover_repo_workflows()

    repo_root = _repo_root()

    specs = discover_repo_workflows(
        logger=logging.getLogger(__name__),
        base_path=repo_root,
    )

    assert discover_calls == [repo_root]
    assert any(spec.get("source") == "bot_discovery" for spec in specs)
