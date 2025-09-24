from __future__ import annotations

import importlib
import sys
from pathlib import Path


def _resolve(path: str) -> Path | None:
    try:
        return Path(path).resolve()
    except (OSError, RuntimeError):
        return None


def test_flat_import_constructs_manager(monkeypatch):
    """``import shared_gpt_memory`` should succeed when run as a script."""

    for name in (
        "menace_sandbox.shared_gpt_memory",
        "shared_gpt_memory",
        "menace_sandbox.shared_knowledge_module",
        "shared_knowledge_module",
        "menace_sandbox.gpt_memory",
        "gpt_memory",
        "menace_sandbox",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)

    package_root = Path(__file__).resolve().parents[1]
    parent_dir = package_root.parent.resolve()

    new_sys_path = [str(package_root)]
    for entry in sys.path:
        resolved = _resolve(entry)
        if resolved is None or resolved != parent_dir:
            new_sys_path.append(entry)

    monkeypatch.setattr(sys, "path", new_sys_path)

    module = importlib.import_module("shared_gpt_memory")

    assert module.GPT_MEMORY_MANAGER is not None
    assert module.GPT_MEMORY_MANAGER.__class__.__name__ == "GPTMemoryManager"
    assert sys.modules["shared_gpt_memory"] is module
    assert sys.modules["menace_sandbox.shared_gpt_memory"] is module
