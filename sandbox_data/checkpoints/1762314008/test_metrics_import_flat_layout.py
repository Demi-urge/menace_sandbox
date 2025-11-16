"""Ensure ``self_improvement.metrics`` supports flat import layouts."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def test_metrics_import_flat_layout(monkeypatch):
    """Import the metrics module with only the repo root on ``sys.path``."""

    repo_root = Path(__file__).resolve().parents[2]

    # Ensure an import happens from the repository root rather than the package
    # layout produced by ``menace_sandbox`` being present in ``sys.modules``.
    monkeypatch.delitem(sys.modules, "menace_sandbox", raising=False)
    monkeypatch.delitem(sys.modules, "self_improvement", raising=False)
    monkeypatch.delitem(sys.modules, "self_improvement.metrics", raising=False)

    # Restrict ``sys.path`` to only the repository root so absolute imports are
    # exercised in environments that execute from the root directory.
    monkeypatch.setattr(sys, "path", [str(repo_root)])

    module = importlib.import_module("self_improvement.metrics")

    assert module is not None
    assert hasattr(module, "SandboxSettings")
    assert hasattr(module, "resolve_path")
