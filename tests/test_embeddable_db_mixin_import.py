"""Regression tests for the embeddable DB mixin bootstrap."""

from __future__ import annotations

import runpy
from pathlib import Path


def test_embeddable_db_mixin_run_as_script() -> None:
    module_path = Path(__file__).resolve().parents[1] / "embeddable_db_mixin.py"
    namespace = runpy.run_path(str(module_path))
    func = namespace.get("log_embedding_metrics")
    assert callable(func)
