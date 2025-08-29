"""Metrics helpers for the self-improvement package."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable

from ..sandbox_settings import SandboxSettings


def _update_alignment_baseline(settings: SandboxSettings | None = None) -> None:
    """Write current test counts and complexity scores to baseline metrics file."""
    try:
        settings = settings or SandboxSettings()
        path_str = getattr(settings, "alignment_baseline_metrics_path", "")
        if not path_str:
            return
        repo = Path(SandboxSettings().sandbox_repo_path)
        test_count = 0
        total_complexity = 0
        for file in repo.rglob("*.py"):
            rel = file.relative_to(repo)
            name = rel.name
            rel_posix = rel.as_posix()
            if (
                rel_posix.startswith("tests")
                or name.startswith("test_")
                or name.endswith("_test.py")
            ):
                test_count += 1
            try:
                code = file.read_text(encoding="utf-8")
                tree = ast.parse(code)
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    score = 1
                    for sub in ast.walk(node):
                        if isinstance(
                            sub,
                            (
                                ast.If,
                                ast.For,
                                ast.While,
                                ast.Try,
                                ast.With,
                                ast.BoolOp,
                                ast.IfExp,
                            ),
                        ):
                            score += 1
                    total_complexity += score
        Path(path_str).write_text(
            f"tests={test_count}\ncomplexity={total_complexity}\n",
            encoding="utf-8",
        )
    except Exception:  # pragma: no cover - best effort
        return


__all__ = ["_update_alignment_baseline"]
