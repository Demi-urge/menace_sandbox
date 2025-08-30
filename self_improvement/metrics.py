"""Metrics helpers for the self-improvement package."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

try:  # pragma: no cover - radon is an optional dependency
    from radon.complexity import cc_visit
    from radon.metrics import mi_visit
except Exception:  # pragma: no cover
    cc_visit = mi_visit = None

from ..sandbox_settings import SandboxSettings


def _collect_metrics(files: Iterable[Path], repo: Path) -> tuple[Dict[str, Dict[str, float]], int, float, int]:
    """Return per-file metrics, total complexity, avg maintainability and tests."""

    per_file: Dict[str, Dict[str, float]] = {}
    total_complexity = 0
    mi_total = 0.0
    mi_count = 0
    test_count = 0

    for file in files:
        rel = file.relative_to(repo).as_posix()
        name = file.name
        if rel.startswith("tests") or name.startswith("test_") or name.endswith("_test.py"):
            test_count += 1
        try:
            code = file.read_text(encoding="utf-8")
        except Exception:
            continue

        file_complexity = 0
        file_mi = 0.0

        if cc_visit and mi_visit:
            try:
                blocks = cc_visit(code)
                file_complexity = int(sum(b.complexity for b in blocks))
                file_mi = float(mi_visit(code, False))
            except Exception:
                pass
        else:  # fallback to AST-based estimation
            try:
                tree = ast.parse(code)
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
                        file_complexity += score
                file_mi = 100.0
            except Exception:
                pass

        per_file[rel] = {"complexity": file_complexity, "maintainability": file_mi}
        total_complexity += file_complexity
        mi_total += file_mi
        mi_count += 1

    avg_mi = mi_total / mi_count if mi_count else 0.0
    return per_file, total_complexity, avg_mi, test_count


def get_alignment_metrics(settings: SandboxSettings | None = None) -> Dict[str, Any]:
    """Return stored baseline metrics if available."""

    try:
        settings = settings or SandboxSettings()
        path_str = getattr(settings, "alignment_baseline_metrics_path", "")
        if not path_str:
            return {}
        return yaml.safe_load(Path(path_str).read_text()) or {}
    except Exception:  # pragma: no cover - best effort
        return {}


def _update_alignment_baseline(settings: SandboxSettings | None = None) -> Dict[str, Any]:
    """Compute and persist current code metrics to the baseline file."""

    try:
        settings = settings or SandboxSettings()
        path_str = getattr(settings, "alignment_baseline_metrics_path", "")
        if not path_str:
            return {}
        repo = Path(SandboxSettings().sandbox_repo_path)
        per_file, total_complexity, avg_mi, test_count = _collect_metrics(
            repo.rglob("*.py"), repo
        )
        data: Dict[str, Any] = {
            "tests": test_count,
            "complexity": total_complexity,
            "maintainability": avg_mi,
            "files": per_file,
        }
        Path(path_str).write_text(yaml.safe_dump(data), encoding="utf-8")
        return data
    except Exception:  # pragma: no cover - best effort
        return {}


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entry point for updating or displaying baseline metrics."""

    parser = argparse.ArgumentParser(description="Self-improvement metrics utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("update", help="Recalculate and store baseline metrics")
    sub.add_parser("show", help="Display stored baseline metrics")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "update":
        metrics = _update_alignment_baseline()
        print(yaml.safe_dump(metrics))
    elif args.cmd == "show":
        print(yaml.safe_dump(get_alignment_metrics()))


__all__ = ["_update_alignment_baseline", "get_alignment_metrics", "main"]


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
