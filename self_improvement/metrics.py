"""Metrics helpers for the self-improvement package."""
from __future__ import annotations

import argparse
import ast
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import yaml

try:  # pragma: no cover - radon is an optional dependency
    from radon.complexity import cc_visit
    from radon.metrics import mi_visit
except ImportError:  # pragma: no cover
    cc_visit = mi_visit = None

from ..sandbox_settings import SandboxSettings

logger = logging.getLogger(__name__)

_SKIP_DIRS = {
    ".git",
    "bin",
    "build",
    "dist",
    "node_modules",
    "site-packages",
    "venv",
    ".venv",
    "vendor",
    "third_party",
    "__pycache__",
}


def _collect_metrics(
    files: Iterable[Path],
    repo: Path,
) -> tuple[Dict[str, Dict[str, float]], int, float, int]:
    """Return per-file metrics, total complexity, avg maintainability and tests."""

    per_file: Dict[str, Dict[str, float]] = {}
    total_complexity = 0
    mi_total = 0.0
    mi_count = 0
    test_count = 0

    for file in files:
        try:
            rel_path = file.relative_to(repo)
        except ValueError:
            rel_path = file
        if any(part in _SKIP_DIRS for part in rel_path.parts):
            continue
        rel = rel_path.as_posix()
        name = file.name
        if rel.startswith("tests") or name.startswith("test_") or name.endswith("_test.py"):
            test_count += 1
        try:
            code = file.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Failed to read %s: %s", rel, exc)
            continue

        file_complexity = 0
        file_mi = 0.0

        if cc_visit and mi_visit:
            try:
                blocks = cc_visit(code)
                file_complexity = int(sum(b.complexity for b in blocks))
                file_mi = float(mi_visit(code, False))
            except Exception as exc:  # pragma: no cover - radon may raise various errors
                logger.warning("Radon metrics failed for %s: %s", rel, exc)
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
            except (SyntaxError, ValueError) as exc:
                logger.warning("AST metrics failed for %s: %s", rel, exc)

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
    except (OSError, yaml.YAMLError) as exc:  # pragma: no cover - best effort
        logger.warning("Failed to load baseline metrics: %s", exc)
        return {}


def _update_alignment_baseline(
    settings: SandboxSettings | None = None,
    files: Sequence[Path | str] | None = None,
) -> Dict[str, Any]:
    """Compute and persist current code metrics to the baseline file."""

    try:
        settings = settings or SandboxSettings()
        path_str = getattr(settings, "alignment_baseline_metrics_path", "")
        if not path_str:
            return {}
        repo = Path(SandboxSettings().sandbox_repo_path)
        if files is None:
            file_iter: Iterable[Path] = repo.rglob("*.py")
        else:
            tmp: list[Path] = []
            for f in files:
                p = Path(f)
                tmp.append(p if p.is_absolute() else repo / p)
            file_iter = tmp
        per_file, _, _, _ = _collect_metrics(file_iter, repo)

        baseline_path = Path(path_str)
        try:
            existing = yaml.safe_load(baseline_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            logger.warning("Failed to read existing baseline: %s", exc)
            existing = {}

        if files is None:
            files_data = per_file
        else:
            files_data = existing.get("files", {})
            files_data.update(per_file)

        total_complexity = sum(f["complexity"] for f in files_data.values())
        total_mi = sum(f["maintainability"] for f in files_data.values())
        avg_mi = total_mi / len(files_data) if files_data else 0.0
        test_count = sum(
            1
            for path in files_data
            if path.startswith("tests")
            or Path(path).name.startswith("test_")
            or Path(path).name.endswith("_test.py")
        )

        data: Dict[str, Any] = {
            "tests": test_count,
            "complexity": total_complexity,
            "maintainability": avg_mi,
            "files": files_data,
        }
        baseline_path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return data
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Failed to update baseline: %s", exc)
        return {}


def main(argv: Iterable[str] | None = None) -> None:
    """CLI entry point for updating or displaying baseline metrics."""

    parser = argparse.ArgumentParser(description="Self-improvement metrics utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)
    update = sub.add_parser("update", help="Recalculate and store baseline metrics")
    update.add_argument("files", nargs="*", help="Specific files to update")
    sub.add_parser("show", help="Display stored baseline metrics")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "update":
        metrics = _update_alignment_baseline(files=args.files)
        print(yaml.safe_dump(metrics))
    elif args.cmd == "show":
        print(yaml.safe_dump(get_alignment_metrics()))


__all__ = ["_update_alignment_baseline", "get_alignment_metrics", "main"]


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
