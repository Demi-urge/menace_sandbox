"""Metrics helpers for the self-improvement package."""
from __future__ import annotations

import argparse
import ast
import io
import keyword
import logging
import os
import math
import tokenize
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import yaml
from statistics import fmean

try:  # pragma: no cover - radon is an optional dependency
    from radon.complexity import cc_visit
    from radon.metrics import mi_visit
except ImportError:  # pragma: no cover - fall back to AST-based metrics
    cc_visit = mi_visit = None

from ..sandbox_settings import SandboxSettings
from ..logging_utils import setup_logging

logger = logging.getLogger(__name__)


def _collect_metrics(
    files: Iterable[Path],
    repo: Path,
    settings: SandboxSettings | None = None,
) -> tuple[Dict[str, Dict[str, float]], int, float, int, float, float]:
    """Return per-file metrics and aggregated summaries.

    The helper now additionally computes token level entropy and diversity for
    each file.  The return tuple has been extended with the average entropy and
    diversity across processed files.
    """

    if settings is None or not hasattr(settings, "metrics_skip_dirs"):
        skip_dirs = set(SandboxSettings().metrics_skip_dirs)
    else:
        skip_dirs = set(settings.metrics_skip_dirs)

    per_file: Dict[str, Dict[str, float]] = {}
    total_complexity = 0
    mi_total = 0.0
    mi_count = 0
    test_count = 0
    entropy_total = 0.0
    diversity_total = 0.0

    for file in files:
        try:
            rel_path = file.relative_to(repo)
        except ValueError:
            rel_path = file
        if any(part in skip_dirs for part in rel_path.parts):
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

        # Token statistics used for entropy and maintainability metrics
        token_counts: Counter[str] = Counter()
        ops: set[str] = set()
        operands: set[str] = set()
        N1 = N2 = 0
        sloc_lines: set[int] = set()
        try:
            for tok in tokenize.generate_tokens(io.StringIO(code).readline):
                if tok.type in (
                    tokenize.NL,
                    tokenize.NEWLINE,
                    tokenize.INDENT,
                    tokenize.DEDENT,
                    tokenize.COMMENT,
                    tokenize.ENCODING,
                ):
                    continue
                sloc_lines.add(tok.start[0])
                token_counts[tok.string] += 1
                if tok.type == tokenize.OP or (
                    tok.type == tokenize.NAME and tok.string in keyword.kwlist
                ):
                    ops.add(tok.string)
                    N1 += 1
                elif tok.type in (
                    tokenize.NAME,
                    tokenize.NUMBER,
                    tokenize.STRING,
                ):
                    operands.add(tok.string)
                    N2 += 1
        except tokenize.TokenError as exc:  # pragma: no cover - best effort
            logger.warning("Tokenisation failed for %s: %s", rel, exc)

        total_tokens = sum(token_counts.values())
        if total_tokens:
            token_entropy = -sum(
                (count / total_tokens) * math.log2(count / total_tokens)
                for count in token_counts.values()
            )
            token_diversity = len(token_counts) / total_tokens
        else:
            token_entropy = 0.0
            token_diversity = 0.0

        if cc_visit and mi_visit:
            try:
                blocks = cc_visit(code)
                file_complexity = int(sum(b.complexity for b in blocks))
                file_mi = float(mi_visit(code, False))
            except Exception as exc:  # pragma: no cover - radon may raise various errors
                logger.warning("Radon metrics failed for %s: %s", rel, exc)
        else:  # fallback to AST-based estimation using MI formula
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

                n1, n2 = len(ops), len(operands)
                n = n1 + n2
                N = N1 + N2
                volume = N * math.log2(n) if n else 0.0
                sloc = len(sloc_lines)
                if volume > 0 and sloc > 0:
                    file_mi = max(
                        0.0,
                        (
                            171
                            - 5.2 * math.log(volume)
                            - 0.23 * file_complexity
                            - 16.2 * math.log(sloc)
                        )
                        * 100
                        / 171,
                    )
                else:
                    file_mi = 100.0
            except (SyntaxError, ValueError) as exc:
                logger.warning("AST metrics failed for %s: %s", rel, exc)

        per_file[rel] = {
            "complexity": file_complexity,
            "maintainability": file_mi,
            "token_entropy": token_entropy,
            "token_diversity": token_diversity,
        }
        total_complexity += file_complexity
        mi_total += file_mi
        mi_count += 1
        entropy_total += token_entropy
        diversity_total += token_diversity

    avg_mi = mi_total / mi_count if mi_count else 0.0
    avg_entropy = entropy_total / mi_count if mi_count else 0.0
    avg_diversity = diversity_total / mi_count if mi_count else 0.0
    return per_file, total_complexity, avg_mi, test_count, avg_entropy, avg_diversity


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


def compute_entropy_metrics(
    files: Sequence[Path | str],
    settings: SandboxSettings | None = None,
) -> tuple[float, float]:
    """Return ``(code_diversity, token_complexity)`` for ``files``.

    ``code_diversity`` is the average token entropy across the supplied files
    while ``token_complexity`` reflects the average cyclomatic complexity.  Both
    values are normalised so callers can compare entropy trends across
    self‑improvement cycles.
    """

    repo = Path(SandboxSettings().sandbox_repo_path)
    file_iter: list[Path] = []
    for f in files:
        p = Path(f)
        file_iter.append(p if p.is_absolute() else repo / p)
    per_file, total_complexity, _, _, avg_entropy, _ = _collect_metrics(
        file_iter, repo, settings=settings
    )
    avg_complexity = total_complexity / len(per_file) if per_file else 0.0
    return avg_entropy, avg_complexity


def compute_code_entropy(
    files: Sequence[Path | str],
    settings: SandboxSettings | None = None,
) -> float:
    """Return combined code entropy derived from complexity and maintainability.

    The helper aggregates cyclomatic complexity and maintainability index for
    ``files`` and returns the mean of average complexity and the complement of
    maintainability.  Higher values indicate a more disorderly code base.
    """

    repo = Path(SandboxSettings().sandbox_repo_path)
    file_iter: list[Path] = []
    for f in files:
        p = Path(f)
        file_iter.append(p if p.is_absolute() else repo / p)
    per_file, total_complexity, avg_mi, _, _, _ = _collect_metrics(
        file_iter, repo, settings=settings
    )
    if not per_file:
        return 0.0
    avg_complexity = total_complexity / len(per_file)
    mi_entropy = 100.0 - avg_mi
    return fmean([avg_complexity, mi_entropy])


def compute_entropy_delta(
    code_diversity: float,
    token_complexity: float,
    *,
    settings: SandboxSettings | None = None,
) -> tuple[float, float]:
    """Return ``(delta, moving_avg)`` for the current entropy metrics.

    The moving average is derived from the ``entropy_history`` stored in the
    alignment baseline. ``delta`` represents the deviation of the current
    entropy from this moving average.
    """

    history = get_alignment_metrics(settings).get("entropy_history") or []
    entropies: list[float] = []
    for entry in history:
        try:
            if isinstance(entry, dict):
                cd = float(entry.get("code_diversity", 0.0))
                tc = float(entry.get("token_complexity", 0.0))
                entropies.append(fmean([cd, tc]))
            else:
                entropies.append(float(entry))
        except Exception:  # pragma: no cover - best effort
            continue
    moving_avg = fmean(entropies) if entropies else 0.0
    current = fmean([float(code_diversity), float(token_complexity)])
    return current - moving_avg, moving_avg


def record_entropy(
    code_diversity: float,
    token_complexity: float,
    *,
    roi: float | None = None,
    settings: SandboxSettings | None = None,
) -> None:
    """Append an entropy record to the baseline metrics file.

    Parameters
    ----------
    code_diversity:
        Shannon entropy of code changes for the cycle.
    token_complexity:
        Token level complexity observed in the cycle.
    roi:
        Optional ROI associated with the cycle so entropy history can be
        correlated with investment data.
    settings:
        Sandbox settings providing the baseline metrics path. When omitted the
        default :class:`SandboxSettings` is used.
    """

    try:
        settings = settings or SandboxSettings()
        path_str = getattr(settings, "alignment_baseline_metrics_path", "")
        if not path_str:
            return
        baseline_path = Path(path_str)
        try:
            data = yaml.safe_load(baseline_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            data = {}

        history = list(data.get("entropy_history", []))
        history.append(
            {
                "code_diversity": float(code_diversity),
                "token_complexity": float(token_complexity),
                "roi": float(roi) if roi is not None else None,
                "entropy": fmean([float(code_diversity), float(token_complexity)]),
            }
        )
        data["entropy_history"] = history
        baseline_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Failed to record entropy: %s", exc)


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
        per_file, _, _, _, _, _ = _collect_metrics(file_iter, repo, settings=settings)

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
        if "entropy_history" in existing:
            data["entropy_history"] = existing["entropy_history"]
        baseline_path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return data
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("Failed to update baseline: %s", exc)
        return {}


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for updating or displaying baseline metrics."""

    settings = SandboxSettings()
    os.environ["SANDBOX_CENTRAL_LOGGING"] = (
        "1" if settings.sandbox_central_logging else "0"
    )
    setup_logging()

    parser = argparse.ArgumentParser(description="Self-improvement metrics utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)
    update = sub.add_parser("update", help="Recalculate and store baseline metrics")
    update.add_argument("files", nargs="*", help="Specific files to update")
    sub.add_parser("show", help="Display stored baseline metrics")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "update":
        metrics = _update_alignment_baseline(files=args.files)
        logger.info(yaml.safe_dump(metrics))
    elif args.cmd == "show":
        logger.info(yaml.safe_dump(get_alignment_metrics()))


__all__ = [
    "_update_alignment_baseline",
    "get_alignment_metrics",
    "compute_entropy_metrics",
    "compute_code_entropy",
    "compute_entropy_delta",
    "record_entropy",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
