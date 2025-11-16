import contextlib
import sqlite3
import sys
import types
from pathlib import Path

import pytest
import textwrap


class _CtxConnWrapper:
    """Minimal wrapper exposing a ``_connect`` context manager."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    @contextlib.contextmanager
    def _connect(self):  # pragma: no cover - simple helper
        yield self.conn


# Provide stubs so the module under test can import ``CodeDB`` and
# ``PatchHistoryDB`` without pulling in the full implementations.
sys.modules['code_database'] = types.SimpleNamespace(
    CodeDB=_CtxConnWrapper, PatchHistoryDB=_CtxConnWrapper
)

from enhancement_classifier import EnhancementClassifier, EnhancementSuggestion


def test_scan_repo_generates_suggestions() -> None:
    code_conn = sqlite3.connect(":memory:")
    code_conn.execute("CREATE TABLE code (id INTEGER PRIMARY KEY)")
    code_conn.execute("INSERT INTO code (id) VALUES (1)")

    patch_conn = sqlite3.connect(":memory:")
    patch_conn.execute(
        """
        CREATE TABLE patch_history (
            code_id INTEGER,
            filename TEXT,
            roi_delta REAL,
            errors_before INTEGER,
            errors_after INTEGER,
            complexity_delta REAL
        )
        """
    )
    patch_conn.executemany(
        "INSERT INTO patch_history VALUES (1, 'mod.py', ?, ?, ?, ?)",  # path-ignore
        [
            (-0.1, 1, 3, 0.5),
            (-0.2, 2, 5, 0.3),
            (-0.05, 1, 2, 0.4),
        ],
    )

    config_path = Path(__file__).resolve().parent.parent / "enhancement_classifier_config.json"
    classifier = EnhancementClassifier(
        code_db=_CtxConnWrapper(code_conn),
        patch_db=_CtxConnWrapper(patch_conn),
        config_path=str(config_path),
    )

    suggestions = list(classifier.scan_repo())
    assert len(suggestions) == 1
    sugg = suggestions[0]
    assert isinstance(sugg, EnhancementSuggestion)
    assert sugg.path == "mod.py"  # path-ignore
    # Frequency=3, ROI=-0.116..., errors=2, complexity=0.4 -> score ~5.52
    assert sugg.score > 5.5
    assert "module mod.py refactored 3 times" in sugg.rationale  # path-ignore
    assert "ROI dropped 0.12%" in sugg.rationale
    assert "errors increased by 2.00" in sugg.rationale
    assert classifier.thresholds["min_patches"] == 3


def test_ast_metrics_and_scoring(tmp_path: Path) -> None:
    code_src = textwrap.dedent('''
    def a(x):
        if x > 1:
            return 1
        else:
            return 2

    def b(x):
        if x > 1:
            return 1
        else:
            return 2

    def long_function():
        """long"""
    {}''').format("\n    pass" * 60)

    code_conn = sqlite3.connect(":memory:")
    code_conn.execute("CREATE TABLE code (id INTEGER PRIMARY KEY, code TEXT)")
    code_conn.execute("INSERT INTO code (id, code) VALUES (1, ?)", (code_src,))

    patch_conn = sqlite3.connect(":memory:")
    patch_conn.execute(
        """
        CREATE TABLE patch_history (
            code_id INTEGER,
            filename TEXT,
            roi_delta REAL,
            errors_before INTEGER,
            errors_after INTEGER,
            complexity_delta REAL
        )
        """
    )
    patch_conn.executemany(
        "INSERT INTO patch_history VALUES (1, 'mod.py', ?, ?, ?, ?)",  # path-ignore
        [(-1.0, 0, 1, 0.5), (-1.0, 0, 1, 0.5), (-1.0, 0, 1, 0.5)],
    )

    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        """{\n  \"weights\": {\n    \"frequency\": 1.0,\n    \"roi\": 1.0,\n    \"errors\": 1.0,\n    \"complexity\": 1.0,\n    \"cyclomatic\": 1.0,\n    \"duplication\": 1.0,\n    \"length\": 1.0,\n    \"anti\": 1.0\n  },\n  \"thresholds\": {\n    \"min_patches\": 5,\n    \"roi_cutoff\": -0.5,\n    \"complexity_delta\": 0.0\n  }\n}\n"""
    )

    classifier = EnhancementClassifier(
        code_db=_CtxConnWrapper(code_conn),
        patch_db=_CtxConnWrapper(patch_conn),
        config_path=str(cfg),
    )

    metrics = classifier._gather_metrics(1)
    assert metrics is not None
    (
        filename,
        patches,
        avg_roi,
        avg_errors,
        avg_complexity,
        avg_cc,
        dup_ratio,
        long_funcs,
        neg_roi_ratio,
        error_prone_ratio,
        func_churn,
        style_violations,
        refactor_count,
        anti_pattern_hits,
        notes,
        roi_volatility,
        raroi,
    ) = metrics
    assert filename == "mod.py"  # path-ignore
    assert dup_ratio == pytest.approx(1 / 3)
    assert long_funcs == 1
    assert avg_cc > 1.0
    assert notes

    suggestions = list(classifier.scan_repo())
    assert suggestions and suggestions[0].path == "mod.py"  # path-ignore
    assert classifier.thresholds["min_patches"] == 5
    expected = (
        classifier.weights["frequency"] * patches
        + classifier.weights["roi"] * (-avg_roi)
        + classifier.weights["raroi"] * (-raroi)
        + classifier.weights["errors"] * avg_errors
        + classifier.weights["complexity"] * avg_complexity
        + classifier.weights["cyclomatic"] * avg_cc
        + classifier.weights["duplication"] * dup_ratio
        + classifier.weights["length"] * long_funcs
        + classifier.weights["history"] * (neg_roi_ratio + error_prone_ratio) * (avg_cc + dup_ratio)
        + classifier.weights["churn"] * func_churn
        + classifier.weights["style"] * style_violations
        + classifier.weights["refactor"] * refactor_count
        + classifier.weights["anti_log"] * anti_pattern_hits
    )
    assert suggestions[0].score == pytest.approx(expected)
    assert "duplicated" in suggestions[0].rationale
    assert "long" in suggestions[0].rationale


def test_env_overrides(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    code_conn = sqlite3.connect(":memory:")
    code_conn.execute("CREATE TABLE code (id INTEGER PRIMARY KEY)")
    patch_conn = sqlite3.connect(":memory:")
    patch_conn.execute(
        "CREATE TABLE patch_history (code_id INTEGER, filename TEXT, roi_delta REAL, errors_before INTEGER, errors_after INTEGER, complexity_delta REAL)"
    )
    cfg = tmp_path / "cfg.json"
    cfg.write_text("{}")
    monkeypatch.setenv("ENHANCEMENT_WEIGHT_FREQUENCY", "2.5")
    monkeypatch.setenv("ENHANCEMENT_THRESHOLD_MIN_PATCHES", "1.5")
    classifier = EnhancementClassifier(
        code_db=_CtxConnWrapper(code_conn),
        patch_db=_CtxConnWrapper(patch_conn),
        config_path=str(cfg),
    )
    assert classifier.weights["frequency"] == 2.5
    assert classifier.thresholds["min_patches"] == 1.5

