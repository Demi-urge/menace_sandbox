import contextlib
import sqlite3
import contextlib
import sqlite3
import sys
import types
from pathlib import Path


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
        "INSERT INTO patch_history VALUES (1, 'mod.py', ?, ?, ?, ?)",
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
    assert sugg.path == "mod.py"
    # Frequency=3, ROI=-0.116..., errors=2 -> score ~5.12
    assert sugg.score > 5
    assert "avg ROI delta -0.12" in sugg.rationale

