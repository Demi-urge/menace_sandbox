import types
import logging
import threading
import contextlib
from tests.test_self_debugger_sandbox import sds, DummyTelem, DummyEngine, DummyBuilder


def test_z_score_prefers_better_patch(tmp_path):
    class DummyPatchDB:
        def __init__(self, path):
            self.path = path
            self.records = []

        def add(self, rec):
            self.records.append(rec)

        def filter(self, filename=None, reverted=None):
            return list(self.records)

        def get_weights(self):
            return None

    patch_db = DummyPatchDB(tmp_path / "p.db")
    patch_db.add(
        types.SimpleNamespace(
            ts="1",
            errors_before=10,
            errors_after=8,
            roi_delta=0.1,
            complexity_delta=0.05,
            synergy_roi=0.0,
            synergy_efficiency=0.0,
            coverage_delta=0.1,
        )
    )
    patch_db.add(
        types.SimpleNamespace(
            ts="2",
            errors_before=11,
            errors_after=9,
            roi_delta=0.1,
            complexity_delta=0.05,
            synergy_roi=0.0,
            synergy_efficiency=0.0,
            coverage_delta=0.1,
        )
    )

    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(),
        DummyEngine(),
        context_builder=DummyBuilder(),
        score_weights=(1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
        weight_update_interval=0.0,
    )
    dbg._score_db = patch_db
    dbg._history_conn = None
    dbg._load_history_stats = lambda limit=50: None
    dbg._update_score_weights(patch_db)

    lower = dbg._composite_score(0.12, 2.0, 0.12, 0.0, 0.0, 0.05, 0.0)
    higher = dbg._composite_score(0.25, 5.0, 0.25, 0.0, 0.0, 0.02, 0.0)
    assert higher > lower


def test_complexity_penalizes_score():
    dbg = sds.SelfDebuggerSandbox(
        DummyTelem(),
        DummyEngine(),
        context_builder=DummyBuilder(),
        score_weights=(1.0, 1.0, 1.0, 1.0, 0.0, 0.0),
    )
    dbg._update_score_weights = lambda *a, **k: None
    dbg._metric_stats = {
        "coverage": (0.1, 0.05),
        "error": (2.0, 1.0),
        "roi": (0.1, 0.05),
        "complexity": (0.05, 0.02),
        "synergy_roi": (0.0, 1.0),
        "synergy_efficiency": (0.0, 1.0),
        "synergy_resilience": (0.0, 1.0),
        "synergy_antifragility": (0.0, 1.0),
    }

    good = dbg._composite_score(0.2, 3.0, 0.15, 0.0, 0.0, 0.04, 0.0)
    bad = dbg._composite_score(0.2, 3.0, 0.15, 0.0, 0.0, 0.2, 0.0)


def test_entropy_penalizes_score():
    dbg = sds.SelfDebuggerSandbox.__new__(sds.SelfDebuggerSandbox)
    dbg.context_builder = DummyBuilder()
    dbg.score_weights = (1.0, 1.0, 1.0, 1.0, 0.0, 0.0)
    dbg._baseline_tracker = types.SimpleNamespace(
        update=lambda s: (0.0, 0.0), stats=lambda: (0.0, 0.0)
    )
    dbg._metric_stats = {
        "coverage": (0.1, 0.05),
        "error": (2.0, 1.0),
        "roi": (0.1, 0.05),
        "complexity": (0.05, 0.02),
        "synergy_roi": (0.0, 1.0),
        "synergy_efficiency": (0.0, 1.0),
        "synergy_resilience": (0.0, 1.0),
        "synergy_antifragility": (0.0, 1.0),
    }
    dbg.logger = logging.getLogger("test")
    dbg.engine = types.SimpleNamespace()
    dbg._score_db = None
    dbg._db_lock = threading.Lock()
    dbg._history_db = lambda: contextlib.nullcontext(None)
    dbg._update_score_weights = lambda *a, **k: None

    low = dbg._composite_score(0.2, 3.0, 0.15, 0.0, 0.0, 0.04, 0.0)
    high = dbg._composite_score(0.2, 3.0, 0.15, 0.0, 0.0, 0.04, 0.5)
    assert low > high
