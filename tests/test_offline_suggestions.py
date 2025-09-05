import os
import sys
import types

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")


def _stub(name, **attrs):
    mod = types.SimpleNamespace(**attrs)
    sys.modules.setdefault(name, mod)
    return mod


_stub("memory_logging", log_with_tags=lambda *a, **k: None)
_stub("memory_aware_gpt_client", ask_with_memory=lambda *a, **k: "")
_stub("vector_service", Retriever=object, FallbackResult=object, EmbeddableDBMixin=object)
_stub("foresight_tracker", ForesightTracker=object)
_stub("db_router", DBRouter=object, GLOBAL_ROUTER=None, init_db_router=lambda *a, **k: None)
_stub("analytics", adaptive_roi_model=None)
_stub("adaptive_roi_predictor", load_training_data=lambda *a, **k: None)
_stub(
    "metrics_exporter",
    orphan_modules_reintroduced_total=None,
    orphan_modules_tested_total=None,
    orphan_modules_failed_total=None,
    orphan_modules_redundant_total=None,
    orphan_modules_legacy_total=None,
    orphan_modules_reclassified_total=None,
)
_stub(
    "relevancy_radar",
    RelevancyRadar=object,
    track_usage=lambda *a, **k: None,
    evaluate_final_contribution=lambda *a, **k: None,
    record_output_impact=lambda *a, **k: None,
    radar=types.SimpleNamespace(track=lambda fn: fn),
)
_stub("sandbox_runner.resource_tuner", ResourceTuner=object)
_stub(
    "sandbox_runner.orphan_discovery",
    append_orphan_cache=lambda *a, **k: None,
    append_orphan_classifications=lambda *a, **k: None,
    prune_orphan_cache=lambda *a, **k: None,
    load_orphan_cache=lambda *a, **k: None,
    load_orphan_traces=lambda *a, **k: None,
    append_orphan_traces=lambda *a, **k: None,
    discover_recursive_orphans=lambda *a, **k: None,
)

from menace.sandbox_runner.cycle import _choose_suggestion


class _DummyDB:
    def __init__(self, path):
        self.data = {}

    def add(self, rec):
        self.data.setdefault(rec.module, rec.description)

    def best_match(self, module):
        return self.data.get(module)


class _Rec:
    def __init__(self, module: str, description: str, safe: bool = True):
        self.module = module
        self.description = description
        self.safe = safe


_stub("patch_suggestion_db", PatchSuggestionDB=_DummyDB, SuggestionRecord=_Rec)
import patch_suggestion_db as psdb


def test_choose_suggestion_uses_db(tmp_path):
    db = psdb.PatchSuggestionDB(tmp_path / "s.db")
    db.add(psdb.SuggestionRecord(module="mod.py", description="add logging"))  # path-ignore
    db.add(psdb.SuggestionRecord(module="mod.py", description="add logging"))  # path-ignore
    db.add(psdb.SuggestionRecord(module="mod.py", description="improve error"))  # path-ignore
    ctx = types.SimpleNamespace(suggestion_cache={"mod.py": "fallback"}, suggestion_db=db)  # path-ignore
    assert _choose_suggestion(ctx, "mod.py") == "add logging"  # path-ignore


def test_choose_suggestion_fallback():
    ctx = types.SimpleNamespace(suggestion_cache={"mod.py": "fallback"}, suggestion_db=None)  # path-ignore
    assert _choose_suggestion(ctx, "mod.py") == "fallback"  # path-ignore

