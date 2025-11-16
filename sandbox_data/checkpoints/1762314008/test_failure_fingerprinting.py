import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# dummy vector service used for fingerprint store and manager tests


class DummyVectorStore:
    def __init__(self):
        self.records: dict[str, list[float]] = {}
        self.meta: dict[str, dict] = {}

    def add(self, kind, record_id, vector, *, origin_db=None, metadata=None):
        self.records[record_id] = list(vector)
        self.meta[record_id] = metadata or {}

    def query(self, vector, top_k=5):
        # return all records with dummy distance
        return [(rid, 0.0) for rid in self.records]


class DummyVectorService:
    def __init__(self):
        self.vector_store = DummyVectorStore()

    def vectorise(self, kind: str, record: dict) -> list[float]:
        # deterministic non-zero embedding so cosine similarity works
        return [1.0, 0.0]


# expose dummy vector service before importing store
vec_module = types.ModuleType("vector_service")
vec_module.SharedVectorService = DummyVectorService
sys.modules.setdefault("vector_service", vec_module)

# stubs for modules that perform complex imports
pkg = types.ModuleType("menace_sandbox")
pkg.__path__ = [str(Path.cwd())]
pkg.RAISE_ERRORS = False
sys.modules.setdefault("menace_sandbox", pkg)
sys.modules.setdefault("menace", pkg)

cd_mod = types.ModuleType("menace_sandbox.config_discovery")
cd_mod.ConfigDiscovery = type("ConfigDiscovery", (), {})
sys.modules.setdefault("config_discovery", cd_mod)
sys.modules.setdefault("menace_sandbox.config_discovery", cd_mod)

# minimal prompt optimizer stub
po_mod = types.ModuleType("prompt_optimizer")
po_mod.PromptOptimizer = type(
    "PromptOptimizer",
    (),
    {
        "__init__": lambda self, *a, **k: setattr(
            self,
            "stats",
            {(
                "m",
                "a",
                "neutral",
                ("H",),
                "start",
                False,
                False,
                False,
            ): types.SimpleNamespace(success=0)},
        )
    },
)
sys.modules.setdefault("prompt_optimizer", po_mod)

from failure_fingerprint_store import FailureFingerprint, FailureFingerprintStore  # noqa: E402
from prompt_optimizer import PromptOptimizer  # noqa: E402


def make_store(tmp_path: Path) -> FailureFingerprintStore:
    svc = DummyVectorService()
    return FailureFingerprintStore(
        path=tmp_path / "fps.jsonl",
        vector_service=svc,
        similarity_threshold=0.9,
        compact_interval=0,
    )


# ---------------------------------------------------------------------------
# basic store behaviour

def test_log_writes_jsonl_and_embeddings(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    fp = FailureFingerprint("a.py", "f", "err", "trace", "prompt")  # path-ignore
    store.log(fp)
    # JSONL line written
    data = json.loads(store.path.read_text().strip())
    assert data["filename"] == "a.py"  # path-ignore
    # embedding persisted via vector store
    rid = store._id_for(fp)
    assert rid in store.vector_service.vector_store.records


def test_find_similar_above_threshold(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    fp1 = FailureFingerprint("a.py", "f", "err", "trace", "p1")  # path-ignore
    store.log(fp1)
    fp2 = FailureFingerprint("b.py", "g", "err", "trace", "p2")  # path-ignore
    matches = store.find_similar(fp2, threshold=0.0)
    assert matches and matches[0].filename == "a.py"  # path-ignore


# ---------------------------------------------------------------------------
# run_patch similarity handling

@dataclass
class TestResult:
    stdout: str
    stderr: str
    success: bool
    duration: float = 0.0
    failure: dict | None = None


def _load_manager(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, initial_results: list[TestResult] | None = None
):
    """Prepare SelfCodingManager with heavily mocked dependencies."""

    # package stub to avoid running heavy __init__
    pkg = types.ModuleType("menace_sandbox")
    pkg.__path__ = [str(Path.cwd())]
    pkg.RAISE_ERRORS = False
    monkeypatch.setitem(sys.modules, "menace_sandbox", pkg)
    monkeypatch.setitem(sys.modules, "menace", pkg)

    # stub config_discovery to avoid relative import errors
    cd_mod = types.ModuleType("menace_sandbox.config_discovery")
    cd_mod.ConfigDiscovery = type("ConfigDiscovery", (), {})
    monkeypatch.setitem(sys.modules, "config_discovery", cd_mod)
    monkeypatch.setitem(sys.modules, "menace_sandbox.config_discovery", cd_mod)

    # stub sandbox_runner.test_harness with sequential results
    results: list[TestResult] = list(initial_results or [])

    def run_tests(repo, changed, backend=None):
        return results.pop(0) if results else TestResult("1 passed", "", True)

    th = types.ModuleType("menace_sandbox.sandbox_runner.test_harness")
    th.run_tests = run_tests
    th.TestHarnessResult = TestResult
    sr = types.ModuleType("menace_sandbox.sandbox_runner")
    sr.test_harness = th
    monkeypatch.setitem(sys.modules, "menace_sandbox.sandbox_runner", sr)
    monkeypatch.setitem(sys.modules, "menace_sandbox.sandbox_runner.test_harness", th)

    # simple engine capturing descriptions
    class Engine:
        def __init__(self):
            self.last_prompt_text = "prompt"
            self.calls: list[str] = []
            self.cognition_layer = types.SimpleNamespace(
                context_builder=types.SimpleNamespace(
                    query=lambda desc, exclude_tags=None: ("ctx", "sid")
                )
            )
            self.patch_suggestion_db = None

        def apply_patch(self, path, desc, **kwargs):
            self.calls.append(desc)
            return 1, False, None

    eng_mod = types.ModuleType("menace_sandbox.self_coding_engine")
    eng_mod.SelfCodingEngine = Engine
    monkeypatch.setitem(sys.modules, "menace_sandbox.self_coding_engine", eng_mod)

    # minimal pipeline and related stubs
    @dataclass
    class AutoRes:
        success: bool = True

    class Pipeline:
        def run(self, bot_name, energy=1):
            return AutoRes(True)

    mp_mod = types.ModuleType("menace_sandbox.model_automation_pipeline")
    mp_mod.ModelAutomationPipeline = Pipeline
    mp_mod.AutomationResult = AutoRes
    monkeypatch.setitem(sys.modules, "menace_sandbox.model_automation_pipeline", mp_mod)

    db_mod = types.ModuleType("menace_sandbox.data_bot")
    db_mod.DataBot = type(
        "DataBot",
        (),
        {"roi": lambda self, name: 0.0, "average_errors": lambda self, name: 0.0},
    )
    monkeypatch.setitem(sys.modules, "menace_sandbox.data_bot", db_mod)

    ae_mod = types.ModuleType("menace_sandbox.advanced_error_management")
    ae_mod.FormalVerifier = type("FV", (), {})
    ae_mod.AutomatedRollbackManager = type("ARM", (), {})
    monkeypatch.setitem(sys.modules, "menace_sandbox.advanced_error_management", ae_mod)

    ml_mod = types.ModuleType("menace_sandbox.mutation_logger")
    ml_mod.log_mutation = lambda *a, **k: 0
    ml_mod.record_mutation_outcome = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "menace_sandbox.mutation_logger", ml_mod)

    rb_mod = types.ModuleType("menace_sandbox.rollback_manager")
    rb_mod.RollbackManager = type("RM", (), {})
    monkeypatch.setitem(sys.modules, "menace_sandbox.rollback_manager", rb_mod)

    ps_mod = types.ModuleType("menace_sandbox.patch_suggestion_db")
    ps_mod.PatchSuggestionDB = type(
        "PatchSuggestionDB",
        (),
        {
            "log_repo_scan": lambda self: None,
            "add_failed_strategy": lambda self, tag: None,
        },
    )
    monkeypatch.setitem(sys.modules, "menace_sandbox.patch_suggestion_db", ps_mod)

    # attach store and vector service to package namespace
    import failure_fingerprint_store as ffs

    monkeypatch.setitem(sys.modules, "menace_sandbox.failure_fingerprint_store", ffs)
    monkeypatch.setitem(sys.modules, "menace_sandbox.vector_service", vec_module)

    # avoid real git clone
    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)

    # import manager class
    import importlib

    scm = importlib.reload(importlib.import_module("menace_sandbox.self_coding_manager"))

    store = make_store(tmp_path)
    engine = Engine()
    manager = scm.SelfCodingManager(
        engine,
        Pipeline(),
        data_bot=db_mod.DataBot(),
        failure_store=store,
        skip_similarity=0.8,
    )
    return manager, store, engine


def test_run_patch_skips_on_high_similarity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial = [
        TestResult("1 passed", "", True),
        TestResult(
            "File \"mod.py\", line 1, in bad\nValueError: boom",  # path-ignore
            "",
            False,
            failure={
                "stack": 'File "mod.py", line 1, in bad\nValueError: boom',  # path-ignore
                "strategy_tag": "tag",
            },
        ),
    ]
    manager, store, _engine = _load_manager(monkeypatch, tmp_path, initial)

    target = Path("dummy_patch_target.py")  # path-ignore
    target.write_text("x = 1", encoding="utf-8")
    import subprocess

    def fake_run(cmd, check=True, cwd=None):
        if cmd[:2] == ["git", "clone"]:
            dest = Path(cmd[-1])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / target.name).write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
        return None

    monkeypatch.setattr(subprocess, "run", fake_run)
    import failure_fingerprint as fp_mod

    def fake_from_failure(filename, function_name, stack, error, prompt):
        return fp_mod.FailureFingerprint(
            filename, function_name, error, stack, prompt, embedding=[1.0, 0.0]
        )

    monkeypatch.setattr(
        fp_mod.FailureFingerprint, "from_failure", staticmethod(fake_from_failure)
    )
    try:
        fp_match = FailureFingerprint(
            "mod.py", "bad", "ValueError: boom", "trace", "prompt", embedding=[1.0, 0.0]  # path-ignore
        )
        calls = {"n": 0}

        def fake_find(fp, threshold=None):
            calls["n"] += 1
            return [] if calls["n"] == 1 else [fp_match]

        store.find_similar = fake_find
        with pytest.raises(RuntimeError):
            manager.run_patch(target, "desc", max_attempts=2)
    finally:
        target.unlink(missing_ok=True)


def test_run_patch_warns_on_low_similarity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial = [
        TestResult("1 passed", "", True),
        TestResult(
            "File \"mod.py\", line 1, in bad\nValueError: boom",  # path-ignore
            "",
            False,
            failure={
                "stack": 'File "mod.py", line 1, in bad\nValueError: boom',  # path-ignore
                "strategy_tag": "tag",
            },
        ),
        TestResult("1 passed", "", True),
    ]
    manager, store, engine = _load_manager(monkeypatch, tmp_path, initial)

    target = Path("dummy_patch_target.py")  # path-ignore
    target.write_text("x = 1", encoding="utf-8")
    import subprocess

    def fake_run(cmd, check=True, cwd=None):
        if cmd[:2] == ["git", "clone"]:
            dest = Path(cmd[-1])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / target.name).write_text(
                target.read_text(encoding="utf-8"), encoding="utf-8"
            )
        return None

    monkeypatch.setattr(subprocess, "run", fake_run)
    import failure_fingerprint as fp_mod

    def fake_from_failure(filename, function_name, stack, error, prompt):
        return fp_mod.FailureFingerprint(
            filename, function_name, error, stack, prompt, embedding=[1.0, 0.0]
        )

    monkeypatch.setattr(
        fp_mod.FailureFingerprint, "from_failure", staticmethod(fake_from_failure)
    )
    try:
        fp_match = FailureFingerprint(
            "mod.py", "bad", "ValueError: boom", "trace", "prompt", embedding=[0.0, 1.0]  # path-ignore
        )
        calls = {"n": 0}

        def fake_find(fp, threshold=None):
            calls["n"] += 1
            return [] if calls["n"] == 1 else [fp_match]

        store.find_similar = fake_find
        manager.run_patch(target, "desc", max_attempts=3)
        assert any("avoid repeating failure" in d for d in engine.calls)
    finally:
        target.unlink(missing_ok=True)


def test_provisional_fingerprint_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager, store, engine = _load_manager(monkeypatch, tmp_path, [])

    import failure_fingerprint as fp_mod

    def fake_from_failure(filename, function_name, stack, error, prompt):
        return fp_mod.FailureFingerprint(
            filename, function_name, error, stack, prompt, embedding=[1.0, 0.0]
        )

    monkeypatch.setattr(
        fp_mod.FailureFingerprint, "from_failure", staticmethod(fake_from_failure)
    )

    def fake_find(fp, threshold=None):
        return [
            fp_mod.FailureFingerprint("f.py", "", "e", "t", "p", embedding=[1.0, 0.0])  # path-ignore
        ]

    store.find_similar = fake_find

    target = Path("dummy_patch_target.py")  # path-ignore
    target.write_text("x = 1", encoding="utf-8")
    import subprocess

    def fake_run(cmd, check=True, cwd=None):
        if cmd[:2] == ["git", "clone"]:
            dest = Path(cmd[-1])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / target.name).write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
        return None

    monkeypatch.setattr(subprocess, "run", fake_run)
    try:
        with pytest.raises(RuntimeError):
            manager.run_patch(target, "desc", max_attempts=1)
        assert not engine.calls
    finally:
        target.unlink(missing_ok=True)


def test_provisional_fingerprint_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initial = [TestResult("1 passed", "", True), TestResult("1 passed", "", True)]
    manager, store, engine = _load_manager(monkeypatch, tmp_path, initial)

    import failure_fingerprint as fp_mod

    def fake_from_failure(filename, function_name, stack, error, prompt):
        return fp_mod.FailureFingerprint(
            filename, function_name, error, stack, prompt, embedding=[1.0, 0.0]
        )

    monkeypatch.setattr(
        fp_mod.FailureFingerprint, "from_failure", staticmethod(fake_from_failure)
    )

    def fake_find(fp, threshold=None):
        return [
            fp_mod.FailureFingerprint("f.py", "", "e", "t", "p", embedding=[0.0, 1.0])  # path-ignore
        ]

    store.find_similar = fake_find

    target = Path("dummy_patch_target.py")  # path-ignore
    target.write_text("x = 1", encoding="utf-8")
    import subprocess

    def fake_run(cmd, check=True, cwd=None):
        if cmd[:2] == ["git", "clone"]:
            dest = Path(cmd[-1])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / target.name).write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
        return None

    monkeypatch.setattr(subprocess, "run", fake_run)
    try:
        manager.run_patch(target, "desc", max_attempts=1)
        assert any("avoid repeating failure" in d for d in engine.calls)
    finally:
        target.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# prompt optimizer penalties

def test_prompt_optimizer_penalty(tmp_path: Path) -> None:
    success = tmp_path / "success.jsonl"
    failure = tmp_path / "failure.jsonl"
    success.write_text(
        json.dumps(
            {
                "module": "m",
                "action": "a",
                "prompt": "# H\nExample",
                "success": True,
                "roi": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    failure.write_text("", encoding="utf-8")
    store = make_store(tmp_path)
    fp1 = FailureFingerprint("m", "a", "err", "trace", "# H\nExample")
    fp2 = FailureFingerprint("m", "a", "err", "trace", "# H\nExample")
    store.add(fp1)
    store.add(fp2)
    opt = PromptOptimizer(
        success,
        failure,
        stats_path=tmp_path / "stats.json",
        failure_store=store,
        fingerprint_threshold=1,
    )
    key = (
        "m",
        "a",
        "neutral",
        ("H",),
        "start",
        False,
        False,
        False,
    )
    assert opt.stats[key].penalty_factor < 1.0
