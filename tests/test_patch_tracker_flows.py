import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Stub heavy dependencies
sys.modules.setdefault("menace.self_coding_engine", types.SimpleNamespace(SelfCodingEngine=object))
sys.modules.setdefault(
    "menace.model_automation_pipeline",
    types.SimpleNamespace(ModelAutomationPipeline=object, AutomationResult=object),
)
sys.modules.setdefault("menace.data_bot", types.SimpleNamespace(DataBot=object))
sys.modules.setdefault(
    "menace.advanced_error_management",
    types.SimpleNamespace(FormalVerifier=object, AutomatedRollbackManager=object),
)
sys.modules.setdefault("menace.mutation_logger", types.SimpleNamespace())
sys.modules.setdefault("menace.rollback_manager", types.SimpleNamespace(RollbackManager=object))
sys.modules.setdefault("menace.sandbox_settings", types.SimpleNamespace(SandboxSettings=object))
sys.modules.setdefault(
    "menace.failure_fingerprint_store",
    types.SimpleNamespace(
        FailureFingerprint=type("FailureFingerprint", (object,), {}),
        FailureFingerprintStore=object,
    ),
)
sys.modules.setdefault(
    "menace.failure_retry_utils",
    types.SimpleNamespace(
        check_similarity_and_warn=lambda fp, store, threshold, desc: (desc, False, 0.0, [], None),
        record_failure=lambda fp, store: None,
    ),
)
sys.modules.setdefault(
    "menace.patch_suggestion_db",
    types.SimpleNamespace(PatchSuggestionDB=object),
)
sys.modules.setdefault(
    "menace.error_parser",
    types.SimpleNamespace(
        FailureCache=type(
            "FailureCache",
            (),
            {
                "__init__": lambda self: None,
                "add": lambda self, report: None,
                "seen": lambda self, trace: False,
            },
        ),
        ErrorReport=type("ErrorReport", (), {"__init__": lambda self, trace, tags: None}),
        ErrorParser=types.SimpleNamespace(
            parse_failure=lambda trace: {}, parse=lambda trace: {}
        ),
    ),
)


def _thr_init(self, success, stdout, stderr, duration, failure=None, path=None):
    self.success = success
    self.stdout = stdout
    self.stderr = stderr
    self.duration = duration
    self.failure = failure
    self.path = path


TestHarnessResult = type("TestHarnessResult", (), {"__init__": _thr_init})

sys.modules.setdefault(
    "sandbox_runner.test_harness",
    types.SimpleNamespace(run_tests=lambda *a, **k: None, TestHarnessResult=TestHarnessResult),
)
sys.modules.setdefault(
    "menace.self_improvement.baseline_tracker",
    types.SimpleNamespace(
        BaselineTracker=type(
            "BaselineTracker",
            (),
            {
                "__init__": lambda self, *a, **k: None,
                "get": lambda self, key: 0.0,
                "std": lambda self, key: 0.0,
                "update": lambda self, **kwargs: None,
            },
        )
    ),
)

spec_adb = importlib.util.spec_from_file_location(
    "menace.automated_debugger", ROOT / "automated_debugger.py"  # path-ignore
)
automated_debugger = importlib.util.module_from_spec(spec_adb)
sys.modules["menace.automated_debugger"] = automated_debugger
assert spec_adb.loader is not None
spec_adb.loader.exec_module(automated_debugger)
AutomatedDebugger = automated_debugger.AutomatedDebugger

spec_scm = importlib.util.spec_from_file_location(
    "menace.self_coding_manager", ROOT / "self_coding_manager.py"  # path-ignore
)
self_coding_manager = importlib.util.module_from_spec(spec_scm)
sys.modules["menace.self_coding_manager"] = self_coding_manager
assert spec_scm.loader is not None
spec_scm.loader.exec_module(self_coding_manager)
SelfCodingManager = self_coding_manager.SelfCodingManager

from menace.self_improvement.target_region import TargetRegion  # noqa: E402
from menace.vector_service.context_builder import ContextBuilder  # noqa: E402


class DummyEngine:
    def __init__(self):
        self.calls = []
        self.last_prompt_text = ""
        self.cognition_layer = types.SimpleNamespace(
            context_builder=types.SimpleNamespace(
                query=lambda desc, exclude_tags=None: ("ctx", "sid")
            )
        )

    def apply_patch(
        self,
        path: Path,
        description: str,
        *,
        reason: str | None = None,
        trigger: str | None = None,
        target_region: TargetRegion | None = None,
        **_: object,
    ) -> tuple[int, bool, str]:
        self.calls.append(target_region)
        return 1, False, ""


class DummyTelemetry:
    def __init__(self, log: str):
        self.log = log

    def recent_errors(self, limit: int = 5):  # pragma: no cover - simple
        return [self.log]


class DummyBuilder(ContextBuilder):
    def __init__(self):
        pass

    def build_context(self, query: str, **kwargs):
        return {}

    def refresh_db_weights(self):
        pass


class DummyPipeline:
    def run(self, bot_name: str, energy: int = 1):  # pragma: no cover - simple
        return types.SimpleNamespace(roi=0.0)


class DummyDataBot:
    def roi(self, bot_name: str) -> float:  # pragma: no cover - simple
        return 0.0
    def average_errors(self, bot_name: str) -> float:  # pragma: no cover - simple
        return 0.0


def test_automated_debugger_escalation_and_reset(tmp_path, monkeypatch):
    mod = tmp_path / "buggy.py"  # path-ignore
    mod.write_text("def f():\n    raise ValueError('boom')\n")
    log = (
        "Traceback (most recent call last):\n  File \""
        f"{mod}\", line 2, in f\nValueError: boom"
    )

    engine = DummyEngine()
    debugger = AutomatedDebugger(DummyTelemetry(log), engine, DummyBuilder())

    call = {"n": 0}

    def fake_run(cmd, capture_output=True, text=True):
        call["n"] += 1
        rc = 1 if call["n"] < 5 else 0
        return types.SimpleNamespace(returncode=rc, stdout="", stderr="")

    monkeypatch.setattr(AutomatedDebugger, "_recent_logs", lambda self, limit=5: [log])
    monkeypatch.setattr(automated_debugger.subprocess, "run", fake_run)

    for _ in range(5):
        debugger.analyse_and_fix()

    # region -> region -> function -> function -> module
    assert [getattr(r, "start_line", None) for r in engine.calls] == [2, 2, 1, 1, None]

    region = TargetRegion(start_line=2, end_line=2, function="f", filename=str(mod))
    func_region = TargetRegion(start_line=1, end_line=2, function="f", filename=str(mod))
    assert debugger._tracker.level_for(region, func_region)[0] == "region"


def test_self_coding_manager_escalation_and_reset(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    mod = repo / "buggy.py"  # path-ignore
    mod.write_text("def f():\n    raise ValueError('boom')\n")

    manager = SelfCodingManager(DummyEngine(), DummyPipeline(), data_bot=DummyDataBot())
    manager.baseline_tracker = types.SimpleNamespace(
        get=lambda key: 0.0, std=lambda key: 0.0, update=lambda **kw: None
    )
    monkeypatch.chdir(repo)

    # patch similarity and fingerprint helpers
    monkeypatch.setattr(
        self_coding_manager,
        "check_similarity_and_warn",
        lambda *a, **k: (a[3], False, 0.0, [], None),
    )
    monkeypatch.setattr(
        self_coding_manager,
        "record_failure",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        self_coding_manager,
        "record_failed_tags",
        lambda tags: None,
    )
    monkeypatch.setattr(
        self_coding_manager,
        "MutationLogger",
        types.SimpleNamespace(
            log_mutation=lambda *a, **k: 1,
            record_mutation_outcome=lambda *a, **k: None,
        ),
    )
    monkeypatch.setattr(
        self_coding_manager,
        "RollbackManager",
        lambda: types.SimpleNamespace(rollback=lambda *a, **k: None),
    )

    class DummyFP:
        hash = "h"
        timestamp = 0.0

    monkeypatch.setattr(
        self_coding_manager.FailureFingerprint,
        "from_failure",
        staticmethod(lambda *a, **k: DummyFP()),
        raising=False,
    )

    def fake_parse_failure(trace):
        return {"strategy_tag": "tag"}

    def fake_parse(trace):
        return {
            "trace": trace,
            "target_region": types.SimpleNamespace(
                file=str(mod), start_line=2, end_line=2, function="f"
            ),
        }

    monkeypatch.setattr(
        self_coding_manager.ErrorParser,
        "parse_failure",
        staticmethod(fake_parse_failure),
    )
    monkeypatch.setattr(
        self_coding_manager.ErrorParser,
        "parse",
        staticmethod(fake_parse),
    )

    from sandbox_runner.test_harness import TestHarnessResult

    counter = {"n": -1}

    def fake_run_tests(repo_path, changed_path, backend="venv"):
        counter["n"] += 1
        if counter["n"] == 0:
            return TestHarnessResult(True, "", "", 0.0)
        elif counter["n"] <= 5:
            trace = f'File "{changed_path}", line 2, in f\nValueError'
            return TestHarnessResult(False, trace, "", 0.0)
        else:
            return TestHarnessResult(True, "", "", 0.0)

    monkeypatch.setattr(self_coding_manager, "run_tests", fake_run_tests)

    import shutil

    def fake_subprocess_run(cmd, check=False, capture_output=False, text=False, cwd=None):
        if cmd[:2] == ["git", "clone"]:
            shutil.copytree(cmd[2], cmd[3], dirs_exist_ok=True)
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(self_coding_manager.subprocess, "run", fake_subprocess_run)

    reset_called = {"v": False}
    orig_reset = self_coding_manager.PatchAttemptTracker.reset

    def fake_reset(self, region):
        reset_called["v"] = True
        orig_reset(self, region)

    monkeypatch.setattr(self_coding_manager.PatchAttemptTracker, "reset", fake_reset)

    manager.run_patch(mod, "fix", max_attempts=6)

    calls = manager.engine.calls
    assert [getattr(r, "start_line", None) for r in calls] == [None, 2, 2, 0, 0, None]
    assert reset_called["v"]
