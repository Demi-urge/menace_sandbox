import importlib
import sys
import types
from pathlib import Path

import pytest


class _PromptStub(types.SimpleNamespace):
    def __init__(self) -> None:
        super().__init__(text="demo", metadata={}, system="", examples=[])


def _exercise_engine(module, monkeypatch):
    class DummyROITracker:
        def __init__(self, *args, **kwargs):
            self.metrics = {}

        def update_db_metrics(self, metrics):
            self.metrics.update(metrics)

        def origin_db_deltas(self):
            return {}

    class DummyAuditTrail:
        def __init__(self, *args, **kwargs):
            self.records = []

        def log(self, *args, **kwargs):
            self.records.append((args, kwargs))

    class DummyPatchAttemptTracker:
        def __init__(self, *args, **kwargs):
            self.calls = []

        def record(self, *args, **kwargs):  # pragma: no cover - defensive
            self.calls.append((args, kwargs))

    class DummyContextBuilder:
        def __init__(self) -> None:
            self.roi_tracker = None

        def refresh_db_weights(self):
            return None

        def build_context(self, *args, **kwargs):
            return {}

    monkeypatch.setattr(module, "ROITracker", DummyROITracker)
    monkeypatch.setattr(module, "AuditTrail", DummyAuditTrail)
    monkeypatch.setattr(module, "PatchAttemptTracker", DummyPatchAttemptTracker)
    monkeypatch.setattr(module, "GPT_MEMORY_MANAGER", object())
    builder = DummyContextBuilder()
    patch_logger = types.SimpleNamespace(roi_tracker=None)
    cognition = types.SimpleNamespace(roi_tracker=None, context_builder=builder)
    optimizer = types.SimpleNamespace(refresh=lambda: None)
    evolution_memory = types.SimpleNamespace(log=lambda *a, **k: None)

    engine = module.SelfCodingEngine(
        code_db=object(),
        memory_mgr=object(),
        context_builder=builder,
        patch_logger=patch_logger,
        cognition_layer=cognition,
        llm_client=types.SimpleNamespace(gpt_memory=None, context_builder=builder),
        prompt_memory=types.SimpleNamespace(),
        prompt_optimizer=optimizer,
        prompt_evolution_memory=evolution_memory,
        knowledge_service=None,
    )
    assert engine.context_builder is builder
    assert isinstance(engine.roi_tracker, DummyROITracker)
    engine.simplify_prompt(_PromptStub())
    return engine


@pytest.mark.parametrize(
    "import_name",
    ["menace_sandbox.self_coding_engine", "self_coding_engine"],
)
def test_self_coding_engine_imports_bootstrap(monkeypatch, import_name):
    saved = {
        key: sys.modules.get(key)
        for key in (
            "menace_sandbox",
            "menace_sandbox.self_coding_engine",
            "self_coding_engine",
            "menace_sandbox.self_improvement",
            "menace_sandbox.self_improvement.baseline_tracker",
            "menace_sandbox.self_improvement.init",
            "menace_sandbox.self_improvement.prompt_memory",
            "menace_sandbox.self_improvement.target_region",
            "self_improvement",
            "self_improvement.baseline_tracker",
            "self_improvement.init",
            "self_improvement.prompt_memory",
            "self_improvement.target_region",
            "dynamic_path_router",
            "menace_sandbox.patch_attempt_tracker",
            "patch_attempt_tracker",
        )
    }
    for key in list(saved):
        sys.modules.pop(key, None)

    dyn_router = saved.get("dynamic_path_router") or types.SimpleNamespace()
    setattr(dyn_router, "resolve_path", getattr(dyn_router, "resolve_path", Path))
    setattr(dyn_router, "resolve_dir", getattr(dyn_router, "resolve_dir", Path))
    setattr(
        dyn_router,
        "resolve_module_path",
        getattr(
            dyn_router,
            "resolve_module_path",
            lambda name: Path(name.replace(".", "/") + ".py"),
        ),
    )
    setattr(dyn_router, "path_for_prompt", getattr(dyn_router, "path_for_prompt", lambda p: str(p)))
    setattr(dyn_router, "repo_root", getattr(dyn_router, "repo_root", lambda: Path(".")))
    setattr(dyn_router, "get_project_root", getattr(dyn_router, "get_project_root", lambda: Path(".")))
    sys.modules["dynamic_path_router"] = dyn_router

    baseline_stub = types.ModuleType("menace_sandbox.self_improvement.baseline_tracker")

    class DummyBaselineTracker:
        def __init__(self, *args, **kwargs):
            pass

        def update(self, **metrics):
            return None

    self_improvement_pkg = types.ModuleType("menace_sandbox.self_improvement")
    self_improvement_pkg.__path__ = []  # type: ignore[attr-defined]
    baseline_stub.BaselineTracker = DummyBaselineTracker
    baseline_stub.TRACKER = DummyBaselineTracker()
    setattr(self_improvement_pkg, "baseline_tracker", baseline_stub)
    sys.modules["menace_sandbox.self_improvement"] = self_improvement_pkg
    sys.modules["self_improvement"] = self_improvement_pkg
    sys.modules["menace_sandbox.self_improvement.baseline_tracker"] = baseline_stub
    sys.modules["self_improvement.baseline_tracker"] = baseline_stub

    prompt_memory_stub = types.ModuleType("menace_sandbox.self_improvement.prompt_memory")
    prompt_memory_stub.log_prompt_attempt = lambda *a, **k: None
    setattr(self_improvement_pkg, "prompt_memory", prompt_memory_stub)
    sys.modules["menace_sandbox.self_improvement.prompt_memory"] = prompt_memory_stub
    sys.modules["self_improvement.prompt_memory"] = prompt_memory_stub

    init_stub = types.ModuleType("menace_sandbox.self_improvement.init")

    class _FileLock:
        def __init__(self, *args, **kwargs):
            pass

        def acquire(self):  # pragma: no cover - compatibility
            return True

        def release(self):  # pragma: no cover - compatibility
            return None

    def _atomic_write(*args, **kwargs):
        return None

    init_stub.FileLock = _FileLock
    init_stub._atomic_write = _atomic_write
    setattr(self_improvement_pkg, "init", init_stub)
    sys.modules["menace_sandbox.self_improvement.init"] = init_stub
    sys.modules["self_improvement.init"] = init_stub

    target_region_stub = types.ModuleType("menace_sandbox.self_improvement.target_region")

    class _TargetRegion:
        pass

    target_region_stub.TargetRegion = _TargetRegion
    setattr(self_improvement_pkg, "target_region", target_region_stub)
    sys.modules["menace_sandbox.self_improvement.target_region"] = target_region_stub
    sys.modules["self_improvement.target_region"] = target_region_stub

    patch_attempt_stub = types.ModuleType("menace_sandbox.patch_attempt_tracker")

    class _PatchedTracker:
        def __init__(self, *args, **kwargs):
            pass

    patch_attempt_stub.PatchAttemptTracker = _PatchedTracker
    sys.modules["menace_sandbox.patch_attempt_tracker"] = patch_attempt_stub
    sys.modules["patch_attempt_tracker"] = patch_attempt_stub

    try:
        module = importlib.import_module(import_name)
        assert module.__package__ == "menace_sandbox"
        assert sys.modules["menace_sandbox.self_coding_engine"] is module
        _exercise_engine(module, monkeypatch)
    finally:
        for key in (
            "menace_sandbox.self_improvement",
            "menace_sandbox.self_improvement.baseline_tracker",
            "menace_sandbox.self_improvement.init",
            "menace_sandbox.self_improvement.prompt_memory",
            "menace_sandbox.self_improvement.target_region",
            "self_improvement",
            "self_improvement.baseline_tracker",
            "self_improvement.init",
            "self_improvement.prompt_memory",
            "self_improvement.target_region",
            "dynamic_path_router",
            "menace_sandbox.patch_attempt_tracker",
            "patch_attempt_tracker",
        ):
            sys.modules.pop(key, None)
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
