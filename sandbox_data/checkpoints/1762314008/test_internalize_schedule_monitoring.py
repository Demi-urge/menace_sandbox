import types
import sys
from pathlib import Path
import pytest

menace_pkg = types.ModuleType("menace")
menace_pkg.__path__ = [str(Path("."))]
sys.modules.setdefault("menace", menace_pkg)

qfe_stub = types.ModuleType("quick_fix_engine")

class QuickFixEngineError(Exception):
    pass

qfe_stub.QuickFixEngine = object
qfe_stub.QuickFixEngineError = QuickFixEngineError
qfe_stub.generate_patch = lambda *a, **k: None
sys.modules["quick_fix_engine"] = qfe_stub
sys.modules["menace.quick_fix_engine"] = qfe_stub
stub_dpr = types.SimpleNamespace(
    resolve_path=lambda p: Path(p),
    path_for_prompt=lambda p: p,
    repo_root=lambda: Path("."),
    resolve_dir=lambda p: Path(p),
    get_project_root=lambda: Path("."),
    resolve_module_path=lambda m: Path(m),
)
sys.modules["dynamic_path_router"] = stub_dpr
sys.modules["menace.dynamic_path_router"] = stub_dpr

stub_cbi = types.ModuleType("coding_bot_interface")
stub_cbi.self_coding_managed = lambda *a, **k: (lambda cls: cls)
stub_cbi.manager_generate_helper = lambda *_a, **_k: None
sys.modules["coding_bot_interface"] = stub_cbi
sys.modules["menace.coding_bot_interface"] = stub_cbi

stub_engine = types.ModuleType("self_coding_engine")
class _DummyEngine:  # noqa: WPS431 - simple placeholder
    pass
stub_engine.SelfCodingEngine = _DummyEngine
sys.modules["self_coding_engine"] = stub_engine
sys.modules["menace.self_coding_engine"] = stub_engine

stub_harness = types.ModuleType("sandbox_runner.test_harness")
stub_harness.run_tests = lambda *a, **k: None
class _DummyResult:  # noqa: WPS431 - simple placeholder
    pass
stub_harness.TestHarnessResult = _DummyResult
sandbox_runner_pkg = types.ModuleType("sandbox_runner")
sandbox_runner_pkg.test_harness = stub_harness
sys.modules["sandbox_runner"] = sandbox_runner_pkg
sys.modules["sandbox_runner.test_harness"] = stub_harness
menace_sandbox_runner_pkg = types.ModuleType("menace.sandbox_runner")
menace_sandbox_runner_pkg.test_harness = stub_harness
sys.modules["menace.sandbox_runner"] = menace_sandbox_runner_pkg
sys.modules["menace.sandbox_runner.test_harness"] = stub_harness

import menace.self_coding_manager as scm


def test_schedule_monitoring_invoked_without_event_bus(monkeypatch):
    calls = []

    class DummyDataBot:
        def __init__(self):
            self.settings = types.SimpleNamespace(bot_thresholds={})
            self.event_bus = None

        def schedule_monitoring(self, bot: str) -> None:
            calls.append(bot)

    class DummyManager:
        def __init__(self, *_a, **_k):
            self.quick_fix = object()
            self.logger = types.SimpleNamespace(
                exception=lambda *a, **k: None
            )

    class DummyRegistry:
        def register_bot(self, *_a, **_k):
            return None

    monkeypatch.setattr(scm, "SelfCodingManager", DummyManager)
    monkeypatch.setattr(scm, "persist_sc_thresholds", lambda *a, **k: None)

    data_bot = DummyDataBot()
    scm.internalize_coding_bot(
        "bot",
        engine=object(),
        pipeline=object(),
        data_bot=data_bot,
        bot_registry=DummyRegistry(),
    )

    assert calls == ["bot"]


def test_internalization_requires_quick_fix_engine(monkeypatch):
    class DummyDataBot:
        def __init__(self):
            self.settings = types.SimpleNamespace(bot_thresholds={})

    class DummyManager:
        def __init__(self, *_a, **_k):
            self.quick_fix = None
            self.logger = types.SimpleNamespace(exception=lambda *a, **k: None)

    class DummyRegistry:
        def register_bot(self, *_a, **_k):
            return None

    monkeypatch.setattr(scm, "SelfCodingManager", DummyManager)
    monkeypatch.setattr(scm, "persist_sc_thresholds", lambda *a, **k: None)
    monkeypatch.setattr(scm, "QuickFixEngine", None, raising=False)

    with pytest.raises(ImportError, match="QuickFixEngine"):
        scm.internalize_coding_bot(
            "bot",
            engine=object(),
            pipeline=object(),
            data_bot=DummyDataBot(),
            bot_registry=DummyRegistry(),
        )
