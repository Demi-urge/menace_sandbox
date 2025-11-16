import importlib.util
import sys
import types
import os
import logging
from pathlib import Path

import pytest

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
import sandbox_runner


def _stub_module(monkeypatch, name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    monkeypatch.setitem(sys.modules, name, mod)
    pkg_name, _, sub = name.partition(".")
    pkg = sys.modules.get(pkg_name)
    if pkg and sub:
        setattr(pkg, sub, mod)
    return mod


class DummyBot:
    def __init__(self, *a, **k):
        pass


class FailingBot:
    def __init__(self, *a, **k):
        raise RuntimeError("boom")


class DummyClient:
    def __init__(self, *a, **k):
        pass


class DummyConfig:
    def __init__(self, *a, **k):
        pass

    def load(self):
        pass

    def start_auto_refresh(self):
        pass


def _setup_mm_stubs(monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    stub_cycle = types.ModuleType("sandbox_runner.cycle")
    stub_cycle._async_track_usage = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "sandbox_runner.cycle", stub_cycle)

    pkg = types.ModuleType("menace")
    pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "menace", pkg)

    _stub_module(
        monkeypatch, "menace.unified_config_store", UnifiedConfigStore=DummyConfig
    )
    _stub_module(monkeypatch, "menace.dependency_self_check", self_check=lambda: None)
    _stub_module(monkeypatch, "menace.menace_orchestrator", MenaceOrchestrator=DummyBot)
    _stub_module(
        monkeypatch,
        "menace.self_coding_manager",
        PatchApprovalPolicy=DummyBot,
        SelfCodingManager=DummyBot,
    )
    _stub_module(
        monkeypatch,
        "menace.advanced_error_management",
        AutomatedRollbackManager=DummyBot,
    )
    _stub_module(
        monkeypatch, "menace.environment_bootstrap", EnvironmentBootstrapper=DummyBot
    )
    _stub_module(
        monkeypatch,
        "menace.auto_env_setup",
        ensure_env=lambda *a, **k: None,
        interactive_setup=lambda *a, **k: None,
    )
    _stub_module(
        monkeypatch,
        "menace.auto_resource_setup",
        ensure_proxies=lambda *a, **k: None,
        ensure_accounts=lambda *a, **k: None,
    )
    _stub_module(
        monkeypatch,
        "menace.external_dependency_provisioner",
        ExternalDependencyProvisioner=DummyBot,
    )
    _stub_module(monkeypatch, "menace.disaster_recovery", DisasterRecovery=DummyBot)
    _stub_module(monkeypatch, "menace.chatgpt_idea_bot", ChatGPTClient=DummyClient)
    _stub_module(
        monkeypatch,
        "menace.chatgpt_enhancement_bot",
        ChatGPTEnhancementBot=DummyBot,
        EnhancementDB=DummyBot,
    )
    _stub_module(monkeypatch, "menace.self_learning_service", main=lambda **k: None)
    _stub_module(
        monkeypatch, "menace.self_service_override", SelfServiceOverride=DummyBot
    )
    _stub_module(monkeypatch, "menace.resource_allocation_optimizer", ROIDB=DummyBot)
    _stub_module(monkeypatch, "menace.data_bot", MetricsDB=DummyBot)
    _stub_module(monkeypatch, "menace.unified_event_bus", UnifiedEventBus=DummyBot)
    _stub_module(monkeypatch, "menace.retry_utils", retry=lambda *a, **k: (lambda f: f))
    _stub_module(
        monkeypatch,
        "menace.override_policy",
        OverridePolicyManager=DummyBot,
        OverrideDB=DummyBot,
    )
    _stub_module(
        monkeypatch, "menace.unified_update_service", UnifiedUpdateService=DummyBot
    )
    _stub_module(
        monkeypatch, "menace.self_improvement", SelfImprovementEngine=DummyBot
    )
    _stub_module(monkeypatch, "menace.self_test_service", SelfTestService=DummyBot)
    _stub_module(monkeypatch, "menace.self_debugger_sandbox", SelfDebuggerSandbox=DummyBot)

    bot_modules = {
        "bot_development_bot": ("BotDevelopmentBot", DummyBot),
        "bot_testing_bot": ("BotTestingBot", DummyBot),
        "chatgpt_prediction_bot": ("ChatGPTPredictionBot", DummyBot),
        "chatgpt_research_bot": ("ChatGPTResearchBot", DummyBot),
        "competitive_intelligence_bot": ("CompetitiveIntelligenceBot", DummyBot),
        "contrarian_model_bot": ("ContrarianModelBot", DummyBot),
        "conversation_manager_bot": ("ConversationManagerBot", DummyBot),
        "database_steward_bot": ("DatabaseStewardBot", DummyBot),
        "deployment_bot": ("DeploymentBot", DummyBot),
        "enhancement_bot": ("EnhancementBot", DummyBot),
        "error_bot": ("ErrorBot", DummyBot),
        "ga_prediction_bot": ("GAPredictionBot", DummyBot),
        "genetic_algorithm_bot": ("GeneticAlgorithmBot", DummyBot),
        "ipo_bot": ("IPOBot", DummyBot),
        "implementation_optimiser_bot": ("ImplementationOptimiserBot", DummyBot),
        "mirror_bot": ("MirrorBot", DummyBot),
        "niche_saturation_bot": ("NicheSaturationBot", DummyBot),
        "market_manipulation_bot": ("MarketManipulationBot", DummyBot),
        "passive_discovery_bot": ("PassiveDiscoveryBot", DummyBot),
        "preliminary_research_bot": ("PreliminaryResearchBot", DummyBot),
        "report_generation_bot": ("ReportGenerationBot", DummyBot),
        "resource_allocation_bot": ("ResourceAllocationBot", DummyBot),
        "resources_bot": ("ResourcesBot", DummyBot),
        "scalability_assessment_bot": ("ScalabilityAssessmentBot", DummyBot),
        "strategy_prediction_bot": ("StrategyPredictionBot", DummyBot),
        "structural_evolution_bot": ("StructuralEvolutionBot", DummyBot),
        "text_research_bot": ("TextResearchBot", DummyBot),
        "video_research_bot": ("VideoResearchBot", DummyBot),
        "ai_counter_bot": ("AICounterBot", DummyBot),
        "dynamic_resource_allocator_bot": ("DynamicResourceAllocator", DummyBot),
        "diagnostic_manager": ("DiagnosticManager", DummyBot),
        "idea_search_bot": ("KeywordBank", DummyBot),
        "newsreader_bot": ("NewsDB", DummyBot),
    }
    for mod, (name, cls) in bot_modules.items():
        _stub_module(monkeypatch, f"menace.{mod}", **{name: cls})

    def start_auto_refresh(self):
        pass


def test_init_unused_bot_logs_failure(monkeypatch, caplog):
    _setup_mm_stubs(monkeypatch)
    _stub_module(monkeypatch, "menace.bot_development_bot", BotDevelopmentBot=FailingBot)

    path = Path(__file__).resolve().parents[1] / "menace_master.py"  # path-ignore
    spec = importlib.util.spec_from_file_location("menace_master", path)
    mm = importlib.util.module_from_spec(spec)
    sys.modules["menace_master"] = mm
    spec.loader.exec_module(mm)

    monkeypatch.setattr(
        sys.modules["menace.override_policy"].OverridePolicyManager,
        "run_continuous",
        lambda self, *a, **k: type("T", (), {"join": lambda self, timeout=None: None})(),
        raising=False,
    )

    caplog.set_level(logging.ERROR)
    mm._init_unused_bots()
    assert "Failed to instantiate FailingBot" in caplog.text
    assert "boom" in caplog.text


def test_main_run_cycles(monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    monkeypatch.setenv("RUN_CYCLES", "2")
    monkeypatch.setenv("SLEEP_SECONDS", "0")
    monkeypatch.setenv("AUTO_UPDATE", "0")
    monkeypatch.setenv("AUTO_BOOTSTRAP", "0")
    monkeypatch.setenv("AUTO_BACKUP", "0")
    monkeypatch.setenv("AUTO_SANDBOX", "0")

    _setup_mm_stubs(monkeypatch)

    calls = []

    class DummyOrch:
        def create_oversight(self, *a, **k):
            pass

        def start_scheduled_jobs(self):
            pass

        def run_cycle(self, models):
            calls.append(models)
            return {m: "ok" for m in models}

    class DummyThread:
        def join(self, timeout=None):
            pass

    class DummyService:
        def __init__(self, *a, **k):
            pass

        def adjust(self):
            pass

        def run_continuous(self, *a, **k):
            return DummyThread()

    _stub_module(monkeypatch, "menace.menace_orchestrator", MenaceOrchestrator=DummyOrch)
    _stub_module(monkeypatch, "menace.self_service_override", SelfServiceOverride=DummyService)
    _stub_module(monkeypatch, "menace.override_policy", OverridePolicyManager=DummyService, OverrideDB=DummyBot)
    _stub_module(monkeypatch, "menace.unified_update_service", UnifiedUpdateService=DummyService)

    path = Path(__file__).resolve().parents[1] / "menace_master.py"  # path-ignore
    spec = importlib.util.spec_from_file_location("menace_master", path)
    mm = importlib.util.module_from_spec(spec)
    sys.modules["menace_master"] = mm
    spec.loader.exec_module(mm)

    mm._init_unused_bots = lambda: None
    mm._start_dependency_watchdog = lambda event_bus=None: None

    mm.main([])

    assert len(calls) == 2


def test_auto_service_setup_root(monkeypatch):
    _setup_mm_stubs(monkeypatch)
    called = []
    _stub_module(
        monkeypatch,
        "menace.service_installer",
        _install_systemd=lambda: called.append("systemd"),
        _install_windows=lambda: called.append("windows"),
    )
    path = Path(__file__).resolve().parents[1] / "menace_master.py"  # path-ignore
    spec = importlib.util.spec_from_file_location("menace_master", path)
    mm = importlib.util.module_from_spec(spec)
    sys.modules["menace_master"] = mm
    spec.loader.exec_module(mm)
    monkeypatch.setattr(mm, "_install_user_systemd", lambda: called.append("user"))
    monkeypatch.setattr(mm, "_install_task_scheduler", lambda: called.append("task"))
    monkeypatch.setattr(mm.platform, "system", lambda: "Linux")
    monkeypatch.setattr(mm.os, "geteuid", lambda: 0, raising=False)

    mm._auto_service_setup()
    assert called


def test_auto_service_setup_user(monkeypatch, caplog):
    _setup_mm_stubs(monkeypatch)
    called = []
    _stub_module(
        monkeypatch,
        "menace.service_installer",
        _install_systemd=lambda: called.append("systemd"),
        _install_windows=lambda: called.append("windows"),
    )
    path = Path(__file__).resolve().parents[1] / "menace_master.py"  # path-ignore
    spec = importlib.util.spec_from_file_location("menace_master", path)
    mm = importlib.util.module_from_spec(spec)
    sys.modules["menace_master"] = mm
    spec.loader.exec_module(mm)
    monkeypatch.setattr(mm, "_install_user_systemd", lambda: called.append("user"))
    monkeypatch.setattr(mm, "_install_task_scheduler", lambda: called.append("task"))
    monkeypatch.setattr(mm.platform, "system", lambda: "Linux")
    monkeypatch.setattr(mm.os, "geteuid", lambda: 1000, raising=False)

    caplog.set_level(logging.INFO)
    mm._auto_service_setup()
    assert "user" in called
    assert "systemd" not in called and "windows" not in called


def test_first_run_triggers_sandbox(monkeypatch, tmp_path):
    _setup_mm_stubs(monkeypatch)
    path = Path(__file__).resolve().parents[1] / "menace_master.py"  # path-ignore
    spec = importlib.util.spec_from_file_location("menace_master", path)
    mm = importlib.util.module_from_spec(spec)
    sys.modules["menace_master"] = mm
    spec.loader.exec_module(mm)

    called = []
    monkeypatch.setattr(mm, "_start_dependency_watchdog", lambda event_bus=None: None)
    monkeypatch.setattr(mm, "_init_unused_bots", lambda: None)
    monkeypatch.setattr(sandbox_runner, "_run_sandbox", lambda args: called.append("sandbox"))
    monkeypatch.setattr(mm, "run_once", lambda models: called.append("once"))

    class DummyOrch:
        def create_oversight(self, *a, **k):
            pass

        def start_scheduled_jobs(self):
            pass

        def run_cycle(self, *a, **k):
            return {}

    monkeypatch.setattr(mm, "MenaceOrchestrator", DummyOrch, raising=False)

    monkeypatch.setattr(
        sys.modules["menace.override_policy"].OverridePolicyManager,
        "run_continuous",
        lambda self, *a, **k: type("T", (), {"join": lambda self, timeout=None: None})(),
        raising=False,
    )

    flag = tmp_path / "first.flag"
    monkeypatch.setenv("MENACE_FIRST_RUN_FILE", str(flag))
    monkeypatch.setenv("AUTO_SANDBOX", "1")
    monkeypatch.setenv("AUTO_BOOTSTRAP", "0")
    monkeypatch.setenv("AUTO_UPDATE", "0")
    monkeypatch.setenv("AUTO_BACKUP", "0")
    monkeypatch.setenv("RUN_CYCLES", "1")

    mm.main([])

    assert "once" in called and "sandbox" in called
    assert flag.exists()


def test_first_run_failure_no_flag(monkeypatch, tmp_path):
    _setup_mm_stubs(monkeypatch)
    path = Path(__file__).resolve().parents[1] / "menace_master.py"  # path-ignore
    spec = importlib.util.spec_from_file_location("menace_master", path)
    mm = importlib.util.module_from_spec(spec)
    sys.modules["menace_master"] = mm
    spec.loader.exec_module(mm)

    called = []
    monkeypatch.setattr(mm, "_start_dependency_watchdog", lambda event_bus=None: None)
    monkeypatch.setattr(mm, "_init_unused_bots", lambda: None)
    monkeypatch.setattr(sandbox_runner, "_run_sandbox", lambda args: called.append("sandbox"))

    def fail_once(models):
        raise RuntimeError("boom")

    monkeypatch.setattr(mm, "run_once", fail_once)

    class DummyOrch:
        def create_oversight(self, *a, **k):
            pass

        def start_scheduled_jobs(self):
            pass

        def run_cycle(self, *a, **k):
            return {}

    monkeypatch.setattr(mm, "MenaceOrchestrator", DummyOrch, raising=False)

    monkeypatch.setattr(
        sys.modules["menace.override_policy"].OverridePolicyManager,
        "run_continuous",
        lambda self, *a, **k: type("T", (), {"join": lambda self, timeout=None: None})(),
        raising=False,
    )

    flag = tmp_path / "first.flag"
    monkeypatch.setenv("MENACE_FIRST_RUN_FILE", str(flag))
    monkeypatch.setenv("AUTO_SANDBOX", "1")
    monkeypatch.setenv("AUTO_BOOTSTRAP", "0")
    monkeypatch.setenv("AUTO_UPDATE", "0")
    monkeypatch.setenv("AUTO_BACKUP", "0")
    monkeypatch.setenv("RUN_CYCLES", "1")

    mm.main([])

    assert not called
    assert not flag.exists()


def test_sandbox_skipped_on_run_once_error(monkeypatch, tmp_path, caplog):
    _setup_mm_stubs(monkeypatch)
    path = Path(__file__).resolve().parents[1] / "menace_master.py"  # path-ignore
    spec = importlib.util.spec_from_file_location("menace_master", path)
    mm = importlib.util.module_from_spec(spec)
    sys.modules["menace_master"] = mm
    spec.loader.exec_module(mm)

    called = []
    monkeypatch.setattr(mm, "_start_dependency_watchdog", lambda event_bus=None: None)
    monkeypatch.setattr(mm, "_init_unused_bots", lambda: None)
    monkeypatch.setattr(sandbox_runner, "_run_sandbox", lambda args: called.append("sandbox"))

    class FailingOrch:
        calls = 0

        def create_oversight(self, *a, **k):
            pass

        def start_scheduled_jobs(self):
            pass

        def run_cycle(self, *a, **k):
            if FailingOrch.calls == 0:
                FailingOrch.calls += 1
                raise RuntimeError("boom")
            return {}

    monkeypatch.setattr(mm, "MenaceOrchestrator", FailingOrch, raising=False)

    monkeypatch.setattr(
        sys.modules["menace.override_policy"].OverridePolicyManager,
        "run_continuous",
        lambda self, *a, **k: type("T", (), {"join": lambda self, timeout=None: None})(),
        raising=False,
    )

    flag = tmp_path / "first.flag"
    monkeypatch.setenv("MENACE_FIRST_RUN_FILE", str(flag))
    monkeypatch.setenv("AUTO_SANDBOX", "1")
    monkeypatch.setenv("AUTO_BOOTSTRAP", "0")
    monkeypatch.setenv("AUTO_UPDATE", "0")
    monkeypatch.setenv("AUTO_BACKUP", "0")
    monkeypatch.setenv("RUN_CYCLES", "1")

    caplog.set_level(logging.ERROR)
    mm.main([])

    assert not called
    assert not flag.exists()
    assert "run_once failed" in caplog.text


def test_deploy_patch_requires_self_test_summary(monkeypatch, tmp_path):
    _setup_mm_stubs(monkeypatch)
    path = Path(__file__).resolve().parents[1] / "menace_master.py"  # path-ignore
    spec = importlib.util.spec_from_file_location("menace_master", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["menace_master"] = module
    spec.loader.exec_module(module)

    class DummyEngine:
        def __init__(self) -> None:
            self.last_added_modules: list[str] = []
            self.added_modules: list[str] = []

    class DummyManager:
        def __init__(self) -> None:
            self.summary_payload: dict[str, object] = {}
            self.bot_name = "MenaceMaster"
            self.context_builder = None
            self.engine = DummyEngine()

        def auto_run_patch(self, path, description):
            return {
                "summary": self.summary_payload,
                "patch_id": 1,
                "commit": "abc123",
                "result": None,
            }

    manager = DummyManager()

    monkeypatch.setattr(module, "SelfCodingEngine", lambda *a, **k: manager.engine, raising=False)
    monkeypatch.setattr(module, "ModelAutomationPipeline", lambda *a, **k: types.SimpleNamespace(), raising=False)
    monkeypatch.setattr(module, "DataBot", lambda *a, **k: types.SimpleNamespace(), raising=False)
    monkeypatch.setattr(module, "CapitalManagementBot", lambda *a, **k: types.SimpleNamespace(), raising=False)
    monkeypatch.setattr(module, "SystemEvolutionManager", lambda *a, **k: types.SimpleNamespace(), raising=False)
    monkeypatch.setattr(module, "UnifiedEventBus", lambda *a, **k: types.SimpleNamespace(), raising=False)
    monkeypatch.setattr(module, "BotRegistry", lambda *a, **k: types.SimpleNamespace(), raising=False)
    monkeypatch.setattr(
        module,
        "EvolutionOrchestrator",
        lambda *a, **k: types.SimpleNamespace(provenance_token="token"),
        raising=False,
    )
    monkeypatch.setattr(module, "internalize_coding_bot", lambda *a, **k: manager, raising=False)
    monkeypatch.setattr(
        module,
        "get_thresholds",
        lambda name: types.SimpleNamespace(
            roi_drop=0.0, error_increase=0.0, test_failure_increase=0.0
        ),
        raising=False,
    )
    monkeypatch.setattr(module, "persist_sc_thresholds", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(
        module,
        "AutomatedRollbackManager",
        lambda *a, **k: types.SimpleNamespace(auto_rollback=lambda *a, **k: None),
        raising=False,
    )
    monkeypatch.setattr(
        sandbox_runner,
        "try_integrate_into_workflows",
        lambda *a, **k: None,
        raising=False,
    )

    builder = types.SimpleNamespace(refresh_db_weights=lambda: None)
    file_path = tmp_path / "module.py"  # path-ignore
    file_path.write_text("print('hi')\n")

    with pytest.raises(RuntimeError, match="self test summary unavailable"):
        module.deploy_patch(file_path, "desc", builder)

    manager.summary_payload = {"self_tests": {"failed": 0}}
    module.deploy_patch(file_path, "desc", builder)
