import types
import sys
import importlib.util
from pathlib import Path

from menace.unified_event_bus import UnifiedEventBus


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


class DummyProv:
    def __init__(self):
        pass

    def provision(self):
        raise RuntimeError("boom")


class DummyObj:
    def __init__(self, *a, **k):
        pass


def test_provision_failure_emits_event(monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    pkg = types.ModuleType("menace")
    pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "menace", pkg)
    monkeypatch.setenv("PROVISION_ATTEMPTS", "1")
    monkeypatch.setenv("PROVISION_RETRY_DELAY", "0")
    monkeypatch.setenv("DEPENDENCY_ENDPOINTS", "svc=http://x")

    _stub_module(monkeypatch, "menace.unified_config_store", UnifiedConfigStore=DummyObj)
    _stub_module(monkeypatch, "menace.dependency_self_check", self_check=lambda: None)
    _stub_module(monkeypatch, "menace.menace_orchestrator", MenaceOrchestrator=DummyObj)
    _stub_module(monkeypatch, "menace.self_coding_manager", PatchApprovalPolicy=DummyObj, SelfCodingManager=DummyObj)
    _stub_module(monkeypatch, "menace.advanced_error_management", AutomatedRollbackManager=DummyObj)
    _stub_module(monkeypatch, "menace.environment_bootstrap", EnvironmentBootstrapper=DummyObj)
    _stub_module(monkeypatch, "menace.auto_env_setup", ensure_env=lambda *a, **k: None, interactive_setup=lambda *a, **k: None)
    _stub_module(monkeypatch, "menace.auto_resource_setup", ensure_proxies=lambda *a, **k: None, ensure_accounts=lambda *a, **k: None)
    _stub_module(monkeypatch, "menace.external_dependency_provisioner", ExternalDependencyProvisioner=lambda: DummyProv())
    class DummyWatchdog:
        def __init__(self, *a, **k):
            pass
        def check(self):
            pass
    _stub_module(monkeypatch, "menace.dependency_watchdog", DependencyWatchdog=DummyWatchdog)
    _stub_module(monkeypatch, "menace.disaster_recovery", DisasterRecovery=DummyObj)
    _stub_module(monkeypatch, "menace.chatgpt_idea_bot", ChatGPTClient=DummyObj)
    _stub_module(monkeypatch, "menace.self_learning_service", main=lambda **k: None)
    _stub_module(monkeypatch, "menace.self_service_override", SelfServiceOverride=DummyObj)
    _stub_module(monkeypatch, "menace.resource_allocation_optimizer", ROIDB=DummyObj)
    _stub_module(monkeypatch, "menace.data_bot", MetricsDB=DummyObj)

    bot_modules = {
        "bot_development_bot": "BotDevelopmentBot",
        "bot_testing_bot": "BotTestingBot",
        "chatgpt_enhancement_bot": "ChatGPTEnhancementBot",
        "chatgpt_prediction_bot": "ChatGPTPredictionBot",
        "chatgpt_research_bot": "ChatGPTResearchBot",
        "competitive_intelligence_bot": "CompetitiveIntelligenceBot",
        "contrarian_model_bot": "ContrarianModelBot",
        "conversation_manager_bot": "ConversationManagerBot",
        "database_steward_bot": "DatabaseStewardBot",
        "deployment_bot": "DeploymentBot",
        "enhancement_bot": "EnhancementBot",
        "error_bot": "ErrorBot",
        "ga_prediction_bot": "GAPredictionBot",
        "genetic_algorithm_bot": "GeneticAlgorithmBot",
        "ipo_bot": "IPOBot",
        "implementation_optimiser_bot": "ImplementationOptimiserBot",
        "mirror_bot": "MirrorBot",
        "niche_saturation_bot": "NicheSaturationBot",
        "market_manipulation_bot": "MarketManipulationBot",
        "passive_discovery_bot": "PassiveDiscoveryBot",
        "preliminary_research_bot": "PreliminaryResearchBot",
        "report_generation_bot": "ReportGenerationBot",
        "resource_allocation_bot": "ResourceAllocationBot",
        "resources_bot": "ResourcesBot",
        "scalability_assessment_bot": "ScalabilityAssessmentBot",
        "strategy_prediction_bot": "StrategyPredictionBot",
        "structural_evolution_bot": "StructuralEvolutionBot",
        "text_research_bot": "TextResearchBot",
        "video_research_bot": "VideoResearchBot",
        "ai_counter_bot": "AICounterBot",
        "dynamic_resource_allocator_bot": "DynamicResourceAllocator",
        "diagnostic_manager": "DiagnosticManager",
        "idea_search_bot": "KeywordBank",
        "newsreader_bot": "NewsDB",
    }

    for mod, name in bot_modules.items():
        _stub_module(monkeypatch, f"menace.{mod}", **{name: DummyObj})

    path = Path(__file__).resolve().parents[1] / "menace_master.py"  # path-ignore
    spec = importlib.util.spec_from_file_location("menace_master", path)
    mm = importlib.util.module_from_spec(spec)
    sys.modules["menace_master"] = mm
    spec.loader.exec_module(mm)

    monkeypatch.setattr(mm, "requests", types.SimpleNamespace(get=lambda *a, **k: (_ for _ in ()).throw(Exception("down"))))
    monkeypatch.setattr(mm.time, "sleep", lambda s: None)
    import menace.retry_utils as ru
    monkeypatch.setattr(ru.time, "sleep", lambda s: None)
    bus = UnifiedEventBus()
    events = []
    bus.subscribe("dependency:provision_failed", lambda t, e: events.append(e))
    class DummyThread:
        def __init__(self, target=None, daemon=None):
            self.target = target
        def start(self):
            pass

    monkeypatch.setattr(mm.threading, "Thread", DummyThread)
    mm._start_dependency_watchdog(event_bus=bus)
    assert events and events[-1]["error"] == "boom"


def test_missing_requests_installs(monkeypatch):
    monkeypatch.setenv("MENACE_LIGHT_IMPORTS", "1")
    pkg = types.ModuleType("menace")
    pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "menace", pkg)
    monkeypatch.setenv("DEPENDENCY_ENDPOINTS", "svc=http://x")

    _stub_module(monkeypatch, "menace.unified_config_store", UnifiedConfigStore=DummyObj)
    _stub_module(monkeypatch, "menace.dependency_self_check", self_check=lambda: None)
    _stub_module(monkeypatch, "menace.menace_orchestrator", MenaceOrchestrator=DummyObj)
    _stub_module(monkeypatch, "menace.self_coding_manager", PatchApprovalPolicy=DummyObj, SelfCodingManager=DummyObj)
    _stub_module(monkeypatch, "menace.advanced_error_management", AutomatedRollbackManager=DummyObj)
    _stub_module(monkeypatch, "menace.environment_bootstrap", EnvironmentBootstrapper=DummyObj)
    _stub_module(monkeypatch, "menace.auto_env_setup", ensure_env=lambda *a, **k: None, interactive_setup=lambda *a, **k: None)
    _stub_module(monkeypatch, "menace.auto_resource_setup", ensure_proxies=lambda *a, **k: None, ensure_accounts=lambda *a, **k: None)
    _stub_module(monkeypatch, "menace.external_dependency_provisioner", ExternalDependencyProvisioner=lambda: DummyObj())

    class DummyWatchdog:
        def __init__(self, *a, **k):
            pass
        def check(self):
            pass

    _stub_module(monkeypatch, "menace.dependency_watchdog", DependencyWatchdog=DummyWatchdog)
    _stub_module(monkeypatch, "menace.disaster_recovery", DisasterRecovery=DummyObj)
    _stub_module(monkeypatch, "menace.chatgpt_idea_bot", ChatGPTClient=DummyObj)
    _stub_module(monkeypatch, "menace.self_learning_service", main=lambda **k: None)
    _stub_module(monkeypatch, "menace.self_service_override", SelfServiceOverride=DummyObj)
    _stub_module(monkeypatch, "menace.resource_allocation_optimizer", ROIDB=DummyObj)
    _stub_module(monkeypatch, "menace.data_bot", MetricsDB=DummyObj)

    bot_modules = {
        "bot_development_bot": "BotDevelopmentBot",
        "bot_testing_bot": "BotTestingBot",
        "chatgpt_enhancement_bot": "ChatGPTEnhancementBot",
        "chatgpt_prediction_bot": "ChatGPTPredictionBot",
        "chatgpt_research_bot": "ChatGPTResearchBot",
        "competitive_intelligence_bot": "CompetitiveIntelligenceBot",
        "contrarian_model_bot": "ContrarianModelBot",
        "conversation_manager_bot": "ConversationManagerBot",
        "database_steward_bot": "DatabaseStewardBot",
        "deployment_bot": "DeploymentBot",
        "enhancement_bot": "EnhancementBot",
        "error_bot": "ErrorBot",
        "ga_prediction_bot": "GAPredictionBot",
        "genetic_algorithm_bot": "GeneticAlgorithmBot",
        "ipo_bot": "IPOBot",
        "implementation_optimiser_bot": "ImplementationOptimiserBot",
        "mirror_bot": "MirrorBot",
        "niche_saturation_bot": "NicheSaturationBot",
        "market_manipulation_bot": "MarketManipulationBot",
        "passive_discovery_bot": "PassiveDiscoveryBot",
        "preliminary_research_bot": "PreliminaryResearchBot",
        "report_generation_bot": "ReportGenerationBot",
        "resource_allocation_bot": "ResourceAllocationBot",
        "resources_bot": "ResourcesBot",
        "scalability_assessment_bot": "ScalabilityAssessmentBot",
        "strategy_prediction_bot": "StrategyPredictionBot",
        "structural_evolution_bot": "StructuralEvolutionBot",
        "text_research_bot": "TextResearchBot",
        "video_research_bot": "VideoResearchBot",
        "ai_counter_bot": "AICounterBot",
        "dynamic_resource_allocator_bot": "DynamicResourceAllocator",
        "diagnostic_manager": "DiagnosticManager",
        "idea_search_bot": "KeywordBank",
        "newsreader_bot": "NewsDB",
    }

    for mod, name in bot_modules.items():
        _stub_module(monkeypatch, f"menace.{mod}", **{name: DummyObj})

    class DummyThread:
        def __init__(self, target=None, daemon=None):
            self.target = target
        def start(self):
            pass

    class DummySP:
        def __init__(self, *, packages=None):
            self.packages = list(packages or [])
        def ensure_packages(self):
            installs.append(self.packages)

    _stub_module(monkeypatch, "menace.system_provisioner", SystemProvisioner=DummySP)

    installs = []

    path = Path(__file__).resolve().parents[1] / "menace_master.py"  # path-ignore
    spec = importlib.util.spec_from_file_location("menace_master", path)
    mm = importlib.util.module_from_spec(spec)
    sys.modules["menace_master"] = mm
    spec.loader.exec_module(mm)

    mm.requests = None
    monkeypatch.setattr(mm.importlib, "import_module", lambda name: types.ModuleType(name))
    monkeypatch.setattr(mm.threading, "Thread", DummyThread)

    thread = mm._start_dependency_watchdog()

    assert installs == [["python3-requests", "requests"]]
    assert isinstance(thread, DummyThread)
