import json
import subprocess
import sys
import textwrap


def test_pipeline_imports_with_lazy_prediction_manager():
    """Ensure pipeline import defers PredictionManager instantiation."""

    script = textwrap.dedent(
        """
        import importlib
        import json
        import sys
        import types

        def stub_module(name, **attrs):
            module = types.ModuleType(name)
            for key, value in attrs.items():
                setattr(module, key, value)
            sys.modules[name] = module
            return module

        class _DummyConn:
            def execute(self, *args, **kwargs):
                return types.SimpleNamespace(lastrowid=1, fetchall=lambda: [])

            def commit(self):
                return None

        class _DummyRouter:
            menace_id = "router"

            def get_connection(self, name):
                return _DummyConn()

        def _self_coding_managed(*args, **kwargs):
            def decorator(obj):
                return obj

            return decorator

        stub_module("menace_sandbox.bot_registry", BotRegistry=type("BotRegistry", (), {}))
        stub_module("menace_sandbox.coding_bot_interface", self_coding_managed=_self_coding_managed)

        class _ResourceMetrics:
            def __init__(self, cpu=0.0, memory=0.0, disk=0.0, time=0.0):
                self.cpu = cpu
                self.memory = memory
                self.disk = disk
                self.time = time

        class _TemplateDB:
            def __init__(self, *args, **kwargs):
                pass

            def query(self, task):
                return []

            def add(self, task, metrics):
                return None

            def save(self):
                return None

        stub_module(
            "menace_sandbox.resource_prediction_bot",
            ResourceMetrics=_ResourceMetrics,
            TemplateDB=_TemplateDB,
            ResourcePredictionBot=type("ResourcePredictionBot", (), {}),
        )
        stub_module("menace_sandbox.data_bot", DataBot=type("DataBot", (), {"__init__": lambda self, *a, **k: None}))
        stub_module("menace_sandbox.retry_utils", retry=lambda *a, **k: (lambda f: f))
        stub_module("menace_sandbox.databases", MenaceDB=type("MenaceDB", (), {}))
        stub_module("menace_sandbox.contrarian_db", ContrarianDB=type("ContrarianDB", (), {}))
        stub_module("db_router", GLOBAL_ROUTER=None, init_db_router=lambda *a, **k: _DummyRouter())
        stub_module("snippet_compressor", compress_snippets=lambda items: items)

        stub_module(
            "menace_sandbox.task_handoff_bot",
            TaskHandoffBot=type("TaskHandoffBot", (), {"__init__": lambda self, *a, **k: None}),
            TaskInfo=type("TaskInfo", (), {}),
            TaskPackage=type("TaskPackage", (), {"__init__": lambda self, tasks=None: setattr(self, "tasks", tasks or [])}),
            WorkflowDB=type("WorkflowDB", (), {"__init__": lambda self, *a, **k: None}),
        )
        stub_module(
            "menace_sandbox.implementation_optimiser_bot",
            ImplementationOptimiserBot=type("ImplementationOptimiserBot", (), {"__init__": lambda self, *a, **k: None}),
        )
        stub_module("menace_sandbox.chatgpt_enhancement_bot", EnhancementDB=type("EnhancementDB", (), {}))
        stub_module(
            "menace_sandbox.database_manager",
            DB_PATH="db.sqlite",
            update_model=lambda *a, **k: None,
            init_db=lambda *a, **k: None,
        )
        stub_module("menace_sandbox.unified_event_bus", UnifiedEventBus=type("UnifiedEventBus", (), {}))
        stub_module("menace_sandbox.strategy_prediction_bot", StrategyPredictionBot=type("StrategyPredictionBot", (), {}))
        stub_module(
            "menace_sandbox.bot_database",
            BotDB=type(
                "BotDB",
                (),
                {
                    "__init__": lambda self, *a, **k: setattr(
                        self,
                        "conn",
                        types.SimpleNamespace(execute=lambda *aa, **kk: types.SimpleNamespace(fetchall=lambda: [])),
                    ),
                },
            ),
        )
        stub_module("menace_sandbox.code_database", CodeDB=type("CodeDB", (), {}))

        for module_name, attr in {
            "menace_sandbox.performance_assessment_bot": "PerformanceAssessmentBot",
            "menace_sandbox.communication_maintenance_bot": "CommunicationMaintenanceBot",
            "menace_sandbox.operational_monitor_bot": "OperationalMonitoringBot",
            "menace_sandbox.central_database_bot": "CentralDatabaseBot",
            "menace_sandbox.sentiment_bot": "SentimentBot",
            "menace_sandbox.query_bot": "QueryBot",
            "menace_sandbox.memory_bot": "MemoryBot",
            "menace_sandbox.communication_testing_bot": "CommunicationTestingBot",
            "menace_sandbox.discrepancy_detection_bot": "DiscrepancyDetectionBot",
            "menace_sandbox.finance_router_bot": "FinanceRouterBot",
            "menace_sandbox.meta_genetic_algorithm_bot": "MetaGeneticAlgorithmBot",
            "menace_sandbox.offer_testing_bot": "OfferTestingBot",
            "menace_sandbox.research_fallback_bot": "ResearchFallbackBot",
            "menace_sandbox.resource_allocation_optimizer": "ResourceAllocationOptimizer",
            "menace_sandbox.ai_counter_bot": "AICounterBot",
            "menace_sandbox.dynamic_resource_allocator_bot": "DynamicResourceAllocator",
            "menace_sandbox.diagnostic_manager": "DiagnosticManager",
            "menace_sandbox.idea_search_bot": "KeywordBank",
            "menace_sandbox.newsreader_bot": "NewsDB",
            "menace_sandbox.investment_engine": "AutoReinvestmentBot",
            "menace_sandbox.unified_learning_engine": "UnifiedLearningEngine",
            "menace_sandbox.action_planner": "ActionPlanner",
        }.items():
            stub_module(module_name, **{attr: type(attr, (), {})})

        stub_module(
            "menace_sandbox.bot_db_utils",
            wrap_bot_methods=lambda obj: obj,
        )
        stub_module(
            "menace_sandbox.neuroplasticity",
            Outcome=type("Outcome", (), {}),
            PathwayDB=type("PathwayDB", (), {}),
            PathwayRecord=type("PathwayRecord", (), {}),
        )
        stub_module(
            "menace_sandbox.revenue_amplifier",
            RevenueSpikeEvaluatorBot=type("RevenueSpikeEvaluatorBot", (), {}),
            CapitalAllocationBot=type("CapitalAllocationBot", (), {}),
            RevenueEventsDB=type("RevenueEventsDB", (), {}),
        )
        stub_module(
            "menace_sandbox.shared.model_pipeline_core",
            ModelAutomationPipeline=type("ModelAutomationPipeline", (), {}),
        )

        vec_pkg = types.ModuleType("vector_service")
        ctx_mod = types.ModuleType("vector_service.context_builder")

        class _DummyContextBuilder:
            def __init__(self, *args, **kwargs):
                pass

            def refresh_db_weights(self):
                return None

        ctx_mod.ContextBuilder = _DummyContextBuilder
        ctx_mod.FallbackResult = type("FallbackResult", (), {})
        ctx_mod.ErrorResult = type("ErrorResult", (), {})
        vec_pkg.context_builder = ctx_mod
        sys.modules["vector_service"] = vec_pkg
        sys.modules["vector_service.context_builder"] = ctx_mod

        class RaisingPredictionManager:
            calls = 0

            def __init__(self, *args, **kwargs):
                type(self).calls += 1
                raise RuntimeError("prediction manager bootstrap failed")

        pred_mod = types.ModuleType("menace_sandbox.prediction_manager_bot")
        pred_mod.PredictionManager = RaisingPredictionManager
        sys.modules["menace_sandbox.prediction_manager_bot"] = pred_mod

        for name in [
            "menace_sandbox.resource_allocation_bot",
            "menace_sandbox.efficiency_bot",
            "menace_sandbox.pre_execution_roi_bot",
            "menace_sandbox.model_automation_pipeline",
        ]:
            sys.modules.pop(name, None)

        pipeline = importlib.import_module("menace_sandbox.model_automation_pipeline")
        resource_mod = importlib.import_module("menace_sandbox.resource_allocation_bot")
        efficiency_mod = importlib.import_module("menace_sandbox.efficiency_bot")
        pre_roi_mod = importlib.import_module("menace_sandbox.pre_execution_roi_bot")

        cls_resource = resource_mod._prediction_manager_cls()
        cls_efficiency = efficiency_mod._prediction_manager_cls()

        assert cls_resource is RaisingPredictionManager
        assert cls_efficiency is RaisingPredictionManager

        try:
            cls_resource()
        except RuntimeError:
            pass
        else:
            raise AssertionError("resource helper should raise")

        try:
            cls_efficiency()
        except RuntimeError:
            pass
        else:
            raise AssertionError("efficiency helper should raise")

        bot = pre_roi_mod.PreExecutionROIBot(prediction_manager=None)
        if bot.prediction_manager is not None:
            raise AssertionError("prediction manager fallback expected")

        print(json.dumps({"calls": RaisingPredictionManager.calls}))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip())
    assert payload["calls"] >= 2
