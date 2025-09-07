import os
import sys
import types
from pathlib import Path

sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
sys.modules.setdefault("cryptography.hazmat.primitives", types.ModuleType("primitives"))
sys.modules.setdefault("cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric"))
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric.ed25519", types.ModuleType("ed25519")
)
ed = sys.modules["cryptography.hazmat.primitives.asymmetric.ed25519"]
ed.Ed25519PrivateKey = types.SimpleNamespace(generate=lambda: object())
ed.Ed25519PublicKey = object
serialization = types.ModuleType("serialization")
primitives = sys.modules["cryptography.hazmat.primitives"]
primitives.serialization = serialization
sys.modules.setdefault("cryptography.hazmat.primitives.serialization", serialization)
sys.modules.setdefault("jinja2", types.ModuleType("jinja2"))
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))
sys.modules.setdefault("env_config", types.SimpleNamespace(DATABASE_URL="sqlite:///"))
sys.modules.setdefault("httpx", types.ModuleType("httpx"))

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")
os.environ["MENACE_LOCAL_DB_PATH"] = "/tmp/menace_local.db"
os.environ["MENACE_SHARED_DB_PATH"] = "/tmp/menace_shared.db"

sys.modules.setdefault(
    "dynamic_path_router",
    types.SimpleNamespace(
        resolve_path=lambda p: Path(p),
        resolve_dir=lambda p: Path(p),
        resolve_module_path=lambda m: Path(m.replace(".", "/") + ".py"),
        path_for_prompt=lambda p: Path(p).as_posix(),
    ),
)

_stubs = [
    "menace.environment_bootstrap",
    "menace.code_database",
    "code_database",
    "menace.menace_memory_manager",
    "menace.model_automation_pipeline",
    "menace.self_coding_engine",
    "menace.self_coding_manager",
    "menace.quick_fix_engine",
    "menace.error_bot",
    "menace.error_logger",
    "menace.bot_development_bot",
    "menace.implementation_pipeline",
    "menace.task_handoff_bot",
    "menace.scalability_pipeline",
    "menace.ipo_implementation_pipeline",
    "menace.bot_testing_bot",
    "menace.deployment_bot",
]

for name in _stubs:
    if name not in sys.modules:
        stub = types.ModuleType(name)
        if name.endswith("task_handoff_bot"):
            class TaskInfo:
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)

            stub.TaskInfo = TaskInfo
        if name.endswith("deployment_bot"):
            class DeploymentSpec:
                def __init__(self, *a, **k):
                    pass

            class DeploymentBot:
                def deploy(self, *a, **k):
                    pass

            stub.DeploymentSpec = DeploymentSpec
            stub.DeploymentBot = DeploymentBot
        if name.endswith("environment_bootstrap"):
            class EnvironmentBootstrapper:
                def bootstrap(self):
                    pass

            stub.EnvironmentBootstrapper = EnvironmentBootstrapper
        if name.endswith("code_database"):
            class CodeDB:
                pass

            stub.CodeDB = CodeDB
        if name.endswith("menace_memory_manager"):
            class MenaceMemoryManager:
                def store(self, *a, **k):
                    pass

            stub.MenaceMemoryManager = MenaceMemoryManager
        if name.endswith("model_automation_pipeline"):
            class ModelAutomationPipeline:
                pass

            stub.ModelAutomationPipeline = ModelAutomationPipeline
        if name.endswith("self_coding_engine"):
            class SelfCodingEngine:
                def __init__(self, *a, **k):
                    self.memory_mgr = types.SimpleNamespace(store=lambda *a, **k: None)

            stub.SelfCodingEngine = SelfCodingEngine
        if name.endswith("self_coding_manager"):
            class SelfCodingManager:
                def __init__(self, *a):
                    pass

            stub.SelfCodingManager = SelfCodingManager
        if name.endswith("quick_fix_engine"):
            class QuickFixEngine:
                def __init__(self, *a):
                    pass

                def run_and_validate(self, *a):
                    pass

            stub.QuickFixEngine = QuickFixEngine
        if name.endswith("error_bot"):
            class ErrorDB:
                def __init__(self, *a, **k):
                    class Conn:
                        def execute(self, *a, **k):
                            class C:
                                def fetchall(self):
                                    return []

                            return C()

                    self.conn = Conn()

            stub.ErrorDB = ErrorDB
        if name.endswith("self_debugger_sandbox"):
            class SelfDebuggerSandbox:
                def __init__(self, *a, **k):
                    pass

                def analyse_and_fix(self):
                    pass

            stub.SelfDebuggerSandbox = SelfDebuggerSandbox
        if name.endswith("error_logger"):
            class ErrorLogger:
                def __init__(self, *a, **k):
                    pass

            stub.ErrorLogger = ErrorLogger
        if name.endswith("bot_development_bot"):
            class BotDevelopmentBot:
                def __init__(self, *a, **k):
                    pass

            stub.BotDevelopmentBot = BotDevelopmentBot
        if name.endswith("implementation_pipeline"):
            class ImplementationPipeline:
                def __init__(self, *a, **k):
                    pass

                def run(self, *a):
                    pass

            stub.ImplementationPipeline = ImplementationPipeline
        if name.endswith("scalability_pipeline"):
            class ScalabilityPipeline:
                def __init__(self, *a, **k):
                    pass

                def run(self, *a):
                    pass

            stub.ScalabilityPipeline = ScalabilityPipeline
        if name.endswith("ipo_implementation_pipeline"):
            class IPOImplementationPipeline:
                def __init__(self, *a, **k):
                    pass

                def run(self, *a):
                    pass

            stub.IPOImplementationPipeline = IPOImplementationPipeline
        if name.endswith("bot_testing_bot"):
            class BotTestingBot:
                def run_unit_tests(self, *a):
                    pass

            stub.BotTestingBot = BotTestingBot
        sys.modules[name] = stub

import menace.launch_menace_bots as lmb  # noqa: E402
sys.modules.pop("menace.self_debugger_sandbox", None)


def test_extract_docstrings_helper():
    code = '''"""Module doc"""

def a():
    """Func A"""
    pass

def b():
    pass
'''
    desc, docs = lmb._extract_docstrings(code)
    assert desc == "Module doc"
    assert docs["a"] == "Func A"
    assert "b" not in docs


def test_debug_and_deploy_runs_sandbox(monkeypatch, tmp_path):
    calls = []

    class DummySandbox:
        def __init__(self, *a, **k):
            pass

        def analyse_and_fix(self):
            calls.append("fix")

    monkeypatch.setattr(lmb, "SelfDebuggerSandbox", DummySandbox)
    (tmp_path / "mod.py").write_text("def run():\n    pass\n")  # path-ignore
    builder = types.SimpleNamespace(refresh_db_weights=lambda: None)
    lmb.debug_and_deploy(tmp_path, context_builder=builder)
    assert "fix" in calls
