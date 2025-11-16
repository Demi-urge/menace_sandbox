import sys
import types
import threading
import json

class DummyBus:
    pass

stub_modules = {
    "menace.unified_event_bus": types.ModuleType("menace.unified_event_bus"),
    "menace.data_bot": types.ModuleType("menace.data_bot"),
    "menace.neuroplasticity": types.ModuleType("menace.neuroplasticity"),
    "menace.learning_engine": types.ModuleType("menace.learning_engine"),
    "menace.unified_learning_engine": types.ModuleType("menace.unified_learning_engine"),
    "menace.action_learning_engine": types.ModuleType("menace.action_learning_engine"),
    "menace.evaluation_manager": types.ModuleType("menace.evaluation_manager"),
    "menace.error_bot": types.ModuleType("menace.error_bot"),
    "menace.curriculum_builder": types.ModuleType("menace.curriculum_builder"),
    "sandbox_settings": types.ModuleType("sandbox_settings"),
    "sandbox_runner.bootstrap": types.ModuleType("sandbox_runner.bootstrap"),
    "menace.self_improvement": types.ModuleType("menace.self_improvement"),
    "menace.self_improvement.init": types.ModuleType("menace.self_improvement.init"),
}

stub_modules["menace.unified_event_bus"].EventBus = DummyBus

class MetricsDB:  # minimal stub
    def __init__(self, *a, **k):
        pass

stub_modules["menace.data_bot"].MetricsDB = MetricsDB

class PathwayRecord:
    def __init__(self, **kw):
        self.__dict__.update(kw)

class Outcome:
    SUCCESS = "SUCCESS"

stub_modules["menace.neuroplasticity"].PathwayRecord = PathwayRecord
stub_modules["menace.neuroplasticity"].Outcome = Outcome

stub_modules["menace.learning_engine"].LearningEngine = object
stub_modules["menace.unified_learning_engine"].UnifiedLearningEngine = object
stub_modules["menace.action_learning_engine"].ActionLearningEngine = object

class EvaluationManager:
    def __init__(self, *a, **k):
        pass

stub_modules["menace.evaluation_manager"].EvaluationManager = EvaluationManager
stub_modules["menace.error_bot"].ErrorBot = object
stub_modules["menace.curriculum_builder"].CurriculumBuilder = object

class SandboxSettings:
    def __init__(self):
        import os
        self.self_learning_eval_interval = 0
        self.self_learning_summary_interval = 0
        self.sandbox_data_dir = os.getenv("SANDBOX_DATA_DIR", ".")

stub_modules["sandbox_settings"].SandboxSettings = SandboxSettings
stub_modules["sandbox_settings"].load_sandbox_settings = lambda: SandboxSettings()
stub_modules["sandbox_runner.bootstrap"].initialize_autonomous_sandbox = lambda settings=None: settings
from filelock import FileLock as RealFileLock


def atomic_write(path, data, *, lock=None, binary=False):
    lock = lock or RealFileLock(str(path) + ".lock")
    mode = "wb" if binary else "w"
    with lock:
        with open(path, mode) as fh:
            fh.write(data)


stub_modules["menace.self_improvement.init"].FileLock = RealFileLock
stub_modules["menace.self_improvement.init"]._atomic_write = atomic_write
stub_modules["menace.self_improvement"].init = stub_modules["menace.self_improvement.init"]

for name, mod in stub_modules.items():
    sys.modules.setdefault(name, mod)

from menace.self_learning_coordinator import SelfLearningCoordinator


def test_state_file_concurrent_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDBOX_DATA_DIR", str(tmp_path))
    coord = SelfLearningCoordinator(DummyBus())

    errors: list[Exception] = []
    barrier = threading.Barrier(6)

    def writer(val):
        barrier.wait()
        coord._train_count = val
        coord._save_state()

    def reader():
        barrier.wait()
        try:
            coord._load_state()
        except Exception as exc:  # pragma: no cover - unexpected
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
    threads.append(threading.Thread(target=reader))

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    data = json.loads((tmp_path / "self_learning_state.json").read_text())
    assert data["train_count"] in {0, 1, 2, 3, 4}
