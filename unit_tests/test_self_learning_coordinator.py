import ast
import asyncio
from dynamic_path_router import resolve_path


def _build_coordinator():
    src = resolve_path("self_learning_coordinator.py").read_text()
    tree = ast.parse(src)
    class_node = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "SelfLearningCoordinator"
    )
    wanted = {"__init__", "start", "stop", "_on_memory"}
    methods = [
        m
        for m in class_node.body
        if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and m.name in wanted
    ]
    inner = [n for n in class_node.body if isinstance(n, ast.ClassDef) and n.name == "MemoryEvent"]
    new_class = ast.ClassDef("SelfLearningCoordinator", [], [], inner + methods, [])
    module = ast.Module([new_class], type_ignores=[])
    module = ast.fix_missing_locations(module)

    class BaseModel:
        def __init__(self, **data):
            for k, v in data.items():
                setattr(self, k, v)

        @classmethod
        def model_validate(cls, data):
            if not isinstance(data, dict):
                raise ValidationError("invalid")
            return cls(**data)

    class ValidationError(Exception):
        pass

    class PathwayRecord:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class Outcome:
        SUCCESS = "SUCCESS"

    import logging

    class EvaluationManager:
        def __init__(self, *a, **k):
            pass

    ns = {
        "BaseModel": BaseModel,
        "ValidationError": ValidationError,
        "EventBus": object,
        "MetricsDB": object,
        "PathwayRecord": PathwayRecord,
        "Outcome": Outcome,
        "LearningEngine": object,
        "UnifiedLearningEngine": object,
        "ActionLearningEngine": object,
        "EvaluationManager": EvaluationManager,
        "ErrorBot": object,
        "CurriculumBuilder": object,
        "logger": logging.getLogger("slc"),
    }
    exec(compile(module, "<ast>", "exec"), ns)
    cls = ns["SelfLearningCoordinator"]
    # simple __init__

    def __init__(self, event_bus):
        self.event_bus = event_bus
        self.running = False
        self._train_count = 0
        self._subs = []

    cls.__init__ = __init__
    # simplify training to just count calls

    async def _train_all(self, rec, *, source="unknown"):
        self._train_count += 1
        self.last_record = rec

    cls._train_all = _train_all
    for name in [
        "_on_code",
        "_on_pathway",
        "_on_workflow",
        "_on_error",
        "_on_telemetry",
        "_on_metrics",
        "_on_transaction",
        "_on_curriculum",
    ]:
        async def _noop(self, t, p):
            return None

        setattr(cls, name, _noop)
    return cls


class DummyBus:
    def __init__(self):
        self.subs = {}
        self.loop = asyncio.new_event_loop()

    def subscribe_async(self, topic, cb):
        self.subs.setdefault(topic, []).append(cb)

    def publish(self, topic, payload):
        for cb in list(self.subs.get(topic, [])):
            self.loop.run_until_complete(cb(topic, payload))

    def unsubscribe(self, topic, cb):
        lst = self.subs.get(topic, [])
        if cb in lst:
            lst.remove(cb)
        if not lst:
            self.subs.pop(topic, None)


def test_memory_event_triggers_training():
    Bus = DummyBus()
    Coord = _build_coordinator()
    coord = Coord(Bus)
    coord.start()
    Bus.publish("memory:new", {"key": "x"})
    assert coord._train_count == 1
    assert coord.last_record.actions == "x"


def test_stop_unsubscribes():
    Bus = DummyBus()
    Coord = _build_coordinator()
    coord = Coord(Bus)
    coord.start()
    coord.stop()
    assert Bus.subs == {}
    Bus.publish("memory:new", {"key": "y"})
    assert coord._train_count == 0


def test_train_record_restarts_failed_tasks(tmp_path):
    import os
    import asyncio
    import ast
    import logging

    src = resolve_path("self_learning_coordinator.py").read_text()
    tree = ast.parse(src)
    class_node = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SelfLearningCoordinator"
    )
    wanted = {"_train_record", "_partial_train"}
    methods = [
        m for m in class_node.body if isinstance(m, ast.AsyncFunctionDef) and m.name in wanted
    ]
    new_cls = ast.ClassDef("SelfLearningCoordinator", [], [], methods, [])
    module = ast.Module([new_cls], type_ignores=[])
    module = ast.fix_missing_locations(module)
    class PathwayRecord:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    ns = {"asyncio": asyncio, "logger": logging.getLogger("slc"), "PathwayRecord": PathwayRecord}
    exec(compile(module, "<ast>", "exec"), ns)
    SelfLearningCoordinator = ns["SelfLearningCoordinator"]

    def __init__(self, event_bus=None, *, learning_engine=None, unified_engine=None, action_engine=None):
        self.event_bus = event_bus
        self.learning_engine = learning_engine
        self.unified_engine = unified_engine
        self.action_engine = action_engine

    SelfLearningCoordinator.__init__ = __init__
    class Outcome:
        FAILURE = "FAILURE"

    os.environ["SANDBOX_DATA_DIR"] = str(tmp_path)

    class FailOnceEngine:
        def __init__(self) -> None:
            self.calls = 0

        def partial_train(self, rec) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("boom")

    class OkEngine:
        def __init__(self) -> None:
            self.calls = 0

        def partial_train(self, rec) -> None:
            self.calls += 1

    failing = FailOnceEngine()
    ok = OkEngine()
    coord = SelfLearningCoordinator(object(), learning_engine=failing, unified_engine=ok)
    rec = PathwayRecord(
        actions="",
        inputs="",
        outputs="",
        exec_time=0.0,
        resources="",
        outcome=Outcome.FAILURE,
        roi=0.0,
    )
    asyncio.run(coord._train_record(rec))
    assert failing.calls == 2
    assert ok.calls == 1
