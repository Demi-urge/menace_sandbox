import ast
from pathlib import Path


def _build_coordinator():
    src = Path("self_learning_coordinator.py").read_text()
    tree = ast.parse(src)
    class_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SelfLearningCoordinator")
    wanted = {"__init__", "start", "stop", "_on_memory"}
    methods = [m for m in class_node.body if isinstance(m, ast.FunctionDef) and m.name in wanted]
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

    # simplify training to just count calls
    def _train_all(self, rec, *, source="unknown"):
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
        setattr(cls, name, lambda self, t, p: None)
    return cls


class DummyBus:
    def __init__(self):
        self.subs = {}

    def subscribe(self, topic, cb):
        self.subs.setdefault(topic, []).append(cb)

    def publish(self, topic, payload):
        for cb in list(self.subs.get(topic, [])):
            cb(topic, payload)

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

