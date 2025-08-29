import importlib
import importlib.util
import pathlib
import sys
import logging

# Load package to allow relative imports
spec = importlib.util.spec_from_file_location(
    'menace', pathlib.Path(__file__).resolve().parents[1] / '__init__.py'
)
menace_pkg = importlib.util.module_from_spec(spec)
sys.modules['menace'] = menace_pkg
spec.loader.exec_module(menace_pkg)

scm_module = importlib.import_module('menace.self_coding_manager')
sie_module = importlib.import_module('menace.self_improvement')
mutation_logger = importlib.import_module('menace.mutation_logger')


class DummyPipeline:
    def run(self, bot_name: str, energy: int = 1):
        class Result:
            roi = type('Roi', (), {'roi': 1.5})()
        return Result()


class DummyDataBot:
    def __init__(self):
        self.calls = 0

    def roi(self, bot_name: str) -> float:
        self.calls += 1
        return [0.5, 1.5][self.calls - 1]

    def log_evolution_cycle(self, *args, **kwargs):
        pass


class DummyCodingEngine:
    patch_db = None

    def apply_patch(self, *args, **kwargs):
        return 1, False, 1.0


def test_run_patch_records_outcome(monkeypatch):
    recorded = []

    monkeypatch.setattr(mutation_logger, 'log_mutation', lambda **_: 99)
    def capture(event_id, after_metric, roi, performance):
        recorded.append((event_id, after_metric, roi, performance))
    monkeypatch.setattr(mutation_logger, 'record_mutation_outcome', capture)

    manager = scm_module.SelfCodingManager(
        DummyCodingEngine(),
        DummyPipeline(),
        data_bot=DummyDataBot(),
    )
    manager.run_patch(pathlib.Path('x.py'), 'test')

    assert recorded == [(99, 1.5, 1.5, 1.0)]


class DummySelfCodingEngine:
    patch_db = None

    def apply_patch(self, *args, **kwargs):
        return 5, False, 2.0


def test_optimize_self_records_outcome(monkeypatch):
    recorded = []

    monkeypatch.setattr(mutation_logger, 'log_mutation', lambda **_: 42)
    def capture(event_id, after_metric, roi, performance):
        recorded.append((event_id, after_metric, roi, performance))
    monkeypatch.setattr(mutation_logger, 'record_mutation_outcome', capture)

    engine = sie_module.SelfImprovementEngine.__new__(sie_module.SelfImprovementEngine)
    engine.self_coding_engine = DummySelfCodingEngine()
    engine._last_patch_id = None
    engine._last_mutation_id = None
    engine.logger = logging.getLogger('test')

    engine._optimize_self()

    assert recorded == [(42, 2.0, 2.0, 2.0)]
