import ast
import types
from pathlib import Path


def load_energy_score_engine():
    path = Path(__file__).resolve().parents[1] / 'capital_management_bot.py'  # path-ignore
    source = path.read_text()
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == 'EnergyScoreEngine':
            mod = types.ModuleType('EnergyScoreEngine')
            code = compile(ast.Module(body=[node], type_ignores=[]), path.as_posix(), 'exec')
            exec(code, mod.__dict__)
            return mod.EnergyScoreEngine
    raise AssertionError('EnergyScoreEngine not found')


EnergyScoreEngine = load_energy_score_engine()


def test_history_trim():
    engine = EnergyScoreEngine(history_limit=5)
    for i in range(7):
        engine.compute(
            capital=100.0,
            profit_trend=0.0,
            load=0.1,
            success_rate=0.9,
            deploy_eff=0.8,
            failure_rate=0.1,
            reward=float(i),
        )
    assert len(engine.feature_history) == 5
    assert len(engine.reward_history) == 5
    assert engine.reward_history == [2.0, 3.0, 4.0, 5.0, 6.0]
