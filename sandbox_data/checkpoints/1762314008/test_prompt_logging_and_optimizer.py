import json
from pathlib import Path
import json

from menace_sandbox.prompt_types import Prompt
from menace_sandbox.prompt_optimizer import PromptOptimizer
from menace_sandbox.prompt_evolution_memory import PromptEvolutionMemory
from menace_sandbox.self_coding_engine import SelfCodingEngine


class DummyCodeDB:
    def search(self, _desc: str):
        return []


class DummyMemory:
    def log_interaction(self, *args, **kwargs):
        pass


def make_engine(tmp_path: Path):
    success_log = tmp_path / "success.json"
    failure_log = tmp_path / "failure.json"
    stats_path = tmp_path / "stats.json"
    logger = PromptEvolutionMemory(success_path=success_log, failure_path=failure_log)
    optimizer = PromptOptimizer(success_log, failure_log, stats_path=stats_path)
    engine = SelfCodingEngine(
        DummyCodeDB(),
        DummyMemory(),
        prompt_evolution_memory=logger,
        prompt_optimizer=optimizer,
        audit_trail_path=str(tmp_path / "audit.log"),
        context_builder=types.SimpleNamespace(
            build_context=lambda *a, **k: {},
            refresh_db_weights=lambda *a, **k: None,
        ),
    )
    return engine, success_log, failure_log, optimizer


def read_lines(path: Path):
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_prompt_evolution_logging(tmp_path):
    engine, success_log, failure_log, optimizer = make_engine(tmp_path)

    engine._last_prompt = Prompt(
        user="do it",
        system="sys",
        examples=["ex1"],
    )
    engine.prompt_engine.last_metadata = {
        "tone": "neutral",
        "headers": ["H"],
        "example_order": ["success"],
    }

    class R:
        stdout = "ok"
        runtime = 1.0

    engine._log_prompt_evolution(
        "p1",
        True,
        R(),
        roi_delta=1.2,
        coverage=0.8,
        module="mod1",
        action="act1",
    )

    engine._last_prompt = Prompt(
        user="fail",
        system="sys",
        examples=["ex2"],
    )
    engine.prompt_engine.last_metadata = {
        "tone": "neutral",
        "headers": ["H"],
        "example_order": ["failure"],
    }

    class R2:
        stdout = "no"
        runtime = 2.0

    engine._log_prompt_evolution(
        "p2",
        False,
        R2(),
        roi_delta=-0.5,
        coverage=0.4,
        module="mod2",
        action="act2",
    )

    success_entries = read_lines(success_log)
    failure_entries = read_lines(failure_log)

    assert success_entries and failure_entries
    s = success_entries[0]
    f = failure_entries[0]

    assert s["module"] == "mod1"
    assert s["action"] == "act1"
    assert s["prompt"]["system"] == "sys"
    assert s["prompt"]["user"] == "do it"
    assert s["prompt"]["examples"] == ["ex1"]
    assert s["prompt_text"]
    assert "tone" in s["prompt"]["metadata"]
    assert "roi_delta" in s["roi"] and "coverage" in s["roi"]
    assert s["roi"].get("runtime_improvement") == 0.0

    assert f["module"] == "mod2"
    assert f["action"] == "act2"
    assert f["prompt"]["user"] == "fail"
    assert f["prompt"]["examples"] == ["ex2"]
    assert f["roi"].get("runtime_improvement") == -1.0

    entries = optimizer._load_logs()
    successes = {e["success"] for e in entries}
    assert successes == {True, False}


def test_prompt_optimizer_ranking(tmp_path):
    success_log = tmp_path / "success.json"
    failure_log = tmp_path / "failure.json"

    s_entries = [
        {
            "module": "m",
            "action": "a",
            "prompt": "# H1\nExample: foo",
            "success": True,
            "roi": 2.0,
        },
        {
            "module": "m",
            "action": "a",
            "prompt": "# H2\nExample: foo",
            "success": True,
            "roi": 5.0,
        },
    ]
    f_entries = [
        {
            "module": "m",
            "action": "a",
            "prompt": "# H1\nExample: foo",
            "success": False,
            "roi": -1.0,
        },
    ]
    success_log.write_text("\n".join(json.dumps(e) for e in s_entries))
    failure_log.write_text("\n".join(json.dumps(e) for e in f_entries))

    opt = PromptOptimizer(success_log, failure_log, stats_path=tmp_path / "stats.json")
    suggestions = opt.suggest_format("m", "a", limit=2)
    assert suggestions[0]["structured_sections"] == ["H2"]
    assert suggestions[1]["structured_sections"] == ["H1"]
    assert suggestions[0]["success_rate"] > suggestions[1]["success_rate"]
    assert suggestions[0]["weighted_roi"] > suggestions[1]["weighted_roi"]
    assert "avg_runtime_improvement" in suggestions[0]


def test_prompt_optimizer_runtime_weighting(tmp_path):
    success_log = tmp_path / "success.json"
    failure_log = tmp_path / "failure.json"

    # Two configurations with differing runtime improvements
    s_entries = [
        {
            "module": "m",
            "action": "a",
            "prompt": "# H1\nExample: foo",
            "success": True,
            "roi": 1.0,
            "runtime_improvement": 1.0,
        },
        {
            "module": "m",
            "action": "a",
            "prompt": "# H1\nExample: foo",
            "success": True,
            "roi": 3.0,
            "runtime_improvement": 1.0,
        },
        {
            "module": "m",
            "action": "a",
            "prompt": "# H2\nExample: foo",
            "success": True,
            "roi": 1.0,
            "runtime_improvement": 0.5,
        },
        {
            "module": "m",
            "action": "a",
            "prompt": "# H2\nExample: foo",
            "success": True,
            "roi": 3.0,
            "runtime_improvement": 2.0,
        },
    ]
    success_log.write_text("\n".join(json.dumps(e) for e in s_entries))
    failure_log.write_text("")

    opt = PromptOptimizer(
        success_log,
        failure_log,
        stats_path=tmp_path / "stats.json",
        weight_by="runtime",
    )
    suggestions = opt.suggest_format("m", "a", limit=2)
    assert suggestions[0]["structured_sections"] == ["H2"]
    assert suggestions[1]["structured_sections"] == ["H1"]
    assert suggestions[0]["weighted_roi"] > suggestions[1]["weighted_roi"]
    assert suggestions[0]["avg_runtime_improvement"] > suggestions[1]["avg_runtime_improvement"]
