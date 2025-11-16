import menace.error_bot as eb
import menace.error_logger as elog
import types
from roi_calculator import propose_fix


class DummyBuilder:
    def refresh_db_weights(self):
        pass


class DummyManager:
    def __init__(self):
        self.calls = []
        self.evolution_orchestrator = types.SimpleNamespace(provenance_token="tok", event_bus=None)

    def generate_patch(self, module, description="", context_builder=None, provenance_token="", **kwargs):  # pragma: no cover - stub
        self.calls.append(module)
        return 1


def test_propose_fix_highlights_missing_metrics():
    profile = {
        "weights": {
            "profitability": 0.4,
            "efficiency": 0.6,
        }
    }
    metrics = {"profitability": 0.8}
    fixes = propose_fix(metrics, profile)
    assert [m for m, _ in fixes[:2]] == ["efficiency", "profitability"]


def test_propose_fix_prioritises_vetoed_bottlenecks():
    profile = {
        "weights": {
            "profitability": 0.5,
            "efficiency": 0.3,
            "security": 0.2,
        },
        "veto": {"security": {"min": 0.5}},
    }
    metrics = {"profitability": 0.1, "efficiency": 0.9}
    fixes = propose_fix(metrics, profile)
    assert [m for m, _ in fixes[:3]] == ["security", "profitability", "efficiency"]


def test_log_fix_suggestions_attaches_and_triggers_hooks(tmp_path, monkeypatch, caplog):
    db = eb.ErrorDB(tmp_path / "e.db")
    manager = DummyManager()
    logger = elog.ErrorLogger(db, context_builder=DummyBuilder(), manager=manager)
    monkeypatch.setattr(
        elog,
        "propose_fix",
        lambda m, p: [("mod1", "hint1"), ("", "generic hint")],
    )
    ticket_file = tmp_path / "tickets.txt"
    monkeypatch.setenv("FIX_TICKET_FILE", str(ticket_file))
    caplog.set_level("INFO", logger=elog.__name__)

    events = logger.log_fix_suggestions({"a": 0.1}, {}, task_id="t1", bot_id="b1")

    assert [e.fix_suggestions for e in events] == [["hint1"], ["generic hint"]]
    assert [e.bottlenecks for e in events] == [["mod1"], []]
    assert manager.calls == ["mod1"]
    assert ticket_file.read_text().strip()
    assert not any("Codex prompt" in r.message for r in caplog.records)
