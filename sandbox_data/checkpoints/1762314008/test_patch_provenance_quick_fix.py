import os
import sys
import types
import importlib.util
from pathlib import Path

import pytest

# Create lightweight package stubs to load QuickFixEngine without heavy deps
package = types.ModuleType("menace")
package.__path__ = [str(Path(__file__).resolve().parents[2])]
sys.modules["menace"] = package

error_bot = types.ModuleType("menace.error_bot")
error_bot.ErrorDB = object
sys.modules["menace.error_bot"] = error_bot

scm = types.ModuleType("menace.self_coding_manager")
scm.SelfCodingManager = object
sys.modules["menace.self_coding_manager"] = scm

kg = types.ModuleType("menace.knowledge_graph")
kg.KnowledgeGraph = object
sys.modules["menace.knowledge_graph"] = kg

adv = types.ModuleType("menace.advanced_error_management")
adv.AutomatedRollbackManager = object
sys.modules["menace.advanced_error_management"] = adv

flagger = types.ModuleType("menace.human_alignment_flagger")
flagger._collect_diff_data = lambda *a, **k: {}
sys.modules["menace.human_alignment_flagger"] = flagger

agent = types.ModuleType("menace.human_alignment_agent")


class _Agent:
    def evaluate_changes(self, *_a, **_k):
        return {}


agent.HumanAlignmentAgent = _Agent
sys.modules["menace.human_alignment_agent"] = agent

viol = types.ModuleType("menace.violation_logger")
viol.log_violation = lambda *a, **k: None
sys.modules["menace.violation_logger"] = viol

# Load QuickFixEngine
spec = importlib.util.spec_from_file_location(
    "menace.quick_fix_engine",
    Path(__file__).resolve().parents[2] / "quick_fix_engine.py",  # path-ignore
)
quick_fix = importlib.util.module_from_spec(spec)
sys.modules["menace.quick_fix_engine"] = quick_fix
spec.loader.exec_module(quick_fix)
QuickFixEngine = quick_fix.QuickFixEngine


@pytest.fixture(scope="module", autouse=True)
def _cleanup_quick_fix_module():
    yield
    for name in [
        "menace.quick_fix_engine",
        "quick_fix_engine",
        "menace.error_bot",
        "menace.self_coding_manager",
        "menace.knowledge_graph",
        "menace.advanced_error_management",
        "menace.human_alignment_flagger",
        "menace.human_alignment_agent",
        "menace.violation_logger",
        "menace",
    ]:
        sys.modules.pop(name, None)


class DummyGraph:
    def add_telemetry_event(self, *a, **k):
        pass

    def update_error_stats(self, *a, **k):
        pass


def test_quick_fix_records_license_and_alerts(tmp_path, monkeypatch):
    class Manager:
        def __init__(self):
            self.bot_registry = types.SimpleNamespace()
            self.data_bot = types.SimpleNamespace()
            self._last_commit_hash = "hash"
            self._last_patch_id = 1

        def run_patch(self, path, desc, context_meta=None, **kw):
            return types.SimpleNamespace(patch_id=1)

        def register_bot(self, *a, **k):
            return None

        def auto_run_patch(self, path, desc, *, context_meta=None, context_builder=None):
            return {
                "result": object(),
                "commit": self._last_commit_hash,
                "patch_id": self._last_patch_id,
                "summary": {"self_tests": {"failed": 0}},
            }

    class Retriever:
        def search(self, query, top_k, session_id):
            return [
                {
                    "origin_db": "db1",
                    "record_id": "vec1",
                    "score": 0.5,
                    "license": "mit",
                    "semantic_alerts": ["unsafe"],
                }
            ]

    class DummyBuilder:
        def refresh_db_weights(self):
            return None

        def build(self, query, session_id=None, include_vectors=False):
            return ""

    class DummyPatchLogger:
        def __init__(self):
            self.calls = []

        def track_contributors(
            self,
            ids,
            result,
            *,
            patch_id="",
            session_id="",
            contribution=0.0,
            retrieval_metadata=None,
        ):
            self.calls.append((ids, retrieval_metadata or {}))

    patch_logger = DummyPatchLogger()
    engine = QuickFixEngine(
        error_db=None,
        manager=Manager(),
        threshold=1,
        graph=DummyGraph(),
        retriever=Retriever(),
        patch_logger=patch_logger,
        context_builder=DummyBuilder(),
        helper_fn=lambda *a, **k: "",
    )
    (tmp_path / "mod.py").write_text("x=1\n")  # path-ignore
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(engine, "_top_error", lambda bot: ("err", "mod", {}, 1))
    monkeypatch.setattr(quick_fix.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(quick_fix, "generate_code_diff", lambda *a, **k: {})
    monkeypatch.setattr(quick_fix, "flag_risky_changes", lambda *a, **k: [])
    monkeypatch.setattr(quick_fix, "resolve_path", lambda name: tmp_path / name)
    monkeypatch.setattr(quick_fix, "path_for_prompt", lambda p: p)

    engine.run("bot")
    assert patch_logger.calls, "patch logger not invoked"
    _ids, metadata = patch_logger.calls[0]
    assert metadata.get("db1:vec1", {}).get("license") == "mit"
    assert metadata.get("db1:vec1", {}).get("semantic_alerts") == ["unsafe"]


def test_quick_fix_runs_post_patch_cycle(monkeypatch, tmp_path):
    events = []

    class Graph:
        def __init__(self) -> None:
            self.telemetry = []

        def add_telemetry_event(self, *_, **kwargs):
            self.telemetry.append(kwargs)

        def update_error_stats(self, *_a, **_k):
            return None

    class Manager:
        def __init__(self) -> None:
            self.bot_registry = types.SimpleNamespace()
            self.data_bot = types.SimpleNamespace()
            self.event_bus = types.SimpleNamespace(
                publish=lambda topic, payload: events.append((topic, payload))
            )
            self.evolution_orchestrator = types.SimpleNamespace(provenance_token="prov")
            self.engine = types.SimpleNamespace()
            self._last_commit_hash = None
            self._last_patch_id = None
            self._last_validation_summary = None
            self.post_calls = []

        def register_bot(self, *_a, **_k) -> None:
            return None

        def auto_run_patch(self, path, desc, *, context_meta=None, context_builder=None):
            self._last_commit_hash = "commit"
            self._last_patch_id = 99
            return {
                "result": object(),
                "commit": self._last_commit_hash,
                "patch_id": self._last_patch_id,
                "summary": None,
            }

        def run_post_patch_cycle(
            self,
            module_path,
            description,
            *,
            provenance_token,
            context_meta=None,
        ):
            call = (
                Path(module_path),
                description,
                provenance_token,
                dict(context_meta or {}),
            )
            self.post_calls.append(call)
            return {"self_tests": {"failed": 0, "passed": 4}}

    graph = Graph()
    manager = Manager()

    class Builder:
        def refresh_db_weights(self):
            return None

        def build(self, *_a, **_k):
            return ""

    def _best_cluster(_module):
        return None, [], 5

    monkeypatch.setattr(quick_fix, "create_context_builder", lambda: Builder())
    monkeypatch.setattr(quick_fix, "ensure_fresh_weights", lambda _b: None)
    monkeypatch.setattr(quick_fix, "ErrorClusterPredictor", lambda _db: types.SimpleNamespace(best_cluster=_best_cluster))
    monkeypatch.setattr(quick_fix, "generate_code_diff", lambda *a, **k: {})
    monkeypatch.setattr(quick_fix, "flag_risky_changes", lambda *a, **k: [])
    monkeypatch.setattr(quick_fix, "retry_with_backoff", lambda fn, *a, **k: fn())
    monkeypatch.setattr(quick_fix, "compress_snippets", lambda data: data)
    monkeypatch.setattr(quick_fix, "_collect_diff_data", lambda *a, **k: {})
    monkeypatch.setattr(quick_fix, "HumanAlignmentAgent", type("Agent", (), {"evaluate_changes": lambda *_a: {}}))
    monkeypatch.setattr(quick_fix, "extract_target_region", lambda *_a, **_k: None)
    monkeypatch.setattr(quick_fix, "resolve_path", lambda name: tmp_path / name)
    monkeypatch.setattr(quick_fix, "path_for_prompt", lambda path: str(path))

    patch_logger = types.SimpleNamespace(track_contributors=lambda *a, **k: None)
    engine = QuickFixEngine(
        error_db=types.SimpleNamespace(),
        manager=manager,
        threshold=1,
        graph=graph,
        context_builder=Builder(),
        helper_fn=lambda *a, **k: "",
        patch_logger=patch_logger,
    )

    monkeypatch.setattr(engine, "_top_error", lambda _bot: ("etype", "module", {"module": 1}, 5))
    module_path = tmp_path / "module.py"
    module_path.write_text("def foo():\n    return 1\n")
    monkeypatch.chdir(tmp_path)

    engine.run("bot")

    assert manager.post_calls, "post patch cycle not invoked"
    call_path, call_desc, call_token, call_meta = manager.post_calls[0]
    assert call_path == module_path
    assert call_token == "prov"
    assert call_meta.get("module", "").endswith("module.py")
    assert manager._last_validation_summary == {"self_tests": {"failed": 0, "passed": 4}}
    assert any(topic == "quick_fix:patch_start" for topic, _ in events)
    assert not any(topic == "quick_fix:patch_failed" for topic, _ in events)
    assert graph.telemetry and graph.telemetry[0].get("resolved") is True


def test_quick_fix_reports_service_failures(monkeypatch, tmp_path):
    events = []

    class Graph:
        def __init__(self) -> None:
            self.telemetry = []

        def add_telemetry_event(self, *_, **kwargs):
            self.telemetry.append(kwargs)

        def update_error_stats(self, *_a, **_k):
            return None

    class EngineWithRollback:
        def __init__(self) -> None:
            self.rollbacks: list[str] = []

        def rollback_patch(self, patch_id: str) -> None:
            self.rollbacks.append(patch_id)

    class Manager:
        def __init__(self) -> None:
            self.bot_registry = types.SimpleNamespace()
            self.data_bot = types.SimpleNamespace()
            self.event_bus = types.SimpleNamespace(
                publish=lambda topic, payload: events.append((topic, payload))
            )
            self.evolution_orchestrator = types.SimpleNamespace(provenance_token="prov")
            self.engine = EngineWithRollback()
            self._last_commit_hash = None
            self._last_patch_id = None
            self._last_validation_summary = None
            self.post_calls = []

        def register_bot(self, *_a, **_k) -> None:
            return None

        def auto_run_patch(self, path, desc, *, context_meta=None, context_builder=None):
            self._last_commit_hash = "commit"
            self._last_patch_id = 7
            return {
                "result": object(),
                "commit": self._last_commit_hash,
                "patch_id": self._last_patch_id,
                "summary": None,
            }

        def run_post_patch_cycle(
            self,
            module_path,
            description,
            *,
            provenance_token,
            context_meta=None,
        ):
            call = (
                Path(module_path),
                description,
                provenance_token,
                dict(context_meta or {}),
            )
            self.post_calls.append(call)
            return {"self_tests": {"failed": 2, "passed": 3}}

    graph = Graph()
    manager = Manager()

    class Builder:
        def refresh_db_weights(self):
            return None

        def build(self, *_a, **_k):
            return ""

    def _best_cluster(_module):
        return None, [], 5

    monkeypatch.setattr(quick_fix, "create_context_builder", lambda: Builder())
    monkeypatch.setattr(quick_fix, "ensure_fresh_weights", lambda _b: None)
    monkeypatch.setattr(quick_fix, "ErrorClusterPredictor", lambda _db: types.SimpleNamespace(best_cluster=_best_cluster))
    monkeypatch.setattr(quick_fix, "generate_code_diff", lambda *a, **k: {})
    monkeypatch.setattr(quick_fix, "flag_risky_changes", lambda *a, **k: [])
    monkeypatch.setattr(quick_fix, "retry_with_backoff", lambda fn, *a, **k: fn())
    monkeypatch.setattr(quick_fix, "compress_snippets", lambda data: data)
    monkeypatch.setattr(quick_fix, "_collect_diff_data", lambda *a, **k: {})
    monkeypatch.setattr(quick_fix, "HumanAlignmentAgent", type("Agent", (), {"evaluate_changes": lambda *_a: {}}))
    monkeypatch.setattr(quick_fix, "extract_target_region", lambda *_a, **_k: None)
    monkeypatch.setattr(quick_fix, "resolve_path", lambda name: tmp_path / name)
    monkeypatch.setattr(quick_fix, "path_for_prompt", lambda path: str(path))

    patch_logger = types.SimpleNamespace(track_contributors=lambda *a, **k: None)
    engine = QuickFixEngine(
        error_db=types.SimpleNamespace(),
        manager=manager,
        threshold=1,
        graph=graph,
        context_builder=Builder(),
        helper_fn=lambda *a, **k: "",
        patch_logger=patch_logger,
    )

    monkeypatch.setattr(engine, "_top_error", lambda _bot: ("etype", "module", {"module": 1}, 5))
    module_path = tmp_path / "module.py"
    module_path.write_text("def foo():\n    return 1\n")
    monkeypatch.chdir(tmp_path)

    engine.run("bot")

    assert manager.post_calls, "post patch cycle not invoked"
    assert manager._last_validation_summary == {"self_tests": {"failed": 2, "passed": 3}}
    assert graph.telemetry and graph.telemetry[0].get("resolved") is False
    assert manager.engine.rollbacks == ["7"], "rollback not triggered"
    failed_events = [payload for topic, payload in events if topic == "quick_fix:patch_failed"]
    assert failed_events and failed_events[0]["summary"] == {"self_tests": {"failed": 2, "passed": 3}}
    assert failed_events[0]["failed_tests"] == 2
