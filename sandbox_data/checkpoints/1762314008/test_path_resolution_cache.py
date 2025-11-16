import importlib


import importlib
import types


class DummyBuilder:
    def refresh_db_weights(self):
        pass


class DummyManager:
    def __init__(self):
        self.evolution_orchestrator = types.SimpleNamespace(provenance_token="tok", event_bus=None)

    def generate_patch(self, module, description="", context_builder=None, provenance_token="", **kwargs):  # pragma: no cover - stub
        return 1


def test_resolve_path_updates_after_repo_move(monkeypatch, tmp_path):
    repo_a = tmp_path / "repo_a"
    (repo_a / ".git").mkdir(parents=True)
    file_a = repo_a / "sandbox_runner.py"  # path-ignore
    file_a.write_text("a\n")

    repo_b = tmp_path / "repo_b"
    (repo_b / ".git").mkdir(parents=True)
    file_b = repo_b / "nested" / "sandbox_runner.py"  # path-ignore
    file_b.parent.mkdir(parents=True)
    file_b.write_text("b\n")

    monkeypatch.setenv("MENACE_ROOT", str(repo_a))
    import menace.dynamic_path_router as dpr
    dpr.clear_cache()
    assert dpr.resolve_path("sandbox_runner.py") == file_a.resolve()  # path-ignore

    monkeypatch.setenv("MENACE_ROOT", str(repo_b))
    # Cache still points to repo_a
    assert dpr.resolve_path("sandbox_runner.py") == file_a.resolve()  # path-ignore

    dpr.clear_cache()
    resolved = dpr.resolve_path("sandbox_runner.py")  # path-ignore
    assert resolved == file_b.resolve()
    assert dpr.path_for_prompt("sandbox_runner.py") == resolved.as_posix()  # path-ignore


def test_error_logger_path_for_prompt_cache(monkeypatch, tmp_path):
    repo_a = tmp_path / "a"
    (repo_a / ".git").mkdir(parents=True)
    file_a = repo_a / "pkg" / "mod.py"  # path-ignore
    file_a.parent.mkdir(parents=True)
    file_a.write_text("a\n")

    repo_b = tmp_path / "b"
    (repo_b / ".git").mkdir(parents=True)
    file_b = repo_b / "pkg" / "mod.py"  # path-ignore
    file_b.parent.mkdir(parents=True)
    file_b.write_text("b\n")

    monkeypatch.setenv("MENACE_ROOT", str(repo_a))
    import menace.dynamic_path_router as dpr
    dpr.clear_cache()

    el = importlib.reload(importlib.import_module("menace.error_logger"))
    monkeypatch.setattr(el, "cdh", None)
    monkeypatch.setattr(
        el,
        "propose_fix",
        lambda metrics, profile: [("pkg/mod.py", "hint")],  # path-ignore
    )

    class DummyDB:
        def __init__(self):
            self.events = []

        def add_telemetry(self, event):
            self.events.append(event)

    logger = el.ErrorLogger(db=DummyDB(), context_builder=DummyBuilder(), manager=DummyManager())
    events1 = logger.log_fix_suggestions({}, {})
    assert events1 and events1[0].module == file_a.resolve().as_posix()

    monkeypatch.setenv("MENACE_ROOT", str(repo_b))
    events2 = logger.log_fix_suggestions({}, {})
    assert events2 and events2[0].module == file_a.resolve().as_posix()

    dpr.clear_cache()
    events3 = logger.log_fix_suggestions({}, {})
    expected = file_b.resolve().as_posix()
    assert events3 and events3[0].module == expected
    assert dpr.path_for_prompt("pkg/mod.py") == expected  # path-ignore
