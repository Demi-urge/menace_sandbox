from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

from self_improvement import meta_planning


def _counter() -> types.SimpleNamespace:
    def _labels(**_kwargs: object) -> types.SimpleNamespace:
        return counter

    def _inc(*_args: object, **_kwargs: object) -> None:
        return None

    counter = types.SimpleNamespace(labels=_labels, inc=_inc)
    return counter


def test_initialize_autonomous_sandbox_handles_missing_meta_planner(
    monkeypatch, tmp_path, capsys
) -> None:
    import sandbox_runner.bootstrap as bootstrap

    # Make bootstrap side effects inert for the test harness.
    monkeypatch.setattr(bootstrap, "auto_configure_env", lambda settings: None)
    monkeypatch.setattr(bootstrap, "ensure_vector_service", lambda: None)
    monkeypatch.setattr(bootstrap, "_ensure_sqlite_db", lambda path: None)
    monkeypatch.setattr(bootstrap, "_start_optional_services", lambda *a, **k: None)
    monkeypatch.setattr(bootstrap, "_self_improvement_warmup", lambda: None)
    monkeypatch.setattr(bootstrap, "repo_root", lambda: tmp_path)
    monkeypatch.setattr(bootstrap, "resolve_path", lambda value: Path(value))
    monkeypatch.setattr(bootstrap, "_INITIALISED", False)
    monkeypatch.setattr(bootstrap, "_SELF_IMPROVEMENT_THREAD", None)

    # Replace Prometheus counters with lightweight stubs.
    monkeypatch.setattr(bootstrap, "sandbox_restart_total", _counter())
    monkeypatch.setattr(bootstrap, "environment_failure_total", _counter())
    monkeypatch.setattr(bootstrap, "sandbox_crashes_total", _counter())

    class _DummyRegistry:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def mark_missing(self, **info: object) -> None:
            self.calls.append(("missing", dict(info)))

        def mark_available(self, **info: object) -> None:
            self.calls.append(("available", dict(info)))

        def summary(self) -> dict[str, object]:
            return {}

    monkeypatch.setattr(bootstrap, "dependency_registry", _DummyRegistry())

    monkeypatch.setattr(meta_planning, "_META_PLANNER_RESOLVED", False)
    monkeypatch.setattr(meta_planning, "MetaWorkflowPlanner", None)

    def _start_cycle_stub(*_args: object, **_kwargs: object):
        meta_planning.resolve_meta_workflow_planner()

        class _Thread:
            def __init__(self) -> None:
                self._thread = self

            def start(self) -> None:
                return None

            def join(self, *_args: object, **_kwargs: object) -> None:
                return None

            def stop(self) -> None:
                return None

            def is_alive(self) -> bool:
                return True

        return _Thread()

    monkeypatch.setattr(meta_planning, "start_self_improvement_cycle", _start_cycle_stub)
    monkeypatch.setattr(meta_planning, "stop_self_improvement_cycle", lambda: None)

    api_stub = types.ModuleType("self_improvement.api")
    api_stub.init_self_improvement = lambda settings: settings
    api_stub.start_self_improvement_cycle = meta_planning.start_self_improvement_cycle
    api_stub.stop_self_improvement_cycle = meta_planning.stop_self_improvement_cycle

    monkeypatch.setitem(sys.modules, "self_improvement.api", api_stub)
    monkeypatch.setitem(sys.modules, "menace.self_improvement.api", api_stub)

    attempted: list[str] = []
    original_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name.endswith("meta_workflow_planner"):
            attempted.append(name)
            raise ModuleNotFoundError(name)
        return original_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    for mod_name in (
        "meta_workflow_planner",
        "menace.meta_workflow_planner",
    ):
        if mod_name in sys.modules:
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    data_dir = tmp_path / "sandbox-data"
    settings = types.SimpleNamespace(
        sandbox_data_dir=str(data_dir),
        optional_service_versions={},
        sandbox_required_db_files=(),
        alignment_baseline_metrics_path="",
        menace_env_file=str(tmp_path / ".env"),
        menace_mode="development",
        required_system_tools=[],
        required_python_packages=[],
        optional_python_packages=[],
    )

    bootstrap.initialize_autonomous_sandbox(
        settings,
        start_services=False,
        start_self_improvement=True,
    )

    out = capsys.readouterr().out
    assert "🧬 K: self-improvement startup evaluation" in out
    assert "🧱 SI-3: modules imported" in out
    assert attempted, "meta_workflow_planner import should be attempted"
