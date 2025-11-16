import importlib
import sys
import types
from pathlib import Path

import dynamic_path_router as dpr  # noqa: E402

SR_NAME = Path(dpr.resolve_path("sandbox_runner.py")).name


def _make_repo(tmp_path: Path, layout: str) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    if layout != "no_git":
        (root / ".git").mkdir(parents=True)
    else:
        root.mkdir()
    if layout == "submodule":
        sub = root / "submodule"
        (sub / ".git").mkdir(parents=True)
        target = sub / SR_NAME
        target.parent.mkdir(parents=True, exist_ok=True)
    elif layout == "nested":
        target = root / "other" / SR_NAME
        target.parent.mkdir(parents=True, exist_ok=True)
    else:
        target = root / SR_NAME
    target.write_text("print('hi')\n")
    return root, target


def test_env_var_overrides(monkeypatch, tmp_path):
    repo, target = _make_repo(tmp_path, "standard")
    for env in ["MENACE_ROOT", "SANDBOX_REPO_PATH"]:
        monkeypatch.setenv(env, str(repo))
        dpr.clear_cache()
        assert dpr.get_project_root() == repo.resolve()
        assert dpr.resolve_path("sandbox_runner.py") == target.resolve()
        monkeypatch.delenv(env)


def test_repo_without_git_uses_os_walk(monkeypatch, tmp_path):
    repo, target = _make_repo(tmp_path, "nested")
    monkeypatch.setenv("MENACE_ROOT", str(repo))
    dpr.clear_cache()
    calls = []
    real_walk = dpr.os.walk

    def track(path, *args, **kw):
        calls.append(path)
        return real_walk(path, *args, **kw)

    monkeypatch.setattr(dpr.os, "walk", track)
    resolved = dpr.resolve_path("sandbox_runner.py")
    assert resolved == target.resolve()
    assert calls


def test_nested_repo_with_submodule(monkeypatch, tmp_path):
    repo, target = _make_repo(tmp_path, "submodule")
    monkeypatch.chdir(repo / "submodule")
    monkeypatch.setenv("MENACE_ROOT", str(repo))
    dpr.clear_cache()
    resolved = dpr.resolve_path("sandbox_runner.py")
    assert resolved == target.resolve()


def test_self_coding_scheduler_invokes_resolve_path(monkeypatch, tmp_path):
    repo, target = _make_repo(tmp_path, "standard")
    monkeypatch.setenv("MENACE_ROOT", str(repo))
    dpr.clear_cache()
    calls = []

    def spy(name: str):
        calls.append(name)
        return target if name == "sandbox_metrics.yaml" else repo / name

    monkeypatch.setattr(dpr, "resolve_path", spy)
    menace_pkg = types.ModuleType("menace")
    menace_pkg.__path__ = [str(Path(dpr.resolve_path(".")))]
    sys.modules["menace"] = menace_pkg
    stubs = {
        "menace.self_coding_manager": types.SimpleNamespace(
            SelfCodingManager=object
        ),
        "menace.data_bot": types.SimpleNamespace(DataBot=object),
        "menace.advanced_error_management": types.SimpleNamespace(
            AutomatedRollbackManager=object
        ),
        "menace.sandbox_settings": types.SimpleNamespace(
            SandboxSettings=type(
                "S", (), {
                    "self_coding_interval": 1,
                    "self_coding_roi_drop": 0.0,
                    "self_coding_error_increase": 0.0,
                }
            )
        ),
        "menace.error_parser": types.SimpleNamespace(ErrorParser=object),
        "sandbox_runner.workflow_sandbox_runner": types.SimpleNamespace(
            WorkflowSandboxRunner=object
        ),
    }
    for name, mod in stubs.items():
        sys.modules[name] = mod
    scheduler_mod = importlib.import_module("menace.self_coding_scheduler")
    manager = types.SimpleNamespace(bot_name="bot", engine=object())
    data_bot = types.SimpleNamespace(
        roi=lambda _: 0.0, db=types.SimpleNamespace(fetch=lambda _: [])
    )
    scheduler_mod.SelfCodingScheduler(manager, data_bot)
    assert calls
