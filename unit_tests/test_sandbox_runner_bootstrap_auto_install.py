import sys
import types
from pathlib import Path

import pytest

_COUNTER = types.SimpleNamespace(
    labels=lambda *a, **k: types.SimpleNamespace(inc=lambda *args, **kwargs: None)
)

_METRICS_STUB = types.SimpleNamespace(
    sandbox_restart_total=_COUNTER,
    environment_failure_total=_COUNTER,
    sandbox_crashes_total=types.SimpleNamespace(inc=lambda *a, **k: None),
)
sys.modules.setdefault("metrics_exporter", _METRICS_STUB)
sys.modules.setdefault("sandbox_runner.metrics_exporter", _METRICS_STUB)
sys.modules.setdefault(
    "sandbox_runner.cycle", types.SimpleNamespace(ensure_vector_service=lambda: None)
)
sys.modules.setdefault("sandbox_runner.cli", types.SimpleNamespace(main=lambda _=None: None))
menace_stub = sys.modules.setdefault("menace", types.SimpleNamespace())
menace_stub.auto_env_setup = getattr(
    menace_stub, "auto_env_setup", types.SimpleNamespace(ensure_env=lambda *_: None)
)
menace_stub.default_config_manager = getattr(
    menace_stub,
    "default_config_manager",
    types.SimpleNamespace(
        DefaultConfigManager=lambda *_: types.SimpleNamespace(apply_defaults=lambda: None)
    ),
)
sys.modules.setdefault("menace.auto_env_setup", menace_stub.auto_env_setup)
sys.modules.setdefault("menace.default_config_manager", menace_stub.default_config_manager)

import sandbox_runner.bootstrap as bootstrap


class DummyConfigManager:
    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def apply_defaults(self) -> None:  # pragma: no cover - trivial
        return None


def _make_settings(tmp_path: Path) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        menace_env_file=str(tmp_path / "env"),
        sandbox_data_dir=str(tmp_path / "data"),
        sandbox_repo_path=str(tmp_path),
        alignment_baseline_metrics_path=str(tmp_path / "metrics.json"),
        sandbox_required_db_files=[],
        optional_service_versions={},
        required_env_vars=[],
        menace_mode="test",
    )


def test_bootstrap_environment_auto_installs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)

    monkeypatch.setattr(bootstrap, "load_sandbox_settings", lambda: settings)
    monkeypatch.setattr(bootstrap, "ensure_env", lambda *_: None)
    monkeypatch.setattr(bootstrap, "DefaultConfigManager", lambda *_: DummyConfigManager())

    verify_calls: list[dict[str, list[str]]] = []

    def fake_verify(_: bootstrap.SandboxSettings) -> dict[str, list[str]]:
        verify_calls.append({})
        if len(verify_calls) == 1:
            return {"python": ["pkg-one"], "optional": ["pkg-two"]}
        return {}

    monkeypatch.setattr(bootstrap, "_verify_required_dependencies", fake_verify)

    install_calls: list[dict[str, list[str]]] = []

    def fake_auto_install(errors: dict[str, list[str]]) -> bool:
        install_calls.append(errors.copy())
        return True

    monkeypatch.setattr(bootstrap, "_auto_install_missing_python_packages", fake_auto_install)

    sentinel = object()
    monkeypatch.setattr(bootstrap, "initialize_autonomous_sandbox", lambda s: sentinel)

    result = bootstrap._bootstrap_environment(auto_install=True)

    assert result is sentinel
    assert len(verify_calls) == 2
    assert len(install_calls) == 1
    assert install_calls[0]["python"] == ["pkg-one"]
    assert install_calls[0]["optional"] == ["pkg-two"]


def test_bootstrap_environment_auto_install_skipped_for_system(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = _make_settings(tmp_path)

    monkeypatch.setattr(bootstrap, "load_sandbox_settings", lambda: settings)
    monkeypatch.setattr(bootstrap, "ensure_env", lambda *_: None)
    monkeypatch.setattr(bootstrap, "DefaultConfigManager", lambda *_: DummyConfigManager())

    monkeypatch.setattr(
        bootstrap,
        "_verify_required_dependencies",
        lambda *_: {"system": ["ffmpeg"]},
    )

    def fail_auto_install(_: dict[str, list[str]]) -> bool:
        raise AssertionError("auto-install should not run when only system packages are missing")

    monkeypatch.setattr(bootstrap, "_auto_install_missing_python_packages", fail_auto_install)

    with pytest.raises(SystemExit) as exc:
        bootstrap._bootstrap_environment(auto_install=True)

    assert "Missing system packages" in str(exc.value)


def test_auto_install_helper_invokes_pip(monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    def fake_run(cmd, check):
        commands.append(cmd)
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(bootstrap.subprocess, "run", fake_run)

    attempted = bootstrap._auto_install_missing_python_packages(
        {"python": ["pkg-a", "pkg-b"], "optional": ["pkg-b", "pkg-c"]}
    )

    assert attempted is True
    expected = [
        [bootstrap.sys.executable, "-m", "pip", "install", "pkg-a"],
        [bootstrap.sys.executable, "-m", "pip", "install", "pkg-b"],
        [bootstrap.sys.executable, "-m", "pip", "install", "pkg-c"],
    ]
    assert commands == expected


def test_auto_install_helper_no_packages() -> None:
    attempted = bootstrap._auto_install_missing_python_packages({"system": ["ffmpeg"]})
    assert attempted is False
