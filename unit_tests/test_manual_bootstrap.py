import logging

import manual_bootstrap


class DummySettings:
    sandbox_repo_path = "repo"
    sandbox_data_dir = "data"
    menace_env_file = "settings.env"


def test_configure_environment_defaults():
    env: dict[str, str] = {}
    manual_bootstrap._configure_environment(manual_bootstrap._REPO_ROOT, env)
    assert env["SANDBOX_REPO_PATH"] == str(manual_bootstrap._REPO_ROOT)
    assert env["SANDBOX_DATA_DIR"] == str(manual_bootstrap._DEFAULT_DATA_DIR)


def test_main_skips_environment_bootstrap(monkeypatch):
    calls: list[str] = []

    def fake_bootstrap_environment(*, auto_install: bool):
        calls.append(f"sandbox:{auto_install}")
        return DummySettings()

    class FakeBootstrapper:
        def bootstrap(self) -> None:
            calls.append("environment")

    monkeypatch.setattr(manual_bootstrap, "bootstrap_environment", fake_bootstrap_environment)
    monkeypatch.setattr(manual_bootstrap, "ensure_autonomous_launch", lambda *a, **k: None)
    monkeypatch.setattr(manual_bootstrap, "EnvironmentBootstrapper", lambda: FakeBootstrapper())
    monkeypatch.setattr(manual_bootstrap, "_configure_environment", lambda *a, **k: None)

    exit_code = manual_bootstrap.main(["--skip-environment"])

    assert exit_code == 0
    assert calls == ["sandbox:False"]


def test_main_runs_environment_only(monkeypatch):
    calls: list[str] = []

    class FakeBootstrapper:
        def bootstrap(self) -> None:
            calls.append("environment")

    def fail_bootstrap_environment(**_: object) -> None:
        raise AssertionError("sandbox bootstrap should be skipped")

    monkeypatch.setattr(manual_bootstrap, "bootstrap_environment", fail_bootstrap_environment)
    monkeypatch.setattr(manual_bootstrap, "ensure_autonomous_launch", lambda *a, **k: None)
    monkeypatch.setattr(manual_bootstrap, "EnvironmentBootstrapper", lambda: FakeBootstrapper())
    monkeypatch.setattr(manual_bootstrap, "_configure_environment", lambda *a, **k: None)

    exit_code = manual_bootstrap.main(["--skip-sandbox"])

    assert exit_code == 0
    assert calls == ["environment"]


def test_main_auto_install_flag(monkeypatch):
    seen: list[bool] = []

    def fake_bootstrap_environment(*, auto_install: bool):
        seen.append(auto_install)
        return DummySettings()

    class FakeBootstrapper:
        def bootstrap(self) -> None:
            pass

    monkeypatch.setattr(manual_bootstrap, "bootstrap_environment", fake_bootstrap_environment)
    monkeypatch.setattr(manual_bootstrap, "ensure_autonomous_launch", lambda *a, **k: None)
    monkeypatch.setattr(manual_bootstrap, "EnvironmentBootstrapper", lambda: FakeBootstrapper())
    monkeypatch.setattr(manual_bootstrap, "_configure_environment", lambda *a, **k: None)

    manual_bootstrap.main(["--skip-environment"])
    manual_bootstrap.main(["--skip-environment", "--auto-install"])

    assert seen == [False, True]


def test_main_dependency_failure_logs(monkeypatch, caplog):
    def fake_bootstrap_environment(*, auto_install: bool):
        raise SystemExit("missing packages")

    class FakeBootstrapper:
        def bootstrap(self) -> None:
            raise AssertionError("environment bootstrap should be skipped")

    monkeypatch.setattr(manual_bootstrap, "bootstrap_environment", fake_bootstrap_environment)
    monkeypatch.setattr(manual_bootstrap, "ensure_autonomous_launch", lambda *a, **k: None)
    monkeypatch.setattr(manual_bootstrap, "EnvironmentBootstrapper", lambda: FakeBootstrapper())
    monkeypatch.setattr(manual_bootstrap, "_configure_environment", lambda *a, **k: None)

    with caplog.at_level(logging.ERROR):
        exit_code = manual_bootstrap.main([])

    assert exit_code == 1
    assert any("missing packages" in rec.message for rec in caplog.records)
