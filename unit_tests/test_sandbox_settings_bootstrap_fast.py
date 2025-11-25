from menace_sandbox.sandbox_settings import SandboxSettings


def test_sandbox_settings_bootstrap_fast_allows_initialisation():
    settings = SandboxSettings(bootstrap_fast=True, build_groups=False)

    assert settings.bootstrap_fast is True
