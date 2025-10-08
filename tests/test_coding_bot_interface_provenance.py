"""Regression tests for unsigned provenance fallbacks in ``coding_bot_interface``."""

from __future__ import annotations

import importlib
import json
import logging
import os
import sys
from pathlib import Path

import pytest


def _reload_coding_bot_interface() -> object:
    """Reload ``coding_bot_interface`` to honour updated environment variables."""

    module_names = (
        "menace_sandbox.coding_bot_interface",
        "menace.coding_bot_interface",
        "coding_bot_interface",
    )
    for name in module_names:
        sys.modules.pop(name, None)
    return importlib.import_module("coding_bot_interface")


def _clear_provenance_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove provenance related environment variables for test isolation."""

    for key in (
        "MENACE_ALLOW_UNSIGNED_PROVENANCE",
        "MENACE_REQUIRE_SIGNED_PROVENANCE",
        "PATCH_PROVENANCE_FILE",
        "PATCH_PROVENANCE_PUBKEY",
        "PATCH_PROVENANCE_PUBLIC_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


def test_unsigned_provenance_enabled_when_signed_artifacts_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Unsigned provenance should be allowed when signed artefacts are misconfigured."""

    _clear_provenance_env(monkeypatch)
    monkeypatch.setenv("PATCH_PROVENANCE_FILE", str(tmp_path / "missing.json"))
    monkeypatch.setenv("PATCH_PROVENANCE_PUBKEY", str(tmp_path / "missing.pem"))

    caplog.set_level(logging.WARNING)
    module = _reload_coding_bot_interface()

    assert module._unsigned_provenance_allowed() is True  # type: ignore[attr-defined]
    assert "falling back to unsigned provenance" in caplog.text


def test_signed_provenance_configuration_short_circuits_unsigned_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Valid signed provenance configuration must disable unsigned fallbacks."""

    _clear_provenance_env(monkeypatch)
    prov_file = tmp_path / "provenance.json"
    prov_file.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("PATCH_PROVENANCE_FILE", os.fspath(prov_file))
    monkeypatch.setenv(
        "PATCH_PROVENANCE_PUBKEY",
        "-----BEGIN PUBLIC KEY-----\nZm9vYmFy\n-----END PUBLIC KEY-----\n",
    )

    module = _reload_coding_bot_interface()

    assert module._unsigned_provenance_allowed() is False  # type: ignore[attr-defined]


def test_signed_provenance_matches_abbreviated_commits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Signed provenance should match when commits are abbreviated."""

    _clear_provenance_env(monkeypatch)

    full_commit = "0123456789abcdef0123456789abcdef01234567"
    short_commit = full_commit[:12]
    prov_payload = {
        "data": {"patch_id": 915, "commit": short_commit},
        "signature": "deadbeef",  # content is irrelevant for the cache loader
    }
    prov_file = tmp_path / "signed.json"
    prov_file.write_text(json.dumps(prov_payload), encoding="utf-8")
    monkeypatch.setenv("PATCH_PROVENANCE_FILE", os.fspath(prov_file))
    monkeypatch.setenv("PATCH_PROVENANCE_PUBKEY", "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQ")

    module = _reload_coding_bot_interface()
    context = module._RepositoryContext(
        patch_id=None,
        commit=full_commit,
        repo_root=None,
        relative_path=None,
        canonical_path=None,
    )
    monkeypatch.setattr(
        module,
        "_resolve_repository_context",
        lambda *_, **__: context,
    )

    decision = module._resolve_provenance_decision(
        "ExampleBot",
        "module.py",
        [],
        (None, None),
    )

    assert decision.available is True
    assert decision.mode == "signed"
    assert decision.patch_id == 915
    assert decision.commit == full_commit


def test_signed_provenance_matches_windows_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Signed provenance should match by Windows-style paths when commits are missing."""

    _clear_provenance_env(monkeypatch)

    prov_payload = [
        {
            "patch_id": 321,
            "commit": "abc123def456",
            "files": [r"menace_sandbox\\task_validation_bot.py"],
        }
    ]
    prov_file = tmp_path / "signed.json"
    prov_file.write_text(json.dumps(prov_payload), encoding="utf-8")
    monkeypatch.setenv("PATCH_PROVENANCE_FILE", os.fspath(prov_file))
    monkeypatch.setenv(
        "PATCH_PROVENANCE_PUBKEY", "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQ"
    )
    monkeypatch.setenv("MENACE_REQUIRE_SIGNED_PROVENANCE", "1")

    module = _reload_coding_bot_interface()

    repo_path = "menace_sandbox/task_validation_bot.py"
    context = module._RepositoryContext(
        patch_id=None,
        commit=None,
        repo_root=tmp_path,
        relative_path=Path(repo_path),
        canonical_path=repo_path,
    )
    monkeypatch.setattr(
        module,
        "_resolve_repository_context",
        lambda *_, **__: context,
    )

    decision = module._resolve_provenance_decision(
        "TaskValidationBot",
        r"C:\\menace_sandbox\\menace_sandbox\\task_validation_bot.py",
        [],
        (None, None),
    )

    assert decision.available is True
    assert decision.mode == "signed"
    assert decision.patch_id == 321
    assert decision.commit == "abc123def456"
    assert decision.source == "signed:path"
