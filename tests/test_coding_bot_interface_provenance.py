"""Regression tests for unsigned provenance fallbacks in ``coding_bot_interface``."""

from __future__ import annotations

import importlib
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
