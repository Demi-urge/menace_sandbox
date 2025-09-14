from __future__ import annotations

import json
import logging

import pytest

from menace_sandbox.bot_registry import BotRegistry
from menace_sandbox.override_validator import generate_signature


ORIG_VERIFY = BotRegistry._verify_signed_provenance


def _make_prov(tmp_path, patch_id: int, commit: str, monkeypatch) -> None:
    key = tmp_path / "key"
    key.write_text("secret")
    data = {"patch_id": patch_id, "commit": commit}
    sig = generate_signature(data, str(key))
    prov = tmp_path / "prov.json"
    prov.write_text(json.dumps({"data": data, "signature": sig}))
    monkeypatch.setenv("PATCH_PROVENANCE_FILE", str(prov))
    monkeypatch.setenv("PATCH_PROVENANCE_PUBKEY", str(key))


def test_verify_signed_provenance_valid(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(BotRegistry, "_verify_signed_provenance", ORIG_VERIFY)
    _make_prov(tmp_path, 1, "abc", monkeypatch)
    reg = BotRegistry()
    with caplog.at_level(logging.INFO):
        assert reg._verify_signed_provenance(1, "abc")
    assert "patch_id=1 commit=abc" in caplog.text


def test_verify_signed_provenance_missing_signature(tmp_path, monkeypatch):
    monkeypatch.setattr(BotRegistry, "_verify_signed_provenance", ORIG_VERIFY)
    key = tmp_path / "key"
    key.write_text("secret")
    data = {"patch_id": 1, "commit": "abc"}
    prov = tmp_path / "prov.json"
    prov.write_text(json.dumps({"data": data}))
    monkeypatch.setenv("PATCH_PROVENANCE_FILE", str(prov))
    monkeypatch.setenv("PATCH_PROVENANCE_PUBKEY", str(key))
    reg = BotRegistry()
    with pytest.raises(RuntimeError):
        reg._verify_signed_provenance(1, "abc")


def test_verify_signed_provenance_bad_signature(tmp_path, monkeypatch):
    monkeypatch.setattr(BotRegistry, "_verify_signed_provenance", ORIG_VERIFY)
    key = tmp_path / "key"
    key.write_text("secret")
    data = {"patch_id": 1, "commit": "abc"}
    prov = tmp_path / "prov.json"
    prov.write_text(json.dumps({"data": data, "signature": "dead"}))
    monkeypatch.setenv("PATCH_PROVENANCE_FILE", str(prov))
    monkeypatch.setenv("PATCH_PROVENANCE_PUBKEY", str(key))
    reg = BotRegistry()
    with pytest.raises(RuntimeError):
        reg._verify_signed_provenance(1, "abc")
