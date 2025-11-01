import logging
import pytest

from compliance.license_fingerprint import check as license_check
from menace_sandbox.embeddable_db_mixin import EmbeddableDBMixin


GPL_TEXT = """\
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""

GPL2_TEXT = "This code is released under the GNU General Public License version 2."
NON_COMMERCIAL_TEXT = "This dataset is for non-commercial use only."


def test_detect_license_gpl():
    assert license_check(GPL_TEXT) == "GPL-3.0"


def test_detect_license_gpl_v2():
    assert license_check(GPL2_TEXT) == "GPL-2.0"


def test_detect_non_commercial_phrase():
    assert license_check(NON_COMMERCIAL_TEXT) == "CC-BY-NC-4.0"


class DummyDB(EmbeddableDBMixin):
    def __init__(self, tmp_path):
        super().__init__(index_path=tmp_path / "x.ann", metadata_path=tmp_path / "x.json")
        self.calls = 0

    def vector(self, record):  # pragma: no cover - should not be called for GPL
        self.calls += 1
        return [0.0]

    def iter_records(self):  # pragma: no cover - not used
        return iter([])


def test_add_embedding_skips_gpl(tmp_path, caplog):
    db = DummyDB(tmp_path)
    with caplog.at_level(logging.WARNING):
        db.add_embedding("1", GPL_TEXT, "code")
    assert db.calls == 0
    meta = db._metadata.get("1")
    assert meta and meta.get("license") == "GPL-3.0"
    assert any("license" in rec.message.lower() for rec in caplog.records)


def test_add_embedding_detects_semantic_risk(tmp_path, caplog):
    db = DummyDB(tmp_path)
    risky = "eval('data')"
    with caplog.at_level(logging.WARNING):
        db.add_embedding("2", risky, "code")
    assert db.calls == 0
    meta = db._metadata.get("2")
    alerts = meta.get("semantic_risks") if meta else None
    assert alerts and any("eval" in a[0] for a in alerts)
    assert any("semantic" in rec.message.lower() for rec in caplog.records)


class BackfillDB(EmbeddableDBMixin):
    def __init__(self, tmp_path):
        super().__init__(index_path=tmp_path / "b.ann", metadata_path=tmp_path / "b.json")
        self.calls = 0

    def vector(self, record):
        self.calls += 1
        return [0.0]

    def iter_records(self):
        return iter([
            ("1", "safe", "code"),
            ("2", "eval('data')", "code"),
        ])


def test_backfill_detects_semantic_risk(tmp_path, caplog):
    db = BackfillDB(tmp_path)
    with caplog.at_level(logging.WARNING):
        db.backfill_embeddings()
    assert db.calls == 1
    meta = db._metadata.get("2")
    alerts = meta.get("semantic_risks") if meta else None
    assert alerts and any("eval" in a[0] for a in alerts)
    assert any("semantic" in rec.message.lower() for rec in caplog.records)
