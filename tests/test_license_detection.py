import logging

import logging
import pytest

from compliance.license_fingerprint import check as license_check
from embeddable_db_mixin import EmbeddableDBMixin


GPL_TEXT = """\
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""


def test_detect_license_gpl():
    assert license_check(GPL_TEXT) == "GPL-3.0"


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
