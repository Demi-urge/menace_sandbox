import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.orchestrator import SandboxOrchestrator  # noqa: E402
from neurosales.sql_db import create_session, RLFeedback  # noqa: E402
from context_builder_util import create_context_builder  # noqa: E402


def test_orchestrator_persistence(tmp_path):
    db_file = tmp_path / "tmp.db"
    url = f"sqlite:///{db_file}"
    Session = create_session(url)

    with patch("neurosales.embedding.embed_text", return_value=[0.0]):
        orch = SandboxOrchestrator(
            context_builder=create_context_builder(),
            persistent=True,
            session_factory=Session,
        )
        orch.handle_chat("u1", "hello")
        orch.handle_chat("u1", "thanks")

    with patch("neurosales.embedding.embed_text", return_value=[0.0]):
        orch2 = SandboxOrchestrator(
            context_builder=create_context_builder(),
            persistent=True,
            session_factory=Session,
        )

    mem = orch2._get_memory("u1")
    msgs = [m.content for m in mem.get_recent_messages()]
    assert "hello" in msgs
    assert "thanks" in msgs

    with Session() as s:
        fb_rows = s.query(RLFeedback).all()
    assert len(fb_rows) >= 2

    buf = orch2.ranker._buffer("u1")
    assert len(buf) >= 1

    db_file.unlink()
