import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.orchestrator import SandboxOrchestrator  # noqa: E402
from neurosales.sql_db import create_session as create_sql_session  # noqa: E402
from context_builder_util import create_context_builder  # noqa: E402


def test_followup_updates_rl():
    Session = create_sql_session("sqlite://")
    orch = SandboxOrchestrator(
        context_builder=create_context_builder(),
        persistent=True,
        session_factory=Session,
    )

    first_reply, _ = orch.handle_chat("u1", "hello")
    second_reply, _ = orch.handle_chat("u1", "thanks")

    pairs = orch._get_reactions("u1").get_pairs()
    assert (first_reply, "thanks") in pairs

    buf = orch.ranker._buffer("u1")
    assert len(buf) >= 1
