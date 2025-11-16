import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unittest.mock import patch  # noqa: E402

import pytest
try:  # pragma: no cover - skip if vector_service missing
    from neurosales.orchestrator import SandboxOrchestrator  # noqa: E402
    from neurosales.sql_db import create_session as create_sql_session  # noqa: E402
    from context_builder_util import create_context_builder  # noqa: E402
except Exception:  # pragma: no cover - dependency missing
    pytest.skip("vector_service not installed", allow_module_level=True)


def test_orchestrator_chat_and_learning():
    Session = create_sql_session("sqlite://")
    orch = SandboxOrchestrator(
        context_builder=create_context_builder(),
        persistent=True,
        session_factory=Session,
    )

    with patch("neurosales.embedding.embed_text", return_value=[0.0]):
        with patch.object(orch.ranker, "rank", wraps=orch.ranker.rank) as mock_rank:
            reply, conf = orch.handle_chat("u1", "hello")
            assert mock_rank.called

    assert isinstance(reply, str) and reply
    assert isinstance(conf, float)

    events = orch.learner.buffer.get_events()
    assert len(events) == 1
    assert events[0].text == "hello"
    assert events[0].issue == reply
