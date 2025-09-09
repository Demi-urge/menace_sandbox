import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.orchestrator import SandboxOrchestrator  # noqa: E402
from neurosales.sql_db import create_session  # noqa: E402
from context_builder_util import create_context_builder  # noqa: E402


def test_handle_chat_benchmark(benchmark):
    Session = create_session("sqlite://")
    orch = SandboxOrchestrator(
        context_builder=create_context_builder(),
        persistent=True,
        session_factory=Session,
    )
    with patch("neurosales.embedding.embed_text", return_value=[0.0]):
        reply, conf = benchmark.pedantic(
            lambda: orch.handle_chat("u1", "hello"),
            iterations=100,
            rounds=1,
        )
    assert isinstance(reply, str)
    assert isinstance(conf, float)
    assert benchmark.stats.stats.mean < 0.03
