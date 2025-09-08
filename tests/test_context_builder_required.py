import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
try:  # pragma: no cover - import validation
    import menace_sandbox.chatgpt_enhancement_bot as ceb
    import menace_sandbox.chatgpt_research_bot as crb
except Exception as exc:  # pragma: no cover - optional dependency missing
    pytest.skip(f"bot modules unavailable: {exc}", allow_module_level=True)


def test_enhancement_bot_requires_context_builder():
    with pytest.raises(TypeError):
        ceb.ChatGPTEnhancementBot(None)


def test_research_bot_requires_context_builder():
    with pytest.raises(TypeError):
        crb.ChatGPTResearchBot()
