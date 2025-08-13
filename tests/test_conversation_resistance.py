import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import menace_sandbox.neurosales as ns
sys.modules["neurosales"] = ns
import menace_sandbox.neurosales.memory_queue as mq
import menace_sandbox.neurosales.memory_stack as ms
import menace_sandbox.conversation_manager_bot as cmb


class DummyClient:
    def ask(self, messages):
        return {"choices": [{"message": {"content": "ok"}}]}


def test_resistance_triggers_tone_shift(monkeypatch):
    monkeypatch.setattr(mq, "_default_queue", mq.MemoryQueue(), raising=False)
    monkeypatch.setattr(ms, "_default_stack", ms.MemoryStack(), raising=False)
    bot = cmb.ConversationManagerBot(DummyClient())
    calls = []
    bot.on_resistance(lambda objs, chain: calls.append((objs, chain)))
    ns.add_message("no thanks")
    ns.add_message("don't want to")
    bot._detect_resistance()
    assert bot.strategy == "conciliatory"
    assert calls and len(calls[0][0]) == 2
    ns.add_message("no again")
    ns.add_message("won't do")
    bot._detect_resistance()
    assert bot.strategy == "assertive"
    assert len(calls) == 2


def test_tone_shift_requires_repeated_resistance(monkeypatch):
    monkeypatch.setattr(mq, "_default_queue", mq.MemoryQueue(), raising=False)
    monkeypatch.setattr(ms, "_default_stack", ms.MemoryStack(), raising=False)
    bot = cmb.ConversationManagerBot(DummyClient())
    ns.add_message("not interested")
    bot._detect_resistance()
    assert bot.strategy == "neutral"
    ns.add_message("no thanks")
    bot._detect_resistance()
    assert bot.strategy == "conciliatory"
