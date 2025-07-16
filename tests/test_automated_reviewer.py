import os

os.environ.setdefault("MENACE_LIGHT_IMPORTS", "1")

import menace.automated_reviewer as ar

class DummyEscalation:
    def __init__(self) -> None:
        self.messages = []
    def handle(self, msg, attachments=None):
        self.messages.append(msg)

class DummyDB:
    def __init__(self):
        self.updated = []
    def update_bot(self, bot_id, **fields):
        self.updated.append((bot_id, fields))


def test_escalation_on_critical():
    esc = DummyEscalation()
    db = DummyDB()
    reviewer = ar.AutomatedReviewer(bot_db=db, escalation_manager=esc)
    reviewer.handle({"bot_id": "7", "severity": "critical"})
    assert db.updated and db.updated[0][0] == 7
    assert esc.messages and "review for bot 7" in esc.messages[0]
