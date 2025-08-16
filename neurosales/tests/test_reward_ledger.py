import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.reward_ledger import RewardLedger, DatabaseRewardLedger
from neurosales.sql_db import create_session as create_sql_session


def test_stamp_and_audit():
    ledger = RewardLedger()
    line_id = ledger.stamp_line(
        "u1",
        "hello",
        sentiment_before=0.0,
        followups=1,
        session_delta=0.5,
        pref_match=0.4,
        confidence=1.0,
    )
    bal = ledger.ledgers["u1"]
    assert bal.green > 0
    assert bal.violet >= 0
    ledger.audit_line(line_id, session_delta=2.0, lost_user=True)
    audit_bal = ledger.ledgers["audit"]
    assert audit_bal.green != 0
    assert audit_bal.iron < 0


def test_reward_persistence():
    Session = create_sql_session("sqlite://")
    ledger1 = DatabaseRewardLedger(session_factory=Session)
    line_id = ledger1.stamp_line(
        "u1",
        "hi",
        sentiment_before=0.0,
        session_delta=1.0,
    )
    ledger1.audit_line(line_id, session_delta=2.0)

    ledger2 = DatabaseRewardLedger(session_factory=Session)
    assert line_id in ledger2.lines
    assert abs(ledger2.lines[line_id].coins.green - ledger1.lines[line_id].coins.green) < 1e-6
    total1 = ledger1.ledgers["u1"].green + ledger1.ledgers["audit"].green
    assert abs(ledger2.ledgers["u1"].green - total1) < 1e-6


def test_reward_db_url(tmp_path):
    db_file = tmp_path / "rewards.db"
    url = f"sqlite:///{db_file}"
    ledger1 = DatabaseRewardLedger(db_url=url)
    line_id = ledger1.stamp_line("u1", "hello", sentiment_before=0.0)
    ledger2 = DatabaseRewardLedger(db_url=url)
    assert line_id in ledger2.lines

