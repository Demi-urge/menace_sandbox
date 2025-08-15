from __future__ import annotations

from types import MethodType
from pathlib import Path

import analytics.retrieval_ranker as arr
from menace.bot_database import BotDB, BotRecord
from menace.task_handoff_bot import WorkflowDB, WorkflowRecord
from menace.universal_retriever import RetrievalWeights, UniversalRetriever


def _const_encoder(self, text: str):
    return [0.0, 0.0]


def test_model_ranking_changes_order(tmp_path):
    bot_db = BotDB(path=tmp_path / "bot.db", vector_index_path=tmp_path / "bot.idx")
    bot_db.encode_text = MethodType(_const_encoder, bot_db)
    first = bot_db.add_bot(BotRecord(name="one", purpose="a"))
    second = bot_db.add_bot(BotRecord(name="two", purpose="b"))

    weights = RetrievalWeights(similarity=1.0, context=0.0, win=0.0, regret=0.0)
    plain = UniversalRetriever(
        bot_db=bot_db,
        weights=weights,
        enable_model_ranking=False,
        enable_reliability_bias=False,
    )
    ranked = UniversalRetriever(
        bot_db=bot_db,
        weights=weights,
        enable_model_ranking=True,
        enable_reliability_bias=False,
    )

    def fake_freq(self, bid: int) -> float:
        return {first: 0.0, second: 5.0}[int(bid)]

    plain._bot_deploy_freq = MethodType(fake_freq, plain)
    ranked._bot_deploy_freq = MethodType(fake_freq, ranked)
    ranked._ranker_model = {
        "features": ["context_score"],
        "coef": [10.0],
        "intercept": 0.0,
    }

    baseline = [h.metadata["id"] for h in plain.retrieve("q", top_k=2)]
    scored = [h.metadata["id"] for h in ranked.retrieve("q", top_k=2)]

    assert baseline != scored
    assert scored[0] == second


def test_reliability_stats_alter_order(tmp_path):
    bot_db = BotDB(path=tmp_path / "bot.db", vector_index_path=tmp_path / "bot.idx")
    wf_db = WorkflowDB(path=tmp_path / "wf.db", vector_index_path=tmp_path / "wf.idx")
    for db in (bot_db, wf_db):
        db.encode_text = MethodType(_const_encoder, db)
    bot_db.add_bot(BotRecord(name="a", purpose="b"))
    wf_db.add(WorkflowRecord(workflow=["x"], title="w"))

    weights = RetrievalWeights(similarity=1.0, context=0.0)
    retriever = UniversalRetriever(
        bot_db=bot_db,
        workflow_db=wf_db,
        weights=weights,
        enable_model_ranking=False,
        enable_reliability_bias=True,
    )

    retriever._bot_deploy_freq = MethodType(lambda self, bid: 0.0, retriever)
    retriever._workflow_usage = MethodType(lambda self, wf: 0.0, retriever)
    retriever._load_reliability_stats = MethodType(lambda self: self._reliability_stats, retriever)

    retriever._reliability_stats = {
        "bot": {"win_rate": 1.0, "regret_rate": 0.0, "reliability": 1.0},
        "workflow": {"win_rate": 0.0, "regret_rate": 0.0, "reliability": 0.0},
    }
    order1 = [h.origin_db for h in retriever.retrieve("x", top_k=2)]
    retriever._reliability_stats = {
        "bot": {"win_rate": 0.0, "regret_rate": 0.0, "reliability": 0.0},
        "workflow": {"win_rate": 1.0, "regret_rate": 0.0, "reliability": 1.0},
    }
    order2 = [h.origin_db for h in retriever.retrieve("x", top_k=2)]

    assert order1[0] == "bot"
    assert order2[0] == "workflow"


def test_training_set_labeling():
    base = Path(__file__).resolve().parent / "fixtures"
    stats = base / "retrieval_ranker_stats.jsonl"
    labels = base / "patch_outcomes.jsonl"
    ts = arr.build_training_set(stats, labels)
    assert list(ts.y) == [1, 0]
    assert "db_bot" in ts.X.columns
