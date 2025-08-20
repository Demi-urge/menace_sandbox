from vector_metrics_db import VectorMetricsDB


def test_log_retrieval_feedback_rates(tmp_path):
    db = VectorMetricsDB(tmp_path / "vm.db")
    db.log_retrieval_feedback("a", win=True, roi=0.5)
    db.log_retrieval_feedback("a", regret=True, roi=-0.2)
    assert db.retriever_win_rate("a") == 0.5
    assert db.retriever_regret_rate("a") == 0.5

