import pytest

pytest.importorskip("torch")

from menace.error_forecaster import ErrorForecaster
from menace.knowledge_graph import KnowledgeGraph
from menace.data_bot import MetricsDB, MetricRecord


def test_train_and_predict(tmp_path):
    mdb = MetricsDB(tmp_path / "m.db")
    for i in range(6):
        mdb.add(
            MetricRecord(
                bot="A",
                cpu=float(i),
                memory=float(i * 2),
                response_time=0.0,
                disk_io=0.0,
                net_io=0.0,
                errors=1 if i % 2 == 0 else 0,
                revenue=float(i),
                expense=float(i) / 2,
            )
        )
    forecaster = ErrorForecaster(mdb, seq_len=2, epochs=1)
    assert forecaster.lstm.input_size == 9
    assert forecaster.dropout.p == pytest.approx(0.1)
    data = forecaster._dataset()
    assert len(data[0][1][0]) == 9
    trained = forecaster.train()
    assert trained
    probs = forecaster.predict_error_prob("A", steps=2)
    assert len(probs) == 2
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_transformer_model(tmp_path):
    mdb = MetricsDB(tmp_path / "m.db")
    for i in range(6):
        mdb.add(
            MetricRecord(
                bot="A",
                cpu=float(i),
                memory=float(i),
                response_time=0.0,
                disk_io=0.0,
                net_io=0.0,
                errors=0,
            )
        )
    forecaster = ErrorForecaster(mdb, seq_len=2, epochs=1, model="transformer")
    assert forecaster.transformer is not None
    forecaster.train()
    probs = forecaster.predict_error_prob("A", steps=1)
    assert len(probs) == 1


def test_predict_failure_chain(tmp_path):
    mdb = MetricsDB(tmp_path / "m.db")
    for i in range(3):
        mdb.add(
            MetricRecord(
                bot="B",
                cpu=1.0,
                memory=1.0,
                response_time=0.0,
                disk_io=0.0,
                net_io=0.0,
                errors=1 if i == 2 else 0,
            )
        )
    kg = KnowledgeGraph()
    kg.graph.add_node("bot:B")
    kg.graph.add_edge("bot:B", "module:M1")
    kg.graph.add_edge("module:M1", "module:M2")
    fc = ErrorForecaster(mdb, seq_len=1, epochs=1)
    fc.train()
    mods = fc.predict_failure_chain("B", kg, steps=2)
    assert "module:M1" in mods
    assert "module:M2" in mods


def test_preemptive_patch_selection(tmp_path):
    mdb = MetricsDB(tmp_path / "m.db")
    for i in range(2):
        mdb.add(
            MetricRecord(
                bot="C",
                cpu=1.0,
                memory=1.0,
                response_time=0.0,
                disk_io=0.0,
                net_io=0.0,
                errors=0,
            )
        )
    kg = KnowledgeGraph()
    kg.graph.add_node("bot:C")
    kg.graph.add_node("error_type:E", weight=2)
    kg.graph.add_edge("error_type:E", "bot:C", type="telemetry")
    kg.graph.add_edge("error_type:E", "module:M", type="module", weight=2)
    kg.graph.add_edge("error_type:E", "patch:P2", type="patch")
    fc = ErrorForecaster(mdb, seq_len=1, epochs=1)
    fc.train()
    fc.predict_error_prob = lambda bot, steps=1: [1.0]
    patches = fc.suggest_patches("C", kg)
    assert "patch:P2" in patches
