import json

import pandas as pd
import menace.vector_metrics_db as vdb
import menace.retrieval_ranker as rr


def test_prepare_and_train(tmp_path):
    db = vdb.VectorMetricsDB(tmp_path / "v.db")
    db.add(
        vdb.VectorMetric(
            event_type="retrieval",
            db="db1",
            tokens=0,
            wall_time_ms=0.0,
            hit=True,
            rank=1,
            contribution=0.3,
            win=True,
            regret=False,
        )
    )
    db.add(
        vdb.VectorMetric(
            event_type="retrieval",
            db="db1",
            tokens=0,
            wall_time_ms=0.0,
            hit=False,
            rank=2,
            contribution=0.1,
            win=False,
            regret=True,
        )
    )
    extra = pd.DataFrame({"feat": [1.0, 0.5]})
    df = rr.prepare_training_dataframe(db, extra_features=extra)
    assert "feat" in df.columns
    model, feats = rr.train_retrieval_ranker(df, target="win")
    model_path = tmp_path / "model.json"
    rr.save_model(model, feats, model_path)
    saved = json.loads(model_path.read_text())
    assert saved["features"] == feats
    cli_path = tmp_path / "cli.json"
    rr.main([
        "--db-path",
        str(tmp_path / "v.db"),
        "--model-path",
        str(cli_path),
        "--target",
        "win",
    ])
    assert cli_path.exists()
