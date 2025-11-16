import json
import os
import threading
import time
from pathlib import Path
from typing import Optional, List, Dict
import csv

from joblib import dump
from dynamic_path_router import resolve_path

from .sql_db import create_session, RLFeedback, RewardEntry, ensure_schema
from .rl_integration import DatabaseRLResponseRanker
from .policy_learning import PolicyLearner
from .engagement_dataset import collect_engagement_logs


def export_feedback(
    *,
    session_factory: Optional[callable] = None,
    db_url: Optional[str] = None,
    mark_processed: bool = False,
) -> List[Dict[str, float]]:
    """Return unprocessed feedback rows as dictionaries and optionally mark them."""
    if session_factory is None:
        from .sql_db import ensure_schema

        ensure_schema(db_url or os.environ.get("NEURO_DB_URL", "sqlite://"))
        session_factory = create_session(db_url)
    Session = session_factory
    with Session() as s:
        fb_rows = (
            s.query(RLFeedback)
            .filter(RLFeedback.processed.is_(False))
            .order_by(RLFeedback.id)
            .all()
        )
        reward_rows = s.query(RewardEntry).order_by(RewardEntry.id).all()
        data = []
        for idx, fb in enumerate(fb_rows):
            rec = {"text": fb.text, "feedback": fb.feedback, "score": fb.score}
            if idx < len(reward_rows):
                rw = reward_rows[idx]
                ctr = rw.followups / max(1.0, rw.followups + rw.session_delta)
                conversions = 1 if rw.followups > 0 else 0
                sent_shift = rw.sentiment_after - rw.sentiment_before
                rec.update(
                    {
                        "ctr": ctr,
                        "conversions": conversions,
                        "sentiment_shift": sent_shift,
                    }
                )
            data.append(rec)
        if mark_processed and fb_rows:
            for r in fb_rows:
                r.processed = True
            s.commit()
        return data


def save_feedback_dataset(
    path: str,
    *,
    session_factory: Optional[callable] = None,
    db_url: Optional[str] = None,
) -> str:
    """Export RL feedback to ``path`` and return the path."""
    data = export_feedback(
        session_factory=session_factory, db_url=db_url, mark_processed=True
    )
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"Exported {len(data)} feedback rows to {path}")
    return path


def train_models(
    dataset_path: str,
    *,
    session_factory: Optional[callable] = None,
    db_url: Optional[str] = None,
) -> None:
    """Train RL models from a JSON dataset produced by :func:`save_feedback_dataset`."""
    if not dataset_path or not os.path.exists(dataset_path):
        return
    with open(dataset_path) as f:
        if dataset_path.endswith(".jsonl"):
            data = [json.loads(l) for l in f.read().splitlines() if l.strip()]
        else:
            data = json.load(f)
    if not data:
        return

    actions = sorted({d["feedback"] for d in data})
    tactics = {a: "flat" for a in actions}

    learner = PolicyLearner(actions, tactics, state_dim=4)
    ranker = DatabaseRLResponseRanker(session_factory=session_factory, db_url=db_url)

    for rec in data:
        ctr = float(rec.get("ctr", 0.0))
        conv = float(rec.get("conversions", 0.0))
        shift = float(rec.get("sentiment_shift", 0.0))
        state = (len(rec.get("text", "")), ctr, conv, shift)
        action = rec["feedback"]
        reward = float(rec.get("score", 1.0))
        ranker.log_outcome("trainer", state, action, reward, state, actions)
        learner.brain.update(list(state), action, reward)

    weights_path = resolve_path("neurosales") / "policy_params.json"
    with open(weights_path, "w") as f:
        json.dump(learner.brain.params, f)
    print(f"Policy weights saved to {weights_path}")


def train_engagement_model(dataset_path: str) -> None:
    """Train the engagement regressor and overwrite the model file."""
    if not dataset_path or not os.path.exists(dataset_path):
        return
    X: List[List[float]] = []
    y: List[float] = []
    with open(dataset_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            X.append([
                float(row["length"]),
                float(row["exclam"]),
                float(row["question"]),
            ])
            y.append(float(row["engagement"]))
    if not X:
        return
    try:
        from sklearn.linear_model import LinearRegression  # type: ignore
    except Exception:
        return
    import numpy as np

    model = LinearRegression()
    model.fit(np.array(X), np.array(y))
    model_path = resolve_path("neurosales") / "engagement_model.joblib"
    dump(model, model_path)
    print(f"Engagement model saved to {model_path}")


def schedule_periodic_training(
    interval: int = 3600,
    *,
    dataset_path: str = "rl_feedback_dataset.json",
    engagement_path: str = "engagement_train.csv",
    session_factory: Optional[callable] = None,
    db_url: Optional[str] = None,
) -> Optional[threading.Thread]:
    """Run export and training on a timer in a background thread."""

    env_int = os.getenv("NEURO_AUTO_TRAIN_INTERVAL")
    if env_int is not None:
        try:
            interval = int(env_int)
        except ValueError:
            pass
    if interval <= 0:
        print("Automatic training disabled")
        return None

    def _loop() -> None:
        while True:
            fb_path = save_feedback_dataset(
                dataset_path, session_factory=session_factory, db_url=db_url
            )
            eng_path = collect_engagement_logs(
                engagement_path,
                session_factory=session_factory,
                db_url=db_url,
            )
            train_engagement_model(eng_path)
            train_models(fb_path, session_factory=session_factory, db_url=db_url)
            time.sleep(interval)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


def schedule_feedback_export(
    interval: int = 3600,
    *,
    dataset_path: str = "rl_feedback_dataset.json",
    session_factory: Optional[callable] = None,
    db_url: Optional[str] = None,
) -> threading.Thread:
    """Export feedback on a timer in a background thread."""

    def _loop() -> None:
        while True:
            save_feedback_dataset(
                dataset_path, session_factory=session_factory, db_url=db_url
            )
            time.sleep(interval)

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


__all__ = [
    "export_feedback",
    "save_feedback_dataset",
    "train_models",
    "train_engagement_model",
    "schedule_periodic_training",
    "schedule_feedback_export",
]
