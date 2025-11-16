import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.rl_integration import (
    QLearningModule,
    ReplayBuffer,
    MetaLearner,
    RLResponseRanker,
    Experience,
)


def test_q_learning_update_and_predict():
    q = QLearningModule(alpha=1.0, gamma=0.0, epsilon=0.0)
    state = (0,)
    q.update(state, "hi", reward=1.0, next_state=state, next_actions=["hi"])
    assert q.predict(state, "hi") == 1.0


def test_replay_buffer_sample():
    buf = ReplayBuffer(max_size=2)
    buf.add(Experience((0,), "a", 1.0, (1,)))
    buf.add(Experience((1,), "b", 0.5, (2,)))
    assert len(buf) == 2
    sample = buf.sample(1)
    assert len(sample) == 1


def test_meta_learner_aggregates():
    meta = MetaLearner()
    q1 = QLearningModule(alpha=1.0, gamma=0.0, epsilon=0.0)
    q2 = QLearningModule(alpha=1.0, gamma=0.0, epsilon=0.0)
    meta.register("g", q1)
    meta.register("g", q2)
    q1.update((0,), "x", 1.0, (0,), ["x"])
    meta.aggregate("g")
    assert q2.predict((0,), "x") > 0.0


def test_rl_ranker_orders_by_q_value():
    ranker = RLResponseRanker()
    scores = {"a": 0.5, "b": 0.5}
    first = ranker.rank("u1", scores)
    assert set(first) == {"a", "b"}
    ranker.log_outcome("u1", (0,), "b", 1.0, (0,), ["a", "b"])
    second = ranker.rank("u1", scores)
    assert second[0] == "b"

