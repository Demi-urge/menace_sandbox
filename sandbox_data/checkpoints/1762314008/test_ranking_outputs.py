import pandas as pd
from retrieval_ranker import rank_candidates


def test_roi_weighted_cosine_prefers_higher_roi():
    df = pd.DataFrame({"similarity": [0.5, 0.5], "roi": [0.0, 0.2]}, index=["a", "b"])
    scores = rank_candidates(df, "roi_weighted_cosine")
    assert scores["b"] > scores["a"]
