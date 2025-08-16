from menace_sandbox.roi_tracker import ROITracker
from menace_sandbox.universal_retriever import UniversalRetriever


def test_origin_db_contribution_and_bias():
    tracker = ROITracker()
    tracker.update(
        0.0,
        1.0,
        retrieval_metrics=[
            {"origin_db": "A", "hit": True, "tokens": 10},
            {"origin_db": "B", "hit": True, "tokens": 5},
        ],
    )
    tracker.update(
        1.0,
        1.3,
        retrieval_metrics=[{"origin_db": "B", "hit": True, "tokens": 10}],
    )
    db_roi = tracker.roi_by_origin_db()
    assert db_roi["A"] > db_roi["B"]

    class DummyDB:
        pass

    class DummyRetriever(UniversalRetriever):
        def __init__(self):
            super().__init__(bot_db=DummyDB())

        def _retrieve_candidates(self, query, top_k):
            return [
                ("A", 1, {}, {"_distance": 1.0}),
                ("B", 2, {}, {"_distance": 1.0}),
            ]

        def _context_score(self, kind, record):
            return 0.0, {}

    retriever = DummyRetriever()
    results, _, _ = retriever.retrieve("q", top_k=2, link_multiplier=1.0, roi_tracker=tracker)
    assert results[0].origin_db == "A"
