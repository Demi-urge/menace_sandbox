import types
from dataclasses import dataclass

from universal_retriever import UniversalRetriever, ResultBundle
from vector_service.patch_logger import PatchLogger


class DummyEncoder:
    def encode_text(self, text: str):
        return [0.0]


class DummyRetriever(UniversalRetriever):
    def __init__(self):
        super().__init__(code_db=DummyEncoder(), enable_model_ranking=False, enable_reliability_bias=False)

    def _retrieve_candidates(self, query, top_k, db_names=None):
        feats1 = {"distance": 0.0, "similarity": 0.9, "context": 0.0}
        feats2 = {"distance": 0.0, "similarity": 0.5, "context": 0.0}
        item1 = {"vector_id": "v1", "note": "password=abc123"}
        item2 = {"vector_id": "v2", "note": "password=def456"}
        return [
            ("db", "v1", item1, 0.0, feats1),
            ("db", "v2", item2, 0.0, feats2),
        ]


class DummyPatchDB:
    def __init__(self):
        self.ancestry = None

    def record_vector_metrics(self, *args, **kwargs):
        pass

    def record_provenance(self, patch_id, detailed):
        pass

    def log_ancestry(self, patch_id, detailed):
        self.ancestry = detailed

    def log_contributors(self, patch_id, detailed, session_id):
        pass


def test_retriever_redacts_and_logs():
    r = DummyRetriever()
    hits, sid, vectors = r.retrieve("q", top_k=2)
    # metadata secrets redacted
    assert hits[0].metadata["note"] == "[REDACTED]"
    assert hits[1].metadata["note"] == "[REDACTED]"
    # vectors include scores
    assert vectors == [("db", "v1", hits[0].score), ("db", "v2", hits[1].score)]
    pdb = DummyPatchDB()
    pl = PatchLogger(patch_db=pdb)
    ids = {f"{o}:{v}": s for o, v, s in vectors}
    pl.track_contributors(ids, True, patch_id="1", session_id=sid)
    assert pdb.ancestry == [
        ("db", "v1", hits[0].score, None, None),
        ("db", "v2", hits[1].score, None, None),
    ]
