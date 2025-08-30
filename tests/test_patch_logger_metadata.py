import types

from vector_service.patch_logger import PatchLogger


class DummyVectorStore:
    def __init__(self) -> None:
        self.calls = []

    def add(self, kind, record_id, vector, *, origin_db=None, metadata=None):
        self.calls.append(
            {
                "kind": kind,
                "record_id": record_id,
                "origin_db": origin_db,
                "metadata": metadata or {},
            }
        )


class DummyVectorService:
    def __init__(self) -> None:
        self.vector_store = DummyVectorStore()

    def vectorise(self, kind, record):
        return [0.0]

    def vectorise_and_store(self, kind, record_id, record, *, origin_db=None, metadata=None):
        vec = self.vectorise(kind, record)
        self.vector_store.add(kind, record_id, vec, origin_db=origin_db, metadata=metadata)
        return vec


class DummyPatchDB:
    def record_provenance(self, *a, **k):
        pass

    def log_ancestry(self, *a, **k):
        pass

    def log_contributors(self, *a, **k):
        pass

    def record_vector_metrics(self, *a, **k):
        pass

    def get(self, *a, **k):
        return types.SimpleNamespace(description="some patch")


def test_track_contributors_persists_roi_tag_in_metadata():
    svc = DummyVectorService()
    pdb = DummyPatchDB()
    pl = PatchLogger(patch_db=pdb, vector_service=svc)
    pl.track_contributors(["db:v1"], True, patch_id="1", roi_tag="high-ROI")
    assert svc.vector_store.calls, "vector store should be called"
    meta = svc.vector_store.calls[0]["metadata"]
    assert meta.get("roi_tag") == "high-ROI"
