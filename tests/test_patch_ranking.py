import types
import time

from vector_service.context_builder import _ScoredEntry
from vector_service.ranking_utils import rank_patches


class DummyPatchDB:
    def __init__(self):
        now = time.time()
        self.data = {
            "A": [(1, types.SimpleNamespace(roi_delta=0.2, ts=now))],
            "B": [(2, types.SimpleNamespace(roi_delta=-0.1, ts=now - 2 * 86400))],
        }

    def find_by_vector(self, vid):
        return self.data.get(vid, [])


def test_rank_patches_weighted():
    entries = [
        _ScoredEntry({"similarity": 0.5, "id": "A"}, 0.5, "patch", "A", {}),
        _ScoredEntry({"similarity": 0.6, "id": "B"}, 0.6, "patch", "B", {}),
    ]
    ranked, conf = rank_patches(entries, patch_db=DummyPatchDB())
    assert ranked[0].vector_id == "A"
    assert conf == ranked[0].score
