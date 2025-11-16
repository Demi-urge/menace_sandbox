import importlib.util
import sys
import random
from pathlib import Path

import pytest

# helper to load fallback ignoring any installed annoy package
FALLBACK_SPEC = importlib.util.spec_from_file_location(
    "annoy_fallback", Path(__file__).resolve().parents[1] / "annoy" / "__init__.py"  # path-ignore
)
FALLBACK_MOD = importlib.util.module_from_spec(FALLBACK_SPEC)
sys.modules["annoy_fallback"] = FALLBACK_MOD
_site_filtered = [p for p in sys.path if "site-packages" not in p]
_old = sys.path
sys.path = _site_filtered
FALLBACK_SPEC.loader.exec_module(FALLBACK_MOD)  # type: ignore[arg-type]
sys.path = _old
FallbackAnnoy = FALLBACK_MOD.AnnoyIndex
_real_cls = FALLBACK_MOD._try_import_real_annoy()


def _build_sample(index_cls, metric="euclidean"):
    idx = index_cls(3, metric)
    random.seed(0)
    for i in range(10):
        idx.add_item(i, [random.random() for _ in range(3)])
    idx.build(10)
    return idx


def test_persistence(tmp_path):
    idx = _build_sample(FallbackAnnoy)
    q = [0.1, 0.2, 0.3]
    before = idx.get_nns_by_vector(q, 3)
    fn = tmp_path / "index.ann"
    idx.save(fn)

    idx2 = FallbackAnnoy(3, "euclidean")
    idx2.load(fn)
    after = idx2.get_nns_by_vector(q, 3)
    assert before == after


@pytest.mark.parametrize("metric", ["euclidean", "angular", "manhattan", "hamming", "dot"])
def test_metrics(metric):
    idx = _build_sample(FallbackAnnoy, metric)
    q = [random.random() for _ in range(3)]
    assert idx.get_nns_by_vector(q, 1)


def test_accuracy_against_real(tmp_path):
    if _real_cls is None:
        pytest.skip("real annoy not available")
    real_idx = _build_sample(_real_cls)
    fb_idx = _build_sample(FallbackAnnoy)
    q = [random.random() for _ in range(3)]
    assert real_idx.get_nns_by_vector(q, 5) == fb_idx.get_nns_by_vector(q, 5)

