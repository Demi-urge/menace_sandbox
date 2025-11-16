import json
import math
from pathlib import Path

from failure_fingerprint_store import FailureFingerprint, FailureFingerprintStore


class DummyVectorStore:
    def __init__(self, dim: int = 2, path: Path = Path('dummy.ann'), metric: str | None = None):
        self.dim = dim
        self.path = Path(path)
        self.meta_path = self.path.with_suffix('.meta.json')
        self.metric = metric
        self.records: dict[str, list[float]] = {}
        self.meta: dict[str, dict] = {}

    def add(self, kind, record_id, vector, *, origin_db=None, metadata=None):
        self.records[record_id] = list(vector)
        self.meta[record_id] = metadata or {}

    def query(self, vector, top_k=5):
        res = []
        for rid, vec in self.records.items():
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(vector, vec)))
            res.append((rid, dist))
        res.sort(key=lambda x: x[1])
        return res[:top_k]

    def load(self):
        self.records.clear()
        self.meta.clear()


class DummyEmbedder:
    def encode(self, texts):
        result = []
        for text in texts:
            a = sum(ord(c) for c in text) % 7
            b = len(text)
            result.append([float(a), float(b)])
        return result


class DummyVectorService:
    def __init__(self):
        self.text_embedder = DummyEmbedder()
        self.vector_store = DummyVectorStore()

    def vectorise(self, kind: str, record: dict) -> list[float]:
        return self.text_embedder.encode([record.get('text', '')])[0]


def make_store(tmp_path):
    svc = DummyVectorService()
    return FailureFingerprintStore(
        path=tmp_path / 'fps.jsonl',
        vector_service=svc,
        similarity_threshold=0.9,
        compact_interval=0,
    )


def test_log_and_find_similar(tmp_path):
    store = make_store(tmp_path)
    fp1 = FailureFingerprint('a.py', 'func', 'oops', 'trace one', 'p1')  # path-ignore
    store.log(fp1)
    assert fp1.embedding and fp1.embedding_metadata['dim'] == 2
    rid1 = store._id_for(fp1)
    assert 'embedding_meta' in store.vector_service.vector_store.meta[rid1]

    fp2 = FailureFingerprint('b.py', 'func2', 'err', 'trace one', 'p2')  # path-ignore
    matches = store.find_similar(fp2)
    assert matches and matches[0].filename == 'a.py'  # path-ignore


def test_compact_rewrites_store(tmp_path):
    store = make_store(tmp_path)
    fp1 = FailureFingerprint('a.py', 'f', 'e', 'trace one', 'p')  # path-ignore
    fp2 = FailureFingerprint('b.py', 'g', 'e', 'trace two', 'p')  # path-ignore
    store.log(fp1)
    store.log(fp2)
    rid1 = store._id_for(fp1)
    rid2 = store._id_for(fp2)
    del store._cache[rid1]
    store.compact()
    with store.path.open('r', encoding='utf-8') as fh:
        lines = fh.read().strip().splitlines()
    assert len(lines) == 1 and rid2 in lines[0]
    assert rid1 not in store.vector_service.vector_store.records


def test_cluster_assignment(tmp_path):
    store = make_store(tmp_path)
    fp1 = FailureFingerprint('a.py', 'f', 'e', 'trace one', 'p')  # path-ignore
    fp2 = FailureFingerprint('b.py', 'g', 'e', 'trace one', 'p')  # path-ignore
    fp3 = FailureFingerprint('c.py', 'h', 'e', 'xyz', 'p')  # path-ignore
    store.log(fp1)
    store.log(fp2)
    store.log(fp3)

    assert fp1.cluster_id == fp2.cluster_id
    assert fp3.cluster_id != fp1.cluster_id
    cluster = store.get_cluster(fp1.cluster_id)
    assert {f.filename for f in cluster} == {'a.py', 'b.py'}  # path-ignore


def test_duplicate_increments_count(tmp_path):
    store = make_store(tmp_path)
    fp = FailureFingerprint('a.py', 'f', 'e', 'trace one', 'p')  # path-ignore
    store.add(fp)
    store.add(FailureFingerprint('a.py', 'f', 'e', 'trace one', 'p'))  # path-ignore
    rid = store._id_for(fp)
    assert store._cache[rid].count == 2
    with store.path.open('r', encoding='utf-8') as fh:
        lines = fh.read().strip().splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data['count'] == 2
    assert store.vector_service.vector_store.meta[rid]['count'] == 2
