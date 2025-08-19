import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

import neurosales.embedding as emb


def cosine(a, b):
    A = np.array(a)
    B = np.array(b)
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    if denom == 0.0:
        return 0.0
    return float(np.dot(A, B) / denom)


def test_shared_embedder_consistency_and_similarity():
    emb._MODEL = None
    data_path = Path(__file__).parent / "data" / "embed_sentences.txt"
    texts = [line.strip() for line in data_path.read_text().splitlines()]

    with patch("neurosales.embedding.SentenceTransformer") as ST:
        model = MagicMock()
        model.get_sentence_embedding_dimension.return_value = 2
        def fake_encode(txt, convert_to_numpy=True):
            if isinstance(txt, list):
                return np.array(
                    [[1.0 if "hello" in t else 0.0, float(len(t))] for t in txt],
                    dtype="float32",
                )
            flag = 1.0 if "hello" in txt else 0.0
            return np.array([flag, float(len(txt))], dtype="float32")

        model.encode.side_effect = fake_encode
        ST.return_value = model

        vecs = [emb.embed_text(t) for t in texts]
        assert all(len(v) == len(vecs[0]) for v in vecs)
        assert len(vecs[0]) == emb.embedding_dimension()

        sim_close = cosine(vecs[0], vecs[1])
        sim_far = cosine(vecs[0], vecs[2])
        assert sim_close > sim_far


def test_embed_text_skips_on_semantic_risk(monkeypatch):
    """Unsafe patterns should be flagged and prevent embedding."""
    emb._MODEL = None

    class DummyModel:
        def __init__(self, name: str) -> None:
            pass

        def get_sentence_embedding_dimension(self) -> int:
            return 2

        def encode(self, text: str, convert_to_numpy: bool = True):
            raise AssertionError("encode should not be called when risk detected")

    monkeypatch.setattr(emb, "SentenceTransformer", DummyModel)

    called = False

    def fake_embed(text: str, model):
        nonlocal called
        called = True
        return [0.0, 0.0]

    monkeypatch.setattr(emb, "governed_embed", fake_embed)

    with pytest.raises(ValueError):
        emb.embed_text("eval('data')")

    assert not called
