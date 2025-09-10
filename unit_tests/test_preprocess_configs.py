import re

import pytest

from vector_service.text_preprocessor import get_config


def _simulate(text: str, db_key: str):
    cfg = get_config(db_key)
    if cfg.split_sentences:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    else:
        sentences = [text]
    joined = "\n".join(sentences)
    chunks = [joined[i:i + cfg.chunk_size] for i in range(0, len(joined), cfg.chunk_size)]
    return cfg, sentences, chunks


@pytest.mark.parametrize(
    "db_key, text, expected_sentences, expected_chunk_size",
    [
        ("code", "One. Two.", ["One. Two."], 50),
        ("bot", "One. Two.", ["One.", "Two."], 60),
        ("error", "One. Two.", ["One.", "Two."], 70),
        ("workflow", "One. Two.", ["One. Two."], 80),
    ],
)
def test_preprocess_configs(db_key, text, expected_sentences, expected_chunk_size):
    cfg, sentences, chunks = _simulate(text, db_key)
    assert sentences == expected_sentences
    assert cfg.chunk_size == expected_chunk_size
    assert all(len(c) <= expected_chunk_size for c in chunks)
