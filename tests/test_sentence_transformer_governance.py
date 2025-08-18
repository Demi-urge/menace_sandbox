"""Tests ensuring governed embedding usage across the repository."""

import pathlib


# Relative paths (from repo root) that are allowed to use SentenceTransformer
# directly.  These files either implement the governed wrapper or contain test
# code that is known to be safe.
ALLOWLIST = {
    pathlib.Path("governed_embeddings.py"),
    pathlib.Path("tests/test_sentence_transformer_governance.py"),
}


def test_no_direct_sentence_transformer_encode():
    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root)
        if rel in ALLOWLIST:
            continue
        text = path.read_text(encoding="utf-8")
        if (
            "SentenceTransformer" in text
            and ".encode(" in text
            and "governed_embed" not in text
        ):
            offenders.append(str(rel))
    assert offenders == [], f"Ungoverned SentenceTransformer.encode calls: {offenders}"
