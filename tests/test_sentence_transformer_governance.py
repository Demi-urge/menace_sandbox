"""Tests ensuring governed embedding usage across the repository."""

import pathlib


# Relative paths (from repo root) that are allowed to invoke
# ``SentenceTransformer.encode`` directly.  ``governed_embeddings.py`` provides  # path-ignore
# the safe wrapper and this test module contains reference code for the check
# itself.
ALLOWLIST = {
    pathlib.Path("governed_embeddings.py"),  # path-ignore
    pathlib.Path("tests/test_sentence_transformer_governance.py"),  # path-ignore
}


def test_no_direct_sentence_transformer_encode():
    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for path in root.rglob("*.py"):  # path-ignore
        rel = path.relative_to(root)
        if rel in ALLOWLIST:
            continue
        text = path.read_text(encoding="utf-8")
        if "SentenceTransformer" not in text:
            continue
        for line in text.splitlines():
            if ".encode(" not in line:
                continue
            if "tokenizer.encode(" in line:
                continue
            if "governed_embed" in line:
                continue
            offenders.append(str(rel))
            break
    assert offenders == [], f"Ungoverned SentenceTransformer.encode calls: {offenders}"
