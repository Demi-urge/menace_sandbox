import importlib.util
from pathlib import Path


# Import module directly to avoid executing heavy package-level imports
spec = importlib.util.spec_from_file_location(
    "text_preprocessor",
    Path(__file__).resolve().parents[1] / "vector_service" / "text_preprocessor.py",
)
text_preprocessor = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(text_preprocessor)
generalise = text_preprocessor.generalise


def test_generalise_shorter_and_semantic():
    text = "The quick brown foxes were jumping quickly over the lazy dogs."
    result = generalise(text)

    # Output should be lowercase and shorter
    assert result == result.lower()
    assert len(result.split()) < len(text.split())

    tokens = set(result.split())
    # Core meaning should be preserved
    assert {"quick", "brown"} <= tokens
    assert tokens.intersection({"fox", "foxes"})
    assert tokens.intersection({"dog", "dogs"})
    assert "the" not in tokens
