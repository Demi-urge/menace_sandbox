import importlib.util
import sys
from pathlib import Path


# Import module directly to avoid executing heavy package-level imports
spec = importlib.util.spec_from_file_location(
    "text_preprocessor",
    Path(__file__).resolve().parents[1] / "vector_service" / "text_preprocessor.py",
)
text_preprocessor = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = text_preprocessor
spec.loader.exec_module(text_preprocessor)
generalise = text_preprocessor.generalise
PreprocessingConfig = text_preprocessor.PreprocessingConfig
load_stop_words = text_preprocessor.load_stop_words
register_preprocessor = text_preprocessor.register_preprocessor


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


def test_load_stop_words_and_config(tmp_path):
    stop_file = tmp_path / "stop.txt"
    stop_file.write_text("hello\n")
    stop_words = load_stop_words(str(stop_file))
    cfg = PreprocessingConfig(stop_words=stop_words)

    assert generalise("Hello world", config=cfg) == "world"


def test_language_detection_and_stemming():
    cfg = PreprocessingConfig(stop_words={"los"})
    text = "Los gatos corriendo perezosos eran"
    result = generalise(text, config=cfg)
    assert set(result.split()) == {"gat", "corr", "perez", "eran"}


def test_database_specific_configuration():
    cfg = PreprocessingConfig(stop_words={"alpha"})
    register_preprocessor("db1", cfg)
    assert generalise("alpha beta", db_key="db1") == "beta"
