import json
import importlib.util
from pathlib import Path
import sys
import types

tp_spec = importlib.util.spec_from_file_location(
    "text_preprocessor", Path(__file__).resolve().parents[1] / "vector_service" / "text_preprocessor.py"
)
tp = importlib.util.module_from_spec(tp_spec)
assert tp_spec.loader is not None
sys.modules[tp_spec.name] = tp
tp_spec.loader.exec_module(tp)

wf_spec = importlib.util.spec_from_file_location(
    "workflow_vectorizer", Path(__file__).resolve().parents[1] / "workflow_vectorizer.py"
)
wf_mod = importlib.util.module_from_spec(wf_spec)
assert wf_spec.loader is not None
sys.modules[wf_spec.name] = wf_mod
wf_spec.loader.exec_module(wf_mod)
WorkflowVectorizer = wf_mod.WorkflowVectorizer
_EMBED_DIM = wf_mod._EMBED_DIM


def test_load_db_configs_json_yaml(tmp_path):
    cfg_json = tmp_path / "cfg.json"
    cfg_json.write_text(json.dumps({"code": {"chunk_size": 123}}))
    tp.load_db_configs(str(cfg_json))
    assert tp.get_config("code").chunk_size == 123

    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text("workflow:\n  split_sentences: false\n")
    tp.load_db_configs(str(cfg_yaml))
    assert tp.get_config("workflow").split_sentences is False
def test_workflow_transform_respects_config(monkeypatch):
    cfg = tp.PreprocessingConfig(split_sentences=False, filter_semantic_risks=False)
    tp.register_preprocessor("workflow", cfg)

    captured = {}

    def fake_embed(texts):
        captured["texts"] = texts
        return [[0.0] * _EMBED_DIM for _ in texts]

    monkeypatch.setattr("workflow_vectorizer._embed_texts", fake_embed)
    monkeypatch.setattr("workflow_vectorizer.compress_snippets", lambda d: d)

    wf = {"name": "", "description": "A. B."}
    WorkflowVectorizer().transform(wf, config=cfg)
    assert captured["texts"] == ["\nA. B."]

    cfg2 = tp.PreprocessingConfig(split_sentences=True, filter_semantic_risks=True)
    tp.register_preprocessor("workflow", cfg2)
    monkeypatch.setattr(
        "workflow_vectorizer.find_semantic_risks",
        lambda lines: [1] if "B" in lines[0] else [],
    )
    WorkflowVectorizer().transform(wf, config=cfg2)
    assert captured["texts"] == ["A."]

