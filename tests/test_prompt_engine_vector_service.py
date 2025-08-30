import logging

from prompt_engine import PromptEngine, DEFAULT_TEMPLATE
import vector_service.vectorizer as vz
from vector_service.retriever import PatchRetriever
from vector_service.vector_store import AnnoyVectorStore


def _setup_store(monkeypatch, tmp_path, patches, query_vec):
    """Return a PatchRetriever searching *patches* with fixed vectors."""
    monkeypatch.setattr(vz, "load_handlers", lambda: {})

    def vectorise(self, kind, record):
        text = record.get("text", "")
        if text == "goal":
            return query_vec
        return [0.0 for _ in query_vec]

    monkeypatch.setattr(vz.SharedVectorService, "vectorise", vectorise, raising=False)

    store = AnnoyVectorStore(dim=len(query_vec), path=tmp_path / "idx.ann")
    orig_save = store._save
    monkeypatch.setattr(store, "_save", lambda: None)
    for pid, vec, meta in patches:
        store.add("patch", pid, vec, origin_db="patch", metadata=meta)
    store._save = orig_save
    store._save()

    vec_service = vz.SharedVectorService(text_embedder=None, vector_store=store)
    return PatchRetriever(store=store, vector_service=vec_service)


def test_prompt_engine_retrieves_top_n_snippets(monkeypatch, tmp_path):
    patches = [
        ("1", [1.0, 0.0], {"summary": "A1", "tests_passed": True}),
        ("2", [0.8, 0.2], {"summary": "A2", "tests_passed": True}),
        ("3", [0.0, 1.0], {"summary": "A3", "tests_passed": True}),
    ]
    pr = _setup_store(monkeypatch, tmp_path, patches, [1.0, 0.0])
    engine = PromptEngine(retriever=pr, top_n=2)
    prompt = engine.build_prompt("goal")
    assert "Code summary: A1" in prompt
    assert "Code summary: A2" in prompt
    assert "A3" not in prompt


def test_prompt_engine_orders_by_roi_and_recency(monkeypatch, tmp_path):
    patches = [
        ("1", [1.0, 0.0], {"summary": "low", "tests_passed": True, "roi_delta": 0.1}),
        ("2", [1.0, 0.0], {"summary": "high", "tests_passed": True, "roi_delta": 0.9}),
        ("3", [1.0, 0.0], {"summary": "old fail", "tests_passed": False, "ts": 1}),
        ("4", [1.0, 0.0], {"summary": "new fail", "tests_passed": False, "ts": 2}),
    ]
    pr = _setup_store(monkeypatch, tmp_path, patches, [1.0, 0.0])
    engine = PromptEngine(retriever=pr, top_n=4)
    prompt = engine.build_prompt("goal")
    assert prompt.index("Code summary: high") < prompt.index("Code summary: low")
    assert prompt.index("Code summary: new fail") < prompt.index("Code summary: old fail")


def test_retry_trace_injection(monkeypatch, tmp_path):
    patches = [("1", [1.0, 0.0], {"summary": "foo", "tests_passed": True})]
    pr = _setup_store(monkeypatch, tmp_path, patches, [1.0, 0.0])
    engine = PromptEngine(retriever=pr)
    trace = "Traceback: fail"
    prompt = engine.build_prompt("goal", retry_info=trace)
    expected = f"Previous attempt failed with {trace}; seek alternative solution."
    assert expected in prompt


def test_fallback_when_confidence_low(monkeypatch, tmp_path, caplog):
    patches = [("1", [-1.0, 0.0], {"summary": "bad", "tests_passed": True})]
    pr = _setup_store(monkeypatch, tmp_path, patches, [1.0, 0.0])
    engine = PromptEngine(retriever=pr, top_n=1)
    monkeypatch.setattr(engine, "_static_prompt", lambda: DEFAULT_TEMPLATE)
    with caplog.at_level(logging.INFO):
        prompt = engine.build_prompt("goal")
    assert prompt == DEFAULT_TEMPLATE
    assert "falling back" in caplog.text.lower()
