import types

from menace.knowledge_graph import KnowledgeGraph
from menace.log_tags import FEEDBACK
from tests.test_self_improvement_logging import _load_engine


class DummyMemory:
    def __init__(self, kg: KnowledgeGraph) -> None:
        self.entries: list[dict] = []
        self.kg = kg

    def log_interaction(self, prompt: str, response: str, tags: list[str] | None = None) -> int:
        tags = list(tags or [])
        self.entries.append({"prompt": prompt, "response": response, "tags": tags})
        self.kg.add_memory_entry(prompt, tags)
        return len(self.entries)

    def search_context(
        self,
        query: str,
        *,
        limit: int = 5,
        tags: list[str] | None = None,
        **_: object,
    ) -> list[object]:
        results: list[object] = []
        for e in self.entries:
            if query in e["prompt"] and (
                not tags or set(tags) & set(e["tags"])
            ):
                results.append(types.SimpleNamespace(response=e["response"]))
            if len(results) >= limit:
                break
        return results


def test_memory_driven_improvement(monkeypatch):
    kg = KnowledgeGraph()
    mem = DummyMemory(kg)

    mem.log_interaction("patch:modA", "ValueError: boom", tags=[FEEDBACK])

    sie = _load_engine()
    engine = sie.SelfImprovementEngine.__new__(sie.SelfImprovementEngine)
    engine.gpt_memory = mem
    engine.self_coding_engine = object()
    engine.logger = types.SimpleNamespace(info=lambda *a, **k: None, exception=lambda *a, **k: None)

    monkeypatch.setattr(sie, "generate_patch", lambda module, coder, **kw: 1)

    captured: list[str] = []
    orig_log = mem.log_interaction

    def capture(prompt, response, tags=None):
        captured.append(prompt)
        return orig_log(prompt, response, tags=tags)

    monkeypatch.setattr(mem, "log_interaction", capture)

    engine._generate_patch_with_memory("modA", "improve")

    summary = engine._memory_summaries("modA")
    assert "ValueError: boom" in summary

    assert captured and "ValueError: boom" in captured[0]

    assert "memory:patch:modA" in kg.graph
    assert any(node.startswith("memory:improve:modA") for node in kg.graph)
