def test_rest_backend_propagates_metadata(monkeypatch):
    import local_backend
    from llm_interface import LLMClient, Prompt
    import prompt_db
    import types

    captured_payload = {}

    def fake_post(self, payload):
        captured_payload.update(payload)
        return {"text": "ok"}

    monkeypatch.setattr(local_backend._RESTBackend, "_post", fake_post)

    logged = {}

    class DummyDB:
        def __init__(self, *_a, **_k):
            pass

        def log(self, prompt, result, backend=None):
            logged["prompt"] = prompt
            logged["raw"] = result.raw
            logged["backend"] = backend

    monkeypatch.setattr(prompt_db, "PromptDB", DummyDB)

    backend = local_backend._RESTBackend(model="m", base_url="http://x", endpoint="/gen")
    client = LLMClient(model="m", backends=[backend])

    prompt = Prompt(
        text="hi",
        examples=["ex"],
        outcome_tags=["tag"],
        vector_confidence=0.5,
        origin="context_builder",
        metadata={"vector_confidences": [0.5]},
    )
    builder = types.SimpleNamespace(roi_tracker=None)
    client.generate(prompt, context_builder=builder)

    assert captured_payload["tags"] == ["tag"]
    assert captured_payload["vector_confidence"] == 0.5
    assert logged["prompt"].examples == ["ex"]
    assert logged["prompt"].vector_confidence == 0.5
    assert logged["prompt"].outcome_tags == ["tag"]
    assert logged["raw"]["tags"] == ["tag"]
    assert logged["raw"]["vector_confidence"] == 0.5
    assert logged["backend"] == "m"
