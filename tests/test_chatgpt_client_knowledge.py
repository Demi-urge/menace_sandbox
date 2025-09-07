import menace.chatgpt_idea_bot as cib


class DummyResp:
    status_code = 200

    @staticmethod
    def json():
        return {"choices": [{"message": {"content": "ok"}}]}


class DummySession:
    def __init__(self, record):
        self.record = record

    def post(self, url, headers=None, json=None, timeout=0):
        self.record["messages"] = json["messages"]
        return DummyResp()


class DummyKnowledge:
    def __init__(self, record):
        self.record = record
        self.logged = None

    class Entry:
        def __init__(self, prompt: str, response: str) -> None:
            self.prompt = prompt
            self.response = response

    def get_similar_entries(self, query, limit=5, use_embeddings=False):
        self.record["query"] = query
        return [(1.0, self.Entry("p1", "r1"))]

    def log_interaction(self, prompt, response, tags):
        self.logged = (prompt, response, list(tags))


def test_ask_injects_context_and_logs(monkeypatch):
    record = {}
    # stub requests module so ChatGPTClient doesn't require real dependency
    cib.requests = type("R", (), {"Timeout": Exception, "RequestException": Exception})

    session = DummySession(record)

    class DummyBuilder:
        def refresh_db_weights(self):
            pass

        def build(self, query, **_):
            return ""

    client = cib.ChatGPTClient(api_key="key", session=session, context_builder=DummyBuilder())
    knowledge = DummyKnowledge(record)

    resp = client.ask(
        [{"role": "user", "content": "hello"}],
        knowledge=knowledge,
        use_memory=True,
        relevance_threshold=0.5,
        max_summary_length=20,
    )

    assert record["query"] == "hello"
    msgs = record["messages"]
    assert msgs[0]["role"] == "system"
    assert "Prompt: p1" in msgs[0]["content"]
    assert len(msgs[0]["content"]) <= 20
    assert msgs[1]["content"] == "hello"
    assert knowledge.logged == ("hello", "ok", [])
    assert resp["choices"][0]["message"]["content"] == "ok"
