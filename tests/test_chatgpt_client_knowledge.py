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

    def search_context(self, query, limit=5):
        self.record["query"] = query
        return [{"prompt": "p1", "response": "r1"}]

    def log_interaction(self, prompt, response, tags):
        self.logged = (prompt, response, list(tags))


def test_ask_injects_context_and_logs(monkeypatch):
    record = {}
    # stub requests module so ChatGPTClient doesn't require real dependency
    cib.requests = type("R", (), {"Timeout": Exception, "RequestException": Exception})

    session = DummySession(record)
    client = cib.ChatGPTClient(api_key="key", session=session)
    knowledge = DummyKnowledge(record)

    resp = client.ask([{"role": "user", "content": "hello"}], knowledge=knowledge)

    assert record["query"] == "hello"
    msgs = record["messages"]
    assert msgs[0]["role"] == "system"
    assert "Prompt: p1" in msgs[0]["content"]
    assert msgs[1]["content"] == "hello"
    assert knowledge.logged == ("hello", "ok", [])
    assert resp["choices"][0]["message"]["content"] == "ok"
