import intent_clusterer as ic


def test_find_modules_related_to_helper(monkeypatch):
    class Dummy:
        def __init__(self):
            self.args = None

        def find_modules_related_to(self, query, top_k=5):
            self.args = (query, top_k)
            return [{"path": "x.py", "score": 1.0}]

    dummy = Dummy()
    monkeypatch.setattr(ic, "IntentClusterer", lambda: dummy)
    res = ic.find_modules_related_to("demo", top_k=2)
    assert res == [{"path": "x.py", "score": 1.0}]
    assert dummy.args == ("demo", 2)
