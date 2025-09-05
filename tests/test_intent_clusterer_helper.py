import intent_clusterer as ic


def test_find_modules_related_to_helper(monkeypatch):
    class Dummy:
        def __init__(self):
            self.args = None

        def find_modules_related_to(self, query, top_k=5, *, include_clusters=False):
            self.args = (query, top_k, include_clusters)
            return [ic.IntentMatch(path="x.py", similarity=1.0, cluster_ids=[])]  # path-ignore

    dummy = Dummy()
    monkeypatch.setattr(ic, "IntentClusterer", lambda: dummy)
    res = ic.find_modules_related_to("demo", top_k=2)
    assert res == [ic.IntentMatch(path="x.py", similarity=1.0, cluster_ids=[])]  # path-ignore
    assert dummy.args == ("demo", 2, False)
