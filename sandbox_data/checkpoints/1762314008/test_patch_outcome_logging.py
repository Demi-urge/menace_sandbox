import json
from threading import Thread
import types

import patch_score_backend as psb
from patch_score_backend import FilePatchScoreBackend


def test_patch_score_backend_logs_success_and_failure(tmp_path, monkeypatch):
    be = FilePatchScoreBackend(str(tmp_path))
    counter = iter(range(1000, 2000))
    monkeypatch.setattr(psb.time, "time", lambda: next(counter))
    be.store({"description": "p1", "result": "ok", "vectors": [("db1", "v1")]})
    be.store({"description": "p2", "result": "error", "vectors": [("db2", "v2")]})
    files = sorted(tmp_path.glob("*.json"))
    records = [json.loads(f.read_text()) for f in files]
    assert any(r["description"] == "p1" and r["result"] == "ok" for r in records)
    assert any(r["description"] == "p2" and r["result"] == "error" for r in records)


def test_patch_score_backend_concurrent_writes(tmp_path, monkeypatch):
    be = FilePatchScoreBackend(str(tmp_path))
    counter = iter(range(1000, 2000))
    monkeypatch.setattr(psb.time, "time", lambda: next(counter))

    def worker(i: int) -> None:
        be.store({"description": f"p{i}", "result": "ok"})

    threads = [Thread(target=worker, args=(i,)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 10
