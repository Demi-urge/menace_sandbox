import importlib
import sys
import threading


def test_concurrent_resolve_path(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    file_a = repo / "a.txt"
    file_a.write_text("a\n")
    file_b = repo / "dir" / "b.txt"
    file_b.parent.mkdir(parents=True)
    file_b.write_text("b\n")

    monkeypatch.setenv("MENACE_ROOT", str(repo))
    sys.modules.pop("dynamic_path_router", None)
    dpr = importlib.import_module("dynamic_path_router")
    dpr.clear_cache()

    results = []
    errors = []

    def worker(name: str) -> None:
        try:
            results.append(dpr.resolve_path(name))
        except Exception as exc:  # pragma: no cover - shouldn't happen
            errors.append(exc)

    names = ["a.txt", "dir/b.txt"] * 10
    threads = [threading.Thread(target=worker, args=(n,)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert results.count(file_a.resolve()) == 10
    assert results.count(file_b.resolve()) == 10
    cache = dpr.list_files()
    root_key = repo.resolve().as_posix()
    assert cache.get(f"{root_key}:a.txt") == file_a.resolve()
    assert cache.get(f"{root_key}:dir/b.txt") == file_b.resolve()
    assert len(cache) == 2
