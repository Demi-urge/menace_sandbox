import inspect
import logging
import os
import sys
import threading
import types

import menace.governed_embeddings as governed_embeddings
import menace.local_knowledge_module as lkm
import menace.menace_memory_manager as mmm
from menace.governed_embeddings import governed_embed


def test_governed_embed_blocks_gpl(caplog):
    text = "This code is licensed under the GNU General Public License"
    with caplog.at_level(logging.WARNING):
        assert governed_embed(text) is None
    assert any("license" in r.msg for r in caplog.records)


def test_governed_embed_redacts_and_embeds(monkeypatch, caplog):
    recorded = {}

    def fake_redact(text: str) -> str:
        recorded["redacted"] = text
        return "clean"

    def fake_check(text: str):
        recorded["checked"] = text
        return None

    class DummyEmbedder:
        def encode(self, arr):
            assert arr[0] == "clean"
            return [types.SimpleNamespace(tolist=lambda: [0.1, 0.2])]

    monkeypatch.setattr(governed_embeddings, "redact", fake_redact)
    monkeypatch.setattr(governed_embeddings, "license_check", fake_check)
    monkeypatch.setattr(governed_embeddings, "get_embedder", lambda: DummyEmbedder())
    with caplog.at_level(logging.WARNING):
        vec = governed_embed("secret token")
    assert vec == [0.1, 0.2]
    assert recorded["redacted"] == "secret token"
    assert recorded["checked"] == "secret token"
    assert any("redacted" in r.msg for r in caplog.records)


def test_memory_manager_uses_governed_embed(monkeypatch):
    called = {}

    def fake_embed(text: str, embedder=None):
        called["text"] = text
        return [0.0]

    monkeypatch.setattr(mmm, "governed_embed", fake_embed)
    params = inspect.signature(mmm.MenaceMemoryManager).parameters
    kwargs = {"path": ":memory:"} if "path" in params else {}
    mm = mmm.MenaceMemoryManager(**kwargs)
    assert mm._embed("hello") == [0.0]
    assert called["text"] == "hello"


def test_init_local_knowledge_uses_get_embedder(monkeypatch, tmp_path):
    called = {}

    def fake_get(*args, **kwargs):
        called["hit"] = True
        class Dummy:
            def encode(self, arr):
                return [[0.0]]
        return Dummy()

    monkeypatch.setattr(lkm, "get_embedder", fake_get)
    lkm._LOCAL_KNOWLEDGE = None
    lkm.init_local_knowledge(tmp_path / "db.sqlite")
    assert called.get("hit")


def test_get_embedder_initialises_without_token(monkeypatch):
    monkeypatch.delenv("HUGGINGFACE_API_TOKEN", raising=False)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER", None)
    monkeypatch.setattr(governed_embeddings, "_resolve_local_snapshot", lambda *_: None)

    class DummySentenceTransformer:
        def __init__(self, name: str, **_: object) -> None:
            self.name = name

    monkeypatch.setattr(
        governed_embeddings,
        "SentenceTransformer",
        DummySentenceTransformer,
    )

    embedder = governed_embeddings.get_embedder()
    assert isinstance(embedder, DummySentenceTransformer)
    assert embedder.name == "all-MiniLM-L6-v2"


def test_get_embedder_exports_token_when_available(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_API_TOKEN", "secret-token")
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    monkeypatch.delenv("HF_HUB_TOKEN", raising=False)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER", None)
    monkeypatch.setattr(governed_embeddings, "_resolve_local_snapshot", lambda *_: None)

    class DummySentenceTransformer:
        def __init__(self, name: str, **_: object) -> None:
            self.name = name

    monkeypatch.setattr(
        governed_embeddings,
        "SentenceTransformer",
        DummySentenceTransformer,
    )

    embedder = governed_embeddings.get_embedder()
    assert isinstance(embedder, DummySentenceTransformer)
    assert embedder.name == "all-MiniLM-L6-v2"
    assert os.environ["HUGGINGFACEHUB_API_TOKEN"] == "secret-token"
    assert os.environ["HF_HUB_TOKEN"] == "secret-token"


def test_get_embedder_configures_hf_timeouts(monkeypatch):
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER", None)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_INIT_EVENT", threading.Event())
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_INIT_THREAD", None)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_TIMEOUT_LOGGED", False)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_TIMEOUT_REACHED", False)
    monkeypatch.setattr(governed_embeddings, "_HF_TIMEOUT_CONFIGURED", False)
    monkeypatch.setattr(governed_embeddings, "_HF_TIMEOUT_SETTINGS", {})
    monkeypatch.setattr(governed_embeddings, "_resolve_local_snapshot", lambda *_: None)
    monkeypatch.setattr(governed_embeddings, "_cleanup_hf_locks", lambda *a, **k: None)

    class DummySentenceTransformer:
        def __init__(self, name: str, **_: object) -> None:
            self.name = name

    monkeypatch.setattr(
        governed_embeddings,
        "SentenceTransformer",
        DummySentenceTransformer,
    )

    for env in (
        "HF_HUB_TIMEOUT",
        "HF_HUB_READ_TIMEOUT",
        "HF_HUB_CONNECTION_TIMEOUT",
        "HF_HUB_DOWNLOAD_RETRIES",
    ):
        monkeypatch.delenv(env, raising=False)

    monkeypatch.setenv("EMBEDDER_HF_TIMEOUT", "12")
    monkeypatch.setenv("EMBEDDER_HF_RETRIES", "2")

    embedder = governed_embeddings.get_embedder()
    assert isinstance(embedder, DummySentenceTransformer)
    assert embedder.name == "all-MiniLM-L6-v2"
    assert os.environ["HF_HUB_TIMEOUT"] == "12"
    assert os.environ["HF_HUB_READ_TIMEOUT"] == "12"
    assert os.environ["HF_HUB_CONNECTION_TIMEOUT"] == "12"
    assert os.environ["HF_HUB_DOWNLOAD_RETRIES"] == "2"


def test_initialise_embedder_wait_capped(monkeypatch, caplog):
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER", None)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_TIMEOUT_LOGGED", False)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_TIMEOUT_REACHED", False)
    monkeypatch.setattr(
        governed_embeddings, "_EMBEDDER_THREAD_LOCK", threading.RLock()
    )
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_INIT_TIMEOUT", 9999.0)
    monkeypatch.setattr(governed_embeddings, "_MAX_EMBEDDER_WAIT", 0.5)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_WAIT_CAPPED", False)

    recorded = {}

    class DummyEvent:
        def wait(self, timeout: float) -> bool:
            recorded["timeout"] = timeout
            return False

        def is_set(self) -> bool:
            return False

    monkeypatch.setattr(
        governed_embeddings,
        "_ensure_embedder_thread_locked",
        lambda: DummyEvent(),
    )

    with caplog.at_level(logging.WARNING):
        governed_embeddings._initialise_embedder_with_timeout()

    assert recorded["timeout"] == 0.5
    assert any("capping embedder" in rec.msg for rec in caplog.records)


def test_initialise_embedder_soft_wait(monkeypatch, caplog):
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER", None)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_TIMEOUT_LOGGED", False)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_TIMEOUT_REACHED", False)
    monkeypatch.setattr(
        governed_embeddings, "_EMBEDDER_THREAD_LOCK", threading.RLock()
    )
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_INIT_TIMEOUT", 9999.0)
    monkeypatch.setattr(governed_embeddings, "_MAX_EMBEDDER_WAIT", 9999.0)
    monkeypatch.setattr(governed_embeddings, "_SOFT_EMBEDDER_WAIT", 0.2)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_WAIT_CAPPED", False)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_SOFT_WAIT_LOGGED", False)

    recorded = {}

    class DummyEvent:
        def wait(self, timeout: float) -> bool:
            recorded["timeout"] = timeout
            return False

        def is_set(self) -> bool:
            return False

    monkeypatch.setattr(
        governed_embeddings,
        "_ensure_embedder_thread_locked",
        lambda: DummyEvent(),
    )

    with caplog.at_level(logging.WARNING):
        governed_embeddings._initialise_embedder_with_timeout()

    assert recorded["timeout"] == 0.2
    assert any("EMBEDDER_INIT_SOFT_WAIT" in rec.msg for rec in caplog.records)


def test_initialise_embedder_timeout_override(monkeypatch, caplog):
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER", None)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_TIMEOUT_LOGGED", False)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_TIMEOUT_REACHED", False)
    monkeypatch.setattr(
        governed_embeddings, "_EMBEDDER_THREAD_LOCK", threading.RLock()
    )

    recorded = {}

    class DummyEvent:
        def wait(self, timeout: float) -> bool:
            recorded.setdefault("timeouts", []).append(timeout)
            return False

    monkeypatch.setattr(
        governed_embeddings,
        "_ensure_embedder_thread_locked",
        lambda: DummyEvent(),
    )

    with caplog.at_level(logging.DEBUG):
        governed_embeddings._initialise_embedder_with_timeout(
            timeout_override=0.05, suppress_timeout_log=True
        )

    assert recorded["timeouts"] == [0.05]
    messages = [rec.msg for rec in caplog.records]
    if not any("pending" in msg for msg in messages):
        assert governed_embeddings._EMBEDDER is not None


def test_initialise_embedder_timeout_override_skips_wait(monkeypatch):
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER", None)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_TIMEOUT_LOGGED", False)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_TIMEOUT_REACHED", False)
    monkeypatch.setattr(
        governed_embeddings, "_EMBEDDER_THREAD_LOCK", threading.RLock()
    )

    class DummyEvent:
        def __init__(self) -> None:
            self.calls: list[float] = []

        def wait(self, timeout: float) -> bool:
            self.calls.append(timeout)
            return False

        def is_set(self) -> bool:
            return False

    event = DummyEvent()
    monkeypatch.setattr(
        governed_embeddings,
        "_ensure_embedder_thread_locked",
        lambda: event,
    )

    governed_embeddings._initialise_embedder_with_timeout(
        timeout_override=0.1, suppress_timeout_log=True
    )
    governed_embeddings._initialise_embedder_with_timeout(
        timeout_override=0.2, suppress_timeout_log=True
    )

    assert event.calls == [0.1]


def test_timeout_cap_limits_configuration(monkeypatch, caplog):
    original_timeout = governed_embeddings._EMBEDDER_INIT_TIMEOUT
    original_max_wait = governed_embeddings._MAX_EMBEDDER_WAIT
    original_soft_wait = governed_embeddings._SOFT_EMBEDDER_WAIT

    monkeypatch.setenv("EMBEDDER_INIT_TIMEOUT_CAP", "0.5")
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_INIT_TIMEOUT", 5.0)
    monkeypatch.setattr(governed_embeddings, "_MAX_EMBEDDER_WAIT", 5.0)
    monkeypatch.setattr(governed_embeddings, "_SOFT_EMBEDDER_WAIT", 1.0)

    with caplog.at_level(logging.WARNING):
        governed_embeddings._apply_timeout_caps()

    assert governed_embeddings._EMBEDDER_INIT_TIMEOUT == 0.5
    assert governed_embeddings._MAX_EMBEDDER_WAIT == 0.5
    assert governed_embeddings._SOFT_EMBEDDER_WAIT == 0.5
    messages = {rec.getMessage() for rec in caplog.records}
    assert any("EMBEDDER_INIT_TIMEOUT" in msg for msg in messages)
    assert any("EMBEDDER_INIT_MAX_WAIT" in msg for msg in messages)

    governed_embeddings._EMBEDDER_INIT_TIMEOUT = original_timeout
    governed_embeddings._MAX_EMBEDDER_WAIT = original_max_wait
    governed_embeddings._SOFT_EMBEDDER_WAIT = original_soft_wait
    monkeypatch.delenv("EMBEDDER_INIT_TIMEOUT_CAP", raising=False)
    governed_embeddings._apply_timeout_caps()


def test_get_embedder_prefers_cached_snapshot(monkeypatch, tmp_path):
    cache_root = (
        tmp_path
        / "hub"
        / "models--sentence-transformers--all-MiniLM-L6-v2"
    )
    snapshot_dir = cache_root / "snapshots" / "abc123"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "config.json").write_text("{}")

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setattr(governed_embeddings, "_cleanup_hf_locks", lambda *_: None)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER", None)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_LOCK", None)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_INIT_THREAD", None)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_INIT_EVENT", threading.Event())

    recorded = {}

    class DummySentenceTransformer:
        def __init__(self, path: str, **kwargs: object) -> None:
            recorded["path"] = path
            recorded["kwargs"] = kwargs

    monkeypatch.setattr(
        governed_embeddings,
        "SentenceTransformer",
        DummySentenceTransformer,
    )

    embedder = governed_embeddings.get_embedder()
    assert isinstance(embedder, DummySentenceTransformer)
    assert recorded["path"] == str(snapshot_dir)
    # ``cache_folder`` should not be passed when we load directly from a snapshot.
    assert recorded["kwargs"].get("cache_folder") is None


def test_bundled_fallback_uses_stub_when_archive_missing(monkeypatch):
    monkeypatch.setattr(governed_embeddings, "_BUNDLED_EMBEDDER", None)
    monkeypatch.setattr(governed_embeddings, "_STUB_FALLBACK_USED", False)
    monkeypatch.setattr(governed_embeddings, "_bundled_model_archive", lambda: None)
    monkeypatch.setattr(governed_embeddings, "_prepare_bundled_model_dir", lambda: None)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER", None)
    event = threading.Event()
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_INIT_EVENT", event)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_THREAD_LOCK", threading.RLock())

    assert governed_embeddings._activate_bundled_fallback("timeout")
    embedder = governed_embeddings._EMBEDDER
    assert embedder is governed_embeddings._BUNDLED_EMBEDDER
    vectors = embedder.encode(["hello"], convert_to_numpy=False)
    assert isinstance(vectors, list)
    assert vectors[0] == [0.0] * governed_embeddings._STUB_EMBEDDER_DIMENSION
    assert governed_embeddings._EMBEDDER_INIT_EVENT.is_set()
