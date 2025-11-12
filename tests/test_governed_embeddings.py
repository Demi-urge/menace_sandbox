import inspect
import logging
import os
import sys
import threading
from pathlib import Path
import types

import pytest

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
    monkeypatch.setattr(governed_embeddings, "model", None)
    monkeypatch.setattr(governed_embeddings, "_resolve_local_snapshot", lambda *_: None)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_INIT_EVENT", threading.Event())
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_INIT_THREAD", None)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_TIMEOUT_LOGGED", False)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_TIMEOUT_REACHED", False)

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
    assert embedder.name == "sentence-transformers/all-MiniLM-L6-v2"


def test_get_embedder_exports_token_when_available(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_API_TOKEN", "secret-token")
    monkeypatch.delenv("HUGGINGFACEHUB_API_TOKEN", raising=False)
    monkeypatch.delenv("HF_HUB_TOKEN", raising=False)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER", None)
    monkeypatch.setattr(governed_embeddings, "model", None)
    monkeypatch.setattr(governed_embeddings, "_resolve_local_snapshot", lambda *_: None)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_INIT_EVENT", threading.Event())
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_INIT_THREAD", None)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_TIMEOUT_LOGGED", False)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_TIMEOUT_REACHED", False)

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
    assert embedder.name == "sentence-transformers/all-MiniLM-L6-v2"
    assert os.environ["HUGGINGFACEHUB_API_TOKEN"] == "secret-token"
    assert os.environ["HF_HUB_TOKEN"] == "secret-token"


def test_get_embedder_configures_hf_timeouts(monkeypatch):
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER", None)
    monkeypatch.setattr(governed_embeddings, "model", None)
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
    assert embedder.name == "sentence-transformers/all-MiniLM-L6-v2"
    assert os.environ["HF_HUB_TIMEOUT"] == "12"
    assert os.environ["HF_HUB_READ_TIMEOUT"] == "12"
    assert os.environ["HF_HUB_CONNECTION_TIMEOUT"] == "12"
    assert os.environ["HF_HUB_DOWNLOAD_RETRIES"] == "2"


def test_resolve_local_snapshot_requires_metadata(tmp_path):
    model_cache = tmp_path / "cache"
    snapshots = model_cache / "snapshots"

    valid_snapshot = snapshots / "valid"
    valid_snapshot.mkdir(parents=True)
    (valid_snapshot / "config.json").write_text("{}")
    (valid_snapshot / "modules.json").write_text("{}")

    incomplete_snapshot = snapshots / "incomplete"
    incomplete_snapshot.mkdir()
    (incomplete_snapshot / "config.json").write_text("{}")

    resolved = governed_embeddings._resolve_local_snapshot(model_cache)
    assert resolved == valid_snapshot


def test_resolve_local_snapshot_accepts_sentence_bert_config(tmp_path):
    model_cache = tmp_path / "cache"
    snapshots = model_cache / "snapshots"

    legacy_snapshot = snapshots / "legacy"
    legacy_snapshot.mkdir(parents=True)
    (legacy_snapshot / "config.json").write_text("{}")
    (legacy_snapshot / "modules.json").write_text("{}")

    modern_snapshot = snapshots / "modern"
    modern_snapshot.mkdir()
    (modern_snapshot / "config.json").write_text("{}")
    (modern_snapshot / "sentence_bert_config.json").write_text("{}")

    resolved = governed_embeddings._resolve_local_snapshot(model_cache)
    assert resolved == modern_snapshot


def test_get_embedder_prefers_hub_when_snapshot_incomplete(monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSFORMERS_CACHE", str(tmp_path))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER", None)
    monkeypatch.setattr(governed_embeddings, "model", None)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_INIT_EVENT", threading.Event())
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_INIT_THREAD", None)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_TIMEOUT_LOGGED", False)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_TIMEOUT_REACHED", False)
    monkeypatch.setattr(governed_embeddings, "_HF_TIMEOUT_CONFIGURED", False)
    monkeypatch.setattr(governed_embeddings, "_HF_TIMEOUT_SETTINGS", {})
    monkeypatch.setattr(governed_embeddings, "_cleanup_hf_locks", lambda *a, **k: None)

    model_cache = governed_embeddings._cached_model_path(tmp_path, governed_embeddings._MODEL_NAME)
    incomplete_snapshot = model_cache / "snapshots" / "incomplete"
    incomplete_snapshot.mkdir(parents=True)
    (incomplete_snapshot / "config.json").write_text("{}")
    blobs_dir = model_cache / "blobs"
    blobs_dir.mkdir(parents=True)
    (blobs_dir / "dummy.bin").write_bytes(b"0")
    refs_dir = model_cache / "refs"
    refs_dir.mkdir(parents=True)
    (refs_dir / "main").write_text("deadbeef")

    captured = {}

    class DummySentenceTransformer:
        def __init__(self, name: str, **kwargs):
            captured["name"] = name
            captured["kwargs"] = kwargs

    monkeypatch.setattr(governed_embeddings, "SentenceTransformer", DummySentenceTransformer)

    embedder = governed_embeddings.get_embedder()
    assert isinstance(embedder, DummySentenceTransformer)
    assert captured["name"] == governed_embeddings._MODEL_ID
    assert not captured["kwargs"].get("local_files_only")


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
    (snapshot_dir / "modules.json").write_text("{}")

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setattr(governed_embeddings, "_cleanup_hf_locks", lambda *_: None)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER", None)
    monkeypatch.setattr(governed_embeddings, "model", None)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_LOCK", None)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_INIT_THREAD", None)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_INIT_EVENT", threading.Event())
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_TIMEOUT_LOGGED", False)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_TIMEOUT_REACHED", False)

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


def test_corrupted_snapshot_is_purged_before_retry(monkeypatch, tmp_path):
    cache_root = (
        tmp_path
        / "hub"
        / "models--sentence-transformers--all-MiniLM-L6-v2"
    )
    snapshot_dir = cache_root / "snapshots" / "deadbeef"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "config.json").write_text("{}")
    (snapshot_dir / "modules.json").write_text("{}")

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    monkeypatch.setattr(governed_embeddings, "_cleanup_hf_locks", lambda *_: None)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER", None)
    monkeypatch.setattr(governed_embeddings, "model", None)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_LOCK", None)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_INIT_THREAD", None)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_INIT_EVENT", threading.Event())
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_TIMEOUT_LOGGED", False)
    monkeypatch.setattr(governed_embeddings, "_EMBEDDER_TIMEOUT_REACHED", False)

    purged: dict[str, object] = {}

    def fake_purge(path: Path) -> None:
        purged["path"] = path

    monkeypatch.setattr(governed_embeddings, "_purge_corrupted_snapshot", fake_purge)

    calls: list[tuple[str, dict[str, object]]] = []

    class DummySentenceTransformer:
        def __init__(self, identifier: str, **kwargs: object) -> None:
            calls.append((identifier, dict(kwargs)))
            if len(calls) == 1:
                raise RuntimeError(
                    "Can't copy out of meta tensor; no data! Please use torch.nn.Module.to_empty()"
                )
            self.name = identifier

    monkeypatch.setattr(
        governed_embeddings,
        "SentenceTransformer",
        DummySentenceTransformer,
    )

    embedder = governed_embeddings.get_embedder()
    assert isinstance(embedder, DummySentenceTransformer)
    assert calls[0][0] == str(snapshot_dir)
    assert calls[0][1]["device"] == "meta"
    assert purged.get("path") == snapshot_dir
    assert calls[1][0] == governed_embeddings._MODEL_ID
    assert "local_files_only" not in calls[1][1]
    assert calls[1][1].get("cache_folder") == str(tmp_path)


def test_initialise_sentence_transformer_retries_with_meta(monkeypatch):
    calls: list[tuple[str, dict[str, object]]] = []

    class DummySentenceTransformer:
        def __init__(self, identifier: str, **kwargs: object) -> None:
            calls.append((identifier, dict(kwargs)))
            self.identifier = identifier
            self.device = kwargs.get("device")
            if len(calls) == 1:
                raise RuntimeError(
                    "Cannot copy out of meta tensor; no data! Please use torch.nn.Module.to_empty()"
                )

        def to_empty(self, *, device: object):
            self.device = device
            calls.append(("to_empty", device))
            return self

    dummy_torch = types.SimpleNamespace(device=lambda value: value)

    monkeypatch.setattr(
        governed_embeddings,
        "SentenceTransformer",
        DummySentenceTransformer,
    )
    monkeypatch.setattr(governed_embeddings, "torch", dummy_torch)

    result = governed_embeddings.initialise_sentence_transformer("dummy", device="cpu")
    assert isinstance(result, DummySentenceTransformer)
    assert calls[0][0] == "dummy"
    assert calls[0][1]["device"] == "cpu"
    assert calls[1][0] == "dummy"
    assert calls[1][1]["device"] == "meta"
    assert calls[2] == ("to_empty", "cpu")
    assert result.device == "cpu"


def test_initialise_sentence_transformer_force_meta(monkeypatch):
    calls: list[tuple[str | object, dict[str, object]]] = []

    class DummySentenceTransformer:
        def __init__(self, identifier: str, **kwargs: object) -> None:
            calls.append((identifier, dict(kwargs)))
            self.identifier = identifier
            self.device = kwargs.get("device")

        def to_empty(self, *, device: object):
            calls.append(("to_empty", device))
            self.device = device
            return self

    dummy_torch = types.SimpleNamespace(device=lambda value: value)

    monkeypatch.setattr(
        governed_embeddings,
        "SentenceTransformer",
        DummySentenceTransformer,
    )
    monkeypatch.setattr(governed_embeddings, "torch", dummy_torch)

    result = governed_embeddings.initialise_sentence_transformer(
        "dummy", device="cpu", force_meta_initialisation=True
    )

    assert isinstance(result, DummySentenceTransformer)
    assert calls[0][0] == "dummy"
    assert calls[0][1]["device"] == "meta"
    assert calls[1] == ("to_empty", "cpu")
    assert result.device == "cpu"


def test_materialise_sentence_transformer_raises_when_meta_persists(monkeypatch):
    class DummyTensor:
        def __init__(self, is_meta: bool) -> None:
            self.is_meta = is_meta

    class DummySentenceTransformer:
        def __init__(self) -> None:
            self.device = "meta"

        def to_empty(self, *, device: object):
            raise RuntimeError("to_empty unsupported")

        def to(self, device: object):
            raise RuntimeError("Cannot copy out of meta tensor; no data!")

        def parameters(self, recurse: bool = True):
            yield DummyTensor(True)

        def buffers(self, recurse: bool = True):
            if False:  # pragma: no cover - generator form requires yield
                yield DummyTensor(False)

    dummy_torch = types.SimpleNamespace(
        device=lambda value: value,
        nn=types.SimpleNamespace(Module=type("DummyModule", (), {})),
    )

    monkeypatch.setattr(governed_embeddings, "torch", dummy_torch)

    with pytest.raises(RuntimeError) as excinfo:
        governed_embeddings._materialise_sentence_transformer_device(
            DummySentenceTransformer(), "cpu"
        )

    assert "meta tensors" in str(excinfo.value)


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
