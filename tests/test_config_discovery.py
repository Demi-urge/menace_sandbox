import os
from pathlib import Path
from menace import config_discovery as cd


def test_discover(tmp_path, monkeypatch):
    tf = tmp_path / "terraform"
    tf.mkdir()
    (tf / "main.tf").write_text("module \"x\" {}")
    hosts_file = tmp_path / "hosts"
    hosts_file.write_text("h1\nh2\n")
    monkeypatch.chdir(tmp_path)
    for var in ["TERRAFORM_DIR", "CLUSTER_HOSTS", "REMOTE_HOSTS"]:
        monkeypatch.delenv(var, raising=False)
    cd.ConfigDiscovery().discover()
    assert os.environ.get("TERRAFORM_DIR") == str(tf)
    assert os.environ.get("CLUSTER_HOSTS") == "h1,h2"
    assert os.environ.get("REMOTE_HOSTS") == "h1,h2"


def test_bootstrap_uses_discover(monkeypatch):
    called = []
    def fake_discover(self):
        called.append(True)
    monkeypatch.setattr(cd.ConfigDiscovery, "discover", fake_discover)
    monkeypatch.delenv("TERRAFORM_DIR", raising=False)
    import types, sys
    sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
    sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
    primitives_mod = sys.modules.setdefault(
        "cryptography.hazmat.primitives", types.ModuleType("primitives")
    )
    asym_mod = sys.modules.setdefault(
        "cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric")
    )
    ed_mod = types.ModuleType("ed25519")
    class _DummyKey:
        def __init__(self, *a, **k):
            pass
        @staticmethod
        def from_private_bytes(b):
            return _DummyKey()
        def public_key(self):
            return _DummyKey()
        def public_bytes(self, *a, **k):
            return b""
        def sign(self, m):
            return b""

    ed_mod.Ed25519PrivateKey = _DummyKey
    ed_mod.Ed25519PublicKey = _DummyKey
    sys.modules.setdefault(
        "cryptography.hazmat.primitives.asymmetric.ed25519", ed_mod
    )
    ser_mod = types.ModuleType("serialization")
    class _Encoding:
        Raw = object()
    class _PublicFormat:
        Raw = object()
    ser_mod.Encoding = _Encoding
    ser_mod.PublicFormat = _PublicFormat
    sys.modules.setdefault(
        "cryptography.hazmat.primitives.serialization", ser_mod
    )
    primitives_mod.asymmetric = asym_mod
    primitives_mod.serialization = ser_mod
    asym_mod.ed25519 = ed_mod
    sys.modules["cryptography.hazmat"].primitives = primitives_mod
    import menace.environment_bootstrap as eb
    eb.EnvironmentBootstrapper(tf_dir=None)
    assert called


def test_cloud_detection_from_env(monkeypatch):
    monkeypatch.delenv("CLOUD_PROVIDER", raising=False)
    monkeypatch.delenv("TOTAL_CPUS", raising=False)
    monkeypatch.delenv("TOTAL_MEMORY_GB", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    monkeypatch.setattr(cd.os, "cpu_count", lambda: 4)

    class Mem:
        total = 8 * 1024**3

    monkeypatch.setattr(cd.psutil, "virtual_memory", lambda: Mem)
    cd.ConfigDiscovery().discover()
    assert os.environ.get("CLOUD_PROVIDER") == "GCP"
    assert os.environ.get("TOTAL_CPUS") == "4"
    assert os.environ.get("TOTAL_MEMORY_GB") == "8.0"


def test_cloud_detection_metadata(monkeypatch):
    monkeypatch.delenv("CLOUD_PROVIDER", raising=False)
    monkeypatch.delenv("TOTAL_CPUS", raising=False)
    monkeypatch.delenv("TOTAL_MEMORY_GB", raising=False)

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

    def fake_open(req, timeout=0):
        if "169.254.169.254" in req.full_url:
            return Resp()
        raise OSError

    monkeypatch.setattr(cd.urllib.request, "urlopen", fake_open)
    monkeypatch.setattr(cd.os, "cpu_count", lambda: 8)

    class Mem:
        total = 16 * 1024**3

    monkeypatch.setattr(cd.psutil, "virtual_memory", lambda: Mem)
    cd.ConfigDiscovery().discover()
    assert os.environ.get("CLOUD_PROVIDER") == "AWS"
    assert os.environ.get("TOTAL_CPUS") == "8"
    assert os.environ.get("TOTAL_MEMORY_GB") == "16.0"


def test_run_continuous(monkeypatch):
    import time
    import threading

    cd_obj = cd.ConfigDiscovery()
    calls = []

    def fake_disc(self=cd_obj):
        calls.append(True)

    monkeypatch.setattr(cd_obj, "discover", fake_disc)
    stop = threading.Event()
    t = cd_obj.run_continuous(interval=0.01, stop_event=stop)
    time.sleep(0.03)
    stop.set()
    t.join(1)
    assert calls


def test_huggingface_token_from_cache(tmp_path, monkeypatch):
    home = tmp_path / "home"
    token_dir = home / ".huggingface"
    token_dir.mkdir(parents=True)
    (token_dir / "token").write_text("abc123\n")
    monkeypatch.setenv("HOME", str(home))
    for key in [
        "HUGGINGFACE_TOKEN",
        "HUGGINGFACE_API_TOKEN",
        "HF_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
    ]:
        monkeypatch.delenv(key, raising=False)
    disc = cd.ConfigDiscovery()
    disc.reload_tokens()
    assert os.environ.get("HUGGINGFACE_TOKEN") == "abc123"
    assert disc.failure_count == 0


def test_missing_huggingface_token_increments_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    for key in [
        "HUGGINGFACE_TOKEN",
        "HUGGINGFACE_API_TOKEN",
        "HF_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
    ]:
        monkeypatch.delenv(key, raising=False)
    disc = cd.ConfigDiscovery()
    disc.reload_tokens()
    assert disc.failure_count == 1


def test_stack_env_defaults_and_overrides(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stack_dir = tmp_path / "stack_state"
    env_path = tmp_path / ".stack_env"
    env_path.write_text("STACK_DATA_DIR=%s\nSTACK_STREAMING=1\n" % stack_dir)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("STACK_DATA_DIR", raising=False)
    monkeypatch.delenv("STACK_STREAMING", raising=False)
    monkeypatch.delenv("STACK_METADATA_DB", raising=False)
    monkeypatch.delenv("STACK_METADATA_PATH", raising=False)
    monkeypatch.delenv("STACK_CACHE_DIR", raising=False)
    monkeypatch.delenv("STACK_VECTOR_PATH", raising=False)

    disc = cd.ConfigDiscovery()
    disc.reload_stack_settings()

    base = str(Path(stack_dir))
    assert os.environ.get("STACK_DATA_DIR") == base
    assert os.environ.get("STACK_STREAMING") == "1"
    assert os.environ.get("STACK_METADATA_DB") == str(Path(base) / "stack_metadata.db")
    assert os.environ.get("STACK_METADATA_PATH") == str(Path(base) / "stack_metadata.db")
    assert os.environ.get("STACK_CACHE_DIR") == str(Path(base) / "cache")
    assert os.environ.get("STACK_VECTOR_PATH") == str(Path(base) / "stack_vectors")


def test_stack_reload_respects_manual_streaming_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env_path = tmp_path / ".stack_env"
    env_path.write_text("STACK_STREAMING=1\n")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("STACK_STREAMING", raising=False)

    disc = cd.ConfigDiscovery()
    disc.reload_stack_settings()
    assert os.environ.get("STACK_STREAMING") == "1"

    os.environ["STACK_STREAMING"] = "0"
    disc.reload_stack_settings()
    assert os.environ.get("STACK_STREAMING") == "0"

    env_path.write_text("STACK_STREAMING=1\n")
    disc.reload_stack_settings()
    assert os.environ.get("STACK_STREAMING") == "0"


def test_stack_reload_updates_data_dir_when_not_overridden(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env_path = tmp_path / ".stack_env"
    env_path.write_text("STACK_DATA_DIR=%s\n" % (tmp_path / "one"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("STACK_DATA_DIR", raising=False)
    disc = cd.ConfigDiscovery()
    disc.reload_stack_settings()
    assert os.environ.get("STACK_DATA_DIR") == str(tmp_path / "one")

    env_path.write_text("STACK_DATA_DIR=%s\n" % (tmp_path / "two"))
    disc.reload_stack_settings()
    assert os.environ.get("STACK_DATA_DIR") == str(tmp_path / "two")

