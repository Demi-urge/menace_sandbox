import os
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


def test_discover_populates_stack_env(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    for var in [
        "STACK_STREAMING",
        "STACK_HF_TOKEN",
        "STACK_INDEX_PATH",
        "STACK_METADATA_PATH",
        "STACK_CACHE_DIR",
        "STACK_PROGRESS_PATH",
        "HUGGINGFACE_TOKEN",
        "HF_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "HUGGINGFACE_API_TOKEN",
    ]:
        monkeypatch.delenv(var, raising=False)
    caplog.set_level("INFO")
    cd.ConfigDiscovery().discover()
    env_file = tmp_path / ".env.auto"
    assert env_file.exists()
    text = env_file.read_text()
    assert "STACK_STREAMING=0" in text
    assert f"STACK_HF_TOKEN={cd._HF_PLACEHOLDER}" in text
    assert "STACK_INDEX_PATH=" in text
    assert "STACK_METADATA_PATH=" in text
    assert "STACK_CACHE_DIR=" in text
    assert "STACK_PROGRESS_PATH=" in text
    assert f"HUGGINGFACE_TOKEN={cd._HF_PLACEHOLDER}" in text
    assert f"HF_TOKEN={cd._HF_PLACEHOLDER}" in text
    assert os.environ.get("STACK_STREAMING") == "0"
    assert os.environ.get("STACK_HF_TOKEN") == cd._HF_PLACEHOLDER
    assert os.environ.get("HUGGINGFACE_TOKEN") == cd._HF_PLACEHOLDER
    assert os.environ.get("HF_TOKEN") == cd._HF_PLACEHOLDER
    assert os.environ.get("STACK_CACHE_DIR") == ""
    assert os.environ.get("STACK_PROGRESS_PATH") == ""
    assert any("missing Hugging Face credentials" in rec.message for rec in caplog.records)
    assert any("Stack processing disabled" in rec.message for rec in caplog.records)


def test_discover_persists_existing_token(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for var in [
        "STACK_STREAMING",
        "STACK_HF_TOKEN",
        "STACK_INDEX_PATH",
        "STACK_METADATA_PATH",
        "STACK_CACHE_DIR",
        "STACK_PROGRESS_PATH",
        "HUGGINGFACE_TOKEN",
        "HF_TOKEN",
    ]:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HF_TOKEN", "abc123")
    cd.ConfigDiscovery().discover()
    env_file = tmp_path / ".env.auto"
    assert "HUGGINGFACE_TOKEN=abc123" in env_file.read_text()
    assert os.environ.get("HUGGINGFACE_TOKEN") == "abc123"
    assert os.environ.get("STACK_HF_TOKEN") == "abc123"
    assert os.environ.get("HF_TOKEN") == "abc123"
    # Running again should not duplicate entries
    cd.ConfigDiscovery().discover()
    lines = [line for line in env_file.read_text().splitlines() if line.startswith("HUGGINGFACE_TOKEN=")]
    assert len(lines) == 1


def test_discover_reads_stack_hints(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "STACK_STREAMING=1",
                "STACK_HF_TOKEN=hf_stack",
                "STACK_INDEX_PATH=/var/lib/stack/index",
                "STACK_METADATA_PATH=/var/lib/stack/meta.sqlite",
            ]
        )
    )
    (tmp_path / "config" / "stack_context.yaml").write_text(
        """
stack_dataset:
  enabled: true
  index_path: /opt/index-dataset
  metadata_path: /opt/meta-dataset.db
context_builder:
  stack:
    index_path: /opt/index-cb
    metadata_path: /opt/meta-cb.db
"""
    )
    (tmp_path / "config" / "self_coding_thresholds.yaml").write_text(
        """
stack:
  context_builder:
    cache_dir: /var/lib/stack/cache
    progress_path: /var/lib/stack/progress.sqlite
"""
    )
    for var in [
        "STACK_STREAMING",
        "STACK_HF_TOKEN",
        "STACK_INDEX_PATH",
        "STACK_METADATA_PATH",
        "STACK_CACHE_DIR",
        "STACK_PROGRESS_PATH",
        "HUGGINGFACE_TOKEN",
        "HF_TOKEN",
    ]:
        monkeypatch.delenv(var, raising=False)

    cd.ConfigDiscovery().discover()

    assert os.environ.get("STACK_STREAMING") == "1"
    assert os.environ.get("STACK_HF_TOKEN") == "hf_stack"
    assert os.environ.get("HUGGINGFACE_TOKEN") == "hf_stack"
    assert os.environ.get("HF_TOKEN") == "hf_stack"
    # values from the environment file take precedence over config hints
    assert os.environ.get("STACK_INDEX_PATH") == "/var/lib/stack/index"
    assert os.environ.get("STACK_METADATA_PATH") == "/var/lib/stack/meta.sqlite"
    assert os.environ.get("STACK_CACHE_DIR") == "/var/lib/stack/cache"
    assert os.environ.get("STACK_PROGRESS_PATH") == "/var/lib/stack/progress.sqlite"


def test_discover_streaming_hint_from_config(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "stack_context.yaml").write_text("stack_dataset:\n  enabled: true\n")
    for var in [
        "STACK_STREAMING",
        "STACK_HF_TOKEN",
        "STACK_INDEX_PATH",
        "STACK_METADATA_PATH",
        "STACK_CACHE_DIR",
        "STACK_PROGRESS_PATH",
        "HUGGINGFACE_TOKEN",
        "HF_TOKEN",
    ]:
        monkeypatch.delenv(var, raising=False)
    caplog.set_level("INFO")
    cd.ConfigDiscovery().discover()
    assert os.environ.get("STACK_STREAMING") == "1"
    assert any("enabled but no Hugging Face token" in rec.message for rec in caplog.records)


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

