import sys
import json
import http.client
import types

sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("hazmat"))
sys.modules.setdefault(
    "cryptography.hazmat.primitives", types.ModuleType("primitives")
)
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric", types.ModuleType("asymmetric")
)
sys.modules.setdefault(
    "cryptography.hazmat.primitives.asymmetric.ed25519", types.ModuleType("ed25519")
)
ed = sys.modules["cryptography.hazmat.primitives.asymmetric.ed25519"]
ed.Ed25519PrivateKey = types.SimpleNamespace(generate=lambda: object())
ed.Ed25519PublicKey = object
serialization = types.ModuleType("serialization")
primitives = sys.modules["cryptography.hazmat.primitives"]
primitives.serialization = serialization
sys.modules.setdefault("cryptography.hazmat.primitives.serialization", serialization)

jinja = types.ModuleType("jinja2")
jinja.Template = lambda *a, **k: None
sys.modules.setdefault("jinja2", jinja)
yaml = types.ModuleType("yaml")
sys.modules.setdefault("yaml", yaml)
sys.modules.setdefault("numpy", types.ModuleType("numpy"))

from menace.rollback_manager import RollbackManager


def test_rpc_server(tmp_path):
    mgr = RollbackManager(str(tmp_path / "r.db"))
    mgr.start_rpc_server(port=0)
    port = mgr._server.server_address[1]

    conn = http.client.HTTPConnection("localhost", port)
    body = json.dumps({"patch_id": "p1", "node": "n"})
    conn.request("POST", "/register", body, {"Content-Type": "application/json"})
    conn.getresponse().read()

    patches = mgr.applied_patches()
    assert patches and patches[0].patch_id == "p1"

    conn.request("POST", "/rollback", json.dumps({"patch_id": "p1"}), {"Content-Type": "application/json"})
    conn.getresponse().read()
    assert not mgr.applied_patches()
    mgr.stop_rpc_server()


def test_rpc_server_region(tmp_path):
    mgr = RollbackManager(str(tmp_path / "r.db"))
    mgr.start_rpc_server(port=0)
    port = mgr._server.server_address[1]

    conn = http.client.HTTPConnection("localhost", port)
    body = json.dumps(
        {"patch_id": "p2", "node": "n", "file": "f", "start_line": 1, "end_line": 2}
    )
    conn.request("POST", "/register_region", body, {"Content-Type": "application/json"})
    conn.getresponse().read()

    patches = mgr.applied_region_patches()
    assert patches and patches[0].file == "f"

    conn.request(
        "POST",
        "/rollback_region",
        json.dumps({"file": "f", "start_line": 1, "end_line": 2}),
        {"Content-Type": "application/json"},
    )
    conn.getresponse().read()
    assert not mgr.applied_region_patches()
    mgr.stop_rpc_server()
