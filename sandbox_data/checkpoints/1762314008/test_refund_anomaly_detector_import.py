import pathlib
import sys
import types

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def test_import_module() -> None:
    stub = types.ModuleType("menace_sanity_layer")
    stub.record_billing_event = lambda *a, **k: None
    stub.record_payment_anomaly = lambda *a, **k: None
    sys.modules["menace_sanity_layer"] = stub

    import billing.refund_anomaly_detector  # noqa: F401
