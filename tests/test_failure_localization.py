import importlib.util
import traceback

from failure_localization import extract_target_region


def test_extract_target_region(tmp_path):
    source = (
        "def inner():\n"
        "    raise RuntimeError('boom')\n\n"
        "def outer():\n"
        "    inner()\n"
    )
    path = tmp_path / "mod.py"
    path.write_text(source)

    spec = importlib.util.spec_from_file_location("mod", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    try:
        mod.outer()
    except Exception:
        trace = traceback.format_exc()
    region = extract_target_region(trace)
    assert region is not None
    assert region.path == str(path)
    assert region.func_name == "inner"
    assert region.start_line == 1
    assert region.end_line == 2
