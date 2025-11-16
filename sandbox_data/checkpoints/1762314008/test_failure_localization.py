import importlib.util
import traceback

from failure_localization import extract_target_region


def test_extract_target_region_from_string(tmp_path):
    src = (
        "def inner():\n"
        "    x = 1\n"
        "    y = 2\n\n"
        "def outer():\n"
        "    inner()\n"
    )
    path = tmp_path / "m.py"  # path-ignore
    path.write_text(src)

    trace = (
        f'File "{path}", line 5, in outer\n'
        f'File "{path}", line 2, in inner\n'
    )

    region = extract_target_region(trace)
    assert region is not None
    assert region.filename == str(path)
    assert region.function == "inner"
    assert region.start_line == 1
    assert region.end_line == 3


def test_extract_target_region(tmp_path):
    source = (
        "def inner():\n"
        "    raise RuntimeError('boom')\n\n"
        "def outer():\n"
        "    inner()\n"
    )
    path = tmp_path / "mod.py"  # path-ignore
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
    assert region.filename == str(path)
    assert region.function == "inner"
    assert region.start_line == 1
    assert region.end_line == 2
