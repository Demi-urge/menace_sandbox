import subprocess
import sys
from pathlib import Path


from dynamic_path_router import resolve_path


def _run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    script = resolve_path("scripts/check_stripe_imports.py")  # path-ignore
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def test_flags_direct_stripe_import(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"  # path-ignore
    bad.write_text("import stripe\n")
    result = _run([str(bad)])
    assert result.returncode == 1
    assert "Direct Stripe imports detected" in result.stdout


def test_flags_sk_live_key(tmp_path: Path) -> None:
    bad = tmp_path / "bad.py"  # path-ignore
    bad.write_text("API_KEY='sk_live_secret'\n")
    result = _run(["--keys", str(bad)])
    assert result.returncode == 1
    assert "Potential Stripe live keys" in result.stdout


def test_clean_file_passes(tmp_path: Path) -> None:
    good = tmp_path / "good.py"  # path-ignore
    good.write_text("def ok():\n    return 1\n")
    result = _run([str(good)])
    assert result.returncode == 0, result.stdout + result.stderr


def test_flags_partially_redacted_key(tmp_path: Path) -> None:
    bad = tmp_path / "bad.log"
    bad.write_text("oops sk_live_1234****5678\n")
    result = _run(["--keys", str(bad)])
    assert result.returncode == 1
    assert "Potential Stripe live keys" in result.stdout

