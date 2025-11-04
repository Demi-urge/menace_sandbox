"""Tests for the path utilities defined in ``run_autonomous``."""

from __future__ import annotations

import ast
import os
from pathlib import Path, PureWindowsPath
from typing import Callable

import pytest


def _load_expand_path() -> Callable[[str], Path]:
    """Extract and return the ``_expand_path`` function from the source file."""

    module_path = Path("run_autonomous.py")
    source = module_path.read_text()
    parsed = ast.parse(source)
    func_node = next(
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name == "_expand_path"
    )
    func_source = ast.get_source_segment(source, func_node)
    if func_source is None:  # pragma: no cover - defensive guard
        raise AssertionError("Failed to locate _expand_path source")

    namespace: dict[str, object] = {}
    exec("import os\nimport re\nfrom pathlib import Path", namespace)
    exec(func_source, namespace)
    expand_path = namespace.get("_expand_path")
    assert callable(expand_path)
    return expand_path  # type: ignore[return-value]


@pytest.fixture(scope="module")
def expand_path() -> Callable[[str], Path]:
    return _load_expand_path()


def test_expand_path_expands_percent_tokens(monkeypatch: pytest.MonkeyPatch, expand_path: Callable[[str], Path]) -> None:
    monkeypatch.setenv("USERPROFILE", r"C:\\Users\\Tester")
    result = expand_path(r"%USERPROFILE%\\Sandbox")
    assert str(result) == r"C:\\Users\\Tester\\Sandbox"


def test_expand_path_preserves_escaped_tokens(monkeypatch: pytest.MonkeyPatch, expand_path: Callable[[str], Path]) -> None:
    monkeypatch.setenv("FOO", "Value")
    result = expand_path(r"C:\\%%FOO%%\\%foo%")
    assert str(result) == r"C:\\%FOO%\\Value"


def test_expand_path_expands_backslash_prefixed_token(
    monkeypatch: pytest.MonkeyPatch, expand_path: Callable[[str], Path]
) -> None:
    """Ensure Windows-style ``\\%VAR%`` segments expand correctly."""

    monkeypatch.setenv("TOOLROOT", "Tools")
    result = expand_path(r"C:\\%TOOLROOT%\\bin")
    assert str(result) == r"C:\\Tools\\bin"


def test_expand_path_expands_nested_tokens(
    monkeypatch: pytest.MonkeyPatch, expand_path: Callable[[str], Path]
) -> None:
    monkeypatch.setenv("INNER", r"C:\\Lib")
    monkeypatch.setenv("OUTER", r"%INNER%\\site-packages")
    result = expand_path(r"%OUTER%\\menace")
    assert str(result) == r"C:\\Lib\\site-packages\\menace"


def test_expand_path_expands_windows_home_with_backslash(
    monkeypatch: pytest.MonkeyPatch, expand_path: Callable[[str], Path]
) -> None:
    monkeypatch.setenv("HOME", r"C:\\Users\\Alice")
    monkeypatch.delenv("USERPROFILE", raising=False)
    result = expand_path(r"~\\Desktop\\Menace")
    assert PureWindowsPath(os.fspath(result)) == PureWindowsPath(
        r"C:\\Users\\Alice\\Desktop\\Menace"
    )


def test_expand_path_normalises_posix_backslash_home(
    monkeypatch: pytest.MonkeyPatch, expand_path: Callable[[str], Path]
) -> None:
    r"""Ensure ``~\`` inputs collapse to a clean path on POSIX hosts."""

    monkeypatch.setenv("HOME", "/home/tester")
    monkeypatch.delenv("USERPROFILE", raising=False)
    result = expand_path(r"~\\Documents")
    assert result == Path("/home/tester/Documents")


def test_expand_path_prefers_embedded_windows_drive(
    monkeypatch: pytest.MonkeyPatch, expand_path: Callable[[str], Path]
) -> None:
    """Windows drive letters take precedence over the POSIX home prefix."""

    monkeypatch.setenv("HOME", "/home/tester")
    monkeypatch.setenv("USERPROFILE", r"C:\\Users\\Example")
    result = expand_path(r"~\\%USERPROFILE%\\data")
    assert str(result) == r"C:\\Users\\Example\\data"


def test_expand_path_handles_unc_paths(
    monkeypatch: pytest.MonkeyPatch, expand_path: Callable[[str], Path]
) -> None:
    """Ensure UNC style paths survive the home directory substitution."""

    monkeypatch.delenv("USERPROFILE", raising=False)
    result = expand_path(r"\\\\server\\share\\folder")
    assert str(result) == r"\\\\server\\share\\folder"
