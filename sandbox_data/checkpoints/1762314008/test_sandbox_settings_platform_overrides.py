from __future__ import annotations

from unittest import mock

import sandbox_settings_pydantic as sp
import sandbox_settings_fallback as sf


def test_pydantic_windows_platform_suppresses_linux_only_dependencies():
    def fake_override(required_python, required_system):
        removed_python = []
        removed_system = []
        if "pyroute2" in required_python:
            required_python.remove("pyroute2")
            removed_python.append("pyroute2")
        if "qemu-system-x86_64" in required_system:
            required_system.remove("qemu-system-x86_64")
            removed_system.append("qemu-system-x86_64")
        return {key: vals for key, vals in {"python": removed_python, "system": removed_system}.items() if vals}

    with mock.patch.object(sp, "_platform_dependency_overrides", side_effect=fake_override):
        settings = sp.SandboxSettings()

    assert "pyroute2" not in settings.required_python_packages
    assert "qemu-system-x86_64" not in settings.required_system_tools
    suppressed = settings.suppressed_dependencies
    assert "pyroute2" in suppressed.get("python", [])
    assert "qemu-system-x86_64" in suppressed.get("system", [])


def test_fallback_windows_platform_suppresses_linux_only_dependencies():
    def fake_override(data):
        required_python = list(data.get("required_python_packages", []))
        required_system = list(data.get("required_system_tools", []))
        removed_python = []
        removed_system = []
        if "pyroute2" in required_python:
            required_python.remove("pyroute2")
            removed_python.append("pyroute2")
        if "qemu-system-x86_64" in required_system:
            required_system.remove("qemu-system-x86_64")
            removed_system.append("qemu-system-x86_64")
        data["required_python_packages"] = required_python
        data["required_system_tools"] = required_system
        return {key: vals for key, vals in {"python": removed_python, "system": removed_system}.items() if vals}

    with mock.patch.object(sf, "_platform_dependency_overrides", side_effect=fake_override):
        settings = sf.SandboxSettings()

    assert "pyroute2" not in settings.required_python_packages
    assert "qemu-system-x86_64" not in settings.required_system_tools
    suppressed = settings.suppressed_dependencies()
    assert "pyroute2" in suppressed.get("python", [])
    assert "qemu-system-x86_64" in suppressed.get("system", [])
