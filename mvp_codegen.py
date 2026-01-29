"""Single-pass code generation with strict safety constraints."""

from __future__ import annotations

import ast
from typing import Any


def run_generation(task: dict[str, object]) -> str:
    """Generate a safe Python script from a task payload with strict safeguards."""
    fallback_script = 'print("internal error: code generation failed")'

    if not isinstance(task, dict):
        return fallback_script

    objective_value = task.get("objective")
    if not isinstance(objective_value, str):
        return fallback_script
    objective = objective_value.strip()
    if not objective:
        return fallback_script

    constraints_value = task.get("constraints")
    constraints: list[str] = []
    if isinstance(constraints_value, list):
        for item in constraints_value:
            if isinstance(item, str):
                text = item.strip()
            else:
                try:
                    text = str(item).strip()
                except Exception:
                    continue
            if text:
                constraints.append(text)
    elif constraints_value is not None:
        try:
            text = str(constraints_value).strip()
        except Exception:
            text = ""
        if text:
            constraints.append(text)

    constraints_section = "\n".join(f"- {item}" for item in constraints) if constraints else "- none"

    safety_prompt = (
        "System safety rules (mandatory):\n"
        "- Single-pass, non-recursive output only.\n"
        "- Do not use or suggest dangerous imports (os, sys, subprocess, socket, pathlib, shutil, "
        "requests, http, urllib, openai, etc.).\n"
        "- No system calls, subprocess usage, filesystem access, network access, or shell commands.\n"
        "- Do not use open, eval, exec, compile, __import__, or input.\n"
        "Return only runnable Python code.\n"
    )

    prompt_text = (
        f"{safety_prompt}\n"
        f"Objective:\n{objective}\n\n"
        f"Constraints:\n{constraints_section}\n"
    )

    wrapper = task.get("model_wrapper")
    if wrapper is None:
        return fallback_script

    try:
        raw_output: Any = wrapper(prompt_text)
    except Exception:
        return fallback_script

    if isinstance(raw_output, bytes):
        try:
            output_text = raw_output.decode("utf-8")
        except UnicodeDecodeError:
            return fallback_script
    else:
        output_text = str(raw_output)

    output_text = output_text.replace("\x00", "")
    output_text = output_text.replace("\r\n", "\n").replace("\r", "\n").strip()

    if output_text.startswith("```"):
        lines = output_text.split("\n")
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        output_text = "\n".join(lines).strip()

    if not output_text:
        return fallback_script

    max_length = 4000
    if len(output_text) > max_length:
        cut = output_text.rfind("\n", 0, max_length + 1)
        if cut == -1:
            output_text = output_text[:max_length].rstrip()
        else:
            output_text = output_text[:cut].rstrip()

    if not output_text:
        return fallback_script

    if len(output_text) > max_length:
        return fallback_script

    try:
        parsed = ast.parse(output_text)
    except Exception:
        return fallback_script

    banned_modules = {
        "os",
        "sys",
        "subprocess",
        "socket",
        "pathlib",
        "shutil",
        "requests",
        "http",
        "urllib",
        "openai",
    }
    banned_builtins = {"open", "eval", "exec", "compile", "__import__", "input"}
    banned_base_names = {"builtins", "__builtins__"}

    for node in ast.walk(parsed):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                base_name = alias.name.split(".")[0]
                if base_name in banned_modules:
                    return fallback_script
        if isinstance(node, ast.Name) and node.id in banned_builtins:
            return fallback_script
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                if node.value.id in banned_modules:
                    return fallback_script
                if node.value.id in banned_base_names and node.attr in banned_builtins:
                    return fallback_script

    return output_text
