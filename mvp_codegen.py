"""Single-pass code generation with strict safety constraints."""

from __future__ import annotations

import ast
from typing import Any

import local_model_wrapper


def run_generation(task: dict) -> str:
    """Generate safe Python code from a task payload using one model call."""
    fallback_script = 'print("internal error")'

    if not isinstance(task, dict):
        return fallback_script

    objective_value = task.get("objective") if task is not None else None
    if not isinstance(objective_value, str):
        return fallback_script
    objective = objective_value.strip()
    if not objective:
        return fallback_script

    constraints_value = task.get("constraints")
    if isinstance(constraints_value, str):
        constraints = [line.strip() for line in constraints_value.splitlines() if line.strip()]
    elif isinstance(constraints_value, list):
        constraints = [item.strip() for item in constraints_value if isinstance(item, str) and item.strip()]
    else:
        constraints = []

    constraints_section = "\n".join(f"- {item}" for item in constraints) if constraints else "- none"
    prompt_text = f"Objective:\n{objective}\n\nConstraints:\n{constraints_section}\n"

    safety_prompt = (
        "Output only Python code. "
        "Never import or use: os, sys, subprocess, socket, pathlib, shutil. "
        "Never use open, eval, exec, __import__. "
        "No filesystem, environment, or network access."
    )

    output_text = ""
    try:
        model = task.get("model")
        tokenizer = task.get("tokenizer")
        if model is None or tokenizer is None:
            return fallback_script
        wrapper = local_model_wrapper.LocalModelWrapper(model, tokenizer)
        prompt_obj = local_model_wrapper.Prompt(
            user=prompt_text,
            system=safety_prompt,
            metadata={"origin": "mvp_codegen"},
        )
        raw_output: Any = wrapper.generate(
            prompt_obj,
            context_builder=None,
            max_new_tokens=256,
            do_sample=False,
        )
        if isinstance(raw_output, list):
            output_text = str(raw_output[0]) if raw_output else ""
        else:
            output_text = str(raw_output)
    except Exception:
        return fallback_script

    output_text = str(output_text)
    output_text = output_text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")
    output_text = output_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    max_bytes = 4000
    if len(output_text.encode("utf-8")) > max_bytes:
        output_text = output_text.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore").strip()

    if not output_text:
        return fallback_script

    lowered = output_text.lower()
    blacklist_terms = (
        "import os",
        "from os",
        "os.",
        "import sys",
        "from sys",
        "sys.",
        "import subprocess",
        "from subprocess",
        "subprocess.",
        "import socket",
        "from socket",
        "socket.",
        "import pathlib",
        "from pathlib",
        "pathlib.",
        "import shutil",
        "from shutil",
        "shutil.",
        "open(",
        "eval(",
        "exec(",
        "__import__(",
    )
    if any(term in lowered for term in blacklist_terms):
        return fallback_script

    try:
        parsed = ast.parse(output_text)
    except Exception:
        return fallback_script

    banned_names = {"open", "eval", "exec", "__import__"}
    banned_modules = {"os", "sys", "subprocess", "socket", "pathlib", "shutil"}
    for node in ast.walk(parsed):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                name = alias.name.split(".")[0]
                if name in banned_modules:
                    return fallback_script
        if isinstance(node, ast.Name) and node.id in banned_names:
            return fallback_script
        if isinstance(node, ast.Attribute):
            base = node.value
            if isinstance(base, ast.Name) and base.id in banned_modules:
                return fallback_script

    return output_text
