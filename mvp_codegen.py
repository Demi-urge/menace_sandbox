"""Single-pass code generation with safety constraints."""

from __future__ import annotations

from typing import Any

import local_model_wrapper


def run_generation(task: dict[str, object]) -> str:
    """Generate safe Python code from a task payload using a single model call.

    The task must include an ``objective`` string and may include optional
    ``constraints`` (string or list). The function is deterministic, defensive,
    and side-effect-free: it validates inputs, builds a strict safety system
    prompt, invokes the model wrapper exactly once, and sanitizes the generated
    text by replacing any lines that contain blacklisted modules/keywords,
    truncating the output to a fixed size limit, and ensuring a non-empty return
    value. On any failure, it returns a minimal placeholder script that prints
    an internal error message.
    """
    fallback_script = (
        "\"\"\"Internal error: code generation failed.\"\"\"\n"
        "def main() -> None:\n"
        "    print('Internal error: code generation failed.')\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )

    if not isinstance(task, dict):
        return fallback_script

    objective_value = task.get("objective")
    if not isinstance(objective_value, str):
        return fallback_script
    objective = objective_value.strip()
    if not objective:
        return fallback_script

    constraints_value = task.get("constraints")
    constraints_list: list[str] = []
    if isinstance(constraints_value, str):
        constraints_list = [line.strip() for line in constraints_value.splitlines() if line.strip()]
    elif isinstance(constraints_value, list):
        constraints_list = [item.strip() for item in constraints_value if isinstance(item, str) and item.strip()]
    elif constraints_value is not None:
        constraints_list = [str(constraints_value).strip()] if str(constraints_value).strip() else []

    banned_imports = (
        "os",
        "sys",
        "subprocess",
        "socket",
        "pathlib",
        "shlex",
        "shutil",
        "tempfile",
        "ctypes",
        "importlib",
        "inspect",
        "signal",
        "asyncio",
        "multiprocessing",
        "threading",
        "resource",
        "builtins",
        "platform",
        "selectors",
        "ssl",
        "http",
        "urllib",
        "requests",
    )

    safety_prompt = (
        "You must output only plain Python code. "
        "Never import or use dangerous modules or system interfaces. "
        "Forbidden imports include: "
        + ", ".join(banned_imports)
        + ". "
        "Do not access the filesystem, environment variables, or network. "
        "Do not spawn processes, execute shell/system calls, or use reflection. "
        "Do not use eval/exec/compile, __import__, or dynamic imports. "
        "Only use safe, deterministic in-memory logic." 
    )

    constraints_section = "\n".join(f"- {item}" for item in constraints_list) if constraints_list else "- None provided."
    prompt_text = (
        f"Objective:\n{objective}\n\n"
        f"Constraints:\n{constraints_section}\n\n"
        "Respond with only valid Python code."
    )

    model = task.get("model")
    tokenizer = task.get("tokenizer")
    if model is None or tokenizer is None:
        return fallback_script

    output_text = ""
    try:
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
    output_text = output_text.replace("\r\n", "\n").replace("\r", "\n")
    output_text = "".join(char for char in output_text if char.isprintable() or char in {"\n", "\t"})

    blacklist_terms = (
        "import os",
        "from os",
        "os.",
        "import subprocess",
        "from subprocess",
        "subprocess.",
        "import sys",
        "from sys",
        "sys.",
        "import pathlib",
        "from pathlib",
        "pathlib.",
        "import shutil",
        "from shutil",
        "shutil.",
        "import socket",
        "from socket",
        "socket.",
        "import requests",
        "from requests",
        "requests.",
        "http",
        "open(",
        "eval(",
        "exec(",
        "__import__(",
    )

    cleaned_lines: list[str] = []
    for line in output_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        lowered = stripped.lower()
        if any(term in lowered for term in blacklist_terms):
            cleaned_lines.append(f"{line[: len(line) - len(line.lstrip())]}pass")
            continue
        cleaned_lines.append(line)

    max_lines = 200
    trimmed_lines = cleaned_lines[:max_lines]
    cleaned = "\n".join(trimmed_lines).strip()
    if not cleaned or not any(char.isalnum() for char in cleaned):
        return fallback_script

    max_bytes = 4000
    cleaned_bytes = cleaned.encode("utf-8")
    if len(cleaned_bytes) > max_bytes:
        truncated_text = cleaned_bytes[:max_bytes].decode("utf-8", errors="ignore")
        cutoff = truncated_text.rfind("\n")
        cleaned = truncated_text[: cutoff if cutoff != -1 else len(truncated_text)].rstrip()
        if not cleaned:
            return fallback_script

    try:
        compile(cleaned, "<generated>", "exec")
    except Exception:
        return fallback_script

    return cleaned
