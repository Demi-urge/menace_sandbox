"""Single-pass code generation with safety constraints."""

import local_model_wrapper


def run_generation(task: dict) -> str:
    """Generate Python code from a task payload using a single model call.

    The task dictionary may contain an ``objective`` string and optional
    ``constraints`` list or string. Malformed or missing values are handled
    defensively, and model errors or unsafe outputs fall back to a harmless
    placeholder script. The returned value is always UTF-8 text representing
    Python code.
    """
    fallback_script = (
        '"""Safe fallback script."""\n'
        "def main() -> None:\n"
        "    print('No valid code was generated.')\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    error_script = (
        '"""Generation error script."""\n'
        "def main() -> None:\n"
        "    print('Internal error during generation.')\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )

    if not isinstance(task, dict):
        task = {}

    objective_value = task.get("objective")
    if isinstance(objective_value, str):
        objective = objective_value.strip()
    else:
        objective = ""

    constraints_value = task.get("constraints")
    constraints_list: list[str] = []
    if isinstance(constraints_value, str):
        constraints_list = [line.strip() for line in constraints_value.splitlines() if line.strip()]
    elif isinstance(constraints_value, list):
        constraints_list = [str(item).strip() for item in constraints_value if isinstance(item, str) and str(item).strip()]
    elif constraints_value is not None:
        constraints_list = [str(constraints_value).strip()] if str(constraints_value).strip() else []

    safety_instructions = (
        "You are generating Python code only. "
        "Do NOT import or use dangerous modules or system interfaces, including "
        "os, sys, subprocess, socket, pathlib, shlex, shutil, tempfile, or similar. "
        "Do NOT access the filesystem, spawn processes, execute shell/system calls, "
        "or make any network requests outside the provided wrapper. "
        "Avoid eval/exec, dynamic imports, and any reflection that could escape the sandbox."
    )

    if objective:
        objective_section = f"Objective:\n{objective}\n"
    else:
        objective_section = "Objective:\nProvide a minimal safe Python script.\n"

    if constraints_list:
        constraints_section = "Constraints:\n" + "\n".join(
            f"- {item}" for item in constraints_list
        )
    else:
        constraints_section = "Constraints:\n- None provided."

    prompt_text = (
        f"{objective_section}\n"
        f"{constraints_section}\n\n"
        "Respond with only valid Python code."
    )

    max_chars = 4000
    output_text = ""

    try:
        model = task.get("model")
        tokenizer = task.get("tokenizer")
        if model is None or tokenizer is None:
            raise ValueError("model and tokenizer are required for generation")
        wrapper = local_model_wrapper.LocalModelWrapper(model, tokenizer)
        prompt_obj = local_model_wrapper.Prompt(
            user=prompt_text,
            system=safety_instructions,
            metadata={"origin": "mvp_codegen"},
        )
        raw_output = wrapper.generate(
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
        return error_script

    if not isinstance(output_text, str):
        output_text = str(output_text)
    output_text = output_text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")

    cleaned_lines: list[str] = []
    banned_snippets = [
        "import os",
        "import sys",
        "import subprocess",
        "import socket",
        "import pathlib",
        "import shlex",
        "import shutil",
        "import tempfile",
        "from os",
        "from sys",
        "from subprocess",
        "from socket",
        "from pathlib",
        "from shlex",
        "from shutil",
        "from tempfile",
        "os.system",
        "os.popen",
        "subprocess.",
        "popen(",
        "system(",
        "shell=True",
        "socket.",
        "pathlib.",
        "eval(",
        "exec(",
        "__import__(",
        "open(",
    ]

    for line in output_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            continue
        lowered = stripped.lower()
        if any(bad in lowered for bad in banned_snippets):
            continue
        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines).strip()
    if not cleaned or not any(char.isalnum() for char in cleaned) or len(cleaned) < 20:
        return fallback_script

    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars].rstrip()

    return cleaned
