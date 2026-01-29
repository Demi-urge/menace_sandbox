"""Single-pass code generation with strict safety constraints."""

from __future__ import annotations

import ast
import sys



def run_generation(task: dict[str, object]) -> str:
    """Generate a safe Python script from a task payload with strict safeguards.

    The model wrapper is an injected dependency provided via task["model_wrapper"].
    """
    fallback_script = 'print("internal error: code generation failed")'

    if not isinstance(task, dict):
        return fallback_script

    wrapper = task.get("model_wrapper")
    if not callable(wrapper):
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

    timeout_value = task.get("timeout_s") if isinstance(task, dict) else None

    try:
        if timeout_value is not None:
            try:
                raw_output = wrapper(prompt_text, timeout_s=timeout_value)
            except TypeError:
                raw_output = wrapper(prompt_text)
        else:
            raw_output = wrapper(prompt_text)
    except TimeoutError:
        return fallback_script
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

    stdlib_allowlist = getattr(sys, "stdlib_module_names", None)
    if stdlib_allowlist is None:
        stdlib_allowlist = {
            "__future__",
            "_ast",
            "_thread",
            "abc",
            "argparse",
            "array",
            "ast",
            "base64",
            "binascii",
            "bisect",
            "builtins",
            "calendar",
            "collections",
            "cmath",
            "contextlib",
            "copy",
            "csv",
            "dataclasses",
            "datetime",
            "decimal",
            "difflib",
            "dis",
            "enum",
            "errno",
            "fnmatch",
            "fractions",
            "functools",
            "gc",
            "getopt",
            "getpass",
            "gettext",
            "glob",
            "hashlib",
            "heapq",
            "hmac",
            "html",
            "inspect",
            "io",
            "itertools",
            "json",
            "linecache",
            "locale",
            "logging",
            "math",
            "operator",
            "os",
            "pathlib",
            "pickle",
            "platform",
            "pprint",
            "random",
            "re",
            "reprlib",
            "secrets",
            "shlex",
            "shutil",
            "signal",
            "statistics",
            "string",
            "struct",
            "tempfile",
            "subprocess",
            "sys",
            "textwrap",
            "threading",
            "time",
            "traceback",
            "types",
            "typing",
            "uuid",
            "warnings",
            "weakref",
            "asyncio",
            "ftplib",
            "http",
            "imaplib",
            "poplib",
            "selectors",
            "smtplib",
            "socket",
            "ssl",
            "telnetlib",
            "urllib",
        }

    filesystem_modules = {
        "glob",
        "os",
        "pathlib",
        "shutil",
        "tempfile",
    }
    network_modules = {
        "asyncio",
        "ftplib",
        "http",
        "imaplib",
        "poplib",
        "requests",
        "selectors",
        "smtplib",
        "socket",
        "ssl",
        "telnetlib",
        "urllib",
    }
    extra_denied_modules = {
        "ctypes",
        "importlib",
        "subprocess",
        "sys",
        "xmlrpc",
    }
    denied_modules = filesystem_modules | network_modules | extra_denied_modules
    banned_builtins = {"open", "eval", "exec", "compile", "__import__", "input"}
    banned_base_names = {"builtins", "__builtins__"}
    banned_dynamic_call_names = {"__import__", "import_module"}
    banned_module_attrs = {"import_module", "reload"}
    banned_symbol_names = banned_builtins | banned_dynamic_call_names | banned_module_attrs
    banned_reflection_calls = {"getattr", "setattr", "globals", "locals", "vars", "dir"}
    banned_aliases: set[str] = set()
    banned_module_aliases: set[str] = set()

    for node in ast.walk(parsed):
        if isinstance(node, ast.Import):
            for alias in node.names:
                base_name = alias.name.split(".")[0]
                if base_name not in stdlib_allowlist:
                    return fallback_script
                if base_name in denied_modules:
                    return fallback_script
                if alias.asname and base_name in banned_base_names:
                    banned_aliases.add(alias.asname)
                    banned_module_aliases.add(alias.asname)
        if isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            module_base = module_name.split(".")[0]
            if module_base not in stdlib_allowlist:
                return fallback_script
            if module_base in denied_modules:
                return fallback_script
            for alias in node.names:
                if alias.name in banned_symbol_names:
                    return fallback_script
                if alias.asname and alias.asname in banned_symbol_names:
                    return fallback_script
                if module_name in banned_base_names and alias.name in banned_builtins:
                    return fallback_script
                if alias.asname and module_base in banned_base_names:
                    banned_aliases.add(alias.asname)
                    banned_module_aliases.add(alias.asname)
        if isinstance(node, ast.Name) and node.id in banned_builtins:
            return fallback_script
        if isinstance(node, ast.Name) and node.id in banned_aliases:
            return fallback_script
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name):
                if node.value.id in denied_modules:
                    return fallback_script
                if node.value.id in banned_base_names:
                    if node.attr in banned_builtins:
                        return fallback_script
                if node.value.id in banned_module_aliases:
                    return fallback_script
        if isinstance(node, ast.Subscript):
            key_node = node.slice
            if isinstance(key_node, ast.Index):
                key_node = key_node.value
            key = None
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                key = key_node.value
            elif isinstance(key_node, ast.Str):
                key = key_node.s
            if isinstance(node.value, ast.Name) and node.value.id in banned_base_names:
                if key in banned_builtins:
                    return fallback_script
            if isinstance(node.value, ast.Call):
                call = node.value
                if isinstance(call.func, ast.Name) and call.func.id in {"globals", "locals", "vars"}:
                    if key in banned_builtins or key in denied_modules:
                        return fallback_script
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id in banned_symbol_names:
                    return fallback_script
                if func.id in banned_aliases:
                    return fallback_script
                if func.id in banned_reflection_calls:
                    literal = None
                    if func.id in {"getattr", "setattr"} and len(node.args) >= 2:
                        literal_node = node.args[1]
                        if isinstance(literal_node, ast.Index):
                            literal_node = literal_node.value
                        if isinstance(literal_node, ast.Constant) and isinstance(
                            literal_node.value, str
                        ):
                            literal = literal_node.value
                        elif isinstance(literal_node, ast.Str):
                            literal = literal_node.s
                        target = node.args[0]
                        if literal in banned_builtins and isinstance(target, ast.Name):
                            if target.id in banned_base_names or target.id in banned_aliases:
                                return fallback_script
                        if literal in denied_modules and isinstance(target, ast.Name):
                            if target.id in denied_modules or target.id in banned_module_aliases:
                                return fallback_script
                    if func.id == "dir" and node.args:
                        target = node.args[0]
                        if isinstance(target, ast.Name):
                            if target.id in banned_base_names or target.id in denied_modules:
                                return fallback_script
            if isinstance(func, ast.Attribute):
                if isinstance(func.value, ast.Name):
                    if func.value.id in banned_module_aliases and func.attr in banned_module_attrs:
                        return fallback_script

    return output_text
