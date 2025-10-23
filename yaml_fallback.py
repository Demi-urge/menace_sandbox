"""Robust YAML utilities used when the optional :mod:`PyYAML` dependency is missing.

The autonomous sandbox leans heavily on YAML configuration files for ROI
profiles, sandbox settings and telemetry templates.  The production runtime
ships with :mod:`PyYAML` but our execution environment for exercises and
continuous integration purposely omits heavy optional dependencies.  Importing
modules that unconditionally require PyYAML would therefore raise
``ModuleNotFoundError`` and abort the health checks requested by the user.

To guarantee a predictable bootstrap we expose :func:`get_yaml` which mirrors
the interface of :func:`yaml.safe_load`/``safe_dump``.  When PyYAML is
available the real module is returned.  Otherwise we provide a compact yet
fully documented fallback implementation that supports the YAML constructs used
throughout this repository, including:

* Nested mappings and sequences with arbitrary indentation.
* Inline containers (``{}``/``[]``), quoted and unquoted scalars, booleans and
  null values.
* Block scalars using ``|``/``>`` with indentation and chomp modifiers.
* Preservation of Unicode data and round-trippable ``safe_dump`` output.

The fallback is intentionally strict and produces descriptive :class:`YAMLError`
instances when encountering unsupported constructs so issues surface with clear
context instead of silent misconfiguration.
"""

from __future__ import annotations

import ast
import io
import json
import logging
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Iterator
import re
import sys

__all__ = ["get_yaml"]


_LOGGER = logging.getLogger(__name__)


class YAMLError(RuntimeError):
    """Base exception for YAML parsing/serialisation errors in the fallback."""


class YAMLParseError(YAMLError):
    """Raised when the fallback parser encounters malformed YAML input."""


class YAMLSerialisationError(YAMLError):
    """Raised when an object cannot be represented as YAML by the fallback."""


_BOOL_MAP = {
    "true": True,
    "false": False,
    "yes": True,
    "no": False,
    "on": True,
    "off": False,
}

_NULL_TOKENS = {"null", "~", "none", ""}

_BLOCK_SCALAR_RE = re.compile(r"^(?P<style>[|>])(?P<indent>\d+)?(?P<chomp>[+-])?$")


@dataclass
class _Line:
    indent: int
    content: str


def _strip_comments(line: str) -> str:
    """Return *line* without YAML comments, preserving quoted ``#`` characters."""

    if "#" not in line:
        return line.rstrip()

    result: list[str] = []
    in_single = False
    in_double = False
    escape = False
    for char in line:
        if char == "\\" and not escape:
            escape = True
            result.append(char)
            continue

        if char == "'" and not escape and not in_double:
            in_single = not in_single
            result.append(char)
            escape = False
            continue

        if char == '"' and not escape and not in_single:
            in_double = not in_double
            result.append(char)
            escape = False
            continue

        if char == "#" and not in_single and not in_double:
            break

        result.append(char)
        escape = False

    return "".join(result).rstrip()


def _tokenise(text: str) -> list[_Line]:
    """Normalise input into a list of indentation-aware logical lines."""

    lines: list[_Line] = []
    for raw in text.splitlines():
        if raw.strip() == "":
            lines.append(_Line(indent=0, content=""))
            continue
        if "\t" in raw:
            raise YAMLParseError("Tab characters are not allowed in YAML indentation")
        without_comments = _strip_comments(raw)
        if not without_comments.strip():
            lines.append(_Line(indent=0, content=""))
            continue
        indent = len(without_comments) - len(without_comments.lstrip(" "))
        content = without_comments.strip()
        lines.append(_Line(indent=indent, content=content))

    return lines


def _parse_scalar(value: str) -> Any:
    """Convert a YAML scalar token into the corresponding Python object."""

    lowered = value.lower()
    if lowered in _BOOL_MAP:
        return _BOOL_MAP[lowered]
    if lowered in _NULL_TOKENS:
        return None

    if value.startswith(("'", '"')):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError) as exc:  # pragma: no cover - defensive
            raise YAMLParseError(f"Invalid quoted string: {value!r}") from exc

    # Inline JSON collections – try JSON first to keep bool/null semantics.
    if value.startswith(('{', '[')):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass

    # Fallback to Python literal evaluation with YAML boolean/null tokens mapped.
    replacements = {"true": "True", "false": "False", "null": "None", "none": "None", "~": "None"}

    def _replace(match: re.Match[str]) -> str:
        return replacements[match.group(0).lower()]

    candidate = re.sub(r"\b(true|false|null|none|~)\b", _replace, value, flags=re.IGNORECASE)
    try:
        return ast.literal_eval(candidate)
    except (ValueError, SyntaxError):
        pass

    # Numeric detection – allow ints and floats including scientific notation.
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        if re.fullmatch(r"[-+]?\d*\.\d+(e[-+]?\d+)?", value, re.IGNORECASE):
            return float(value)
    except ValueError:  # pragma: no cover - defensive guard
        pass

    return value


def _compose_block_scalar(
    lines: list[_Line],
    *,
    block_indent: int,
    style: str,
    indent_hint: int | None,
    chomp: str,
) -> tuple[str, int]:
    """Compose block scalar content returning ``(value, consumed_lines)``."""

    consumed = 0
    collected: list[str] = []
    base_indent = block_indent + (indent_hint or 2)

    for line in lines:
        if line.content == "" and consumed == 0:
            consumed += 1
            collected.append("")
            continue
        if line.indent <= block_indent:
            break
        if line.content == "":
            consumed += 1
            collected.append("")
            continue
        consumed += 1
        relative = line.indent - base_indent
        if relative < 0:
            relative = 0
        collected.append(" " * relative + line.content)

    text = "\n".join(collected)
    if style == ">":
        text = _fold_block(text)

    if collected:
        if chomp == "+":
            text = text + "\n"
        elif chomp == "":
            text = text + "\n"

    return text, consumed


def _fold_block(text: str) -> str:
    """Implement basic YAML folded block semantics for ``>`` scalars."""

    if not text:
        return ""

    lines = text.split("\n")
    folded: list[str] = []
    empty_run = 0
    for line in lines:
        if not line:
            empty_run += 1
            continue
        if empty_run:
            folded.extend([""] * empty_run)
            empty_run = 0
        if folded and folded[-1] != "":
            folded[-1] += " " + line.strip()
        else:
            folded.append(line.strip())
    if empty_run:
        folded.extend([""] * empty_run)
    return "\n".join(folded)


def _parse_inline_document(snippet: str) -> Any:
    inline_lines = _tokenise(snippet)
    inline_lines = [line for line in inline_lines if line.content != "" or line.indent != 0]
    if not inline_lines:
        return None
    min_indent = min((line.indent for line in inline_lines if line.content != ""), default=0)
    if min_indent:
        inline_lines = [
            _Line(indent=max(line.indent - min_indent, 0) if line.content != "" else 0, content=line.content)
            for line in inline_lines
        ]
    value, consumed = _parse_block(inline_lines, 0, inline_lines[0].indent)
    if consumed != len(inline_lines):
        raise YAMLParseError("Inline YAML snippet could not be fully parsed")
    return value


def _looks_like_inline_mapping(value: str) -> bool:
    if value.startswith(("'", '"')):
        return False
    return ": " in value or value.endswith(":")


def _parse_block(lines: list[_Line], start: int, indent: int) -> tuple[Any, int]:
    """Recursively parse a block starting at *start* with *indent* spaces."""

    if start >= len(lines):
        return None, start

    container: list[tuple[str, Any]] | list[Any] | None = None
    index = start

    while index < len(lines):
        line = lines[index]
        if line.content == "":
            index += 1
            continue
        if line.indent < indent:
            break
        if line.indent > indent:
            raise YAMLParseError(
                f"Unexpected indentation increase at line {index + 1}: {line.content!r}"
            )

        if line.content.startswith("- ") or line.content == "-":
            if container is None:
                container = []
            elif isinstance(container, list) and container and isinstance(container[0], tuple):
                raise YAMLParseError("Cannot mix mapping and sequence entries at the same level")
            value_str = line.content[1:].strip()
            index += 1
            if value_str == "":
                child, index = _parse_block(lines, index, indent + 2)
                container.append(child)  # type: ignore[arg-type]
            else:
                if match := _BLOCK_SCALAR_RE.match(value_str):
                    block, consumed = _compose_block_scalar(
                        lines[index:],
                        block_indent=indent,
                        style=match.group("style"),
                        indent_hint=int(match.group("indent")) if match.group("indent") else None,
                        chomp=match.group("chomp") or "",
                    )
                    index += consumed
                    container.append(block)  # type: ignore[arg-type]
                else:
                    if _looks_like_inline_mapping(value_str):
                        scalar = _parse_inline_document(value_str)
                    else:
                        scalar = _parse_scalar(value_str)
                    # Attach nested structure if the next line is indented further.
                    if index < len(lines) and lines[index].indent > indent:
                        child, index = _parse_block(lines, index, indent + 2)
                        if isinstance(scalar, dict) and isinstance(child, dict):
                            scalar.update(child)
                            container.append(scalar)  # type: ignore[arg-type]
                        elif isinstance(child, list) and isinstance(scalar, list):
                            scalar.extend(child)
                            container.append(scalar)  # type: ignore[arg-type]
                        else:
                            container.append(child)  # type: ignore[arg-type]
                    else:
                        container.append(scalar)  # type: ignore[arg-type]
        else:
            if container is None:
                container = []  # type: ignore[assignment]
            elif isinstance(container, list) and not container:
                pass
            elif isinstance(container, list) and container and not isinstance(container[0], tuple):
                raise YAMLParseError("Cannot mix sequence and mapping entries at the same level")

            if ":" not in line.content:
                raise YAMLParseError(f"Expected ':' in mapping entry at line {index + 1}")

            key, remainder = line.content.split(":", 1)
            key = key.strip()
            value_str = remainder.strip()
            index += 1

            if match := _BLOCK_SCALAR_RE.match(value_str):
                block, consumed = _compose_block_scalar(
                    lines[index:],
                    block_indent=indent,
                    style=match.group("style"),
                    indent_hint=int(match.group("indent")) if match.group("indent") else None,
                    chomp=match.group("chomp") or "",
                )
                index += consumed
                container.append((key, block))  # type: ignore[arg-type]
                continue

            if value_str == "":
                if index < len(lines) and lines[index].indent > indent:
                    child, index = _parse_block(lines, index, indent + 2)
                    container.append((key, child))  # type: ignore[arg-type]
                else:
                    container.append((key, None))  # type: ignore[arg-type]
            else:
                if _looks_like_inline_mapping(value_str):
                    scalar = _parse_inline_document(value_str)
                else:
                    scalar = _parse_scalar(value_str)
                if index < len(lines) and lines[index].indent > indent:
                    child, index = _parse_block(lines, index, indent + 2)
                    if isinstance(scalar, dict) and isinstance(child, dict):
                        scalar.update(child)
                        container.append((key, scalar))  # type: ignore[arg-type]
                    elif isinstance(scalar, list) and isinstance(child, list):
                        scalar.extend(child)
                        container.append((key, scalar))  # type: ignore[arg-type]
                    else:
                        container.append((key, child))  # type: ignore[arg-type]
                else:
                    container.append((key, scalar))  # type: ignore[arg-type]

    if container is None:
        return None, index

    if container and isinstance(container[0], tuple):
        mapping: dict[str, Any] = {}
        for key, value in container:  # type: ignore[list-item]
            if key in mapping:
                raise YAMLParseError(f"Duplicate key '{key}' in mapping")
            mapping[key] = value
        return mapping, index

    return container, index  # type: ignore[return-value]


def _safe_load(stream: io.TextIOBase | str | bytes) -> Any:
    """Parse YAML content from *stream* or *string* using the fallback parser."""

    if isinstance(stream, (io.TextIOBase, io.BufferedIOBase)):
        text = stream.read()
    elif isinstance(stream, bytes):
        text = stream.decode("utf-8")
    else:
        text = str(stream)

    lines = _tokenise(text)
    if not lines:
        return None

    min_indent = min((line.indent for line in lines if line.content != ""), default=0)
    if min_indent:
        lines = [
            _Line(indent=max(line.indent - min_indent, 0) if line.content != "" else 0, content=line.content)
            for line in lines
        ]

    start_indent = min((line.indent for line in lines if line.content != ""), default=0)
    data, index = _parse_block(lines, 0, start_indent)
    if index != len(lines):  # pragma: no cover - defensive guard
        raise YAMLParseError("Trailing content after parsing YAML document")
    return data


def _serialise_scalar(value: Any) -> str:
    if isinstance(value, str):
        if value == "" or value.strip() != value or "\n" in value or value.startswith(("#", "-", ":")):
            return json.dumps(value)
        return value
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    raise YAMLSerialisationError(f"Unsupported scalar type: {type(value)!r}")


def _serialise(data: Any, *, indent: int, level: int, sort_keys: bool) -> Iterator[str]:
    prefix = " " * (indent * level)
    if isinstance(data, dict):
        items = data.items()
        if sort_keys:
            items = sorted(items, key=lambda item: item[0])  # type: ignore[assignment]
        for key, value in items:  # type: ignore[assignment]
            if not isinstance(key, str):
                raise YAMLSerialisationError("Dictionary keys must be strings")
            if isinstance(value, (dict, list)):
                yield f"{prefix}{key}:\n"
                yield from _serialise(value, indent=indent, level=level + 1, sort_keys=sort_keys)
            else:
                yield f"{prefix}{key}: {_serialise_scalar(value)}\n"
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                yield f"{prefix}-\n"
                yield from _serialise(item, indent=indent, level=level + 1, sort_keys=sort_keys)
            else:
                yield f"{prefix}- {_serialise_scalar(item)}\n"
    else:
        yield f"{prefix}{_serialise_scalar(data)}\n"


def _safe_dump(
    data: Any,
    stream: io.TextIOBase | None = None,
    *,
    indent: int = 2,
    sort_keys: bool = False,
) -> str | None:
    """Serialise *data* to YAML using the fallback implementation."""

    output = "".join(_serialise(data, indent=indent, level=0, sort_keys=sort_keys))
    if stream is not None:
        stream.write(output)
        return None
    return output


def _build_fallback_module(component: str, exc: ModuleNotFoundError) -> ModuleType:
    """Create a drop-in module exposing ``safe_load``/``safe_dump`` fallbacks."""

    module = ModuleType("yaml")
    module.YAMLError = YAMLError  # type: ignore[attr-defined]
    module.safe_load = _safe_load  # type: ignore[attr-defined]
    module.load = _safe_load  # type: ignore[attr-defined]
    module.safe_dump = _safe_dump  # type: ignore[attr-defined]
    module.dump = _safe_dump  # type: ignore[attr-defined]
    module.__all__ = ["safe_load", "safe_dump", "load", "dump", "YAMLError"]
    module.__doc__ = (
        "Fallback YAML implementation used because PyYAML is unavailable for "
        f"{component}. Original error: {exc}"
    )
    return module


_FALLBACK_CACHE: dict[str, ModuleType] = {}


def _get_cached_fallback(component: str, *, reason: ModuleNotFoundError) -> ModuleType:
    """Return a cached fallback module instance for ``component``."""

    cache_key = f"{component}:{reason.args[0]}"
    module = _FALLBACK_CACHE.get(cache_key)
    if module is None:
        module = _build_fallback_module(component, reason)
        _FALLBACK_CACHE[cache_key] = module
    return module


def get_yaml(component: str, *, warn: bool = True, force_fallback: bool = False):
    """Return a YAML implementation suitable for ``component``.

    Parameters
    ----------
    component:
        Logical component requesting YAML support. Used in warning messages so
        callers can trace which subsystem fell back to the lightweight
        implementation.
    warn:
        When ``True`` (the default) a warning is emitted if PyYAML is
        unavailable for ``component``.
    force_fallback:
        When ``True`` the lightweight in-repo YAML implementation is returned
        even if PyYAML is installed. This is useful when callers want a
        best-effort parser as a secondary option after PyYAML raises an error.
    """

    if force_fallback:
        # ``ModuleNotFoundError`` mirrors the real failure mode, ensuring the
        # fallback module has context for logging/debugging while avoiding
        # sys.modules tampering when PyYAML is present.
        return _get_cached_fallback(
            component,
            reason=ModuleNotFoundError("forced fallback for component"),
        )

    try:  # pragma: no cover - exercised only when PyYAML is installed
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - executed without PyYAML
        if warn:
            _LOGGER.warning(
                "PyYAML unavailable for %s; enabling lightweight fallback.",
                component,
                exc_info=exc,
            )
        stub = _get_cached_fallback(component, reason=exc)
        sys.modules.setdefault("yaml", stub)
        return stub

    return yaml


if "yaml" not in sys.modules:  # pragma: no cover - import side effect
    try:
        import yaml  # type: ignore  # noqa: F401
    except ModuleNotFoundError as exc:  # pragma: no cover - minimal environments
        sys.modules["yaml"] = _build_fallback_module("yaml_fallback", exc)

