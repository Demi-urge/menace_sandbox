from __future__ import annotations

"""Helpers for creating and persisting workflow specifications.

This module exposes two small helpers:

``to_spec``
    Convert a list of step dictionaries into the lightweight schema used by
    :class:`task_handoff_bot.WorkflowDB`.  Only a subset of keys is required;
    missing fields fall back to sensible defaults so the resulting mapping can
    be serialised to ``.workflow.json`` files.

``save``
    Persist a workflow specification as ``<name>.workflow.json`` inside a
    configurable output directory and register it in ``WorkflowDB``.  If PyYAML
    is available a companion ``.workflow.yaml`` file is also written.
"""

from dataclasses import fields
import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

try:  # Optional dependency for YAML output
    import yaml  # type: ignore
except Exception:  # pragma: no cover - YAML support is optional
    yaml = None  # type: ignore


def _load_thb() -> tuple[Any, Any]:
    """Return ``WorkflowDB`` and ``WorkflowRecord`` classes."""

    try:
        from .task_handoff_bot import WorkflowDB, WorkflowRecord  # type: ignore
        return WorkflowDB, WorkflowRecord
    except Exception:
        try:  # pragma: no cover - package style import failed
            import task_handoff_bot as thb  # type: ignore
            return thb.WorkflowDB, thb.WorkflowRecord
        except Exception:  # pragma: no cover - fallback to manual loading
            path = Path(__file__).with_name("task_handoff_bot.py")
            spec = importlib.util.spec_from_file_location(
                "menace.task_handoff_bot", path, submodule_search_locations=[str(path.parent)]
            )
            module = importlib.util.module_from_spec(spec)
            assert spec and spec.loader
            sys.modules.setdefault("menace.task_handoff_bot", module)
            spec.loader.exec_module(module)  # type: ignore[attr-defined]
            return module.WorkflowDB, module.WorkflowRecord

# ---------------------------------------------------------------------------
# Configuration
_OUTPUT_DIR = Path(os.environ.get("WORKFLOW_OUTPUT_DIR", "workflows"))

# ---------------------------------------------------------------------------

def _as_list(value: Any) -> list[str]:
    """Helper converting ``value`` to a list of strings."""

    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return [str(value)]


def to_spec(workflow: list[dict]) -> dict:
    """Return a workflow specification mapping for ``WorkflowDB``.

    Parameters
    ----------
    workflow:
        Sequence of step dictionaries.  Common keys include ``name``, ``bot``
        and ``args`` but any additional fields are ignored.

    Returns
    -------
    dict
        Mapping following the :class:`WorkflowRecord` schema.
    """

    names: list[str] = []
    chains: list[str] = []
    args: list[str] = []
    bots: list[str] = []
    enhancements: list[str] = []

    for step in workflow:
        if step.get("name"):
            names.append(str(step["name"]))
        if step.get("action_chain") or step.get("chain") or step.get("actions"):
            chains.append(str(step.get("action_chain") or step.get("chain") or step.get("actions")))
        if step.get("argument_string") or step.get("arguments") or step.get("args"):
            val = step.get("argument_string") or step.get("arguments") or step.get("args")
            if isinstance(val, (list, tuple)):
                args.append(", ".join(map(str, val)))
            else:
                args.append(str(val))
        if step.get("assigned_bot") or step.get("bot"):
            bots.append(str(step.get("assigned_bot") or step.get("bot")))
        if step.get("enhancements") or step.get("enhancement"):
            enhancements.extend(_as_list(step.get("enhancements") or step.get("enhancement")))

    title = names[0] if names else ""
    spec = {
        "workflow": names,
        "action_chains": chains,
        "argument_strings": args,
        "assigned_bots": bots,
        "enhancements": enhancements,
        "title": title,
        "description": "",
        "task_sequence": names[:],
        "tags": [],
        "category": "",
        "type": "",
        "status": "pending",
        "rejection_reason": "",
        "workflow_duration": 0.0,
        "performance_data": "",
        "estimated_profit_per_bot": 0.0,
    }
    return spec


def _record_from_spec(spec: dict) -> WorkflowRecord:
    """Create :class:`WorkflowRecord` from ``spec`` ignoring unknown keys."""

    _, WorkflowRecord = _load_thb()
    valid = {f.name for f in fields(WorkflowRecord)}
    data = {k: v for k, v in spec.items() if k in valid}
    if "type" in spec and "type_" not in data:
        data["type_"] = spec["type"]
    return WorkflowRecord(**data)


def save(workflow: dict, path: Path | None = None) -> Path:
    """Persist ``workflow`` and register it with :class:`WorkflowDB`.

    The resulting JSON file is named ``<name>.workflow.json`` where ``name`` is
    taken from the ``title`` or ``name`` field.  Files are stored inside the
    directory specified by the ``WORKFLOW_OUTPUT_DIR`` environment variable
    unless ``path`` overrides the location.
    """

    # Determine output directory
    if path and path.suffix:
        out_dir = path.parent
    elif path:
        out_dir = path
    else:
        out_dir = _OUTPUT_DIR
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    name = workflow.get("name") or workflow.get("title")
    if not name:
        names = workflow.get("workflow") or []
        name = names[0] if names else "workflow"
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", str(name))
    json_path = out_dir / f"{safe_name}.workflow.json"

    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(workflow, fh, indent=2, sort_keys=True)

    if yaml is not None:
        yaml_path = out_dir / f"{safe_name}.workflow.yaml"
        with yaml_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(workflow, fh, sort_keys=False)  # type: ignore[arg-type]

    # Register in WorkflowDB
    try:
        WorkflowDB, _WR = _load_thb()
        db = WorkflowDB(out_dir / "workflows.db")
        db.add(_record_from_spec(workflow))
    except Exception:  # pragma: no cover - best effort
        pass

    return json_path


__all__ = ["to_spec", "save"]
