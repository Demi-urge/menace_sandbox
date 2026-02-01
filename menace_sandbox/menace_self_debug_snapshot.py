"""Snapshot inputs/configs for self-debug cycles to ensure deterministic runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import uuid
from typing import Any, Mapping

from dynamic_path_router import resolve_path


_DEFAULT_SNAPSHOT_DIR = Path(resolve_path("sandbox_data")) / "self_debug_snapshots"


def _coerce_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _coerce_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_coerce_json(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _snapshot_environment(keys: tuple[str, ...]) -> dict[str, str | None]:
    import os

    return {key: os.getenv(key) for key in keys}


def _snapshot_sandbox_settings(settings: Any) -> dict[str, Any]:
    if hasattr(settings, "model_dump"):
        return _coerce_json(settings.model_dump())
    if hasattr(settings, "dict"):
        return _coerce_json(settings.dict())  # type: ignore[call-arg]
    if hasattr(settings, "_data"):
        return _coerce_json(getattr(settings, "_data"))
    return _coerce_json(getattr(settings, "__dict__", {}))


@dataclass(frozen=True)
class SelfDebugSnapshot:
    snapshot_id: str
    path: Path
    created_at: str
    payload: Mapping[str, Any]


def create_snapshot(
    *,
    inputs: Mapping[str, Any],
    configs: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    snapshot_dir: Path | None = None,
) -> SelfDebugSnapshot:
    snapshot_id = f"self-debug-{uuid.uuid4()}"
    created_at = datetime.utcnow().isoformat()
    payload = {
        "snapshot_id": snapshot_id,
        "created_at": created_at,
        "inputs": _coerce_json(inputs),
        "configs": _coerce_json(configs),
        "metadata": _coerce_json(metadata or {}),
    }
    directory = snapshot_dir or _DEFAULT_SNAPSHOT_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{snapshot_id}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return SelfDebugSnapshot(
        snapshot_id=snapshot_id, path=path, created_at=created_at, payload=payload
    )


def load_snapshot(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def freeze_cycle(
    *,
    inputs: Mapping[str, Any],
    configs: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    snapshot_dir: Path | None = None,
) -> SelfDebugSnapshot:
    snapshot = create_snapshot(
        inputs=inputs,
        configs=configs,
        metadata=metadata,
        snapshot_dir=snapshot_dir,
    )
    frozen = load_snapshot(snapshot.path)
    return SelfDebugSnapshot(
        snapshot_id=snapshot.snapshot_id,
        path=snapshot.path,
        created_at=snapshot.created_at,
        payload=frozen,
    )


__all__ = [
    "SelfDebugSnapshot",
    "create_snapshot",
    "freeze_cycle",
    "load_snapshot",
    "_snapshot_environment",
    "_snapshot_sandbox_settings",
]
