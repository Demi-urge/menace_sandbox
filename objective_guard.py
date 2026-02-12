from __future__ import annotations

"""Hard safety rails for objective/reward code during self-coding.

The guard enforces two protections:

1. Protected target denylist: self-coding may not patch configured objective paths.
2. Hash lock: protected objective files are snapshotted at startup and any drift
   triggers a circuit breaker before additional patches run.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

from self_coding_objective_paths import OBJECTIVE_ADJACENT_UNSAFE_PATHS


_DEFAULT_PROTECTED_SPECS: tuple[str, ...] = (
    *OBJECTIVE_ADJACENT_UNSAFE_PATHS,
    "startup_health_check.py",
    "neurosales/neurosales/hierarchical_reward.py",
    "neurosales/neurosales/reward_ledger.py",
)

DEFAULT_OBJECTIVE_HASH_MANIFEST = "config/objective_hash_lock.json"


@dataclass(frozen=True)
class GuardSpec:
    raw: str
    normalized: str
    prefix: bool


class ObjectiveGuardViolation(PermissionError):
    """Raised when a protected objective policy is violated."""

    def __init__(self, reason: str, *, details: dict[str, object] | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}


class ObjectiveGuard:
    """Validate protected objective paths and integrity hashes."""

    def __init__(
        self,
        *,
        repo_root: Path,
        protected_specs: Sequence[str] | None = None,
        hash_specs: Sequence[str] | None = None,
        enabled: bool | None = None,
        manifest_path: Path | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.enabled = self._resolve_enabled(enabled)
        self.protected_specs = tuple(
            self._build_spec(spec)
            for spec in self._parse_specs(
                protected_specs,
                env_var="MENACE_SELF_CODING_PROTECTED_PATHS",
                default=_DEFAULT_PROTECTED_SPECS,
            )
        )
        resolved_hash_specs = self._parse_specs(
            hash_specs,
            env_var="MENACE_SELF_CODING_OBJECTIVE_HASH_PATHS",
            default=tuple(spec.raw for spec in self.protected_specs if not spec.prefix),
        )
        self.hash_specs = tuple(self._build_spec(spec) for spec in resolved_hash_specs)
        self.manifest_path = (
            manifest_path
            if manifest_path is not None
            else self.repo_root / DEFAULT_OBJECTIVE_HASH_MANIFEST
        )

    @staticmethod
    def _resolve_enabled(enabled: bool | None) -> bool:
        if enabled is not None:
            return bool(enabled)
        raw = os.getenv("MENACE_SELF_CODING_OBJECTIVE_GUARD", "1").strip().lower()
        return raw not in {"0", "false", "no", "off"}

    @staticmethod
    def _parse_specs(
        provided: Sequence[str] | None,
        *,
        env_var: str,
        default: Sequence[str],
    ) -> tuple[str, ...]:
        if provided is not None:
            values = tuple(item.strip() for item in provided if item and item.strip())
            return values or tuple(default)
        raw = os.getenv(env_var, "")
        if raw.strip():
            values = tuple(item.strip() for item in raw.split(",") if item and item.strip())
            if values:
                return values
        return tuple(default)

    def _build_spec(self, raw: str) -> GuardSpec:
        normalized = raw.replace("\\", "/").strip().lstrip("./")
        candidate = self.repo_root / normalized
        prefix = normalized.endswith("/") or candidate.is_dir()
        if prefix:
            normalized = normalized.rstrip("/")
        return GuardSpec(raw=raw, normalized=normalized, prefix=prefix)

    def _as_repo_rel(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(self.repo_root).as_posix()
        except Exception:
            return resolved.as_posix()

    def is_protected_target(self, path: Path) -> bool:
        if not self.enabled:
            return False
        rel = self._as_repo_rel(path)
        for spec in self.protected_specs:
            if spec.prefix:
                if rel == spec.normalized or rel.startswith(f"{spec.normalized}/"):
                    return True
            elif rel == spec.normalized:
                return True
        return False

    def _hash_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            digest.update(handle.read())
        return digest.hexdigest()

    def _expected_manifest_paths(self) -> tuple[str, ...]:
        expected: list[str] = []
        for spec in self.hash_specs:
            if spec.prefix:
                continue
            target = (self.repo_root / spec.normalized).resolve()
            if not target.exists() or not target.is_file():
                continue
            expected.append(self._as_repo_rel(target))
        return tuple(sorted(expected))

    def _hash_targets(self, specs: Iterable[GuardSpec]) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for spec in specs:
            target = (self.repo_root / spec.normalized).resolve()
            if spec.prefix:
                continue
            if not target.exists() or not target.is_file():
                continue
            rel = self._as_repo_rel(target)
            hashes[rel] = self._hash_file(target)
        return hashes

    def snapshot_hashes(self) -> dict[str, str]:
        if not self.enabled:
            return {}
        return self._hash_targets(self.hash_specs)

    def write_manifest(self) -> dict[str, str]:
        """Persist hashes for protected objective files.

        This method is intended for explicit operator workflows.
        """

        hashes = self.snapshot_hashes()
        payload = {
            "version": 1,
            "algorithm": "sha256",
            "files": hashes,
        }
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return hashes

    def verify_manifest(self) -> dict[str, object]:
        """Validate protected file hashes against the persisted manifest."""

        manifest = self.manifest_path
        if not manifest.exists():
            raise ObjectiveGuardViolation(
                "manifest_missing",
                details={
                    "manifest_path": self._as_repo_rel(manifest),
                    "expected_files": list(self._expected_manifest_paths()),
                },
            )
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ObjectiveGuardViolation(
                "manifest_invalid",
                details={
                    "manifest_path": self._as_repo_rel(manifest),
                    "error": str(exc),
                },
            ) from exc
        expected = payload.get("files")
        if not isinstance(expected, dict):
            raise ObjectiveGuardViolation(
                "manifest_invalid",
                details={
                    "manifest_path": self._as_repo_rel(manifest),
                    "error": "manifest missing files mapping",
                },
            )
        expected_hashes = {
            str(path): str(digest)
            for path, digest in expected.items()
            if isinstance(path, str) and isinstance(digest, str)
        }
        current_hashes = self.snapshot_hashes()
        deltas: list[dict[str, object]] = []
        for path in sorted(set(expected_hashes) | set(current_hashes)):
            expected_digest = expected_hashes.get(path)
            current_digest = current_hashes.get(path)
            if expected_digest != current_digest:
                deltas.append(
                    {
                        "path": path,
                        "expected": expected_digest,
                        "current": current_digest,
                    }
                )
        if deltas:
            raise ObjectiveGuardViolation(
                "manifest_hash_mismatch",
                details={
                    "manifest_path": self._as_repo_rel(manifest),
                    "deltas": deltas,
                    "changed_files": [entry["path"] for entry in deltas],
                },
            )
        return {
            "manifest_path": self._as_repo_rel(manifest),
            "files": sorted(current_hashes),
        }

    def assert_patch_target_safe(self, path: Path) -> None:
        if self.is_protected_target(path):
            raise ObjectiveGuardViolation(
                "protected_target",
                details={"path": self._as_repo_rel(path)},
            )

    def assert_integrity(self) -> None:
        if not self.enabled:
            return
        # Persisted manifest verification is the primary integrity gate.
        # This check intentionally uses the on-disk manifest on every cycle
        # rather than process-start snapshots so operator refreshes are honored.
        self.verify_manifest()


__all__ = [
    "DEFAULT_OBJECTIVE_HASH_MANIFEST",
    "ObjectiveGuard",
    "ObjectiveGuardViolation",
]
