"""Action log vectorization utilities for Menace Security AI."""

from __future__ import annotations

import json
import os
import math
from datetime import datetime
from typing import Any, Dict, List, Optional
import hashlib
import importlib

from compliance import license_fingerprint
from vector_utils import persist_embedding


def _log_violation(path: str, lic: str, hash_: str) -> None:
    try:  # pragma: no cover - best effort
        CodeDB = importlib.import_module("code_database").CodeDB
        CodeDB().log_license_violation(path, lic, hash_)
    except Exception:
        pass

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy optional
    np = None  # type: ignore

# default bounds used when clipping values
_DEFAULT_BOUNDS: Dict[str, float] = {
    "risk_score": 100.0,
    "reward": 1000.0,
    "code_length": 10000.0,
    "severity": 10.0,
    "num_violations": 10.0,
    "alignment_score": 100.0,
}


class ActionVectorizer:
    """Vectorizer with persistent category mappings and optional scaling."""

    def __init__(
        self,
        max_action_types: int = 10,
        max_domains: int = 10,
        cyclic_time: bool = False,
        weekend_flag: bool = False,
        scaling: str = "clip",
    ) -> None:
        self.max_action_types = max_action_types
        self.max_domains = max_domains
        self.cyclic_time = cyclic_time
        self.weekend_flag = weekend_flag
        self.scaling = scaling  # "clip" or "minmax"
        self.action_index: Dict[str, int] = {"other": 0}
        self.domain_index: Dict[str, int] = {"other": 0}
        self.scaling_params: Dict[str, List[float]] = {
            k: [-v, v] for k, v in _DEFAULT_BOUNDS.items()
        }
        self._update_metadata()

    # ------------------------------------------------------------------
    # helper utilities
    def _update_metadata(self) -> None:
        time_dim = 2 if self.cyclic_time else 1
        if self.weekend_flag:
            time_dim += 1
        self.num_features = 12 + time_dim  # numerical/binary features
        self.metadata = {
            "max_action_types": self.max_action_types,
            "max_domains": self.max_domains,
            "cyclic_time": self.cyclic_time,
            "weekend_flag": self.weekend_flag,
            "scaling": self.scaling,
            "action_index": self.action_index,
            "domain_index": self.domain_index,
            "scaling_params": self.scaling_params,
            "dim": self.max_action_types + self.max_domains + self.num_features,
        }

    @staticmethod
    def _one_hot(idx: int, length: int) -> List[float]:
        vec = [0.0] * length
        if 0 <= idx < length:
            vec[idx] = 1.0
        return vec

    @staticmethod
    def _parse_time(ts: Any) -> Optional[datetime]:
        dt = None
        if isinstance(ts, (int, float)):
            try:
                dt = datetime.utcfromtimestamp(float(ts))
            except Exception:
                dt = None
        elif isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts)
            except Exception:
                dt = None
        return dt

    def _time_features(self, ts: Any) -> List[float]:
        dt = self._parse_time(ts)
        if not dt:
            feats = [0.0, 0.0] if self.cyclic_time else [0.0]
            if self.weekend_flag:
                feats.append(0.0)
            return feats
        feats: List[float]
        if self.cyclic_time:
            angle = dt.hour * 2 * math.pi / 24.0
            feats = [math.sin(angle), math.cos(angle)]
        else:
            feats = [dt.hour / 23.0]
        if self.weekend_flag:
            feats.append(1.0 if dt.weekday() >= 5 else 0.0)
        return feats

    @staticmethod
    def _get_index(value: Any, mapping: Dict[str, int], max_size: int) -> int:
        val = str(value).lower().strip() or "other"
        if val in mapping:
            return mapping[val]
        if len(mapping) < max_size:
            mapping[val] = len(mapping)
            return mapping[val]
        return mapping["other"]

    def _update_scale(self, field: str, value: Any) -> None:
        try:
            f = float(value)
        except Exception:
            return
        params = self.scaling_params.setdefault(field, [f, f])
        if f < params[0]:
            params[0] = f
        if f > params[1]:
            params[1] = f

    def _scale(self, field: str, value: Any) -> float:
        try:
            f = float(value)
        except Exception:
            return 0.0
        mn, mx = self.scaling_params.get(field, [-_DEFAULT_BOUNDS[field], _DEFAULT_BOUNDS[field]])
        if self.scaling == "minmax":
            if mx == mn:
                return 0.0
            f = max(mn, min(mx, f))
            return 2 * (f - mn) / (mx - mn) - 1
        bound = max(abs(mn), abs(mx))
        f = max(-bound, min(bound, f))
        return f / bound

    # ------------------------------------------------------------------
    def fit(self, logs: List[Dict]) -> "ActionVectorizer":
        """Build mapping and scaling stats from ``logs``."""
        for log in logs:
            self._get_index(log.get("action_type", "other"), self.action_index, self.max_action_types)
            self._get_index(log.get("target_domain", "other"), self.domain_index, self.max_domains)
            if self.scaling == "minmax":
                self._update_scale("risk_score", log.get("risk_score"))
                self._update_scale("reward", log.get("reward"))
                code = log.get("generated_code")
                if code:
                    lic = license_fingerprint.check(str(code))
                    if lic:
                        _log_violation(
                            str(log.get("id", "")),
                            lic,
                            hashlib.sha256(str(code).encode("utf-8")).hexdigest(),
                        )
                        continue
                self._update_scale("code_length", len(str(code)) if code else 0)
                violations = log.get("violations", [])
                severity = 0.0
                if isinstance(violations, dict):
                    severity = float(violations.get("severity", 0.0))
                    vlist = violations.get("violations", [])
                elif isinstance(violations, list):
                    vlist = violations
                    severity = float(log.get("violation_severity", log.get("severity", 0.0)))
                else:
                    vlist = []
                self._update_scale("severity", severity)
                self._update_scale("num_violations", len(vlist))
                self._update_scale("alignment_score", log.get("alignment_score", 0.0))
        self._update_metadata()
        return self

    def transform(self, action_log: Dict) -> List[float]:
        """Return a vector representation for ``action_log``."""
        a_idx = self._get_index(action_log.get("action_type", "other"), self.action_index, self.max_action_types)
        d_idx = self._get_index(action_log.get("target_domain", "other"), self.domain_index, self.max_domains)
        vec: List[float] = []
        vec.extend(self._one_hot(a_idx, self.max_action_types))
        vec.extend(self._one_hot(d_idx, self.max_domains))
        vec.append(self._scale("risk_score", action_log.get("risk_score", 0.0)))
        vec.append(self._scale("reward", action_log.get("reward", 0.0)))
        vec.append(1.0 if action_log.get("invoked_security_ai") else 0.0)
        hb_flag = bool(action_log.get("helper_bot_created"))
        code_tmp = str(action_log.get("generated_code", ""))
        if code_tmp:
            lic = license_fingerprint.check(code_tmp)
            if lic:
                _log_violation(
                    str(action_log.get("id", "")),
                    lic,
                    hashlib.sha256(code_tmp.encode("utf-8")).hexdigest(),
                )
                raise ValueError(f"Disallowed license detected: {lic}")
        if not hb_flag:
            hb_flag = any(k in code_tmp for k in ("helper_bot", "spawn_agent"))
        vec.append(1.0 if hb_flag else 0.0)
        vec.append(1.0 if action_log.get("lockdown_triggered") else 0.0)
        vec.append(1.0 if action_log.get("override_applied") or action_log.get("manual_override") else 0.0)
        code = code_tmp if code_tmp else None
        vec.append(1.0 if code else 0.0)
        vec.append(self._scale("code_length", len(str(code)) if code else 0))
        violations = action_log.get("violations", [])
        severity = 0.0
        if isinstance(violations, dict):
            severity = float(violations.get("severity", 0.0))
            vlist = violations.get("violations", [])
        elif isinstance(violations, list):
            vlist = violations
            severity = float(action_log.get("violation_severity", action_log.get("severity", 0.0)))
        else:
            vlist = []
        vec.append(1.0 if vlist else 0.0)
        vec.append(self._scale("severity", severity))
        vec.append(self._scale("num_violations", len(vlist)))
        vec.extend(self._time_features(action_log.get("timestamp")))
        vec.append(self._scale("alignment_score", action_log.get("alignment_score", 0.0)))
        expected_dim = self.metadata["dim"]
        if len(vec) < expected_dim:
            vec.extend([0.0] * (expected_dim - len(vec)))
        else:
            vec = vec[:expected_dim]
        return vec

    def transform_batch(self, logs: List[Dict]) -> List[List[float]]:
        return [self.transform(log) for log in logs]

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        """Persist vectorizer configuration to ``path``."""
        data = {
            "action_index": self.action_index,
            "domain_index": self.domain_index,
            "scaling_params": self.scaling_params,
            "max_action_types": self.max_action_types,
            "max_domains": self.max_domains,
            "cyclic_time": self.cyclic_time,
            "weekend_flag": self.weekend_flag,
            "scaling": self.scaling,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)

    @classmethod
    def load(cls, path: str) -> "ActionVectorizer":
        with open(path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        vec = cls(
            max_action_types=cfg.get("max_action_types", 10),
            max_domains=cfg.get("max_domains", 10),
            cyclic_time=cfg.get("cyclic_time", False),
            weekend_flag=cfg.get("weekend_flag", False),
            scaling=cfg.get("scaling", "clip"),
        )
        vec.action_index = cfg.get("action_index", {"other": 0})
        vec.domain_index = cfg.get("domain_index", {"other": 0})
        vec.scaling_params = cfg.get("scaling_params", vec.scaling_params)
        vec._update_metadata()
        return vec


# ----------------------------------------------------------------------
# Backwards compatible functional API
_DEFAULT_VECTORIZER = ActionVectorizer()


def vectorize_action(action_log: Dict) -> List[float]:
    return _DEFAULT_VECTORIZER.transform(action_log)


def vectorize_batch(logs: List[Dict]) -> List[List[float]]:
    return _DEFAULT_VECTORIZER.transform_batch(logs)


def save_vectors(vectors: List[List[float]], output_path: str) -> None:
    if not vectors:
        return
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if output_path.endswith(".json"):
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(vectors, fh)
    elif output_path.endswith(".npy"):
        if np is None:
            raise RuntimeError("NumPy required for .npy output")
        np.save(output_path, np.asarray(vectors, dtype=float))
    else:
        raise ValueError("Unsupported file extension: use .json or .npy")


def compare_vectors(v1: List[float], v2: List[float]) -> float:
    if np is None:
        raise RuntimeError("NumPy required for vector comparison")
    a = np.asarray(v1, dtype=float)
    b = np.asarray(v2, dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def vectorize_and_store(
    record_id: str,
    action_log: Dict,
    *,
    path: str = "embeddings.jsonl",
    origin_db: str = "action",
    metadata: Dict[str, Any] | None = None,
) -> List[float]:
    """Vectorise ``action_log`` and persist the embedding."""

    vec = _DEFAULT_VECTORIZER.transform(action_log)
    try:
        persist_embedding(
            "action",
            record_id,
            vec,
            path=path,
            origin_db=origin_db,
            metadata=metadata or {},
        )
    except TypeError:  # pragma: no cover - compatibility with older signatures
        persist_embedding("action", record_id, vec, path=path)
    return vec


__all__ = [
    "ActionVectorizer",
    "vectorize_action",
    "vectorize_and_store",
    "vectorize_batch",
    "save_vectors",
    "compare_vectors",
]
