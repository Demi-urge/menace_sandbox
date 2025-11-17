"""ChatGPT Prediction Bot for profitability validation.

This module loads a scikit-learn :class:`~sklearn.pipeline.Pipeline` trained
model when available.  If scikit-learn is missing, a small logistic regression
``Pipeline`` defined here is used instead.  Instantiating
:class:`ChatGPTPredictionBot` will emit a warning when falling back to this
simplified implementation.

Callers must provide a :class:`~vector_service.context_builder.ContextBuilder`
instance which is used to build compressed context from local databases for
all LLM prompts.
"""

from __future__ import annotations

from .bot_registry import BotRegistry
from .data_bot import DataBot

from .coding_bot_interface import self_coding_managed
from .self_coding_engine import SelfCodingEngine
from context_builder import handle_failure, PromptBuildError
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Tuple
import json
import hashlib
import time
from contextlib import nullcontext
from tempfile import NamedTemporaryFile
import logging

from db_router import DBRouter, GLOBAL_ROUTER
import math
import os
import sys
import csv

registry = BotRegistry()
data_bot = DataBot(start_server=False)

try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv
except Exception as exc:  # pragma: no cover - missing optional dependency
    load_dotenv = None
    logging.getLogger(__name__).warning("dotenv not loaded: %s", exc)
else:  # pragma: no cover - environment loading is optional
    load_dotenv()

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FORMAT = os.environ.get(
    "LOG_FORMAT",
    "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

from .error_flags import RAISE_ERRORS  # noqa: E402

from .mirror_bot import sentiment_score  # noqa: E402
try:  # pragma: no cover - optional dependency
    from .chatgpt_idea_bot import ChatGPTClient
except BaseException:  # pragma: no cover - missing or failing dependency
    ChatGPTClient = None  # type: ignore
from gpt_memory_interface import GPTMemoryInterface  # noqa: E402
from snippet_compressor import compress_snippets  # noqa: E402
from vector_service.context_builder import ContextBuilder, FallbackResult  # noqa: E402
try:  # canonical tag constants
    from .log_tags import FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT  # noqa: E402
except Exception:  # pragma: no cover - fallback for flat layout
    from log_tags import (  # type: ignore  # noqa: E402
        FEEDBACK,
        IMPROVEMENT_PATH,
        ERROR_FIX,
        INSIGHT,
    )
try:  # tag helper
    from memory_logging import ensure_tags
except Exception:  # pragma: no cover - fallback when logging unavailable
    def ensure_tags(key: str, tags=None):  # type: ignore
        return [key, *(tags or [])]
try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer, util as st_util  # noqa: E402
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore
    st_util = None  # type: ignore

from security.secret_redactor import redact  # noqa: E402
from license_detector import detect as detect_license  # noqa: E402
from governed_embeddings import governed_embed  # noqa: E402
try:  # pragma: no cover - optional dependency
    from analysis.semantic_diff_filter import find_semantic_risks
except Exception:  # pragma: no cover - best effort optional import
    find_semantic_risks = None  # type: ignore

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    requests = None  # type: ignore

try:
    from filelock import FileLock
except Exception:  # pragma: no cover - optional dependency
    FileLock = None  # type: ignore

from dynamic_path_router import resolve_path  # noqa: E402
try:  # pragma: no cover - optional dependency
    from vector_service import ErrorResult  # type: ignore
except Exception:  # pragma: no cover - fallback when service missing
    class ErrorResult(Exception):
        """Fallback placeholder when vector service is unavailable."""


def _build_prediction_prompt(
    query: str,
    *,
    intent: Mapping[str, Any],
    context_builder: ContextBuilder,
):
    """Return an enriched prompt for the enhancement evaluation."""

    engine = SelfCodingEngine.__new__(SelfCodingEngine)
    engine.logger = logger
    engine._last_retry_trace = None
    engine._last_prompt = None
    engine._last_prompt_metadata = {}
    return engine.build_enriched_prompt(
        query,
        intent=intent,
        context_builder=context_builder,
    )


def _resolve_model_path(path_str: str) -> Path:
    """Resolve *path_str* using :func:`resolve_path`."""

    return resolve_path(path_str)


def _env_model_path() -> Path:
    path_str = os.environ.get(
        "CHATGPT_PREDICTION_MODEL_PATH",
        "chatgpt_prediction_bot/prediction_model.joblib",
    )
    try:
        return _resolve_model_path(path_str)
    except FileNotFoundError:
        return Path(path_str)


@dataclass
class PredictionBotConfig:
    """Runtime configuration loaded from environment variables."""

    fallback_lr: float = float(os.environ.get("CHATGPT_FALLBACK_LR", "0.1"))
    fallback_iters: int = int(os.environ.get("CHATGPT_FALLBACK_ITERS", "100"))
    fallback_l2: float | None = (
        float(os.environ.get("CHATGPT_FALLBACK_L2"))
        if os.environ.get("CHATGPT_FALLBACK_L2") is not None
        else None
    )
    fallback_c: float = float(os.environ.get("CHATGPT_FALLBACK_C", "1.0"))
    fallback_val_steps: int | None = (
        int(os.environ.get("CHATGPT_FALLBACK_VAL_STEPS"))
        if os.environ.get("CHATGPT_FALLBACK_VAL_STEPS")
        else None
    )
    max_rationale_words: int = int(os.environ.get("CHATGPT_MAX_RATIONALE_WORDS", "50"))
    sentiment_weight: float = float(os.environ.get("CHATGPT_SENTIMENT_WEIGHT", "0.5"))
    force_fallback: bool = (
        os.environ.get("CHATGPT_PREDICTION_FORCE_FALLBACK", "0") == "1"
    )
    auto_save: bool = os.environ.get("CHATGPT_PREDICTION_AUTO_SAVE", "1") == "1"
    model_path: Path = field(default_factory=_env_model_path)
    threshold: float = float(os.environ.get("CHATGPT_PREDICTION_THRESHOLD", "0.5"))
    fallback_data: str | None = os.environ.get("CHATGPT_FALLBACK_DATA")


CFG = PredictionBotConfig()

# version string used for saved models and metadata
MODEL_VERSION = "1.0"

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    joblib = None  # type: ignore


def _joblib_dump_with_retry(obj, path: Path, retries: int = 3, delay: float = 0.5) -> None:
    """Safely persist ``obj`` using ``joblib`` with backups."""
    if not (joblib and hasattr(joblib, "dump")):
        logger.warning("joblib not available; cannot save model to %s", path)
        return
    for i in range(retries):
        tmp = None
        try:
            lock = FileLock(str(path) + ".lock") if FileLock else nullcontext()
            with lock:
                with NamedTemporaryFile(dir=str(path.parent), delete=False) as t:
                    tmp = Path(t.name)
                    joblib.dump(obj, tmp)
                backup = path.with_suffix(path.suffix + ".bak")
                if path.exists():
                    path.replace(backup)
                tmp.replace(path)
            return
        except Exception as exc:  # pragma: no cover - optional dependency
            if tmp and tmp.exists():
                tmp.unlink(missing_ok=True)
            if i == retries - 1:
                logger.exception("failed to save model %s: %s", path, exc)
            time.sleep(delay)


def _joblib_load_with_retry(path: Path, retries: int = 3, delay: float = 0.5):
    if not (joblib and hasattr(joblib, "load")):
        raise RuntimeError("joblib not available")
    for i in range(retries):
        try:
            lock = FileLock(str(path) + ".lock") if FileLock else nullcontext()
            with lock:
                return joblib.load(path)
        except Exception as exc:  # pragma: no cover - optional dependency
            if i == retries - 1:
                logger.exception("failed to load model %s: %s", path, exc)
                # try backup before giving up
                bak = path.with_suffix(path.suffix + ".bak")
                if bak.exists():
                    try:
                        logger.warning("loading backup model from %s", bak)
                        return joblib.load(bak)
                    except Exception:
                        pass
                raise
            time.sleep(delay)


def _hash_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_dump_with_retry(data: dict, path: Path, retries: int = 3, delay: float = 0.5) -> None:
    for i in range(retries):
        tmp = None
        try:
            lock = FileLock(str(path) + ".lock") if FileLock else nullcontext()
            with lock:
                with NamedTemporaryFile("w", dir=str(path.parent), delete=False) as t:
                    tmp = Path(t.name)
                    json.dump(data, t)
                if path.exists():
                    path.replace(path.with_suffix(path.suffix + ".bak.json"))
                tmp.replace(path)
            return
        except Exception as exc:  # pragma: no cover - optional dependency
            if tmp and tmp.exists():
                tmp.unlink(missing_ok=True)
            if i == retries - 1:
                logger.exception("failed to write json %s: %s", path, exc)
            time.sleep(delay)


def _fetch_remote_file(url: str) -> Path:
    if not requests:
        raise RuntimeError("requests not available")
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    with NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
        tmp.write(resp.content)
        return Path(tmp.name)



def _define_fallback() -> None:
    """Define lightweight pipeline when scikit-learn is unavailable."""
    global np, _SimpleLogReg, _DictVectorizer, Pipeline
    if "_SimpleLogReg" in globals():
        return
    import math
    import random

    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        np = None  # type: ignore[assignment]

    class _SimpleLogReg:
        """Lightweight logistic regression that works with or without numpy."""

        def __init__(
            self,
            lr: float = CFG.fallback_lr,
            iters: int = CFG.fallback_iters,
            *,
            l2: float | None = CFG.fallback_l2,
            C: float = CFG.fallback_c,
            fit_intercept: bool = True,
            shuffle: bool = True,
            val_steps: int | None = CFG.fallback_val_steps,
        ) -> None:
            self.lr = lr
            self.iters = iters
            self.l2 = float(l2) if l2 is not None else (1.0 / C if C else 0.0)
            self.fit_intercept = fit_intercept
            self.shuffle = shuffle
            self.val_steps = val_steps or 0
            self.coef_: object | None = None
            self.intercept_: float = 0.0
            self._numpy_enabled = np is not None

        def _sigmoid(self, value: float) -> float:
            return 1.0 / (1.0 + math.exp(-value))

        def _prepare_rows(self, X: Iterable[Iterable[float]]) -> list[list[float]]:
            rows = [list(map(float, row)) for row in X]
            if self.fit_intercept:
                return [[1.0, *row] for row in rows]
            return rows

        def fit(self, X: Iterable[Iterable[float]], y: Iterable[int]) -> "_SimpleLogReg":
            X_rows = self._prepare_rows(X)
            y_vals = [float(val) for val in y]
            if not X_rows or not y_vals:
                raise ValueError("Training data is empty")

            if self._numpy_enabled:
                X_arr = np.asarray(X_rows, dtype=float)
                y_arr = np.asarray(y_vals, dtype=float)
                n_features = X_arr.shape[1]
                coef = np.zeros(n_features)
                indices = np.arange(len(X_arr))
                for i in range(self.iters):
                    if self.shuffle:
                        np.random.shuffle(indices)
                    for batch_idx in np.array_split(indices, max(1, len(indices) // 10)):
                        batch_X = X_arr[batch_idx]
                        batch_y = y_arr[batch_idx]
                        logits = batch_X.dot(coef)
                        preds = 1.0 / (1.0 + np.exp(-logits))
                        error = preds - batch_y
                        grad = batch_X.T.dot(error) / len(batch_y)
                        grad += self.l2 * coef
                        coef -= self.lr * grad
                    if self.val_steps and (i + 1) % self.val_steps == 0:
                        preds = (1.0 / (1.0 + np.exp(-X_arr.dot(coef))) >= 0.5).astype(float)
                        acc = float((preds == y_arr).mean())
                        logger.debug("[fallback] iter=%s accuracy=%.3f", i + 1, acc)
                self.coef_ = coef
                return self

            n_features = len(X_rows[0])
            coef = [0.0] * n_features
            indices = list(range(len(X_rows)))
            for _ in range(self.iters):
                if self.shuffle:
                    random.shuffle(indices)
                for idx in indices:
                    row = X_rows[idx]
                    target = y_vals[idx]
                    dot = sum(w * v for w, v in zip(coef, row))
                    pred = self._sigmoid(dot)
                    error = pred - target
                    for i, val in enumerate(row):
                        coef[i] -= self.lr * (error * val + self.l2 * coef[i])
            self.coef_ = coef
            return self

        def predict_proba(self, X: Iterable[Iterable[float]]) -> list[list[float]]:
            if self.coef_ is None:
                raise ValueError("Model is not fitted yet")
            rows = self._prepare_rows(X)
            if self._numpy_enabled:
                coef = np.asarray(self.coef_, dtype=float)
                logits = np.asarray(rows, dtype=float).dot(coef)
                probs = 1.0 / (1.0 + np.exp(-logits))
                return np.column_stack([1 - probs, probs]).tolist()
            coef = list(self.coef_)
            result = []
            for row in rows:
                score = sum(w * v for w, v in zip(coef, row))
                prob = self._sigmoid(score)
                result.append([1 - prob, prob])
            return result

class _DictVectorizer:
        """Very small dict vectorizer supporting numbers and categories."""

        def __init__(self) -> None:
            self.num_index: dict[str, int] = {}
            self.cat_index: dict[str, dict[str, int]] = {}
            self.n_features = 0

        def fit(self, X: Iterable[dict]) -> None:
            for row in X:
                for key, val in row.items():
                    if isinstance(val, (int, float)):
                        if key not in self.num_index:
                            self.num_index[key] = self.n_features
                            self.n_features += 1
                    else:
                        mapping = self.cat_index.setdefault(key, {})
                        if val not in mapping:
                            mapping[val] = self.n_features
                            self.n_features += 1

        def transform(self, X: Iterable[dict]) -> List[List[float]]:
            # allow unseen features to extend the schema on-the-fly
            self.fit(X)
            result = []
            for row in X:
                vec = [0.0] * self.n_features
                for key, val in row.items():
                    if key in self.num_index:
                        vec[self.num_index[key]] = float(val)
                    elif key in self.cat_index:
                        idx = self.cat_index[key].get(val)
                        if idx is not None:
                            vec[idx] = 1.0
                result.append(vec)
            return result

        def fit_transform(self, X: Iterable[dict]) -> List[List[float]]:
            self.fit(X)
            return self.transform(X)

def _generate_synthetic_data(n_samples: int = 50) -> Tuple[List[dict], List[int]]:
    """Generate a simple synthetic training set."""
    import random

    market_types = ["tech", "finance", "health"]
    monetization_models = ["ads", "subscription", "freemium"]
    X: List[dict] = []
    y: List[int] = []
    for _ in range(n_samples):
        mt = random.choice(market_types)
        mm = random.choice(monetization_models)
        startup = float(random.uniform(0, 10))
        skill = float(random.uniform(0, 5))
        competition = float(random.uniform(0, 5))
        uniqueness = float(random.uniform(0, 5))
        feat = {
            "market_type": mt,
            "monetization_model": mm,
            "startup_cost": startup,
            "skill": skill,
            "competition": competition,
            "uniqueness": uniqueness,
        }
        score = uniqueness + skill - competition - 0.1 * startup
        prob = 1.0 / (1.0 + math.exp(-score))
        label = int(prob > 0.5)
        X.append(feat)
        y.append(label)
    return X, y

class DataIngestor:
    """Load training data from file, API or database."""

    def __init__(self, source: str | Path, router: DBRouter | None = None) -> None:
        self.source = str(source)
        self.router = router or GLOBAL_ROUTER

    # --------------------------------------------------
    def load(self) -> Tuple[List[dict], List[int]]:
        if self.source.startswith("http://") or self.source.startswith("https://"):
            return self._from_api(self.source)
        if self.source.startswith("sqlite://"):
            return self._from_db(self.source[len("sqlite://"):])
        return self._from_file(self.source)

    def _from_file(self, path: str) -> Tuple[List[dict], List[int]]:
        p = Path(path)
        if p.suffix.lower() == ".csv":
            X: List[dict] = []
            y: List[int] = []
            with p.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "label" not in row:
                        raise ValueError("CSV missing label column")
                    label = row.pop("label")
                    y.append(int(float(label)))
                    feat: dict[str, int | float | str] = {}
                    for k, v in row.items():
                        if v is None or v == "":
                            continue
                        try:
                            num = float(v)
                            feat[k] = int(num) if num.is_integer() else num
                        except ValueError:
                            feat[k] = v
                    X.append(feat)
            return X, y
        with p.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload.get("X", []), payload.get("y", [])

    def _from_api(self, url: str) -> Tuple[List[dict], List[int]]:
        if not requests:
            raise RuntimeError("requests not available")
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        return payload.get("X", []), payload.get("y", [])

    def _from_db(self, path: str) -> Tuple[List[dict], List[int]]:
        if not self.router:
            raise RuntimeError("DBRouter not initialised")
        conn = self.router.get_connection("training_data")
        cur = conn.execute("SELECT features, label FROM training_data")
        X: List[dict] = []
        y: List[int] = []
        for row in cur.fetchall():
            X.append(json.loads(row[0]))
            y.append(int(row[1]))
        return X, y

    def _generate_training_data(source: str | None) -> Tuple[List[dict], List[int]]:
        """Return training data from ``source`` or generate synthetic samples."""
        if source:
            try:
                loader = DataIngestor(source)
                X, y = loader.load()
                if X and y:
                    return X, y
                logger.warning("data source %s returned no samples", source)
            except Exception as exc:
                logger.exception("failed to ingest training data from %s: %s", source, exc)
        logger.debug("falling back to synthetic data")
        return _generate_synthetic_data()

    class _FallbackPipeline:  # type: ignore
        """Fallback pipeline with a simple logistic regression model."""

        def __init__(
            self, **model_kwargs
        ) -> None:  # pragma: no cover - optional dependency
            self.vectorizer = _DictVectorizer()
            self.model = _SimpleLogReg(**model_kwargs)
            self._trained = False
            self._train_default()

        # --------------------------------------------------
        def save(self, path: str | Path) -> None:
            """Persist the pipeline using ``joblib`` if available."""
            p = Path(path)
            _joblib_dump_with_retry(self, p)
            schema = {
                "numerical": list(self.vectorizer.num_index.keys()),
                "categorical": {k: list(v.keys()) for k, v in self.vectorizer.cat_index.items()},
            }
            meta = {
                "version": MODEL_VERSION,
                "training_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "schema": schema,
            }
            _json_dump_with_retry(meta, p.with_name("model_metadata.json"))
            logger.info("fallback model saved to %s", path)

        @classmethod
        def load(cls, path: str | Path) -> "_FallbackPipeline":
            obj = _joblib_load_with_retry(Path(path))
            if isinstance(obj, cls):
                if hasattr(obj, "vectorizer") and hasattr(obj, "model"):
                    return obj
                logger.error("loaded pipeline missing expected attributes")
                raise RuntimeError("invalid Pipeline structure")
            logger.error("loaded object from %s is not a %s", path, cls.__name__)
            raise RuntimeError("cannot load Pipeline")

        def _train_default(self) -> None:
            X, y = _generate_training_data(CFG.fallback_data)

            if not X or not y:
                logger.error("no training data available for fallback model")
                X, y = [], []
            vecs = self.vectorizer.fit_transform(X)
            self.model.fit(vecs, y)
            logger.info("fallback model trained on %d samples", len(X))
            self._trained = True

        def predict_proba(self, X: Iterable[dict]):
            if not self._trained:
                self._train_default()
            vecs = self.vectorizer.transform(X)
            return self.model.predict_proba(vecs)

    Pipeline = _FallbackPipeline


try:
    if "sklearn" in sys.modules and not hasattr(sys.modules["sklearn"], "__path__"):
        raise ImportError
    import sklearn.pipeline as skl_pipe  # type: ignore

    Pipeline = skl_pipe.Pipeline  # type: ignore[attr-defined]
    _SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _SKLEARN_AVAILABLE = False
_define_fallback()

try:
    DEFAULT_MODEL_PATH = _resolve_model_path(
        "chatgpt_prediction_bot/prediction_model.joblib"
    )
except FileNotFoundError:
    DEFAULT_MODEL_PATH = Path("chatgpt_prediction_bot/prediction_model.joblib")
MODEL_PATH = CFG.model_path
DEFAULT_THRESHOLD = CFG.threshold


@dataclass
class IdeaFeatures:
    """Feature set describing a business idea with schema versioning."""

    market_type: str
    monetization_model: str
    startup_cost: float
    skill: float
    competition: float
    uniqueness: float
    extra_features: dict[str, int | float | str] = field(default_factory=dict)
    schema_version: int = 1
    feature_schema: dict[str, type] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.market_type, str) or not self.market_type:
            raise ValueError("market_type must be a non-empty string")
        if not isinstance(self.monetization_model, str) or not self.monetization_model:
            raise ValueError("monetization_model must be a non-empty string")
        for name, val in [
            ("startup_cost", self.startup_cost),
            ("skill", self.skill),
            ("competition", self.competition),
            ("uniqueness", self.uniqueness),
        ]:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{name} must be a finite number")
            if val < 0:
                raise ValueError(f"{name} must be non-negative")
        if not isinstance(self.extra_features, dict):
            raise ValueError("extra_features must be a dict")
        for k, v in self.extra_features.items():
            if not isinstance(v, (int, float, str)):
                raise ValueError(f"unsupported type for feature {k}")
            if isinstance(v, (int, float)) and not math.isfinite(float(v)):
                raise ValueError(f"feature {k} must be finite")
            if isinstance(v, str) and len(v) > 1000:
                raise ValueError(f"feature {k} string too long")
        for k, t in self.feature_schema.items():
            if k not in self.extra_features:
                raise ValueError(f"missing required feature {k}")
            if not isinstance(self.extra_features[k], t):
                raise ValueError(f"feature {k} must be {t}")

    def to_dict(self) -> dict:
        data = asdict(self)
        extras = data.pop("extra_features", {})
        data.update(extras)
        return data


@dataclass
class EnhancementEvaluation:
    """Evaluation result for a proposed enhancement."""

    description: str
    reason: str
    value: float
    alerts: List[str] = field(default_factory=list)


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class ChatGPTPredictionBot:
    """Evaluate business ideas using a trained ML model.

    A :class:`~vector_service.context_builder.ContextBuilder` is always
    required and supplies compact context for any LLM-driven interactions.  An
    optional ``gpt_memory`` or pre-configured ``client`` can be provided for
    memory-aware predictions.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        threshold: float | None = None,
        *,
        context_builder: ContextBuilder,
        client: ChatGPTClient | None = None,
        gpt_memory: GPTMemoryInterface | None = None,
        **model_kwargs,
    ) -> None:
        """Load a trained model or fall back to the internal pipeline.

        A warning is logged when scikit-learn is unavailable and the simplified
        pipeline is used.
        """

        if context_builder is None:  # pragma: no cover - defensive guard
            raise TypeError("context_builder is required")

        context_builder.refresh_db_weights()

        self.model_path = Path(model_path) if model_path else CFG.model_path
        self.threshold = float(threshold) if threshold is not None else CFG.threshold
        self.context_builder = context_builder
        self.gpt_memory = gpt_memory
        if self.gpt_memory is not None:
            self.gpt_memory.context_builder = context_builder

        if client is not None:
            self.client = client
            self.client.context_builder = context_builder
            existing_mem = getattr(self.client, "gpt_memory", None)
            if existing_mem is None and self.gpt_memory is not None:
                self.client.gpt_memory = self.gpt_memory
                existing_mem = self.gpt_memory
            if existing_mem is not None:
                existing_mem.context_builder = context_builder
        elif self.gpt_memory is not None:
            try:
                self.client = ChatGPTClient(
                    gpt_memory=self.gpt_memory, context_builder=context_builder
                )
            except Exception:  # pragma: no cover - optional dependency
                logger.debug("failed to initialize ChatGPTClient", exc_info=True)
                self.client = None
        else:
            self.client = None
        self._model_hash: str | None = None
        self._model_mtime: float | None = None
        self._feedback: List[Tuple[dict, int]] = []
        self._last_accuracy: float | None = None
        self._feedback_batch = 10
        self._drift_threshold = 0.2

        logger.debug(
            "ChatGPTPredictionBot initialized with model_path=%s threshold=%.3f",
            self.model_path,
            self.threshold,
        )

        use_sklearn = False
        if not CFG.force_fallback and joblib and hasattr(joblib, "load"):
            try:
                if "sklearn" in sys.modules and not hasattr(sys.modules["sklearn"], "__path__"):
                    raise ImportError
                from sklearn.pipeline import Pipeline as SKPipeline  # type: ignore  # noqa: F401
                use_sklearn = True
            except Exception:
                use_sklearn = False

        if use_sklearn:
            try:
                self._load_pipeline()
                return
            except Exception as exc:
                logger.exception("failed to load model %s: %s", self.model_path, exc)
                use_sklearn = False

        logger.warning("scikit-learn unavailable; using simplified Pipeline fallback")
        _define_fallback()
        globals()["_SKLEARN_AVAILABLE"] = False
        if not model_kwargs:
            model_kwargs = {
                "lr": CFG.fallback_lr,
                "iters": CFG.fallback_iters,
                "l2": CFG.fallback_l2,
                "C": CFG.fallback_c,
                "val_steps": CFG.fallback_val_steps,
            }
        self.pipeline = Pipeline(**model_kwargs)
        logger.info("initialized fallback prediction pipeline")
        if CFG.auto_save and joblib and hasattr(joblib, "dump"):
            try:
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                self.pipeline.save(self.model_path)
            except Exception as exc:  # pragma: no cover - persistence optional
                logger.debug("failed to persist fallback model: %s", exc)

    # --------------------------------------------------
    def _load_pipeline(self) -> None:
        path = self.model_path
        if str(path).startswith("http://") or str(path).startswith("https://"):
            path = _fetch_remote_file(str(path))
        obj = _joblib_load_with_retry(path)
        if not hasattr(obj, "predict_proba"):
            raise TypeError("object lacks predict_proba")
        self.pipeline = obj
        self._model_mtime = path.stat().st_mtime
        self._model_hash = _hash_file(path)
        logger.info("loaded prediction model from %s", self.model_path)

    def _maybe_reload(self) -> None:
        path = self.model_path
        if not path.exists() or self._model_mtime is None:
            return
        mtime = path.stat().st_mtime
        if mtime != self._model_mtime or _hash_file(path) != self._model_hash:
            logger.info("model file changed; reloading")
            self._load_pipeline()

    def record_feedback(self, features: IdeaFeatures, actual: bool) -> None:
        self._feedback.append((features.to_dict(), int(actual)))
        if len(self._feedback) >= self._feedback_batch:
            self._check_drift()

    def _check_drift(self) -> None:
        if not self._feedback:
            return
        preds = [
            int(self.pipeline.predict_proba([f])[0][1] >= self.threshold)
            for f, _ in self._feedback
        ]
        y = [lbl for _, lbl in self._feedback]
        acc = sum(int(a == b) for a, b in zip(preds, y)) / len(y)
        if (
            self._last_accuracy is not None
            and self._last_accuracy - acc > self._drift_threshold
        ):
            logger.warning("model drift detected; performing partial retrain")
            self.partial_retrain([f for f, _ in self._feedback], y)
        self._last_accuracy = acc
        self._feedback.clear()

    def partial_retrain(self, X: List[dict], y: List[int]) -> None:
        if hasattr(self.pipeline, "partial_fit"):
            try:
                vecs = (
                    self.pipeline.vectorizer.transform(X)
                    if hasattr(self.pipeline, "vectorizer")
                    else X
                )
                self.pipeline.model.partial_fit(vecs, y)
                if CFG.auto_save:
                    self.pipeline.save(self.model_path)
            except Exception as exc:
                logger.exception("partial retrain failed: %s", exc)

    def predict(self, features: IdeaFeatures) -> Tuple[bool, float]:
        """Return (is_profitable, confidence) for a single idea."""
        try:
            if not hasattr(self, "pipeline"):
                raise RuntimeError("model pipeline is not initialized")
            self._maybe_reload()
            proba = float(self.pipeline.predict_proba([features.to_dict()])[0][1])
        except Exception as exc:  # pragma: no cover - unexpected model failure
            logger.exception("prediction failed: %s", exc)
            if RAISE_ERRORS:
                raise
            return False, 0.0
        return bool(proba >= self.threshold), proba

    def batch_predict(self, ideas: Iterable[IdeaFeatures]) -> List[Tuple[bool, float]]:
        return [self.predict(idea) for idea in ideas]

    def evaluate_enhancement(
        self,
        idea: str,
        rationale: str,
        *,
        context_builder: ContextBuilder,
    ) -> EnhancementEvaluation:
        """Assess an enhancement's impact using NLP heuristics."""
        logger.debug("evaluating enhancement '%s'", idea)
        clean_idea = redact(idea)
        clean_rationale = redact(rationale)
        alerts: List[str] = []

        if context_builder is None:
            raise ValueError("context_builder is required")

        lic_idea = detect_license(clean_idea)
        lic_rat = detect_license(clean_rationale)
        if lic_idea:
            alerts.append(f"license violation in idea: {lic_idea}")
        if lic_rat:
            alerts.append(f"license violation in rationale: {lic_rat}")

        try:
            sentiment = sentiment_score(clean_rationale)
        except Exception as exc:  # pragma: no cover - sentiment failures
            logger.exception("sentiment_score failed: %s", exc)
            sentiment = 0.0
        similarity = 0.0
        if not (lic_idea or lic_rat) and st_util is not None:
            try:
                emb_idea = governed_embed(clean_idea)
                emb_rat = governed_embed(clean_rationale)
                if emb_idea is not None and emb_rat is not None:
                    similarity = float(st_util.cos_sim(emb_idea, emb_rat)[0][0])
            except Exception as exc:  # pragma: no cover - optional dependency
                logger.debug("embedding similarity failed: %s", exc)
                similarity = 0.0
        words = len(clean_rationale.split())
        length_factor = min(words / float(CFG.max_rationale_words), 1.0)
        raw_value = (
            CFG.sentiment_weight * sentiment
            + 0.5 * similarity
            + 0.5 * length_factor
        )
        value = max(-1.0, min(1.0, raw_value))

        if find_semantic_risks is not None:
            try:
                for line, msg, score in find_semantic_risks(clean_rationale.splitlines()):
                    alerts.append(f"{msg}: '{line}' ({score:.2f})")
            except Exception:  # pragma: no cover - best effort analysis
                logger.debug("semantic risk analysis failed", exc_info=True)

        client = getattr(self, "client", None)
        if client is not None:
            prompt = "Evaluate the following enhancement and provide brief feedback."
            mem_ctx = ""
            if self.gpt_memory is not None:
                try:
                    mem_ctx = self.gpt_memory.build_context(
                        "chatgpt_prediction_bot.evaluate_enhancement", limit=5
                    )
                except Exception:
                    mem_ctx = ""
            intent_meta = {"idea": clean_idea, "rationale": clean_rationale}
            if mem_ctx:
                intent_meta["memory_context"] = mem_ctx
            base_tags = [FEEDBACK, IMPROVEMENT_PATH, ERROR_FIX, INSIGHT]
            full_tags = ensure_tags(
                "chatgpt_prediction_bot.evaluate_enhancement", base_tags
            )
            intent_meta.setdefault("intent_tags", list(full_tags))
            try:
                prompt_obj = _build_prediction_prompt(
                    prompt,
                    intent=intent_meta,
                    context_builder=context_builder,
                )
            except Exception as exc:
                if isinstance(exc, PromptBuildError):
                    raise
                handle_failure(
                    "failed to build enhancement evaluation prompt",
                    exc,
                    logger=logger,
                )
            try:
                result = client.generate(
                    prompt_obj,
                    context_builder=context_builder,
                    tags=full_tags,
                )
            except Exception:
                logger.debug("ChatGPT evaluation failed", exc_info=True)
            else:
                text = result.text
                if self.gpt_memory is not None:
                    try:
                        examples = list(getattr(prompt_obj, "examples", []) or [])
                        log_parts = [*examples, prompt_obj.user]
                        log_prompt = "\n\n".join(log_parts)
                        system_msg = getattr(prompt_obj, "system", "")
                        if system_msg:
                            log_prompt = f"{system_msg}\n\n{log_prompt}"
                        self.gpt_memory.log(log_prompt, text, full_tags)
                    except Exception:
                        pass
        return EnhancementEvaluation(
            description=clean_idea, reason=clean_rationale, value=value, alerts=alerts
        )


__all__ = [
    "IdeaFeatures",
    "EnhancementEvaluation",
    "ChatGPTPredictionBot",
]
