"""AI Counter Bot for detecting and neutralizing competitor AI systems."""

from __future__ import annotations

from .bot_registry import BotRegistry
from .data_bot import DataBot

from .coding_bot_interface import self_coding_managed
import json
import os
import time
import urllib.request
import joblib
import pickle

registry = BotRegistry()
data_bot = DataBot(start_server=False)

try:  # pragma: no cover - optional dependency
    import duckdb  # type: ignore
except Exception:  # pragma: no cover - optional
    duckdb = None
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Tuple

from dynamic_path_router import resolve_path

import numpy as np
import logging

from db_router import DBRouter, GLOBAL_ROUTER, init_db_router

from .error_flags import RAISE_ERRORS
from .prediction_manager_bot import PredictionManager
from .strategy_prediction_bot import StrategyPredictionBot

logger = logging.getLogger(__name__)

try:
    from sklearn.linear_model import LogisticRegression  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    LogisticRegression = None  # type: ignore
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    TfidfVectorizer = None  # type: ignore


def _fetch_remote_list(url: str) -> list[str]:
    """Fetch a list of strings from a remote JSON or newline file."""
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = resp.read().decode("utf-8")
        try:
            parsed = json.loads(data)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except Exception:
            pass
        return [ln.strip() for ln in data.splitlines() if ln.strip()]
    except Exception as exc:  # pragma: no cover - optional
        logger.warning("failed to fetch %s: %s", url, exc)
        return []


def _parse_list(val: str | None) -> list[str]:
    if not val:
        return []
    try:
        data = json.loads(val)
        if isinstance(data, list):
            return [str(v).strip() for v in data if str(v).strip()]
    except Exception:
        pass
    return [v.strip() for v in val.split(',') if v.strip()]


def _load_keywords() -> tuple[list[str], list[str]]:
    kw_file = os.getenv("AI_COUNTER_KEYWORDS_FILE")
    kw_env = os.getenv("AI_COUNTER_KEYWORDS")
    kw_url = os.getenv("AI_COUNTER_KEYWORDS_URL")
    ext_file = os.getenv("AI_COUNTER_EXTENDED_FILE")
    ext_env = os.getenv("AI_COUNTER_EXTENDED")
    ext_url = os.getenv("AI_COUNTER_EXTENDED_URL")
    defaults_file = os.getenv("AI_COUNTER_DEFAULTS") or resolve_path(
        "config/default_keywords.json"
    )

    keywords: list[str] = []
    if kw_env:
        keywords = _parse_list(kw_env)
    elif kw_file and Path(kw_file).exists():
        try:
            keywords = [
                ln.strip()
                for ln in Path(kw_file).read_text().splitlines()
                if ln.strip()
            ]
        except Exception as exc:  # pragma: no cover - optional
            logger.warning("failed to load keyword file %s: %s", kw_file, exc)
    elif kw_url:
        keywords = _fetch_remote_list(kw_url)
    if not keywords and Path(defaults_file).exists():
        try:
            data = json.loads(Path(defaults_file).read_text())
            if isinstance(data, dict):
                keywords = [str(v).strip() for v in data.get("keywords", []) if str(v).strip()]
        except Exception as exc:  # pragma: no cover - optional
            logger.warning("failed to load default keywords %s: %s", defaults_file, exc)
    if not keywords and kw_url:
        keywords = _fetch_remote_list(kw_url)
    if not keywords:
        keywords = ["gpt", "automation", "bot", "machine learning", "ai", "neural"]

    ext: list[str] = []
    if ext_env:
        ext = _parse_list(ext_env)
    elif ext_file and Path(ext_file).exists():
        try:
            ext = [
                ln.strip()
                for ln in Path(ext_file).read_text().splitlines()
                if ln.strip()
            ]
        except Exception as exc:  # pragma: no cover - optional
            logger.warning("failed to load extended keyword file %s: %s", ext_file, exc)
    elif ext_url:
        ext = _fetch_remote_list(ext_url)
    if not ext and Path(defaults_file).exists():
        try:
            data = json.loads(Path(defaults_file).read_text())
            if isinstance(data, dict):
                ext = [str(v).strip() for v in data.get("extended", []) if str(v).strip()]
        except Exception as exc:  # pragma: no cover - optional
            logger.warning("failed to load default extended keywords %s: %s", defaults_file, exc)
    if not ext and ext_url:
        ext = _fetch_remote_list(ext_url)
    if not ext:
        ext = ["deep learning", "neural network", "chatgpt", "openai"]

    return keywords, ext


_AI_KEYWORDS, _EXTENDED_SIGNS = _load_keywords()

_MIN_TRAIN_SAMPLES = int(os.getenv("AI_MIN_TRAIN_SAMPLES", "2"))

# basic labelled outputs for reverse engineering
_REVERSE_MODEL = None


def _load_reverse_train() -> list[tuple[str, str]]:
    data_env = os.getenv("AI_REVERSE_TRAIN_DATA")
    data_url = os.getenv("AI_REVERSE_TRAIN_URL")
    if data_env:
        try:
            if Path(data_env).exists():
                return [tuple(item) for item in json.loads(Path(data_env).read_text())]
            return [tuple(item) for item in json.loads(data_env)]
        except Exception as exc:  # pragma: no cover - optional
            logger.warning("failed to parse AI_REVERSE_TRAIN_DATA: %s", exc)
    if data_url:
        fetched = _fetch_remote_list(data_url)
        if fetched:
            pairs: list[tuple[str, str]] = []
            for item in fetched:
                if isinstance(item, str) and "," in item:
                    text, label = item.split(",", 1)
                    pairs.append((text.strip(), label.strip()))
            if pairs:
                return pairs
    default_path = resolve_path("config/reverse_train_data.json")
    if Path(default_path).exists():
        try:
            return [tuple(item) for item in json.loads(Path(default_path).read_text())]
        except Exception as exc:  # pragma: no cover - optional
            logger.warning("failed to load reverse train data %s: %s", default_path, exc)
    return [
        ("training model update accuracy 0.9", "learning"),
        ("random noise generation entropy", "stochastic"),
        ("schedule 9am daily run", "scheduled"),
        ("static rule triggered", "rule-based"),
    ]


def _sanitize_training_data(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Validate reverse training pairs to mitigate poisoning."""
    allowed = {"learning", "stochastic", "scheduled", "rule-based"}
    out: list[tuple[str, str]] = []
    for text, label in pairs:
        text = str(text).strip()
        label = str(label).strip()
        if not text or label not in allowed:
            continue
        counts = Counter(text)
        total = len(text)
        ent = -sum((c / total) * np.log2(c / total) for c in counts.values())
        if ent < 1.0 or ent > 8.0:
            continue
        out.append((text, label))
    return out


_REVERSE_TRAIN: list[tuple[str, str]] = _sanitize_training_data(_load_reverse_train())
if len(_REVERSE_TRAIN) < 4:
    logger.warning(
        "reverse training data insufficient: %s samples", len(_REVERSE_TRAIN)
    )
elif len(_REVERSE_TRAIN) < 10:
    logger.info(
        "reverse training data small: %s samples may reduce accuracy",
        len(_REVERSE_TRAIN),
    )
_TFIDF = None
if LogisticRegression is not None:
    _REVERSE_MODEL = LogisticRegression(multi_class="auto", max_iter=100)
    if TfidfVectorizer is not None:
        try:
            _TFIDF = TfidfVectorizer()
            texts = [t for t, _ in _REVERSE_TRAIN]
            labels = [l for _, l in _REVERSE_TRAIN]
            X = _TFIDF.fit_transform(texts)
            _REVERSE_MODEL.fit(X, labels)
        except Exception:
            _REVERSE_MODEL = None
            _TFIDF = None
    else:
        X: list[list[float]] = []
        y: list[str] = []
        for text, label in _REVERSE_TRAIN:
            toks = text.lower().split()
            length = len(toks)
            rand = sum(1 for w in toks if w in {"random", "noise", "entropy", "rand"}) / max(length, 1)  # noqa: E501
            sched = sum(1 for w in toks if w.endswith("am") or w.endswith("pm") or w == "utc" or "schedule" in w or "daily" in w or "every" in w) / max(length, 1)  # noqa: E501
            learn = sum(1 for w in toks if w in {"learn", "training", "model", "update", "loss", "accuracy"}) / max(length, 1)  # noqa: E501
            rep = 0.0
            num = sum(1 for t in toks if t.isdigit()) / max(length, 1)
            avg_word_len = sum(len(t) for t in toks) / max(length, 1)
            upper_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
            X.append([rand, sched, learn, rep, num, avg_word_len / 10.0, upper_ratio, length / 1000.0])  # noqa: E501
            y.append(label)
        try:  # pragma: no cover - optional fitting
            _REVERSE_MODEL.fit(X, y)
        except Exception:
            _REVERSE_MODEL = None


def _load_threat_config() -> dict:
    cfg_env = os.getenv("AI_THREAT_WEIGHTS")
    cfg_file = os.getenv("AI_THREAT_WEIGHTS_FILE")
    cfg = {
        "adaptation": 50,
        "similarity": 30,
        "bonus": {"learning": 20, "stochastic": 10, "scheduled": 5},
    }
    data = None
    if cfg_env:
        try:
            data = json.loads(cfg_env)
        except Exception as exc:  # pragma: no cover - optional
            logger.warning("failed to parse AI_THREAT_WEIGHTS: %s", exc)
    elif cfg_file and Path(cfg_file).exists():
        try:
            data = json.loads(Path(cfg_file).read_text())
        except Exception as exc:  # pragma: no cover - optional
            logger.warning("failed to load threat weights %s: %s", cfg_file, exc)
    if isinstance(data, dict):
        cfg.update({k: data.get(k, v) for k, v in cfg.items() if k != "bonus"})
        bonus = data.get("bonus")
        if isinstance(bonus, dict):
            cfg["bonus"].update(bonus)
    return cfg


_THREAT_WEIGHTS = _load_threat_config()


@dataclass
class TrafficSample:
    """Observable features from competitor activity."""

    pattern: str
    frequency: int
    timing_std: float
    similarity: float

    def to_vector(self) -> List[float]:
        return [float(self.frequency), self.timing_std, self.similarity]


class CounterDB:
    """Lightweight log of counter operations backed by SQLite or DuckDB."""

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        engine: str | None = None,
        router: DBRouter | None = None,
    ) -> None:
        self.path = Path(
            resolve_path(path or os.getenv("AI_COUNTER_DB", "ai_counter.db"))
        )
        eng = engine or os.getenv("AI_COUNTER_DB_ENGINE", "sqlite").lower()
        self.engine = "duckdb" if eng == "duckdb" and duckdb is not None else "sqlite"
        self.router = router or GLOBAL_ROUTER or init_db_router("ai_counter")
        self._init()

    def _init(self) -> None:
        if self.engine == "duckdb" and duckdb is not None:
            conn = duckdb.connect(str(self.path))
        else:
            conn = self.router.get_connection("events")
        with conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT,
                    detected INTEGER,
                    algorithm TEXT,
                    counter TEXT
                )
                """
            )

    def add(self, text: str, detected: bool, algorithm: str, counter: str) -> int:
        if self.engine == "duckdb" and duckdb is not None:
            conn = duckdb.connect(str(self.path))
        else:
            conn = self.router.get_connection("events")
        with conn:
            cur = conn.execute(
                "INSERT INTO events(text, detected, algorithm, counter) VALUES(?,?,?,?)",
                (text, int(detected), algorithm, counter),
            )
        return int(cur.lastrowid)

    def fetch(self) -> List[Tuple[str, bool, str, str]]:
        if self.engine == "duckdb" and duckdb is not None:
            conn = duckdb.connect(str(self.path))
        else:
            conn = self.router.get_connection("events")
        with conn:
            rows = conn.execute(
                "SELECT text, detected, algorithm, counter FROM events ORDER BY id"
            ).fetchall()
        return [(r[0], bool(r[1]), r[2], r[3]) for r in rows]

    def purge(self, max_entries: int = 1000) -> None:
        """Remove old entries beyond ``max_entries`` to keep DB size stable."""
        try:
            if self.engine == "duckdb" and duckdb is not None:
                conn = duckdb.connect(str(self.path))
            else:
                conn = self.router.get_connection("events")
            with conn:
                ids = conn.execute(
                    "SELECT id FROM events ORDER BY id DESC LIMIT ?",
                    (max_entries,),
                ).fetchall()
                if not ids:
                    return
                min_id = ids[-1][0]
                conn.execute("DELETE FROM events WHERE id < ?", (min_id,))
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("purge failed: %s", exc)


def detect_ai_presence(text: str) -> bool:
    """Heuristically detect AI references or synthetic patterns."""
    lowered = text.lower()
    keyword_hits = sum(1 for k in _AI_KEYWORDS if k in lowered)
    extra_hits = sum(1 for k in _EXTENDED_SIGNS if k in lowered)
    digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
    return keyword_hits + extra_hits > 0 or digit_ratio > 0.05


def reverse_engineer(outputs: Iterable[str]) -> str:
    """Guess competitor algorithm using lightweight classification."""
    outputs = list(outputs)
    if not outputs or not any(o.strip() for o in outputs):
        return "noop"
    data = " ".join(outputs).lower()

    tokens = data.split()
    length = len(tokens)

    randomness_hits = sum(
        1 for w in tokens if w in {"random", "noise", "entropy", "rand"}
    )
    randomness_score = randomness_hits / max(length, 1)

    schedule_hits = 0
    for word in tokens:
        if word.endswith("am") or word.endswith("pm") or word == "utc":
            schedule_hits += 1
        if "schedule" in word or "daily" in word or "every" in word:
            schedule_hits += 1
    schedule_score = schedule_hits / max(length, 1)

    learning_hits = sum(
        1
        for w in tokens
        if w in {"learn", "training", "model", "update", "loss", "accuracy"}
    )
    learning_score = learning_hits / max(length, 1)

    repetition = sum(
        1 for i in range(1, len(outputs)) if outputs[i].strip() == outputs[i - 1].strip()
    ) / max(len(outputs), 1)

    numeric_density = sum(1 for t in tokens if t.isdigit()) / max(length, 1)
    avg_word_len = sum(len(t) for t in tokens) / max(length, 1)
    upper_ratio = sum(1 for c in data if c.isupper()) / max(len(data), 1)

    features = np.array(
        [
            randomness_score,
            schedule_score,
            learning_score,
            repetition,
            numeric_density,
            avg_word_len / 10.0,
            upper_ratio,
            length / 1000.0,
        ],
        dtype=float,
    )

    if _REVERSE_MODEL is not None:
        try:
            if _TFIDF is not None:
                vec = _TFIDF.transform([data])
                pred = _REVERSE_MODEL.predict(vec)[0]
            else:
                pred = _REVERSE_MODEL.predict([features])[0]
            return str(pred)
        except Exception as exc:  # pragma: no cover - prediction failure
            logger.warning("reverse model failed: %s", exc)

    scores = {
        "learning": learning_score + numeric_density + length / 200.0,
        "stochastic": randomness_score + (1.0 - repetition),
        "scheduled": schedule_score + repetition,
        "rule-based": 0.1 + repetition / 2.0,
    }
    return max(scores, key=scores.get)


def choose_countermeasure(algorithm: str, probability: float | None = None) -> str:
    """Pick countermeasure based on estimated algorithm and risk."""
    prob = probability if probability is not None else 0.5
    if algorithm == "learning":
        if prob >= 0.75:
            return "model inversion"
        if prob >= 0.5:
            return "data poisoning"
        return "monitor updates"
    if algorithm == "scheduled":
        return "misdirect timing" if prob >= 0.6 else "schedule interference"
    if algorithm == "stochastic":
        return "noise injection" if prob >= 0.6 else "increase randomness"
    return "counter flood"


@self_coding_managed(bot_registry=registry, data_bot=data_bot)
class AICounterBot:
    """Analyse competitor activity and plan counter actions."""

    prediction_profile = {"scope": ["ai"], "risk": ["medium"]}

    def __init__(
        self,
        db: CounterDB | None = None,
        *,
        prediction_manager: "PredictionManager" | None = None,
        strategy_bot: "StrategyPredictionBot" | None = None,
        bot_weights: dict[str, float] | None = None,
        model_path: str | None = None,
        manager: "SelfCodingManager | None" = None,
    ) -> None:
        self.db = db or CounterDB()
        self.prediction_manager = prediction_manager
        self.strategy_bot = strategy_bot
        self.bot_weights = bot_weights or {}
        self.assigned_prediction_bots = []
        if self.prediction_manager:
            try:
                self.assigned_prediction_bots = self.prediction_manager.assign_prediction_bots(self)
            except Exception as exc:
                logger.exception("Failed to assign prediction bots: %s", exc)
        self.model_path = Path(
            model_path or os.getenv("AI_COUNTER_MODEL_FILE", "ai_counter_model.pkl")
        )
        self.export_path = Path(
            os.getenv("AI_COUNTER_EXPORT_FILE", "models/ai_classifier.pkl")
        )
        if LogisticRegression is not None:
            if self.model_path.exists():
                try:
                    self.model = joblib.load(self.model_path)
                except Exception as exc:  # pragma: no cover - optional
                    logger.warning("failed to load model %s: %s", self.model_path, exc)
                    self.model = LogisticRegression(random_state=0)
            else:
                self.model = LogisticRegression(random_state=0)
        else:  # pragma: no cover - fallback
            self.model = None
            logger.info("LogisticRegression unavailable; using heuristic model")
        self._scale_mean: np.ndarray | None = None
        self._scale_std: np.ndarray | None = None

    def log_line(self, text: str, detected: bool, algorithm: str, counter: str) -> str:
        """Format a log line for storage."""
        return f"{text} | {detected} | {algorithm} | {counter}"

    def escalate_if_needed(self, score: int, text: str | None = None) -> None:
        """Trigger escalation if the threat score is high."""
        threshold = int(os.getenv("AI_COUNTER_ESCALATE", "80"))
        if score > threshold:
            msg = f"Threat score {score} exceeds threshold"
            if text:
                msg += f": {text[:50]}"
            try:
                proto_path = os.getenv("AI_ESCALATION_PROTOCOL", "escalation_protocol")
                proto = __import__(proto_path).escalation_protocol  # type: ignore[attr-defined]
                getattr(proto, "escalate", lambda m, attachments=None: None)(msg)
            except Exception:  # pragma: no cover - best effort
                logger.warning("failed to escalate threat")

    def _export_model(self) -> None:
        """Persist model to export path."""
        if self.model is None:
            return
        try:
            self.export_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.export_path, "wb") as f:
                pickle.dump(self.model, f)
        except Exception as exc:  # pragma: no cover - optional
            logger.warning("failed to export model %s: %s", self.export_path, exc)

    def train_predictor(
        self, samples: Iterable[TrafficSample], labels: Iterable[int]
    ) -> None:
        """Train adaptation prediction model."""
        if self.model is None:
            return
        samples = list(samples)
        labels = list(labels)
        if len(samples) < _MIN_TRAIN_SAMPLES:
            logger.warning("not enough training samples: %s", len(samples))
            return
        X = np.array([s.to_vector() for s in samples], dtype=float)
        y = np.array(labels)
        self._scale_mean = X.mean(axis=0)
        self._scale_std = X.std(axis=0) + 1e-9
        X_norm = (X - self._scale_mean) / self._scale_std
        self.model.fit(X_norm, y)
        try:
            joblib.dump(self.model, self.model_path)
            self._export_model()
        except Exception as exc:  # pragma: no cover - optional
            logger.warning("failed to persist model %s: %s", self.model_path, exc)

    def _apply_prediction_bots(
        self, base: float, sample: TrafficSample
    ) -> float:
        """Combine predictions from assigned prediction bots."""
        if not self.prediction_manager:
            return base
        predictions: list[float] = [float(base)]
        weights: list[float] = [1.0]
        for bot_id in self.assigned_prediction_bots:
            entry = self.prediction_manager.registry.get(bot_id)
            if not entry or not entry.bot:
                continue
            pred = getattr(entry.bot, "predict", None)
            if not callable(pred):
                continue
            weight = self.bot_weights.get(bot_id, getattr(entry.bot, "confidence", 1.0))
            try:
                other = pred(sample.to_vector())
                if isinstance(other, (list, tuple)):
                    other = other[0]
                predictions.append(float(other))
                weights.append(float(weight))
            except Exception:
                logger.exception("prediction bot %s failed", type(entry.bot).__name__)
                continue
        total = sum(weights)
        if total == 0:
            return float(base)
        norm_weights = [w / total for w in weights]
        norm_probs = sum(p * w for p, w in zip(predictions, norm_weights))
        return float(norm_probs)

    def predict_adaptation(self, sample: TrafficSample) -> float:
        """Predict probability competitors adapt."""
        if self.model is None:
            logger.info("heuristic adaptation prediction in use")
            history = 0
            if self.db:
                try:
                    history = sum(1 for _, _, alg, _ in self.db.fetch() if alg == sample.pattern)
                except Exception:  # pragma: no cover - DB access issues
                    history = 0
            ent = 0.0
            if sample.pattern:
                counts = Counter(sample.pattern)
                total = len(sample.pattern)
                ent = -sum((c / total) * np.log2(c / total) for c in counts.values()) / max(np.log2(total), 1e-9)  # noqa: E501
            score = (
                0.05 * sample.frequency
                + 0.5 * sample.similarity
                - 0.1 * sample.timing_std
                + 0.02 * history
                + 0.1 * ent
            )
            prob = float(1.0 / (1.0 + np.exp(-score)))
        else:
            vec = np.array(sample.to_vector(), dtype=float)
            if self._scale_mean is not None:
                vec = (vec - self._scale_mean) / self._scale_std
            prob = float(self.model.predict_proba([vec])[0][1])
        if self.prediction_manager:
            prob = self._apply_prediction_bots(prob, sample)
        return float(prob)

    def calculate_threat_score(
        self, algorithm: str, adaptation_prob: float, similarity: float
    ) -> int:
        """Compute threat score on a 0-100 scale."""
        sample = TrafficSample(
            pattern=algorithm, frequency=0, timing_std=0.0, similarity=similarity
        )
        return self.threat_score(sample, algorithm, adaptation_prob)

    def threat_score(self, sample: TrafficSample, algorithm: str, adaptation_prob: float) -> int:
        """Centralized threat scoring."""
        score = adaptation_prob * float(_THREAT_WEIGHTS.get("adaptation", 50))
        score += sample.similarity * float(_THREAT_WEIGHTS.get("similarity", 30))
        bonus_cfg = _THREAT_WEIGHTS.get("bonus", {})
        alg_bonus = float(bonus_cfg.get(algorithm, 0))
        history = 0
        if self.db:
            try:
                history = sum(1 for _, _, alg, _ in self.db.fetch() if alg == algorithm)
            except Exception:  # pragma: no cover - DB access issues
                history = 0
        total = score + alg_bonus + min(history, 10)
        return int(max(0, min(total, 100)))

    def analyse(self, outputs: Iterable[str]) -> Tuple[bool, str, str]:
        """Analyse outputs and store counter decision."""
        outputs = list(outputs)
        if not outputs or not any(o.strip() for o in outputs):
            logger.info("empty outputs, skipping analysis")
            return False, "noop", "noop"
        max_items = int(os.getenv("AI_COUNTER_MAX_OUTPUTS", "100"))
        max_text = int(os.getenv("AI_COUNTER_MAX_TEXT", "10000"))
        if len(outputs) > max_items:
            raise ValueError("too many outputs to analyse")
        text = " ".join(outputs)
        if len(text) > max_text:
            raise ValueError("input text too large")
        detected = detect_ai_presence(text)
        algorithm = reverse_engineer(outputs)
        sample = TrafficSample(
            pattern=algorithm,
            frequency=len(outputs),
            timing_std=0.0,
            similarity=1.0 if detected else 0.0,
        )
        adapt_prob = self.predict_adaptation(sample)
        threat = self.threat_score(sample, algorithm, adapt_prob)
        counter = choose_countermeasure(algorithm, adapt_prob)
        if self.prediction_manager:
            det_prob = self._apply_prediction_bots(float(detected), sample)
            detected = det_prob >= 0.5
        for attempt in range(3):
            try:
                self.db.add(text, detected, algorithm, counter)
                break
            except Exception as exc:  # pragma: no cover - DB issues
                logger.warning("db write failed: %s", exc)
                time.sleep(0.1 * (2 ** attempt))
        else:
            try:
                day = time.strftime("%Y-%m-%d")
                path = resolve_path("logs/fallback") / f"{day}.txt"
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a", encoding="utf-8") as f:
                    f.write(self.log_line(text, detected, algorithm, counter) + "\n")
            except Exception as exc:
                logger.warning("fallback log failed: %s", exc)
        logger.info("Threat score: %s", threat)
        try:
            ts_path = resolve_path("logs/threat_scores.log")
            ts_path.parent.mkdir(parents=True, exist_ok=True)
            with open(ts_path, "a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} | {threat} | {text[:50]}...\n")  # noqa: E501
        except Exception:  # pragma: no cover - best effort
            logger.debug("failed to log threat score")
        self.escalate_if_needed(threat, text)
        if self.strategy_bot:
            try:
                self.strategy_bot.receive_ai_competition(sample)
            except Exception as exc:
                logger.exception("strategy bot failed: %s", exc)
                if RAISE_ERRORS:
                    raise
        return detected, algorithm, counter


__all__ = [
    "TrafficSample",
    "CounterDB",
    "detect_ai_presence",
    "reverse_engineer",
    "choose_countermeasure",
    "AICounterBot",
]
if TYPE_CHECKING:  # pragma: no cover - typing helper
    from .self_coding_manager import SelfCodingManager
else:  # pragma: no cover - runtime fallback when manager is unused
    SelfCodingManager = object  # type: ignore[assignment]