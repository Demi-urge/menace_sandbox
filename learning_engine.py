from __future__ import annotations

"""Simple learning engine leveraging stored pathway metadata."""

from typing import List, Tuple, Optional, Any, Dict
import time
import json
import sqlite3
from pathlib import Path
import logging
from db_router import DBRouter, GLOBAL_ROUTER, init_db_router

logger = logging.getLogger(__name__)

from .metrics_exporter import learning_cv_score, learning_holdout_score

try:
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # type: ignore
except Exception:  # pragma: no cover - optional
    LogisticRegression = None  # type: ignore
    GradientBoostingClassifier = None  # type: ignore
    RandomForestClassifier = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover - optional
    torch = None  # type: ignore
    nn = None  # type: ignore


class _SimpleLogReg:
    """Fallback logistic regression when scikit-learn is unavailable."""

    def __init__(self, lr: float = 0.1, iters: int = 100) -> None:
        self.lr = lr
        self.iters = iters
        self.coef_: List[float] | None = None

    def _update(self, vec: List[float], target: int) -> None:
        import math

        if self.coef_ is None:
            self.coef_ = [0.0] * (len(vec) + 1)
        z = sum(w * v for w, v in zip(self.coef_[:-1], vec)) + self.coef_[-1]
        p = 1 / (1 + math.exp(-z))
        err = target - p
        for i in range(len(vec)):
            self.coef_[i] += self.lr * err * vec[i]
        self.coef_[-1] += self.lr * err

    def fit(self, X: List[List[float]], y: List[int]) -> None:
        for _ in range(self.iters):
            for vec, target in zip(X, y):
                self._update(vec, target)

    def partial_fit(self, X: List[List[float]], y: List[int]) -> None:
        for vec, target in zip(X, y):
            self._update(vec, target)

    def predict_proba(self, X: List[List[float]]):
        import math

        res = []
        for vec in X:
            z = sum(w * v for w, v in zip(self.coef_[:-1], vec)) + self.coef_[-1]
            p = 1 / (1 + math.exp(-z))
            res.append([1 - p, p])
        return res


class _SimpleNN:
    """Very small neural net using PyTorch if available."""

    def __init__(self, hidden: int = 8, epochs: int = 50, lr: float = 0.01) -> None:
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.model: Optional[nn.Module] = None
        self.input_dim: Optional[int] = None

    # --------------------------------------------------------------
    def _ensure_model(self, dim: int) -> None:
        if self.model is None:
            if torch is None or nn is None:  # pragma: no cover - optional
                raise RuntimeError("PyTorch not available")
            self.input_dim = dim
            self.model = nn.Sequential(
                nn.Linear(dim, self.hidden),
                nn.ReLU(),
                nn.Linear(self.hidden, 2),
            )
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
            self.loss_fn = nn.CrossEntropyLoss()

    def fit(self, X: List[List[float]], y: List[int]) -> None:
        self._ensure_model(len(X[0]))
        if torch is None:  # pragma: no cover - optional
            return
        x = torch.tensor(X, dtype=torch.float32)
        t = torch.tensor(y, dtype=torch.long)
        for _ in range(self.epochs):
            self.optimizer.zero_grad()
            out = self.model(x)
            loss = self.loss_fn(out, t)
            loss.backward()
            self.optimizer.step()

    def partial_fit(self, X: List[List[float]], y: List[int]) -> None:
        if not X:
            return
        self._ensure_model(len(X[0]))
        if torch is None:  # pragma: no cover - optional
            return
        x = torch.tensor(X, dtype=torch.float32)
        t = torch.tensor(y, dtype=torch.long)
        self.optimizer.zero_grad()
        out = self.model(x)
        loss = self.loss_fn(out, t)
        loss.backward()
        self.optimizer.step()

    def predict_proba(self, X: List[List[float]]):
        if self.model is None or torch is None:  # pragma: no cover - optional
            return [[0.5, 0.5] for _ in X]
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32)
            out = self.model(x)
            prob = torch.softmax(out, dim=1).cpu().numpy().tolist()
        return prob


class SequenceModel:
    """LSTM based sequence classifier with optional pretraining."""

    def __init__(self, hidden: int = 16, epochs: int = 5, lr: float = 0.01, *, pretrain_epochs: int = 0) -> None:
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.pretrain_epochs = pretrain_epochs
        self.vocab: Dict[str, int] = {}
        self.embed: Optional[nn.Embedding] = None
        self.lstm: Optional[nn.LSTM] = None
        self.fc: Optional[nn.Linear] = None
        self.decoder: Optional[nn.Linear] = None

    def _build_vocab(self, texts: List[str]) -> None:
        self.vocab = {"<pad>": 0, "<unk>": 1}
        for t in texts:
            for ch in t:
                if ch not in self.vocab:
                    self.vocab[ch] = len(self.vocab)

    def _encode(self, text: str) -> List[int]:
        return [self.vocab.get(ch, 1) for ch in text]

    def _ensure_model(self) -> None:
        if self.embed is not None:
            return
        if torch is None or nn is None:  # pragma: no cover - optional
            raise RuntimeError("PyTorch not available")
        vocab_size = len(self.vocab)
        self.embed = nn.Embedding(vocab_size, self.hidden)
        self.lstm = nn.LSTM(self.hidden, self.hidden, batch_first=True)
        self.fc = nn.Linear(self.hidden, 2)
        self.decoder = nn.Linear(self.hidden, vocab_size)
        self.optim = torch.optim.Adam(
            list(self.embed.parameters())
            + list(self.lstm.parameters())
            + list(self.fc.parameters()),
            lr=self.lr,
        )
        self.pretrain_optim = torch.optim.Adam(
            list(self.embed.parameters()) + list(self.lstm.parameters()) + list(self.decoder.parameters()),
            lr=self.lr,
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def pretrain(self, texts: List[str]) -> None:
        if self.pretrain_epochs <= 0 or torch is None:
            return
        self._build_vocab(texts)
        self._ensure_model()
        for _ in range(self.pretrain_epochs):
            for text in texts:
                tokens = self._encode(text)
                if len(tokens) < 2:
                    continue
                x = torch.tensor(tokens[:-1], dtype=torch.long).unsqueeze(0)
                y = torch.tensor(tokens[1:], dtype=torch.long)
                self.pretrain_optim.zero_grad()
                emb = self.embed(x)
                out, _ = self.lstm(emb)
                logits = self.decoder(out.squeeze(0))
                loss = self.loss_fn(logits, y)
                loss.backward()
                self.pretrain_optim.step()

    def fit(self, X: List[str], y: List[int]) -> None:
        if torch is None:
            return
        if not self.vocab:
            self._build_vocab(X)
        self._ensure_model()
        if self.pretrain_epochs:
            self.pretrain(X)
        for _ in range(self.epochs):
            for text, label in zip(X, y):
                tokens = self._encode(text)
                if not tokens:
                    continue
                x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
                self.optim.zero_grad()
                emb = self.embed(x)
                out, _ = self.lstm(emb)
                logits = self.fc(out[:, -1, :])
                loss = self.loss_fn(logits, torch.tensor([label]))
                loss.backward()
                self.optim.step()

    def partial_fit(self, X: List[str], y: List[int]) -> None:
        if not X:
            return
        self.fit(X, y)

    def predict_proba(self, X: List[str]):
        if torch is None or self.embed is None:
            return [[0.5, 0.5] for _ in X]
        probs = []
        with torch.no_grad():
            for text in X:
                tokens = self._encode(text)
                if not tokens:
                    probs.append([0.5, 0.5])
                    continue
                x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
                emb = self.embed(x)
                out, _ = self.lstm(emb)
                logits = self.fc(out[:, -1, :])
                p = torch.softmax(logits, dim=1)[0].cpu().numpy().tolist()
                probs.append(p)
        return probs


class AutoEncoderModel:
    """Simple feed-forward autoencoder with classification head."""

    def __init__(self, hidden: int = 8, epochs: int = 50, lr: float = 0.01, *, pretrain_epochs: int = 10) -> None:
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.pretrain_epochs = pretrain_epochs
        self.encoder: Optional[nn.Linear] = None
        self.decoder: Optional[nn.Linear] = None
        self.classifier: Optional[nn.Linear] = None

    def _ensure_model(self, dim: int) -> None:
        if self.encoder is not None:
            return
        if torch is None or nn is None:  # pragma: no cover - optional
            raise RuntimeError("PyTorch not available")
        self.encoder = nn.Linear(dim, self.hidden)
        self.decoder = nn.Linear(self.hidden, dim)
        self.classifier = nn.Linear(self.hidden, 2)
        self.pretrain_optim = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=self.lr,
        )
        self.optim = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.classifier.parameters()),
            lr=self.lr,
        )
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def pretrain(self, X: List[List[float]]) -> None:
        if self.pretrain_epochs <= 0 or torch is None:
            return
        self._ensure_model(len(X[0]))
        x = torch.tensor(X, dtype=torch.float32)
        for _ in range(self.pretrain_epochs):
            self.pretrain_optim.zero_grad()
            encoded = torch.relu(self.encoder(x))
            decoded = self.decoder(encoded)
            loss = self.mse(decoded, x)
            loss.backward()
            self.pretrain_optim.step()

    def fit(self, X: List[List[float]], y: List[int]) -> None:
        if torch is None:
            return
        self._ensure_model(len(X[0]))
        if self.pretrain_epochs:
            self.pretrain(X)
        x = torch.tensor(X, dtype=torch.float32)
        t = torch.tensor(y, dtype=torch.long)
        for _ in range(self.epochs):
            self.optim.zero_grad()
            encoded = torch.relu(self.encoder(x))
            logits = self.classifier(encoded)
            loss = self.ce(logits, t)
            loss.backward()
            self.optim.step()

    def partial_fit(self, X: List[List[float]], y: List[int]) -> None:
        if not X:
            return
        self.fit(X, y)

    def predict_proba(self, X: List[List[float]]):
        if torch is None or self.encoder is None:
            return [[0.5, 0.5] for _ in X]
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32)
            encoded = torch.relu(self.encoder(x))
            logits = self.classifier(encoded)
            prob = torch.softmax(logits, dim=1).cpu().numpy().tolist()
        return prob


class _SimpleTransformer:
    """Tiny wrapper around a HuggingFace transformer model."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        epochs: int = 1,
        lr: float = 1e-4,
        *,
        pretrain_epochs: int = 0,
    ) -> None:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("transformers not available") from exc

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.epochs = epochs
        self.lr = lr
        self.pretrain_epochs = pretrain_epochs

    def pretrain(self, texts: List[str]) -> None:
        if self.pretrain_epochs <= 0:
            return
        import torch

        try:
            from transformers import AutoModelForMaskedLM, DataCollatorForLanguageModeling
        except Exception:
            return

        mlm = AutoModelForMaskedLM.from_pretrained(self.tokenizer.name_or_path)
        collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=True)
        optim = torch.optim.AdamW(mlm.parameters(), lr=self.lr)
        for _ in range(self.pretrain_epochs):
            for text in texts:
                enc = self.tokenizer(text, return_tensors="pt")
                batch = collator([enc])
                out = mlm(**batch, labels=batch["labels"])
                loss = out.loss
                loss.backward()
                optim.step()
                optim.zero_grad()
        try:
            self.model.base_model.load_state_dict(mlm.base_model.state_dict())
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("failed to load pretrained weights: %s", exc)

    def fit(self, X: List[str], y: List[int]) -> None:
        import torch

        self.model.train()
        optim = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            for text, label in zip(X, y):
                enc = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                out = self.model(**enc, labels=torch.tensor([label]))
                loss = out.loss
                loss.backward()
                optim.step()
                optim.zero_grad()

    def partial_fit(self, X: List[str], y: List[int]) -> None:
        if not X:
            return
        self.fit(X, y)

    def predict_proba(self, X: List[str]):
        import torch

        self.model.eval()
        probs = []
        with torch.no_grad():
            for text in X:
                enc = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                out = self.model(**enc)
                p = torch.softmax(out.logits, dim=1)[0].cpu().numpy().tolist()
                probs.append(p)
        return probs

from .neuroplasticity import PathwayDB, PathwayRecord
from .preprocessing_utils import normalize_features, fill_float, encode_outcome
from .menace_memory_manager import MenaceMemoryManager


class LearningEngine:
    """Train a small model to predict pathway success."""

    def __init__(
        self,
        pathway_db: PathwayDB,
        memory_mgr: MenaceMemoryManager,
        model: str | LogisticRegression | GradientBoostingClassifier | None = None,
        *,
        persist_path: str | Path | None = None,
        router: DBRouter | None = None,
    ) -> None:
        self.pathway_db = pathway_db
        self.memory_mgr = memory_mgr
        self.evaluation_history: List[Dict[str, float]] = []
        self.persist_path = Path(persist_path) if persist_path else None
        self._persist_conn: sqlite3.Connection | None = None
        self.router: DBRouter | None = router or GLOBAL_ROUTER
        self._auto_train_stop = False
        if self.persist_path and self.persist_path.suffix not in {".json", ".jsonl"}:
            if not self.router:
                self.router = init_db_router("evaluation", str(self.persist_path), str(self.persist_path))
            try:
                self._persist_conn = self.router.get_connection("evaluation")
                self._persist_conn.execute(
                    "CREATE TABLE IF NOT EXISTS evaluation(ts REAL, cv_score REAL, holdout_score REAL)"
                )
                self._persist_conn.commit()
            except Exception:
                self._persist_conn = None
        if isinstance(model, str):
            if model == "nn":
                self.model = _SimpleNN()
            elif model in {"lstm", "sequence"}:
                self.model = SequenceModel()
            elif model in {"ae", "autoencoder"}:
                self.model = AutoEncoderModel()
            elif model in {"transformer", "bert"}:
                self.model = _SimpleTransformer()
            elif model in {"bert-transfer", "transformer-transfer"}:
                self.model = _SimpleTransformer(pretrain_epochs=1)
            elif model in {"rf", "random_forest"} and RandomForestClassifier:
                self.model = RandomForestClassifier()
            else:
                self.model = _SimpleLogReg()
        elif model is not None:
            self.model = model
        elif GradientBoostingClassifier:
            self.model = GradientBoostingClassifier()
        elif LogisticRegression:
            self.model = LogisticRegression()
        else:
            self.model = _SimpleLogReg()

    # ------------------------------------------------------------------
    def _dataset(self) -> Tuple[List[List[float]], List[int]]:
        cur = self.pathway_db.conn.execute(
            """
            SELECT p.actions, m.frequency, m.avg_exec_time, m.avg_roi, m.myelination_score, m.success_rate
            FROM metadata m JOIN pathways p ON p.id=m.pathway_id
            """
        )
        X: List[List[float]] = []
        y: List[int] = []
        for actions, freq, exec_time, roi, score, success in cur.fetchall():
            if not actions:
                continue
            freq = fill_float(freq)
            exec_time = fill_float(exec_time)
            roi = fill_float(roi)
            score = fill_float(score)
            success = fill_float(success)
            emb = self.memory_mgr._embed(actions)  # type: ignore[attr-defined]
            emb_avg = float(sum(emb) / len(emb)) if emb else 0.0
            X.append([freq, exec_time, roi, score, emb_avg])
            y.append(1 if success >= 0.5 else 0)
        X = normalize_features(X)
        return X, y

    def _text_dataset(self) -> Tuple[List[str], List[int]]:
        cur = self.pathway_db.conn.execute(
            """
            SELECT p.actions, m.success_rate
            FROM metadata m JOIN pathways p ON p.id=m.pathway_id
            """
        )
        texts: List[str] = []
        y: List[int] = []
        for actions, success in cur.fetchall():
            if not actions:
                continue
            success = fill_float(success)
            texts.append(actions)
            y.append(1 if success >= 0.5 else 0)
        return texts, y

    def train(self) -> bool:
        if self.model is None:
            return False
        if isinstance(self.model, (_SimpleTransformer, SequenceModel)):
            texts, y = self._text_dataset()
            if not texts or len(set(y)) < 2:
                return False
            if hasattr(self.model, "pretrain"):
                try:
                    self.model.pretrain(texts)  # type: ignore[attr-defined]
                except Exception as exc:
                    logger.warning("pretrain failed: %s", exc)
            self.model.fit(texts, y)
            return True
        X, y = self._dataset()
        if not X or len(set(y)) < 2:
            return False
        if hasattr(self.model, "pretrain"):
            try:
                self.model.pretrain(X)  # type: ignore[attr-defined]
            except Exception as exc:
                logger.warning("pretrain failed: %s", exc)
        self.model.fit(X, y)
        return True

    def evaluate(self, cv: int = 3, test_split: float = 0.2) -> Dict[str, float]:
        """Cross validate and test the model."""
        if self.model is None:
            result = {"cv_score": 0.0, "holdout_score": 0.0}
            self.evaluation_history.append(result)
            return result
        if isinstance(self.model, (_SimpleTransformer, SequenceModel)):
            texts, y = self._text_dataset()
            if not texts or len(set(y)) < 2:
                result = {"cv_score": 0.0, "holdout_score": 0.0}
                self.evaluation_history.append(result)
                return result
            try:
                split = int(len(texts) * (1 - test_split))
                X_train, X_test = texts[:split], texts[split:]
                y_train, y_test = y[:split], y[split:]
                if hasattr(self.model, "pretrain"):
                    try:
                        self.model.pretrain(X_train)  # type: ignore[attr-defined]
                    except Exception as exc:
                        logger.warning("pretrain failed: %s", exc)
                self.model.fit(X_train, y_train)
                preds = [int(p[1] > 0.5) for p in self.model.predict_proba(X_test)]
                holdout_score = sum(int(a == b) for a, b in zip(preds, y_test)) / len(y_test)
                cv_score = holdout_score
            except Exception:
                cv_score = 0.0
                holdout_score = 0.0
            result = {"cv_score": cv_score, "holdout_score": holdout_score}
            result["timestamp"] = time.time()
            self.evaluation_history.append(result)
            return result
        X, y = self._dataset()
        if not X or len(set(y)) < 2:
            result = {"cv_score": 0.0, "holdout_score": 0.0}
            self.evaluation_history.append(result)
            return result
        try:
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.metrics import accuracy_score

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_split, random_state=42
            )
            self.model.fit(X_train, y_train)
            scores = cross_val_score(self.model, X_train, y_train, cv=cv)
            cv_score = float(scores.mean())
            preds = self.model.predict(X_test)
            holdout_score = float(accuracy_score(y_test, preds))
        except Exception:
            try:
                split = int(len(X) * (1 - test_split))
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]
                self.model.fit(X_train, y_train)
                preds = [int(p[1] > 0.5) for p in self.model.predict_proba(X_test)]
                holdout_score = sum(int(a == b) for a, b in zip(preds, y_test)) / len(y_test)
                cv_score = holdout_score
            except Exception:
                cv_score = 0.0
                holdout_score = 0.0
        result = {"cv_score": cv_score, "holdout_score": holdout_score}
        result["timestamp"] = time.time()
        self.evaluation_history.append(result)
        return result

    # ------------------------------------------------------------------
    def tune_hyperparameters(
        self,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        *,
        search: str = "grid",
        n_iter: int = 10,
        cv: int = 3,
    ) -> Dict[str, Any]:
        """Tune model hyperparameters using scikit-learn search utilities."""
        if self.model is None:
            return {}
        if LogisticRegression is None:
            return {}
        if not isinstance(self.model, (LogisticRegression, GradientBoostingClassifier, RandomForestClassifier)):
            return {}
        X: List[Any]
        y: List[int]
        if isinstance(self.model, (LogisticRegression, GradientBoostingClassifier, RandomForestClassifier)):
            X, y = self._dataset()
        else:
            X, y = [], []
        if not X or len(set(y)) < 2:
            return {}
        if param_grid is None:
            if isinstance(self.model, LogisticRegression):
                param_grid = {"C": [0.01, 0.1, 1.0, 10.0], "max_iter": [100, 200]}
            else:
                param_grid = {
                    "n_estimators": [50, 100],
                    "max_depth": [5, 10],
                }
        try:
            from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

            if search == "random":
                searcher = RandomizedSearchCV(
                    self.model,
                    param_grid,
                    n_iter=n_iter,
                    cv=cv,
                    n_jobs=1,
                )
            else:
                searcher = GridSearchCV(self.model, param_grid, cv=cv, n_jobs=1)
            searcher.fit(X, y)
            self.model = searcher.best_estimator_
            return dict(searcher.best_params_)
        except Exception:
            return {}

    # ------------------------------------------------------------------
    def persist_evaluation(self, result: Dict[str, float]) -> None:
        """Persist *result* to ``self.persist_path`` if configured."""
        if learning_cv_score:
            try:
                learning_cv_score.set(float(result.get("cv_score", 0.0)))
                learning_holdout_score.set(
                    float(result.get("holdout_score", 0.0))
                )
            except Exception as exc:
                logger.warning("metrics export failed: %s", exc)
        if not self.persist_path:
            return
        if self.persist_path.suffix in {".json", ".jsonl"}:
            try:
                with open(self.persist_path, "a", encoding="utf-8") as fh:
                    json.dump(result, fh)
                    fh.write("\n")
            except Exception as exc:
                logger.warning("failed to persist evaluation: %s", exc)
            return
        conn = self._persist_conn
        if conn is None:
            if not self.router:
                self.router = init_db_router("evaluation", str(self.persist_path), str(self.persist_path))
            try:
                conn = self.router.get_connection("evaluation")
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS evaluation(ts REAL, cv_score REAL, holdout_score REAL)"
                )
                self._persist_conn = conn
            except Exception as exc:
                logger.warning("failed to open evaluation database: %s", exc)
                return
        try:
            conn.execute(
                "INSERT INTO evaluation(ts, cv_score, holdout_score) VALUES (?,?,?)",
                (
                    float(result.get("timestamp", time.time())),
                    float(result.get("cv_score", 0.0)),
                    float(result.get("holdout_score", 0.0)),
                ),
            )
            conn.commit()
        except Exception as exc:
            logger.warning("failed to store evaluation: %s", exc)

    # ------------------------------------------------------------------
    def auto_train(self, models: List[str] | None = None, *, eval_interval: float = 0.0) -> str:
        """Try multiple models, tune them and keep the best.

        If *eval_interval* is greater than zero the process repeats forever,
        sleeping for the given number of seconds between cycles and persisting
        each evaluation result via :meth:`persist_evaluation`.
        """

        models = models or ["logreg", "nn", "transformer"]
        cfg_path = (
            (self.persist_path.parent if self.persist_path else Path("."))
            / "best_model.json"
        )
        best_name = ""

        while True:
            if cfg_path.exists():
                try:
                    with open(cfg_path, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    name = data.get("model")
                    params = data.get("params", {})
                    self.__init__(self.pathway_db, self.memory_mgr, model=name, persist_path=self.persist_path)
                    if hasattr(self.model, "set_params"):
                        try:
                            self.model.set_params(**params)
                        except Exception as exc:
                            logger.warning("failed to restore parameters: %s", exc)
                    best_name = str(name)
                except Exception as exc:
                    logger.warning("failed loading saved model config: %s", exc)
            best = ("", -1.0, {}, 0.0)
            saved = self.model
            best_scores: Dict[str, float] = {"cv_score": 0.0, "holdout_score": 0.0, "timestamp": time.time()}
            for name in models:
                try:
                    self.__init__(self.pathway_db, self.memory_mgr, model=name, persist_path=self.persist_path)
                    _params = self.tune_hyperparameters()
                    trained = self.train()
                    if not trained:
                        continue
                    scores = self.evaluate()
                    if scores["cv_score"] > best[1]:
                        best = (name, scores["cv_score"], _params, scores["holdout_score"])
                        saved = self.model
                        best_scores = scores
                except Exception:
                    continue
            self.model = saved
            best_name = best[0]
            try:
                with open(cfg_path, "w", encoding="utf-8") as fh:
                    json.dump(
                        {
                            "model": best[0],
                            "params": best[2],
                            "cv_score": best[1],
                            "holdout_score": best[3],
                        },
                        fh,
                    )
            except Exception as exc:
                logger.warning("failed to save best model: %s", exc)

            if eval_interval > 0:
                try:
                    self.persist_evaluation(best_scores)
                except Exception as exc:
                    logger.warning("failed to persist evaluation: %s", exc)
            if eval_interval <= 0 or self._auto_train_stop:
                break
            try:
                time.sleep(eval_interval)
            except KeyboardInterrupt:
                break

        return best_name

    def stop_auto_train(self) -> None:
        """Signal any running :meth:`auto_train` loop to exit."""
        self._auto_train_stop = True

    def partial_train(self, record: "PathwayRecord") -> bool:
        """Update the model with a single new record."""
        if self.model is None:
            return False
        target = 1 if record.outcome.name.startswith("SUCCESS") else 0
        if isinstance(self.model, (_SimpleTransformer, SequenceModel)):
            try:
                self.model.partial_fit([record.actions], [target])
            except Exception:
                return False
            return True
        emb = self.memory_mgr._embed(record.actions)  # type: ignore[attr-defined]
        emb_avg = float(sum(emb) / len(emb)) if emb else 0.0
        vec = [
            1.0,
            float(record.exec_time),
            float(record.roi),
            1.0,
            emb_avg,
        ]
        if hasattr(self.model, "partial_fit"):
            try:
                self.model.partial_fit([vec], [target])
            except Exception:
                return False
        elif isinstance(self.model, _SimpleLogReg):
            self.model.partial_fit([vec], [target])
        else:
            return False
        return True

    def predict_success(
        self,
        freq: float,
        exec_time: float,
        roi: float,
        score: float,
        text: str = "",
    ) -> float:
        if self.model is None:
            return 0.0
        if isinstance(self.model, (_SimpleTransformer, SequenceModel)):
            prob = self.model.predict_proba([text])[0][1]
            return float(prob)
        emb = self.memory_mgr._embed(text)  # type: ignore[attr-defined]
        emb_avg = float(sum(emb) / len(emb)) if emb else 0.0
        prob = self.model.predict_proba([[freq, exec_time, roi, score, emb_avg]])[0][1]
        return float(prob)

    # ------------------------------------------------------------------
    def performance_trend(self, window: int = 10, *, score_key: str = "holdout_score") -> float:
        """Return simple performance trend over the given window."""
        if not self.persist_path:
            return 0.0
        hist = load_score_history(self.persist_path, router=self.router)
        if len(hist) < 2:
            return 0.0
        hist = hist[-window:]
        if len(hist) < 2:
            return 0.0
        start = float(hist[0].get(score_key, 0.0))
        end = float(hist[-1].get(score_key, 0.0))
        return end - start


def load_score_history(path: str | Path, router: DBRouter | None = None) -> List[Dict[str, float]]:
    """Load persisted evaluation results from *path*."""
    p = Path(path)
    if p.suffix in {".json", ".jsonl"}:
        try:
            with open(p, "r", encoding="utf-8") as fh:
                return [json.loads(line) for line in fh if line.strip()]
        except Exception:
            return []
    rtr = router or GLOBAL_ROUTER or init_db_router("evaluation", str(p), str(p))
    try:
        conn = rtr.get_connection("evaluation")
        cur = conn.execute("SELECT ts, cv_score, holdout_score FROM evaluation ORDER BY ts")
        rows = cur.fetchall()
    except Exception:
        return []
    return [
        {"timestamp": float(ts), "cv_score": float(cv), "holdout_score": float(h)}
        for ts, cv, h in rows
    ]


__all__ = [
    "LearningEngine",
    "_SimpleLogReg",
    "_SimpleNN",
    "SequenceModel",
    "AutoEncoderModel",
    "_SimpleTransformer",
    "tune_hyperparameters",
    "load_score_history",
]
