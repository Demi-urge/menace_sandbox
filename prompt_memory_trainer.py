import json
import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Tuple
import sqlite3
import time

from dynamic_path_router import resolve_path

try:  # pragma: no cover - optional dependency
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:  # pragma: no cover - allow running without dependency
    SentimentIntensityAnalyzer = None  # type: ignore

from code_database import PatchHistoryDB
from db_router import init_db_router

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from menace_sandbox.gpt_memory import GPTMemoryManager

ROOT_DIR = resolve_path(".")

class PromptMemoryTrainer:
    STYLE_VERSION = 1
    """Analyse historical prompts and patch outcomes to learn formatting.

    The trainer reads prompts from :class:`GPTMemoryManager` and correlates
    them with patch outcomes stored in :class:`PatchHistoryDB`.  Sentiment
    analysis classifies tone while heuristics extract style features – headers,
    example order, bullet lists, code blocks and explicit ``System:`` / ``User``
    sections.  Success rates are aggregated per style using ROI or patch
    complexity improvement as weights, exposing the resulting metrics via
    :attr:`style_weights`.
    """

    def __init__(
        self,
        *,
        memory: "GPTMemoryManager | None" = None,
        patch_db: PatchHistoryDB | None = None,
        state_path: str | Path | None = ROOT_DIR / "prompt_style_weights.json",
        db_path: str | Path | None = ROOT_DIR / "prompt_styles.db",
    ) -> None:
        if memory is None:
            from menace_sandbox import gpt_memory

            memory = gpt_memory.GPTMemoryManager(db_path=":memory:")
        self.memory = memory
        self.patch_db = patch_db or PatchHistoryDB(":memory:")
        self.state_path = None
        if state_path:
            p = Path(state_path)
            self.state_path = p if p.is_absolute() else ROOT_DIR / p
        self.db_path = None
        if db_path and str(db_path) != ":memory":
            p = Path(db_path)
            self.db_path = p if p.is_absolute() else ROOT_DIR / p
        self._sentiment = (
            SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None
        )
        self.style_weights: Dict[str, Dict[str, float]] = {}
        self._stats: Dict[str, Dict[str, list[int]]] = {
            "headers": defaultdict(lambda: [0, 0]),
            "example_order": defaultdict(lambda: [0, 0]),
            "tone": defaultdict(lambda: [0, 0]),
            "has_bullets": defaultdict(lambda: [0, 0]),
            "has_code": defaultdict(lambda: [0, 0]),
            "example_count": defaultdict(lambda: [0, 0]),
        }

        # optional JSON weight loading for backwards compatibility
        if self.state_path and self.state_path.exists():
            try:
                with self.state_path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if (
                    isinstance(data, dict)
                    and data.get("version") == self.STYLE_VERSION
                    and isinstance(data.get("weights"), dict)
                ):
                    self.style_weights = data["weights"]
            except Exception:
                self.style_weights = {}

        # initialise sqlite persistence
        db_str = str(self.db_path) if self.db_path else ":memory:"
        self.router = init_db_router("prompt_styles", db_str, db_str)
        self._db = self.router.local_conn
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS style_stats(
                feature TEXT NOT NULL,
                value TEXT NOT NULL,
                success REAL NOT NULL,
                total REAL NOT NULL,
                PRIMARY KEY(feature, value)
            )
            """
        )
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS prompt_records(
                ts REAL,
                tone TEXT,
                headers TEXT,
                example_order TEXT,
                has_bullets INTEGER,
                has_code INTEGER,
                example_count INTEGER,
                success INTEGER,
                weight REAL
            )
            """
        )
        self._db.commit()

        # load existing stats from DB
        cur = self._db.execute(
            "SELECT feature, value, success, total FROM style_stats"
        )
        for feat, val, suc, tot in cur.fetchall():
            if feat not in self._stats:
                self._stats[feat] = defaultdict(lambda: [0, 0])
            self._stats[feat][val] = [int(suc), int(tot)]
            if tot:
                self.style_weights.setdefault(feat, {})[val] = suc / tot

        for feat, mapping in self.style_weights.items():
            if feat in self._stats:
                for key, val in mapping.items():
                    self._stats[feat].setdefault(key, [int(round(val)), 1])

    # ------------------------------------------------------------------
    def _extract_style(self, prompt: str) -> Dict[str, Any]:
        """Return formatting features for ``prompt``.

        Besides the original header and example ordering cues, the method now
        applies sentiment analysis for tone and records whether the prompt
        contains fenced code blocks, bullet lists and ``System``/``User``
        sections.  The analyser also looks for common structured sections such
        as "Constraints" or "Resources", counts the number of examples and
        their placement and classifies the overall verbosity of the prompt.
        """

        headers = re.findall(r"^#+\s*(.+)$", prompt, flags=re.MULTILINE)
        example_matches = list(
            re.finditer(r"Example\s*([\w-]+)", prompt, flags=re.IGNORECASE)
        )
        example_order = [m.group(1) for m in example_matches]
        tone = "neutral"
        if self._sentiment:
            score = self._sentiment.polarity_scores(prompt)["compound"]
            if score > 0.05:
                tone = "positive"
            elif score < -0.05:
                tone = "negative"
        has_code = bool(re.search(r"```.+?```", prompt, flags=re.DOTALL))
        has_bullets = bool(
            re.search(r"^\s*(?:[-*]|\d+\.)\s+", prompt, flags=re.MULTILINE)
        )
        sections = [
            m.group(1).lower()
            for m in re.finditer(r"^(System|User):", prompt, flags=re.MULTILINE)
        ]
        structured_sections = [
            name.lower()
            for name in ("Constraints", "Resources")
            if re.search(
                rf"^(?:#+\s*)?{name}\s*:",
                prompt,
                flags=re.IGNORECASE | re.MULTILINE,
            )
        ]
        example_count = len(example_matches)
        if example_count:
            first_line = prompt[: example_matches[0].start()].count("\n")
            total_lines = prompt.count("\n") + 1
            ratio = first_line / max(total_lines, 1)
            if ratio < 0.33:
                example_place = "start"
            elif ratio < 0.66:
                example_place = "middle"
            else:
                example_place = "end"
        else:
            example_place = "none"
        word_count = len(re.findall(r"\w+", prompt))
        if word_count < 50:
            length = "short"
        elif word_count < 150:
            length = "medium"
        else:
            length = "long"
        return {
            "headers": headers,
            "example_order": example_order,
            "tone": tone,
            "has_code": has_code,
            "has_bullets": has_bullets,
            "sections": sections,
            "structured_sections": structured_sections,
            "example_count": example_count,
            "example_placement": example_place,
            "length": length,
        }

    # ------------------------------------------------------------------
    def train(self) -> Dict[str, Dict[str, float]]:
        """Compute success rates for observed prompt styles.

        Each observation is weighted by either ROI improvement or, when ROI is
        unchanged, the reduction in patch complexity.  This emphasises styles
        associated with more impactful patches.
        """

        stats: Dict[str, Dict[str, Mapping[str, float]]] = {
            "headers": defaultdict(lambda: {"success": 0.0, "weight": 0.0}),
            "example_order": defaultdict(lambda: {"success": 0.0, "weight": 0.0}),
            "tone": defaultdict(lambda: {"success": 0.0, "weight": 0.0}),
            "has_code": defaultdict(lambda: {"success": 0.0, "weight": 0.0}),
            "has_bullets": defaultdict(lambda: {"success": 0.0, "weight": 0.0}),
            "sections": defaultdict(lambda: {"success": 0.0, "weight": 0.0}),
            "structured_sections": defaultdict(
                lambda: {"success": 0.0, "weight": 0.0}
            ),
            "example_count": defaultdict(lambda: {"success": 0.0, "weight": 0.0}),
            "example_placement": defaultdict(
                lambda: {"success": 0.0, "weight": 0.0}
            ),
            "length": defaultdict(lambda: {"success": 0.0, "weight": 0.0}),
        }

        cur = self.memory.conn.execute("SELECT prompt FROM interactions")
        for (text,) in cur.fetchall():
            match = re.search(r"PATCH:(\d+)", text)
            if not match:
                continue
            patch_id = int(match.group(1))
            try:
                row = self.patch_db.conn.execute(
                    (
                        "SELECT outcome, roi_before, roi_after, "
                        "complexity_before, complexity_after, "
                        "prompt_headers, prompt_order, prompt_tone "
                        "FROM patch_history WHERE id=?"
                    ),
                    (patch_id,),
                ).fetchone()
                if not row:
                    continue
                outcome, roi_before, roi_after, c_before, c_after, p_headers, p_order, p_tone = row
            except sqlite3.Error:
                row = self.patch_db.conn.execute(
                    (
                        "SELECT outcome, roi_before, roi_after, "
                        "complexity_before, complexity_after "
                        "FROM patch_history WHERE id=?"
                    ),
                    (patch_id,),
                ).fetchone()
                if not row:
                    continue
                outcome, roi_before, roi_after, c_before, c_after = row
                p_headers = p_order = p_tone = None
            roi_before = roi_before or 0.0
            roi_after = roi_after or 0.0
            c_before = c_before or 0.0
            c_after = c_after or 0.0

            roi_improvement = roi_after - roi_before
            complexity_improvement = c_before - c_after
            weight = roi_improvement if roi_improvement > 0 else complexity_improvement
            if weight <= 0:
                weight = 1.0

            success = (outcome or "").upper() == "SUCCESS"
            feats = self._extract_style(text)
            if p_headers:
                try:
                    feats["headers"] = json.loads(p_headers)
                except Exception:
                    feats["headers"] = []
            if p_order:
                try:
                    feats["example_order"] = json.loads(p_order)
                except Exception:
                    feats["example_order"] = []
            if p_tone:
                feats["tone"] = p_tone
            for key, val in feats.items():
                val_key = json.dumps(val) if isinstance(val, list) else str(val)
                d = stats[key][val_key]
                d["weight"] += weight
                if success:
                    d["success"] += weight

        self.style_weights = {
            feat: {
                k: (v["success"] / v["weight"]) if v["weight"] else 0.0
                for k, v in m.items()
            }
            for feat, m in stats.items()
        }

        # persist stats to database for future incremental updates
        self._stats = {
            feat: defaultdict(lambda: [0, 0]) for feat in self._stats.keys()
        }
        self._db.execute("DELETE FROM style_stats")
        for feat, m in stats.items():
            for k, v in m.items():
                self._stats.setdefault(feat, defaultdict(lambda: [0, 0]))
                self._stats[feat][k] = [v["success"], v["weight"]]
                self._db.execute(
                    "INSERT INTO style_stats(feature, value, success, total) VALUES(?,?,?,?)",
                    (feat, k, v["success"], v["weight"]),
                )
        self._db.commit()

        if self.state_path:
            self.save_weights(self.state_path)
        return self.style_weights

    # ------------------------------------------------------------------
    def save_weights(self, path: str | Path) -> None:
        """Persist :attr:`style_weights` and version to ``path`` as JSON."""

        p = Path(path)
        if not p.is_absolute():
            p = ROOT_DIR / p
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {"version": self.STYLE_VERSION, "weights": self.style_weights}
        with p.open("w", encoding="utf-8") as fh:
            json.dump(data, fh)

    # ------------------------------------------------------------------
    @classmethod
    def load_weights(cls, path: str | Path) -> Tuple[int, Dict[str, Dict[str, float]]]:
        """Return ``(version, weights)`` loaded from JSON ``path``."""

        p = Path(path)
        if not p.is_absolute():
            p = ROOT_DIR / p
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        version = data.get("version")
        weights = data.get("weights") if isinstance(data, dict) else None
        if version != cls.STYLE_VERSION or not isinstance(weights, dict):
            raise ValueError("Incompatible prompt style weights version")
        trainer = cls(state_path=None)
        trainer.style_weights = weights
        return version, trainer.style_weights

    # ------------------------------------------------------------------
    def append_records(self, records: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Append ``records`` to the databases and retrain.

        Each record should contain ``prompt``, ``outcome``, ``roi_before``,
        ``roi_after``, ``complexity_before`` and ``complexity_after`` fields.  A
        new row is inserted into both :class:`PatchHistoryDB` and
        :class:`GPTMemoryManager` for every record.  After insertion the trainer
        retrains and persists the updated ``style_weights``.
        """

        for rec in records:
            cur = self.patch_db.conn.execute(
                (
                    "INSERT INTO patch_history(outcome, roi_before, roi_after, "
                    "complexity_before, complexity_after) VALUES(?, ?, ?, ?, ?)"
                ),
                (
                    rec.get("outcome"),
                    rec.get("roi_before"),
                    rec.get("roi_after"),
                    rec.get("complexity_before"),
                    rec.get("complexity_after"),
                ),
            )
            patch_id = cur.lastrowid
            prompt_text = f"PATCH:{patch_id}\n{rec.get('prompt', '')}"
            # ``log_interaction`` stores additional metadata; an empty response
            # keeps the append logic simple and is sufficient for training.
            self.memory.log_interaction(prompt_text, "", tags=None)
        self.patch_db.conn.commit()
        self.memory.conn.commit()
        return self.train()

    # ------------------------------------------------------------------
    def suggest_style(self) -> Dict[str, Any]:
        """Return the highest scoring formatting style."""

        if not self.style_weights:
            self.train()
        suggestion: Dict[str, Any] = {}
        list_feats = {"headers", "example_order", "sections", "structured_sections"}
        int_feats = {"example_count"}
        bool_feats = {"has_code", "has_bullets"}
        for feat, mapping in self.style_weights.items():
            if not mapping:
                continue
            best_val, _ = max(mapping.items(), key=lambda kv: kv[1])
            if feat in list_feats:
                try:
                    suggestion[feat] = json.loads(best_val)
                except Exception:
                    suggestion[feat] = [best_val]
            elif feat in int_feats:
                try:
                    suggestion[feat] = int(float(best_val))
                except Exception:
                    suggestion[feat] = 0
            elif feat in bool_feats:
                suggestion[feat] = best_val.lower() == "true"
            else:
                suggestion[feat] = best_val
        return suggestion

    # ------------------------------------------------------------------
    def record(
        self,
        *,
        tone: str = "",
        headers: Iterable[str] | None = None,
        example_order: Iterable[str] | None = None,
        has_bullets: bool | None = None,
        has_code: bool | None = None,
        example_count: int | None = None,
        success: bool = False,
        weight: float = 1.0,
        **_: Any,
    ) -> bool:
        """Incrementally update style weights and persist when changed.

        Parameters allow overriding features such as bullet usage, code block
        presence and example count so the trainer can update its statistics
        without a full retraining pass.
        """

        updated = False
        feats = {
            "tone": tone or None,
            "headers": json.dumps(list(headers)) if headers else None,
            "example_order": json.dumps(list(example_order)) if example_order else None,
            "has_bullets": str(has_bullets) if has_bullets is not None else None,
            "has_code": str(has_code) if has_code is not None else None,
            "example_count": str(example_count) if example_count is not None else None,
        }

        # log full record for monitoring
        self._db.execute(
            (
                "INSERT INTO prompt_records(ts, tone, headers, example_order, "
                "has_bullets, has_code, example_count, success, weight) VALUES(?,?,?,?,?,?,?,?,?)"
            ),
            (
                time.time(),
                tone or None,
                json.dumps(list(headers)) if headers else None,
                json.dumps(list(example_order)) if example_order else None,
                1 if has_bullets else 0 if has_bullets is not None else None,
                1 if has_code else 0 if has_code is not None else None,
                example_count,
                1 if success else 0,
                float(weight),
            ),
        )

        for feat, key in feats.items():
            if not key:
                continue
            stats = self._stats[feat][key]
            stats[1] += weight
            if success:
                stats[0] += weight
            score = stats[0] / max(stats[1], 1)
            mapping = self.style_weights.setdefault(feat, {})
            prev = mapping.get(key)
            if prev != score:
                mapping[key] = score
                updated = True
            self._db.execute(
                (
                    "INSERT INTO style_stats(feature, value, success, total) "
                    "VALUES(?,?,?,?) ON CONFLICT(feature, value) DO UPDATE SET "
                    "success=excluded.success, total=excluded.total"
                ),
                (feat, key, stats[0], stats[1]),
            )
        self._db.commit()
        if updated and self.state_path:
            self.save_weights(self.state_path)
        return updated

    # ------------------------------------------------------------------
    def style_metrics(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Return success rates and observation counts for each style."""

        metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
        for feat, mapping in self.style_weights.items():
            feat_metrics: Dict[str, Dict[str, float]] = {}
            for key, score in mapping.items():
                count = self._stats.get(feat, {}).get(key, [0, 0])[1]
                feat_metrics[key] = {"score": score, "count": float(count)}
            metrics[feat] = feat_metrics
        return metrics


__all__ = ["PromptMemoryTrainer"]
