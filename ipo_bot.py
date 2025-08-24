"""IPO Bot for planning bot reuse and creation from Stage 3 blueprints."""

from __future__ import annotations

import json
import logging
import re
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Iterable, Optional

from db_router import GLOBAL_ROUTER, init_db_router

import networkx as nx
try:
    from fuzzywuzzy import fuzz  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    fuzz = None  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    import spacy  # type: ignore
except Exception:  # pragma: no cover - optional heavy dep
    spacy = None

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
except Exception:  # pragma: no cover - optional heavy dep
    AutoTokenizer = None  # type: ignore
    AutoModel = None  # type: ignore
    torch = None  # type: ignore


@dataclass
class BlueprintTask:
    """Single task extracted from a blueprint."""

    name: str
    description: str = ""
    role: str = ""
    dependencies: List[str] = field(default_factory=list)


@dataclass
class Blueprint:
    tasks: List[BlueprintTask]


class BlueprintIngestor:
    """Parse blueprints to extract tasks and dependencies."""

    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_sm") if spacy else None

    def ingest(self, text: str) -> Blueprint:
        names: List[str] = []
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in {"ORG", "PRODUCT"} and ent.text.startswith("Bot"):
                    names.append(ent.text)
        if not names:
            names = re.findall(r"(Bot[A-Za-z0-9_]+)", text)
        seen = set()
        ordered = []
        for n in names:
            if n not in seen:
                seen.add(n)
                ordered.append(n)
        names = ordered
        tasks = []
        for name in names:
            tasks.append(BlueprintTask(name=name))
        for i in range(1, len(tasks)):
            tasks[i].dependencies.append(tasks[i - 1].name)
        return Blueprint(tasks)


@dataclass
class BotCandidate:
    name: str
    keywords: str = ""
    reuse: bool = True
    score: float = 0.0


class BotDatabaseSearcher:
    """Query Menace's bots database."""

    def __init__(self, db_path: str = "models.db") -> None:
        self.db_path = db_path

    def search(self, keywords: Iterable[str]) -> List[BotCandidate]:
        router = GLOBAL_ROUTER or init_db_router("ipo_bot", shared_db_path=self.db_path)
        conn = router.get_connection("bots")
        cur = conn.cursor()
        key = "%" + "%".join(keywords) + "%"
        cur.execute(
            "CREATE TABLE IF NOT EXISTS bots (id INTEGER PRIMARY KEY, name TEXT, keywords TEXT, reuse INTEGER)"
        )
        cur.execute(
            "SELECT name, keywords, reuse FROM bots WHERE name LIKE ? OR keywords LIKE ?",
            (key, key),
        )
        rows = cur.fetchall()
        return [BotCandidate(name=r[0], keywords=r[1], reuse=bool(r[2])) for r in rows]


class DecisionMaker:
    """Rank candidates and choose reuse, fork or new build."""

    def __init__(self) -> None:
        self.vec = TfidfVectorizer()

    def rank(self, task: BlueprintTask, candidates: List[BotCandidate]) -> BotCandidate:
        if not candidates:
            return BotCandidate(name=task.name, reuse=False, score=0.0)
        texts = [task.name] + [c.keywords for c in candidates]
        tfidf = self.vec.fit_transform(texts)
        scores = (tfidf[0] @ tfidf[1:].T).toarray()[0]
        best_idx = int(scores.argmax())
        best = candidates[best_idx]
        best.score = float(scores[best_idx]) * 100
        return best

    def decision(self, task: BlueprintTask, candidate: BotCandidate) -> str:
        score = candidate.score
        if score > 80 and candidate.reuse:
            return "fork_existing"
        if score > 50:
            return "fork_existing"
        return "build_new"


class PlanGraphBuilder:
    """Create dependency graph for execution plan."""

    def build(self, tasks: Iterable[BlueprintTask]) -> nx.DiGraph:
        g = nx.DiGraph()
        for t in tasks:
            g.add_node(t.name)
            for dep in t.dependencies:
                g.add_edge(dep, t.name)
        return g


class IPOEnhancementsDB:
    """Simple database for plan enhancements."""

    def __init__(self, path: Path = Path("enhancements.db")) -> None:
        self.path = path
        self.router = GLOBAL_ROUTER or init_db_router("ipo_bot", shared_db_path=str(path))
        self.conn = self.router.get_connection("enhancements")
        c = self.conn.cursor()
        c.execute(
            "CREATE TABLE IF NOT EXISTS enhancements (id INTEGER PRIMARY KEY, blueprint_id TEXT, bot TEXT, action TEXT, reason TEXT, source_menace_id TEXT DEFAULT '')"
        )
        cols = [r[1] for r in c.execute("PRAGMA table_info(enhancements)").fetchall()]
        if "source_menace_id" not in cols:
            c.execute(
                "ALTER TABLE enhancements ADD COLUMN source_menace_id TEXT DEFAULT ''"
            )
        self.conn.commit()

    def log(self, blueprint_id: str, bot: str, action: str, reason: str) -> None:
        menace_id = self.router.menace_id if self.router else os.getenv("MENACE_ID", "")
        self.conn.execute(
            "INSERT INTO enhancements (blueprint_id, bot, action, reason, source_menace_id) VALUES (?,?,?,?,?)",
            (blueprint_id, bot, action, reason, menace_id),
        )
        self.conn.commit()


@dataclass
class PlanAction:
    bot: str
    action: str
    notes: str = ""


@dataclass
class ExecutionPlan:
    actions: List[PlanAction]
    graph: nx.DiGraph


class IPOBot:
    """Main orchestrator for the IPO planning process."""

    def __init__(self, db_path: str = "models.db", enhancements_db: Optional[Path] = None) -> None:
        self.ingestor = BlueprintIngestor()
        self.searcher = BotDatabaseSearcher(db_path)
        self.decider = DecisionMaker()
        self.graph_builder = PlanGraphBuilder()
        self.db = IPOEnhancementsDB(enhancements_db or Path("enhancements.db"))
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("IPO")

    def generate_plan(self, blueprint_text: str, blueprint_id: str = "bp1") -> ExecutionPlan:
        blueprint = self.ingestor.ingest(blueprint_text)
        actions: List[PlanAction] = []
        for task in blueprint.tasks:
            words = re.findall(r"\w+", task.name)
            cands = self.searcher.search(words)
            cand = self.decider.rank(task, cands)
            decision = self.decider.decision(task, cand)
            self.db.log(blueprint_id, task.name, decision, f"score={cand.score:.1f}")
            self.logger.info("%s -> %s (%.1f)", task.name, decision, cand.score)
            actions.append(PlanAction(bot=task.name, action=decision, notes=str(cand.score)))
        graph = self.graph_builder.build(blueprint.tasks)
        return ExecutionPlan(actions=actions, graph=graph)


__all__ = [
    "BlueprintTask",
    "Blueprint",
    "BotCandidate",
    "PlanAction",
    "ExecutionPlan",
    "BlueprintIngestor",
    "BotDatabaseSearcher",
    "DecisionMaker",
    "PlanGraphBuilder",
    "IPOEnhancementsDB",
    "IPOBot",
]
