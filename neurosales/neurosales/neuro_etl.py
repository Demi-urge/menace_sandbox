from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import re

try:
    from celery import Celery
except Exception:  # pragma: no cover - optional dep
    Celery = None  # type: ignore

try:
    import psycopg2  # type: ignore
except Exception:  # pragma: no cover - optional dep
    psycopg2 = None  # type: ignore

try:
    from pymongo import MongoClient  # type: ignore
except Exception:  # pragma: no cover - optional dep
    MongoClient = None  # type: ignore

try:
    import faiss  # type: ignore
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:  # pragma: no cover - optional heavy deps
    faiss = None  # type: ignore
    SentenceTransformer = None  # type: ignore
    np = None  # type: ignore

try:
    from neo4j import GraphDatabase  # type: ignore
except Exception:  # pragma: no cover - optional dep
    GraphDatabase = None  # type: ignore

from .influence_graph import InfluenceGraph

HARVARD_OXFORD_ATLAS = {
    "ventromedial pfc": "Frontal_Medial_Cortex",
    "amygdala": "Amygdala",
    "nucleus accumbens": "Nucleus_Accumbens",
}

PERSUASION_PRIMITIVES: Dict[str, List[str]] = {
    "scarcity": ["scarce", "limited", "only", "scarcity"],
    "social_proof": ["popular", "trending", "everyone"],
    "loss_aversion": ["lose", "loss", "risk"],
}


@dataclass
class NeuroToken:
    study_id: str
    sentence: str
    region: str
    primitive: str
    effect_size: float


class InMemoryPostgres:
    """Simple in-memory stand-in for PostgreSQL."""

    def __init__(self) -> None:
        self.rows: List[NeuroToken] = []

    def insert(self, token: NeuroToken) -> None:
        self.rows.append(token)


class InMemoryMongo:
    """Simple in-memory Mongo collection."""

    def __init__(self) -> None:
        self.docs: List[Dict[str, str]] = []

    def insert(self, study_id: str, paragraph: str) -> None:
        self.docs.append({"study_id": study_id, "paragraph": paragraph})


class InMemoryFaiss:
    """Embeds sentences and stores in an optional FAISS index."""

    def __init__(self) -> None:
        self.sentences: List[str] = []
        self.ids: List[int] = []
        if faiss is not None and SentenceTransformer is not None:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            dim = self.model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatL2(dim)
        else:  # pragma: no cover - heavy deps missing
            self.model = None
            self.index = None

    def add(self, sentence: str) -> int:
        idx = len(self.sentences)
        self.sentences.append(sentence)
        self.ids.append(idx)
        if self.model is not None and self.index is not None and np is not None:
            vec = self.model.encode(sentence, convert_to_numpy=True).astype("float32")
            self.index.add(vec.reshape(1, -1))
        return idx


class InMemoryNeo4j:
    """Store links in an InfluenceGraph instead of real Neo4j."""

    def __init__(self, graph: InfluenceGraph) -> None:
        self.graph = graph

    def link(self, token_id: int, region: str, primitive: str) -> None:
        node = f"token_{token_id}"
        self.graph.add_node(node, "token")
        self.graph.add_node(region, "region")
        self.graph.add_node(primitive, "primitive")
        self.graph.add_edge(node, region, 1.0, 1.0)
        self.graph.add_edge(node, primitive, 1.0, 1.0)


def tokenize_paragraph(paragraph: str, study_id: str) -> List[NeuroToken]:
    """Convert raw paragraph text into NeuroTokens."""
    tokens: List[NeuroToken] = []
    for sent in re.split(r"(?<=[.!?])\s+", paragraph):
        sent = sent.strip()
        if not sent:
            continue
        lower = sent.lower()
        region = ""
        for k, v in HARVARD_OXFORD_ATLAS.items():
            if k in lower:
                region = v
                break
        primitive = ""
        for p, kws in PERSUASION_PRIMITIVES.items():
            if any(kw in lower for kw in kws):
                primitive = p
                break
        m = re.search(r"(-?\d+(?:\.\d+)?)", sent)
        effect = float(m.group(1)) if m else 0.0
        if region or primitive or effect:
            tokens.append(NeuroToken(study_id, sent, region, primitive, effect))
    return tokens


def process_study_sync(
    study_id: str,
    paragraphs: List[str],
    pg: Optional[InMemoryPostgres] = None,
    mongo: Optional[InMemoryMongo] = None,
    faiss_index: Optional[InMemoryFaiss] = None,
    neo: Optional[InMemoryNeo4j] = None,
    graph: Optional[InfluenceGraph] = None,
) -> List[NeuroToken]:
    """Process text and store results in various backends."""
    graph = graph or InfluenceGraph()
    pg = pg or InMemoryPostgres()
    mongo = mongo or InMemoryMongo()
    faiss_index = faiss_index or InMemoryFaiss()
    neo = neo or InMemoryNeo4j(graph)

    results: List[NeuroToken] = []
    for para in paragraphs:
        mongo.insert(study_id, para)
        tokens = tokenize_paragraph(para, study_id)
        for t in tokens:
            pg.insert(t)
            token_id = faiss_index.add(t.sentence)
            neo.link(token_id, t.region or "unknown", t.primitive or "unknown")
            results.append(t)
    return results


# Optional Celery task
if Celery is not None:
    app = Celery("neuro_etl")

    @app.task
    def process_study_task(study_id: str, paragraphs: List[str]) -> List[Dict[str, str]]:
        tokens = process_study_sync(study_id, paragraphs)
        return [asdict(t) for t in tokens]
else:  # pragma: no cover - Celery not available
    app = None

