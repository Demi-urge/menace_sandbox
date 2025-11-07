from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import re
import logging

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

try:
    from analysis.semantic_diff_filter import find_semantic_risks
except Exception:  # pragma: no cover - optional dependency
    def find_semantic_risks(lines, threshold: float = 0.5):  # type: ignore
        return []
try:
    from compliance.license_fingerprint import check as license_check
except Exception:  # pragma: no cover - optional dependency
    def license_check(text: str):  # type: ignore
        return None
try:
    from security.secret_redactor import redact
except Exception:  # pragma: no cover - optional dependency
    def redact(text: str):  # type: ignore
        return text
from governed_embeddings import governed_embed
from .influence_graph import InfluenceGraph

logger = logging.getLogger(__name__)

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
    """Embeds sentences and stores in an optional FAISS index.

    Sentences are redacted and checked for licensing issues and semantic risks
    before embedding. If any alerts are found the sentence is stored without an
    embedding and the alerts are recorded alongside the stored sentence.
    """

    def __init__(self) -> None:
        self.sentences: List[str] = []
        self.ids: List[int] = []
        self.alerts: List[Optional[List[tuple[str, str, float]]]] = []
        if faiss is not None and SentenceTransformer is not None:
            from huggingface_hub import login
            import os

            login(token=os.getenv("HUGGINGFACE_API_TOKEN"))
            self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            dim = self.model.get_sentence_embedding_dimension()
            self.index = faiss.IndexFlatL2(dim)
        else:  # pragma: no cover - heavy deps missing
            self.model = None
            self.index = None

    def add(self, sentence: str) -> int:
        """Add *sentence* to the store and index an embedding when safe.

        The text is redacted, checked for licensing and scanned with
        :func:`find_semantic_risks`. If disallowed content or alerts are
        detected the sentence is stored but not indexed and the alert list is
        saved to :attr:`alerts`.
        """

        idx = len(self.sentences)
        red = redact(sentence.strip())
        lic = license_check(red)
        alerts = []
        if lic:
            logger.warning("license detected: %s", lic)
            alerts.append((red, f"license:{lic}", 1.0))
        sem = find_semantic_risks(red.splitlines())
        if sem:
            logger.warning("semantic risks detected: %s", [a[1] for a in sem])
            alerts.extend(sem)
        self.sentences.append(red)
        self.ids.append(idx)
        self.alerts.append(alerts or None)
        if alerts or self.model is None or self.index is None or np is None:
            return idx
        vec = governed_embed(red, self.model)
        if vec is not None:
            arr = np.array(vec, dtype="float32")
            self.index.add(arr.reshape(1, -1))
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

