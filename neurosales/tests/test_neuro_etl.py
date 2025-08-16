import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.neuro_etl import (
    tokenize_paragraph,
    process_study_sync,
    InMemoryPostgres,
    InMemoryMongo,
    InMemoryFaiss,
    InMemoryNeo4j,
)
from neurosales.influence_graph import InfluenceGraph


def test_tokenize_paragraph_basic():
    text = "Ventromedial PFC predicts purchases with effect size 0.8 due to scarcity."
    tokens = tokenize_paragraph(text, "s1")
    assert tokens
    t = tokens[0]
    assert t.region == "Frontal_Medial_Cortex"
    assert t.primitive == "scarcity"
    assert abs(t.effect_size - 0.8) < 1e-6


def test_process_study_sync_in_memory():
    graph = InfluenceGraph()
    pg = InMemoryPostgres()
    mongo = InMemoryMongo()
    faiss_idx = InMemoryFaiss()
    neo = InMemoryNeo4j(graph)
    para = "Ventromedial PFC drives impulse buys when everyone is doing it (0.6)."
    tokens = process_study_sync("study1", [para], pg, mongo, faiss_idx, neo, graph)
    assert pg.rows and mongo.docs
    assert faiss_idx.sentences
    assert any(n.startswith("token_") for n in graph.nodes)
