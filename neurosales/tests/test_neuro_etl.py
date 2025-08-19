import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import logging

from neurosales.neuro_etl import (
    tokenize_paragraph,
    process_study_sync,
    InMemoryPostgres,
    InMemoryMongo,
    InMemoryFaiss,
    InMemoryNeo4j,
)
from neurosales.influence_graph import InfluenceGraph
import neurosales.neuro_etl as ne

# Disable heavy optional dependencies for tests
ne.faiss = None
ne.SentenceTransformer = None
ne.np = None


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


GPL_TEXT = """
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
"""


def test_faiss_add_filters(caplog):
    faiss_idx = InMemoryFaiss()
    risky = "eval('data')"
    with caplog.at_level(logging.WARNING):
        idx = faiss_idx.add(risky)
    assert faiss_idx.alerts[idx]
    assert any("eval" in a[0] for a in faiss_idx.alerts[idx] or [])
    with caplog.at_level(logging.WARNING):
        idx2 = faiss_idx.add(GPL_TEXT)
    assert faiss_idx.alerts[idx2]
    assert any("GPL-3.0" in a[1] for a in faiss_idx.alerts[idx2] or [])
