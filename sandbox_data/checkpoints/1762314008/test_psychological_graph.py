import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unittest.mock import patch, MagicMock
import numpy as np
from neurosales.psychological_graph import PsychologicalGraph


def test_add_and_similarity():
    with patch('neurosales.psychological_graph.SentenceTransformer') as ST, \
         patch('neurosales.psychological_graph.faiss') as faiss, \
         patch('neurosales.psychological_graph.np', np):
        model = MagicMock()
        model.get_sentence_embedding_dimension.return_value = 2
        model.encode.return_value = np.array([0.0, 1.0], dtype='float32')
        ST.return_value = model
        index = MagicMock()
        faiss.IndexFlatL2.return_value = index
        index.search.return_value = (np.array([[0.0]]), np.array([[0]]))
        g = PsychologicalGraph()
        rid = g.add_rule('if happy', 'say hi')
        match = g.closest_rule('happy')
        assert match and match.rule_id == rid


def test_decay_and_pagerank():
    g = PsychologicalGraph()
    r1 = g.add_rule('a', 'b', weight=2.0)
    r2 = g.add_rule('b', 'c')
    g.link_rules(r1, r2, weight=1.0)
    g.record_interaction(r1, True)
    before = g.rules[r1].weight
    g.decay_weights()
    after = g.rules[r1].weight
    assert after <= before
    scores = g.pagerank_scores()
    assert r2 in scores


def test_shortest_path():
    g = PsychologicalGraph()
    r1 = g.add_rule('a', 'b')
    r2 = g.add_rule('b', 'c')
    r3 = g.add_rule('c', 'd')
    g.link_rules(r1, r2)
    g.link_rules(r2, r3)
    path = g.shortest_influence_path(r1, r3)
    assert path == [r1, r2, r3]


def test_psychological_graph_env(monkeypatch):
    monkeypatch.setenv('NEURO_NEO4J_URI', 'bolt://env')
    monkeypatch.setenv('NEURO_NEO4J_USER', 'u')
    monkeypatch.setenv('NEURO_NEO4J_PASS', 'p')
    with patch('neurosales.psychological_graph.GraphDatabase.driver') as drv, \
         patch('neurosales.psychological_graph.SentenceTransformer') as ST, \
         patch('neurosales.psychological_graph.faiss') as faiss, \
         patch('neurosales.psychological_graph.np', np):
        model = MagicMock()
        model.get_sentence_embedding_dimension.return_value = 2
        ST.return_value = model
        faiss.IndexFlatL2.return_value = MagicMock()
        g = PsychologicalGraph()
    drv.assert_called_with('bolt://env', auth=('u', 'p'))
    assert g.enabled
