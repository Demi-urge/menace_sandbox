from menace.knowledge_graph import KnowledgeGraph
import sqlite3
import types


def test_ingest_gpt_insights_links_entities():
    kg = KnowledgeGraph()
    conn = sqlite3.connect(':memory:')
    conn.execute('CREATE TABLE memory (key TEXT, tags TEXT)')
    conn.execute(
        'INSERT INTO memory VALUES (?, ?)',
        ('idea1', 'bot:alpha,code:module.py,error:ValueError')
    )
    manager = types.SimpleNamespace(conn=conn)
    kg.ingest_gpt_memory(manager)
    inode = 'insight:idea1'
    assert inode in kg.graph
    assert ('insight:idea1', 'bot:alpha') in kg.graph.edges
    assert ('insight:idea1', 'code:module.py') in kg.graph.edges
    assert ('insight:idea1', 'error_category:ValueError') in kg.graph.edges
