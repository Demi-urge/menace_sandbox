import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unittest.mock import patch

from neurosales.sql_db import create_session
from neurosales.memory import DatabaseConversationMemory
from neurosales.embedding_memory import DatabaseEmbeddingMemory
from neurosales.vector_db import DatabaseVectorDB
from neurosales.user_preferences import DatabasePreferenceEngine
from neurosales.reactions import DatabaseReactionHistory


def test_end_to_end_sandbox_persistence(tmp_path):
    url = f"sqlite:///{tmp_path/'sandbox.db'}"
    Session = create_session(url)

    with patch('neurosales.vector_db.PineconeLogger'):
        conv = DatabaseConversationMemory(user_id='u1', session_factory=Session)
        embed = DatabaseEmbeddingMemory(session_factory=Session)
        vect = DatabaseVectorDB(session_factory=Session, sync_interval=1)
    pref = DatabasePreferenceEngine(session_factory=Session)
    react = DatabaseReactionHistory('u1', session_factory=Session)

    conv.add_message('user', 'hi')
    embed.add_message('user', 'hello')
    vect.add_message('user', 'vector')
    pref.add_message('u1', 'I enjoy pears')
    react.add_pair('hi', 'wave')

    with patch('neurosales.vector_db.PineconeLogger'):
        conv2 = DatabaseConversationMemory(user_id='u1', session_factory=Session)
        embed2 = DatabaseEmbeddingMemory(session_factory=Session)
        vect2 = DatabaseVectorDB(session_factory=Session, sync_interval=1)
    pref2 = DatabasePreferenceEngine(session_factory=Session)
    react2 = DatabaseReactionHistory('u1', session_factory=Session)

    assert [m.content for m in conv2.get_recent_messages()] == ['hi']
    e_msgs = [m.content for m in embed2.get_recent_messages()]
    assert 'hello' in e_msgs
    v_msgs = [m.content for m in vect2.get_recent_messages()]
    assert 'vector' in v_msgs
    assert pref2.get_profile('u1').keyword_freq.get('pears', 0) > 0
    assert react2.get_pairs() == [('hi', 'wave')]
