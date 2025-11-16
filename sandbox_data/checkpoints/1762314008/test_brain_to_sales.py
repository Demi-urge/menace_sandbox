import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from neurosales.neuro_etl import NeuroToken
from neurosales.trigger_phrase_db import TriggerPhraseDB
from neurosales.brain_to_sales import BrainToSalesMapper


def test_brain_to_sales_basic():
    token = NeuroToken(
        study_id="s1",
        sentence="NAcc 2.3x during scarcity",
        region="Nucleus_Accumbens",
        primitive="scarcity",
        effect_size=2.3,
    )
    db = TriggerPhraseDB()
    mapper = BrainToSalesMapper(db)
    mapper.process_tokens([token], "reward_seeker")

    script = "Limited-Time Offer"
    assert script in mapper.scripts_for_region("Nucleus_Accumbens")
    assert abs(mapper.expected_value(script, "reward_seeker") - 2.3) < 1e-6
    assert script in db.phrases
