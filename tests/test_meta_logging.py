import pytest
pytest.skip("optional dependencies not installed", allow_module_level=True)
import json
from unittest.mock import MagicMock

import menace.meta_logging as ml


def test_kafka_meta_logger():
    producer = MagicMock()
    logger = ml.KafkaMetaLogger(producer=producer)
    event = ml.LogEvent("input", {"msg": "hi"})
    logger.log(event)
    producer.send.assert_called_once()
    args, kwargs = producer.send.call_args
    assert args[0] == "menace.events.input"
    payload = json.loads(kwargs["value"].decode()) if isinstance(kwargs.get("value"), bytes) else args[1]
    assert payload["event_type"] == "input"
