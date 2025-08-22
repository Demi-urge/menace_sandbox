import logging
from unittest.mock import Mock

import pytest

from vector_service.cognition_layer import CognitionLayer


def test_init_logs_failures(caplog):
    metrics = Mock()
    metrics.get_db_weights.return_value = {"db": 1.0}
    metrics.load_sessions.side_effect = Exception("load boom")

    context_builder = Mock()
    context_builder.refresh_db_weights.side_effect = Exception("refresh boom")

    roi_tracker = Mock()
    patch_logger = Mock()
    patch_logger.roi_tracker = roi_tracker

    with caplog.at_level(logging.WARNING, logger="vector_service.cognition_layer"):
        CognitionLayer(
            retriever=Mock(),
            context_builder=context_builder,
            patch_logger=patch_logger,
            vector_metrics=metrics,
            roi_tracker=roi_tracker,
        )

    assert "refresh context builder db weights" in caplog.text
    assert "load pending sessions" in caplog.text


def test_query_logs_metric_failures(caplog):
    metrics = Mock()
    metrics.get_db_weights.return_value = None
    metrics.load_sessions.return_value = {}
    metrics.log_retrieval.side_effect = Exception("log boom")
    metrics.save_session.side_effect = Exception("save boom")

    builder = Mock()
    builder.build_context.return_value = (
        "ctx",
        "sid",
        [("db", "vec", 0.1)],
        {"db": [{"vector_id": "vec"}]},
        {"tokens": 1, "wall_time_ms": 0.0, "prompt_tokens": 1},
    )

    roi_tracker = Mock()
    patch_logger = Mock()
    patch_logger.roi_tracker = roi_tracker

    layer = CognitionLayer(
        retriever=Mock(),
        context_builder=builder,
        patch_logger=patch_logger,
        vector_metrics=metrics,
        roi_tracker=roi_tracker,
    )

    with caplog.at_level(logging.WARNING, logger="vector_service.cognition_layer"):
        layer.query("prompt")

    assert "log retrieval metrics" in caplog.text
    assert "save retrieval session" in caplog.text

