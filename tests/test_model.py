"""Model smoke tests (forward pass, output shape)."""

from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Requires network model download; enable locally")
def test_nlp_factory():
    from sentiment_analyzer.models import build_model

    m = build_model("prajjwal1/bert-tiny", num_labels=3)
    assert m is not None
