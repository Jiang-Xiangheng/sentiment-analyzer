"""Data layer smoke tests."""

from __future__ import annotations

import pandas as pd

from sentiment_analyzer.data import TextDataset


def test_text_dataset_loads(tmp_path):
    p = tmp_path / "sample.csv"
    pd.DataFrame({"text": ["hello", "world"], "label": [0, 1]}).to_csv(p, index=False)
    ds = TextDataset(p)
    assert len(ds) == 2
    assert ds[0]["label"] == 0
