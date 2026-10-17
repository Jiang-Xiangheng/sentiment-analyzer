"""Data pipeline entrypoint.

Thin wrapper around ``sentiment_analyzer.data.prepare`` so preprocessing /
feature engineering is exposed from the ``ml/`` directory.
"""

from __future__ import annotations

from sentiment_analyzer.data.prepare import prepare_data

__all__ = ["prepare_data"]
