"""Training entrypoint.

Thin wrapper around ``sentiment_analyzer.training.train`` so the training
stage is exposed from the ``ml/`` directory while the reusable
implementation stays inside the package.
"""
from __future__ import annotations

from sentiment_analyzer.training.train import main

__all__ = ["main"]

if __name__ == "__main__":
    main()
