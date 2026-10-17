"""Service entrypoint for the sentiment analysis app.

Start the API server with:

    uvicorn app.main:app --host 0.0.0.0 --port 8000

The actual FastAPI application lives in ``sentiment_analyzer.serving.main``;
this module is the top-level entrypoint expected by the course layout.
"""
from __future__ import annotations

from sentiment_analyzer.serving.main import app

__all__ = ["app"]
