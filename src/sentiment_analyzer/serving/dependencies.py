"""Dependency injection - singleton model loader."""

from __future__ import annotations

import os
from functools import lru_cache

from ..inference.predict import LoadedModel, load_model
from ..utils import get_logger
from .errors import ModelNotLoadedError

log = get_logger(__name__)


@lru_cache(maxsize=1)
def get_model() -> LoadedModel:
    """Load and cache the production model.

    Raises:
        ModelNotLoadedError: when ``MODEL_PATH`` is unset or the checkpoint
            cannot be loaded. FastAPI maps it to a 503 via the handler
            registered in ``serving.main``.
    """
    checkpoint = os.environ.get("MODEL_PATH")
    if not checkpoint:
        raise ModelNotLoadedError("MODEL_PATH env var not set")

    log.info("model.load", path=checkpoint)
    try:
        return load_model(checkpoint)
    except Exception as exc:
        raise ModelNotLoadedError(f"failed to load model from {checkpoint}: {exc}") from exc


def reset_model_cache() -> None:
    """Clear the cached model (used by tests and hot reloads)."""
    get_model.cache_clear()
