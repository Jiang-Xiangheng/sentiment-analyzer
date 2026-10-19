"""FastAPI routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from .. import __version__
from ..inference.predict import LoadedModel, predict
from .dependencies import get_model
from .errors import InferenceError
from .schemas import ClassificationResponse, HealthResponse, TextRequest

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report whether the model is loaded, without failing the probe."""
    try:
        get_model()
        loaded = True
    except Exception:
        loaded = False
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        version=__version__,
    )


@router.post("/classify", response_model=ClassificationResponse)
def classify(
    req: TextRequest,
    model: LoadedModel = Depends(get_model),
) -> ClassificationResponse:
    """Classify the submitted text."""
    try:
        result = predict(model, req.text)
    except InferenceError:
        raise
    except Exception as exc:
        # Any model/tokenizer failure becomes a 503 via the InferenceError handler.
        raise InferenceError(str(exc)) from exc
    return ClassificationResponse(**result)
