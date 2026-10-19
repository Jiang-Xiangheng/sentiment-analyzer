"""Pydantic request/response schemas."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

#: Maximum number of characters accepted per request. The model truncates at
#: 512 sub-word tokens, so this is a generous safety limit rather than the
#: real model window.
MAX_TEXT_LENGTH = 5000

#: Validated text input: whitespace is stripped first, then the length is
#: checked, so blank strings are rejected with a 422 instead of reaching the model.
Text = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
    ),
]


class HealthResponse(BaseModel):
    """Response of ``GET /health``."""

    status: str = "ok"
    model_loaded: bool
    version: str


class TextRequest(BaseModel):
    """Payload accepted by ``POST /classify``."""

    text: Text = Field(
        ...,
        description="Text to classify.",
        examples=["I absolutely love this movie"],
    )


class ClassScore(BaseModel):
    """Probability assigned to a single class."""

    label: str = Field(..., description="Human-readable class label.")
    score: float = Field(..., ge=0.0, le=1.0, description="Probability of this class.")


class ClassificationResponse(BaseModel):
    """Response of ``POST /classify``."""

    pred: int = Field(..., ge=0, description="Index of the most likely class.")
    top_k: list[ClassScore] = Field(
        ...,
        min_length=1,
        description="Per-class scores, sorted from most to least likely.",
    )
