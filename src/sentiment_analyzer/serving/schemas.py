"""Pydantic request/response schemas."""
from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    version: str


class TextRequest(BaseModel):
    text: str


class ClassificationResponse(BaseModel):
    pred: int
    top_k: list[dict]
