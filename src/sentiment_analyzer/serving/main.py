"""FastAPI application."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from prometheus_fastapi_instrumentator import Instrumentator

from ..utils import configure_logging, get_logger
from .dependencies import get_model
from .errors import (
    InferenceError,
    ModelNotLoadedError,
    inference_error_handler,
    model_not_loaded_handler,
)
from .routes import router

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(json_output=True)
    try:
        get_model()
        log.info("startup.model_loaded")
    except Exception as exc:
        log.warning("startup.model_not_loaded", error=str(exc))
    yield


app = FastAPI(
    title="Sentiment Analyzer",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_exception_handler(InferenceError, inference_error_handler)
app.add_exception_handler(ModelNotLoadedError, model_not_loaded_handler)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


app.include_router(router)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the single-page sentiment UI."""
    return FileResponse(STATIC_DIR / "index.html")
