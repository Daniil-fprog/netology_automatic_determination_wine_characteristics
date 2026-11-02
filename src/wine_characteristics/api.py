"""FastAPI application for wine characteristics inference."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException, Request, Response, status

from . import __version__
from .predictor import WinePredictor
from .schemas import (
    BatchPredictionResponse,
    BatchPredictRequest,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    PredictRequest,
)

LOGGER = logging.getLogger("wine_characteristics.api")
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "artifacts"
    / "wine_characteristics_model.joblib"
)


class Metrics:
    def __init__(self) -> None:
        self._lock = Lock()
        self.requests_total = 0
        self.errors_total = 0
        self.predictions_total = 0
        self.latency_seconds_sum = 0.0

    def record(self, latency: float, is_error: bool) -> None:
        with self._lock:
            self.requests_total += 1
            self.errors_total += int(is_error)
            self.latency_seconds_sum += latency

    def add_predictions(self, count: int) -> None:
        with self._lock:
            self.predictions_total += count

    def render(self, model_loaded: bool) -> str:
        with self._lock:
            return "\n".join(
                [
                    "# TYPE wine_api_requests_total counter",
                    f"wine_api_requests_total {self.requests_total}",
                    "# TYPE wine_api_errors_total counter",
                    f"wine_api_errors_total {self.errors_total}",
                    "# TYPE wine_api_predictions_total counter",
                    f"wine_api_predictions_total {self.predictions_total}",
                    "# TYPE wine_api_latency_seconds_sum counter",
                    f"wine_api_latency_seconds_sum {self.latency_seconds_sum:.6f}",
                    "# TYPE wine_model_loaded gauge",
                    f"wine_model_loaded {int(model_loaded)}",
                    "",
                ]
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    app.state.predictor = None
    app.state.model_error = None
    app.state.metrics = Metrics()
    try:
        app.state.predictor = WinePredictor(model_path)
        LOGGER.info("Loaded model from %s", model_path)
    except (FileNotFoundError, ValueError, OSError) as exc:
        app.state.model_error = str(exc)
        LOGGER.exception("Could not load model")
    yield


app = FastAPI(
    title="Wine Characteristics API",
    description=(
        "Predict acidity, tannins, body, taste intensity, aroma notes and taste notes "
        "from a Russian-language wine description."
    ),
    version=__version__,
    lifespan=lifespan,
)


@app.middleware("http")
async def observe_requests(request: Request, call_next):
    started = time.perf_counter()
    is_error = False
    try:
        response = await call_next(request)
        is_error = response.status_code >= 500
        return response
    except Exception:
        is_error = True
        raise
    finally:
        latency = time.perf_counter() - started
        if hasattr(request.app.state, "metrics"):
            request.app.state.metrics.record(latency, is_error)


def get_predictor(request: Request) -> WinePredictor:
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "Model is unavailable",
                "reason": request.app.state.model_error,
            },
        )
    return predictor


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health(request: Request) -> HealthResponse:
    loaded = request.app.state.predictor is not None
    if not loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "unavailable", "model_loaded": False},
        )
    return HealthResponse(status="ok", model_loaded=True)


@app.get("/model/info", response_model=ModelInfoResponse, tags=["model"])
def model_info(request: Request):
    return get_predictor(request).info()


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(payload: PredictRequest, request: Request):
    predictor = get_predictor(request)
    result = predictor.predict(payload.description)
    request.app.state.metrics.add_predictions(1)
    return result


@app.post(
    "/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"]
)
def predict_batch(payload: BatchPredictRequest, request: Request):
    predictor = get_predictor(request)
    predictions = predictor.predict_batch(payload.descriptions)
    request.app.state.metrics.add_predictions(len(predictions))
    return {"predictions": predictions}


@app.get("/metrics", include_in_schema=False, tags=["service"])
def metrics(request: Request) -> Response:
    body = request.app.state.metrics.render(request.app.state.predictor is not None)
    return Response(content=body, media_type="text/plain; version=0.0.4")
