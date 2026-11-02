"""Pydantic contracts for the HTTP API."""

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    description: str = Field(
        min_length=10,
        max_length=20_000,
        examples=[
            "Насыщенное красное вино с ароматом вишни, специй и мягкими танинами."
        ],
    )


class BatchPredictRequest(BaseModel):
    descriptions: list[str]

    @field_validator("descriptions")
    @classmethod
    def validate_descriptions(cls, descriptions: list[str]) -> list[str]:
        if not 1 <= len(descriptions) <= 100:
            raise ValueError("Batch must contain from 1 to 100 descriptions")
        if any(not 10 <= len(item) <= 20_000 for item in descriptions):
            raise ValueError("Each description must contain from 10 to 20000 characters")
        return descriptions


class CategoryPrediction(BaseModel):
    label: str
    confidence: float | None


class NotePrediction(BaseModel):
    label: str
    probability: float


class PredictionResponse(BaseModel):
    categories: dict[str, CategoryPrediction]
    aroma_notes: list[NotePrediction]
    taste_notes: list[NotePrediction]
    model_mode: str


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    model_path: str
    mode: str
    n_samples: int | None
    categorical_targets: list[str]
    multilabel_targets: list[str]
    metadata: dict
