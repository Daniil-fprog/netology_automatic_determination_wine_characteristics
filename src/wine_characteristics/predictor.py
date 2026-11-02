"""Inference wrapper around the model bundle produced by the training notebook."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def clean_text(value: str) -> str:
    """Apply the same lightweight normalization as the training notebook."""
    value = str(value).lower().replace("ё", "е")
    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


class WinePredictor:
    """Load a serialized bundle and expose stable prediction methods."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")
        self.bundle: dict[str, Any] = joblib.load(self.model_path)
        self._validate_bundle()

    def _validate_bundle(self) -> None:
        required = {
            "vectorizer",
            "categorical_models",
            "multilabel_models",
            "metadata",
        }
        missing = required.difference(self.bundle)
        if missing:
            raise ValueError(f"Invalid model bundle; missing keys: {sorted(missing)}")

    @staticmethod
    def _positive_probability(classifier: Any, features: Any) -> float:
        classes = list(classifier.classes_)
        if 1 not in classes:
            return 0.0
        return float(classifier.predict_proba(features)[0, classes.index(1)])

    def predict(self, description: str) -> dict[str, Any]:
        text = clean_text(description)
        if not text:
            raise ValueError("Description must not be empty")

        features = self.bundle["vectorizer"].transform([text])
        result: dict[str, Any] = {
            "categories": {},
            "aroma_notes": [],
            "taste_notes": [],
            "model_mode": self.mode,
        }

        for target, classifier in self.bundle["categorical_models"].items():
            label = str(classifier.predict(features)[0])
            confidence = None
            if hasattr(classifier, "predict_proba"):
                confidence = float(np.max(classifier.predict_proba(features)[0]))
            result["categories"][target] = {
                "label": label,
                "confidence": confidence,
            }

        for target, target_bundle in self.bundle["multilabel_models"].items():
            notes = []
            for label, classifier in target_bundle["models"].items():
                probability = self._positive_probability(classifier, features)
                if probability >= target_bundle["threshold"]:
                    notes.append(
                        {"label": str(label), "probability": round(probability, 4)}
                    )
            result[target] = sorted(
                notes, key=lambda item: item["probability"], reverse=True
            )
        return result

    def predict_batch(self, descriptions: list[str]) -> list[dict[str, Any]]:
        return [self.predict(description) for description in descriptions]

    @property
    def mode(self) -> str:
        return (
            "weak_supervision_demo"
            if self.bundle["metadata"].get("demo_weak_labels")
            else "supervised"
        )

    def info(self) -> dict[str, Any]:
        metadata = dict(self.bundle["metadata"])
        return {
            "model_path": str(self.model_path),
            "mode": self.mode,
            "n_samples": metadata.get("n_samples"),
            "categorical_targets": sorted(self.bundle["categorical_models"]),
            "multilabel_targets": sorted(self.bundle["multilabel_models"]),
            "metadata": metadata,
        }
