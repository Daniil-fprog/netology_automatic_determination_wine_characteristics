from pathlib import Path

from wine_characteristics.predictor import WinePredictor, clean_text


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "wine_characteristics_model.joblib"


def test_clean_text_normalizes_text():
    assert clean_text("  СВЕЖЁЕ\nвино  ") == "свежее вино"


def test_predictor_returns_all_targets():
    predictor = WinePredictor(MODEL_PATH)
    result = predictor.predict(
        "Насыщенное красное вино с ароматом вишни, специй и мягкими танинами."
    )

    assert set(result["categories"]) == {
        "acidity_level",
        "tannin_level",
        "body_level",
        "intensity_level",
    }
    assert isinstance(result["aroma_notes"], list)
    assert isinstance(result["taste_notes"], list)
    assert result["model_mode"] in {"weak_supervision_demo", "supervised"}


def test_predictor_rejects_empty_description():
    predictor = WinePredictor(MODEL_PATH)
    try:
        predictor.predict("   ")
    except ValueError as exc:
        assert "must not be empty" in str(exc)
    else:
        raise AssertionError("Empty description was accepted")
