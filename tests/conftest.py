import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault(
    "MODEL_PATH", str(ROOT / "artifacts" / "wine_characteristics_model.joblib")
)
