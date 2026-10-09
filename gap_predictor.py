"""
Phase 6d: Trained Gap Predictor
-----------------------------------
Loads gap_model.joblib (produced by train_model.py) and exposes a
predict function with the same shape as app.py's rule-based
compute_gap_score() -- this is the "one function swap" the Phase 4
architecture was designed around.

If no trained model file exists yet, predict_gap_score() returns None,
and the caller (app.py) falls back to the heuristic automatically.
"""

import os
import joblib

from features import extract_features

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "gap_model.joblib")
_bundle = None


def _load_bundle():
    global _bundle
    if _bundle is None and os.path.exists(_MODEL_PATH):
        _bundle = joblib.load(_MODEL_PATH)
    return _bundle


def model_available() -> bool:
    return os.path.exists(_MODEL_PATH)


def predict_gap_score(dwell_seconds: float, revisit_count: int, section_text: str,
                       avg_attention_score: float = None):
    """
    Returns a struggle probability in [0, 1], or None if no trained model
    is available yet (caller should fall back to the heuristic in that case).

    avg_attention_score: optional webcam-derived attention score (0-1) for
    this section. Pass None if webcam tracking wasn't used -- the feature
    extractor substitutes a neutral "assume attentive" value in that case,
    so the model still works for readers who never enable the camera.
    """
    bundle = _load_bundle()
    if bundle is None:
        return None

    model = bundle["model"]
    scaler = bundle["scaler"]

    features = [extract_features(dwell_seconds, revisit_count, section_text, avg_attention_score)]
    if scaler is not None:
        features = scaler.transform(features)

    proba = model.predict_proba(features)[0][1]
    return round(float(proba), 2)


if __name__ == "__main__":
    if not model_available():
        print("No trained model found. Run train_model.py first.")
    else:
        # quick smoke test
        score = predict_gap_score(
            dwell_seconds=60,
            revisit_count=2,
            section_text="This is a moderately complex scientific passage about photosynthesis and the Calvin cycle." * 5,
        )
        print(f"Predicted struggle probability: {score}")