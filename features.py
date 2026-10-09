"""
Phase 6a: Feature Engineering
--------------------------------
Extracts the feature vector used by the knowledge-gap prediction model.

Features combine:
  - Behavioural signals (from Phase 4's tracking): dwell time, revisits
  - Textual complexity of the section being read: word count, average
    word length, and standard readability scores (Flesch Reading Ease,
    Flesch-Kincaid Grade Level)

Why readability scores instead of full SciBERT embeddings for this model:
Your report specifies combining SciBERT embeddings with behavioural
features for the deep learning model. That's the right target architecture
-- but training a neural net on 384+ dimensional embeddings needs either
a large labeled dataset or heavy regularization, and there's no real
labeled dataset yet (that requires an actual user study). A small,
interpretable feature set lets you train a working, honestly-evaluated
baseline NOW with a handful of simulated/pilot samples, and is exactly
the kind of classical-ML-first step your own roadmap called for before
jumping to PyTorch. Swapping this feature extractor for SciBERT embeddings
later only requires changing this one file -- train_model.py and
gap_predictor.py don't need to change.
"""

import textstat


FEATURE_NAMES = [
    "dwell_seconds",
    "revisit_count",
    "word_count",
    "avg_word_length",
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "avg_attention_score",
]

NEUTRAL_ATTENTION_SCORE = 1.0  # used when no webcam data is available for this section


def extract_features(dwell_seconds: float, revisit_count: int, section_text: str,
                      avg_attention_score: float = None) -> list[float]:
    """
    Returns a feature vector in the fixed order defined by FEATURE_NAMES.
    Safe against empty/short text (textstat can misbehave on very short inputs).

    avg_attention_score: the average webcam-derived attention score (0-1)
    for this section, from gaze_tracker.py via app.py's tracking. If the
    webcam feature was never used for this section, pass None -- this
    falls back to NEUTRAL_ATTENTION_SCORE (1.0, i.e. "assume attentive")
    rather than 0, so that NOT using the webcam doesn't get misread by
    the model as "very distracted." This keeps the feature optional: the
    model still works for readers who never enable the camera.
    """
    if avg_attention_score is None:
        avg_attention_score = NEUTRAL_ATTENTION_SCORE
    text = section_text.strip() if section_text else ""
    words = text.split()
    word_count = len(words)
    avg_word_length = (sum(len(w) for w in words) / word_count) if word_count > 0 else 0.0

    if word_count >= 10:
        try:
            reading_ease = textstat.flesch_reading_ease(text)
            grade_level = textstat.flesch_kincaid_grade(text)
        except Exception:
            reading_ease, grade_level = 50.0, 10.0  # neutral fallback
    else:
        reading_ease, grade_level = 50.0, 10.0  # not enough text to score reliably

    return [
        float(dwell_seconds),
        float(revisit_count),
        float(word_count),
        float(avg_word_length),
        float(reading_ease),
        float(grade_level),
        float(avg_attention_score),
    ]