"""
Phase 6b: Synthetic Training Data Generator
-----------------------------------------------
Generates a simulated dataset of (behaviour + text features) -> struggle
label, so the model training pipeline can be built and validated NOW,
before a real user study produces labeled data.

This is a legitimate, citable methodological choice, not a shortcut to
hide: several papers in your own literature review used simulated learner
profiles for exactly this reason -- see [17] Kassenkhan et al. (2026),
who validated an adaptive difficulty model on simulated heterogeneous
learner profiles before real deployment. Cite this precedent in your
report's Methodology/Evaluation section when you describe this step.

How the simulation works:
  A hidden ground-truth function combines dwell time, revisits, and text
  complexity into a "true" struggle probability, then samples a binary
  label from it (Bernoulli). This produces a dataset with realistic,
  learnable-but-noisy structure -- similar to what real behavioural data
  looks like -- rather than a trivially separable toy problem.

IMPORTANT -- what to say in your report:
  This model is trained on simulated data and should be described as a
  proof-of-concept / architecture validation step, NOT as evidence of
  real-world predictive accuracy. Section 11 of your report ("Evaluation
  and Analysis") correctly calls for real user studies -- do that next,
  using this same training pipeline, once you have logged real sessions
  from app.py's behaviour tracker.
"""

import random
import csv

from features import FEATURE_NAMES


def _sigmoid(x: float) -> float:
    return 1 / (1 + pow(2.718281828, -x))


def generate_sample(rng: random.Random) -> dict:
    """Generates one simulated (features..., label) row."""
    # Behavioural features -- most readers are fine, some struggle
    dwell_seconds = max(0, rng.gauss(35, 25))          # mean 35s, some long tails
    revisit_count = max(0, int(rng.gauss(0.7, 1.0)))   # mostly 0-1, occasionally more

    # Text complexity features -- simulate a range of section difficulties
    word_count = max(10, int(rng.gauss(150, 60)))
    avg_word_length = max(3.0, rng.gauss(5.2, 0.8))
    flesch_reading_ease = max(0, min(100, rng.gauss(55, 20)))   # lower = harder
    flesch_kincaid_grade = max(1, rng.gauss(11, 3))             # higher = harder

    # Attention feature -- most readers are mostly attentive (mean ~0.85),
    # with a tail of lower-attention sessions. Modeled as its own signal
    # rather than derived from the other features, since real attention
    # lapses (distraction, fatigue) aren't fully explained by dwell time
    # or text difficulty alone.
    avg_attention_score = max(0.0, min(1.0, rng.gauss(0.85, 0.18)))

    features = [dwell_seconds, revisit_count, word_count, avg_word_length,
                flesch_reading_ease, flesch_kincaid_grade, avg_attention_score]

    # Hidden ground-truth relationship (this is what the model has to learn)
    # Struggle increases with: dwell time, revisits, low reading ease,
    # high grade level, and LOW attention (note the negative coefficient
    # sign on avg_attention_score, since lower attention -> more struggle)
    z = (
        0.045 * dwell_seconds
        + 0.9 * revisit_count
        - 0.035 * flesch_reading_ease
        + 0.12 * flesch_kincaid_grade
        - 1.8 * avg_attention_score
        - 1.3  # bias term, keeps baseline struggle rate reasonable
    )
    true_prob = _sigmoid(z)
    label = 1 if rng.random() < true_prob else 0

    return dict(zip(FEATURE_NAMES, features), label=label)


def generate_dataset(n_samples: int = 2000, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    return [generate_sample(rng) for _ in range(n_samples)]


def save_csv(rows: list[dict], path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FEATURE_NAMES + ["label"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    rows = generate_dataset(n_samples=2000)
    save_csv(rows, "synthetic_training_data.csv")
    positive_rate = sum(r["label"] for r in rows) / len(rows)
    print(f"Generated {len(rows)} samples -> synthetic_training_data.csv")
    print(f"Positive (struggling) rate: {positive_rate:.1%}")