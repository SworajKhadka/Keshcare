"""
Generates a seed dataset for the recommendation engine.

Real user-feedback data doesn't exist yet (the app hasn't launched), so we
bootstrap with a heuristically-labeled synthetic dataset: features are
sampled randomly within realistic ranges, and labels are assigned using a
scoring rule (loosely based on common dermatological risk-factor guidance)
plus injected noise so the classification task isn't trivially linear.

This is a standard, disclosed technique for cold-starting a recommender
system before real interaction data is available. Once the app has real
users, training/train_recommender.py can be re-pointed at logged
(features -> outcome) data instead of this synthetic set.

Run:
    python training/generate_recommendation_dataset.py
"""

import os

import numpy as np
import pandas as pd

OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__), "recommendation_seed_data.csv"
)

N_SAMPLES = 4000
RANDOM_SEED = 42


def score_to_category(score: float) -> int:
    if score < 2:
        return 0  # Preventive Care
    elif score < 4:
        return 1  # Mild Care Routine
    elif score < 6:
        return 2  # Active Treatment
    else:
        return 3  # Consult a Specialist


def generate_dataset(n_samples: int = N_SAMPLES, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    predicted_stage = rng.integers(0, 4, size=n_samples)  # 4 classifier stages
    stress_level = rng.integers(1, 6, size=n_samples)  # 1-5
    sleep_hours = rng.uniform(3.0, 9.5, size=n_samples).round(1)
    diet_quality = rng.integers(1, 6, size=n_samples)  # 1-5
    family_history = rng.integers(0, 2, size=n_samples)  # 0/1
    existing_treatment = rng.integers(0, 2, size=n_samples)  # 0/1

    score = predicted_stage.astype(float) * 2.0
    score += np.where(sleep_hours < 6, 1.0, 0.0)
    score += np.where(stress_level >= 4, 1.0, 0.0)
    score += np.where(diet_quality <= 2, 1.0, 0.0)
    score += np.where(family_history == 1, 1.0, 0.0)
    score -= np.where(existing_treatment == 1, 0.5, 0.0)

    # Inject noise so the classifier has to genuinely learn a boundary
    # rather than memorize a perfectly separable rule.
    noise = rng.normal(loc=0.0, scale=0.9, size=n_samples)
    noisy_score = score + noise

    care_category = np.array([score_to_category(s) for s in noisy_score])

    df = pd.DataFrame(
        {
            "predicted_stage": predicted_stage,
            "stress_level": stress_level,
            "sleep_hours": sleep_hours,
            "diet_quality": diet_quality,
            "family_history": family_history,
            "existing_treatment": existing_treatment,
            "care_category": care_category,
        }
    )
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")
    print(df["care_category"].value_counts().sort_index())
