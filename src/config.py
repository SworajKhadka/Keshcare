"""
Central configuration for the Hair & Scalp Health AI app.

Keeping every path and constant in one place means the rest of the
codebase never hardcodes a filename — if you rename or move something,
you only change it here.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODELS_DIR = os.path.join(BASE_DIR, "models")
CLASSIFIER_WEIGHTS_PATH = os.path.join(MODELS_DIR, "best_model.pth")
CLASS_NAMES_PATH = os.path.join(MODELS_DIR, "class_names.json")
RECOMMENDER_MODEL_PATH = os.path.join(MODELS_DIR, "recommender.pkl")

SAMPLE_IMAGES_DIR = os.path.join(BASE_DIR, "data", "sample_images")

# ---------------------------------------------------------------------------
# Classifier defaults
# ---------------------------------------------------------------------------
# These are used ONLY if models/class_names.json does not exist yet (i.e.
# before you've run the Colab training notebook on a real dataset). Once
# you train on your own data with training/train.py, class names are read
# from class_names.json instead, so this list is just a safe fallback that
# lets the app run out of the box in "demo mode".
DEFAULT_CLASS_NAMES = [
    "Healthy / Minimal Loss",
    "Early Stage Thinning",
    "Moderate Hair Loss",
    "Advanced Hair Loss",
]

IMAGE_SIZE = 224  # standard input size for EfficientNet-B0

# ImageNet normalization stats — required because the backbone is
# pretrained on ImageNet.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Recommendation engine
# ---------------------------------------------------------------------------
CARE_CATEGORIES = [
    "Preventive Care",
    "Mild Care Routine",
    "Active Treatment",
    "Consult a Specialist",
]

CARE_CATEGORY_TIPS = {
    "Preventive Care": [
        "Your scalp and hair currently look healthy — the goal now is to keep it that way.",
        "Use a gentle, sulfate-free shampoo 2-3 times a week rather than daily.",
        "Maintain a balanced diet with enough protein, iron, and biotin.",
        "Avoid tight hairstyles that put repeated tension on the same hairline area.",
    ],
    "Mild Care Routine": [
        "Some early thinning is visible — small consistent habits make the biggest difference now.",
        "Consider a scalp massage routine (5 minutes/day) to support blood flow to follicles.",
        "Reduce heat styling and harsh chemical treatments where possible.",
        "Track your progress with a monthly photo so you can objectively see trends.",
    ],
    "Active Treatment": [
        "The pattern suggests it may be worth exploring clinically studied options.",
        "Over-the-counter topical minoxidil is the most evidence-backed non-prescription option — research it and see if it fits your situation.",
        "Prioritize sleep and stress management — both are strongly linked to hair shedding cycles.",
        "Re-check your progress tracker every 4-6 weeks; treatments typically need 3+ months to show visible change.",
    ],
    "Consult a Specialist": [
        "The combination of factors here is significant enough that a dermatologist's input would be genuinely useful.",
        "A specialist can run tests (e.g., for thyroid, iron, hormonal factors) that an app fundamentally cannot.",
        "Bring your progress-tracker history with you — objective photos over time help a specialist a lot.",
        "This app is an informational aid, not a diagnostic tool — please treat this suggestion seriously.",
    ],
}

RECOMMENDER_FEATURE_COLUMNS = [
    "predicted_stage",
    "stress_level",
    "sleep_hours",
    "diet_quality",
    "family_history",
    "existing_treatment",
]

DISCLAIMER = (
    "This tool provides general, educational information only and is not a "
    "medical diagnosis. Hair loss can have many underlying causes (genetic, "
    "hormonal, nutritional, autoimmune, and more) that only a qualified "
    "dermatologist or physician can properly evaluate. If you're concerned "
    "about hair loss, please consult a healthcare professional."
)
