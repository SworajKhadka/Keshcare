"""
Hair & Scalp Health AI — Streamlit app.

Three AI components come together here:
1. A CNN (EfficientNet-B0, transfer learning) that classifies hair/scalp
   condition from an uploaded photo, with Grad-CAM explainability.
2. A trained recommendation model that combines the CNN's output with a
   short lifestyle questionnaire to suggest a care category.
3. (Optional Phase 2/3 additions from the project roadmap: progress
   tracking over time, and a RAG-based hair-care Q&A assistant.)

Run locally:
    streamlit run app.py
"""

from typing import Optional

import streamlit as st
from PIL import Image

from src import config
from src.predict import predict_image
from src.recommend import get_recommendation

st.set_page_config(
    page_title="Hair & Scalp Health AI",
    page_icon="\U0001FAA0",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def _warm_up():
    """Triggers model loading once per server process and caches it via
    Streamlit's resource cache, so it isn't reloaded on every rerun."""
    from src.predict import _get_cached_model

    return _get_cached_model()


def render_header():
    st.title("\U0001FAA0 Hair & Scalp Health AI")
    st.caption(
        "Upload a photo for an AI-based hair/scalp assessment and a "
        "personalized care suggestion."
    )
    st.info(config.DISCLAIMER, icon="ℹ️")


def render_sidebar_questionnaire():
    st.sidebar.header("A few quick questions")
    st.sidebar.caption(
        "These, combined with the photo analysis, personalize your "
        "recommendation."
    )

    stress_level = st.sidebar.slider("Stress level (1 = low, 5 = high)", 1, 5, 3)
    sleep_hours = st.sidebar.slider("Average sleep (hours/night)", 3.0, 10.0, 7.0, 0.5)
    diet_quality = st.sidebar.slider("Diet quality (1 = poor, 5 = excellent)", 1, 5, 3)
    family_history = st.sidebar.checkbox("Family history of significant hair loss")
    existing_treatment = st.sidebar.checkbox("Already using a hair-care treatment")

    return {
        "stress_level": stress_level,
        "sleep_hours": sleep_hours,
        "diet_quality": diet_quality,
        "family_history": family_history,
        "existing_treatment": existing_treatment,
    }


def get_input_image() -> Optional[Image.Image]:
    st.subheader("1. Provide a photo")
    tab_upload, tab_sample = st.tabs(["Upload your own", "Try a sample image"])

    image = None
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload a clear, well-lit photo of your scalp or hairline",
            type=["jpg", "jpeg", "png"],
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)

    with tab_sample:
        st.caption(
            "No photo handy? Try one of these bundled sample images to see "
            "how the app works end-to-end."
        )
        sample_choice = st.selectbox(
            "Sample image",
            ["-- none --", "Healthy example", "Early stage example",
             "Moderate example", "Advanced example"],
        )
        sample_map = {
            "Healthy example": "sample_healthy.jpg",
            "Early stage example": "sample_early_stage.jpg",
            "Moderate example": "sample_moderate.jpg",
            "Advanced example": "sample_advanced.jpg",
        }
        if sample_choice in sample_map:
            import os

            path = os.path.join(config.SAMPLE_IMAGES_DIR, sample_map[sample_choice])
            if os.path.exists(path):
                image = Image.open(path)

    if image is not None:
        st.image(image, caption="Selected image", width=280)

    return image


def render_results(image: Image.Image, questionnaire: dict):
    with st.spinner("Analyzing image..."):
        result = predict_image(image)

    if not result.is_fine_tuned:
        st.warning(
            "**Demo mode**: no fine-tuned weights found in `models/best_model.pth` "
            "yet, so this prediction comes from the untrained classifier head "
            "(ImageNet backbone only). Run the Colab training notebook on a real "
            "dataset to get meaningful predictions — see SETUP_GUIDE.md. The rest "
            "of the pipeline (preprocessing, Grad-CAM, recommendations) already "
            "works end-to-end.",
            icon="⚠️",
        )

    st.subheader("2. AI assessment")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted condition", result.predicted_class, f"{result.confidence:.0%} confidence")
        st.caption("Confidence across all classes:")
        st.bar_chart(result.class_probabilities)

    with col2:
        st.caption("Grad-CAM: highlighted regions influenced the prediction most")
        st.image(result.gradcam_overlay, width=280)

    st.subheader("3. Personalized recommendation")
    recommendation = get_recommendation(
        predicted_stage=result.predicted_index,
        stress_level=questionnaire["stress_level"],
        sleep_hours=questionnaire["sleep_hours"],
        diet_quality=questionnaire["diet_quality"],
        family_history=questionnaire["family_history"],
        existing_treatment=questionnaire["existing_treatment"],
    )

    st.success(f"**Suggested category: {recommendation.category}**")
    for tip in recommendation.tips:
        st.markdown(f"- {tip}")

    if not recommendation.is_model_based:
        st.caption(
            "(Recommendation model file not found — used the rule-based "
            "fallback. Run training/train_recommender.py to enable the "
            "trained model.)"
        )


def main():
    _warm_up()
    render_header()
    questionnaire = render_sidebar_questionnaire()
    image = get_input_image()

    st.divider()

    if image is not None:
        render_results(image, questionnaire)
    else:
        st.caption("Upload a photo or pick a sample image above to get started.")


if __name__ == "__main__":
    main()
