---
title: Hair & Scalp Health AI
emoji: 🪮
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.32.0"
app_file: app.py
pinned: false
---

# Hair & Scalp Health AI

An AI-powered hair and scalp health assessment tool. Upload a photo and
get an AI-generated condition assessment (with Grad-CAM explainability),
plus a personalized care recommendation based on the photo and a short
lifestyle questionnaire.

**This is an educational/portfolio project, not a medical device.** See
the in-app disclaimer and `src/config.py` for the exact wording.

## What's actually happening under the hood

- **Image classifier**: EfficientNet-B0 (transfer learning from ImageNet),
  fine-tuned to classify hair/scalp condition from a photo.
- **Explainability**: Grad-CAM highlights which region of the image drove
  the prediction, so the output isn't a black box.
- **Recommendation engine**: a RandomForest classifier (scikit-learn)
  trained to map (predicted condition + lifestyle answers) to a care
  category, plus templated guidance per category.
- **Interface**: a single Streamlit app tying all of the above together —
  no separate frontend/backend deployment needed.

## Project status

This repository ships **fully wired and runnable out of the box**,
including a genuinely trained recommendation model
(`models/recommender.pkl`). The image classifier, however, needs to be
fine-tuned on a real hair/scalp image dataset before its predictions are
meaningful — until you do that, the app runs in a clearly-labeled **demo
mode** using the untrained classifier head so you can verify the full
pipeline works immediately.

**See `SETUP_GUIDE.md` for the complete, step-by-step path from "unzip
this" to "deployed on Hugging Face Spaces."**

## Quick reference

| Task | Where |
|---|---|
| Run the app locally | `streamlit run app.py` |
| Fine-tune the classifier | Google Colab, using `training/train.py` |
| Retrain the recommender | `python training/generate_recommendation_dataset.py && python training/train_recommender.py` |
| Deploy | Hugging Face Spaces (Streamlit SDK) — free, no credit card |

## Project structure

```
hair-health-ai/
├── app.py                     # Streamlit app (entry point)
├── requirements.txt
├── src/
│   ├── config.py               # paths, constants, disclaimer text
│   ├── model.py                 # EfficientNet-B0 architecture + checkpoint loading
│   ├── gradcam.py                # Grad-CAM explainability
│   ├── predict.py                # image -> prediction pipeline
│   ├── recommend.py              # recommendation engine
│   └── utils.py                  # preprocessing helpers
├── training/
│   ├── train.py                          # fine-tune the classifier (run in Colab)
│   ├── make_synthetic_dataset.py         # generates fake data to smoke-test train.py
│   ├── generate_recommendation_dataset.py
│   ├── train_recommender.py
│   └── Hair_Health_Classifier_Training.ipynb  # ready-to-run Colab notebook
├── models/
│   ├── recommender.pkl          # already trained, included
│   ├── class_names.json         # written by train.py after fine-tuning
│   └── best_model.pth           # written by train.py after fine-tuning (not included yet)
└── data/sample_images/          # a few images so the app is demoable immediately
```

## Disclaimer

This tool provides general, educational information only and is not a
medical diagnosis. Hair loss can have many underlying causes (genetic,
hormonal, nutritional, autoimmune, and more) that only a qualified
dermatologist or physician can properly evaluate.
