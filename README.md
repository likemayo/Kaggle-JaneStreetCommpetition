# Jane Street Market Prediction — Submission Notebook

This repository contains the **inference notebook** used for my final submission to
the Jane Street Real-Time Market Data Forecasting competition (Kaggle, 2024–2025).

**Final placement: 120 / 3757.**

## What is in this repo

`janestreet2024-baseline-submission-v1-cfe0f5.ipynb` — the submission script. It loads
pre-trained model artifacts (LightGBM, XGBoost, CatBoost, and a PyTorch Lightning NN)
from a Kaggle dataset attached at runtime, applies the feature pipeline, and serves
predictions through the competition's inference API.

## What is not in this repo

The training code and the model artifacts are not included — they live in a private
Kaggle dataset and are not redistributable. This notebook is the inference half only.

## Attribution

The notebook imports `winwinjs` (a compiled helper distributed with the shared
competition baseline) and `tabm_reference`, which is third-party TabM code and not mine.