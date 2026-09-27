# Heart AI — Translational Biomedical ML

A portfolio-grade research software project that turns the Kaggle Heart Failure Prediction Dataset into a reproducible, testable and deployable ML workflow.

## Project goals

- modular typed Python rather than notebook-only code;
- reproducible preprocessing + ML pipelines;
- explicit validation, logging and error handling;
- automated tests and GitHub Actions CI;
- FastAPI endpoints for analysis and inference;
- Streamlit dashboard for model/feature selection and plots;
- Docker packaging and local multi-service orchestration.

**This is a research/portfolio demonstration, not a clinical decision-support system.**

## Dataset

Source: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

The source describes 918 final observations, 11 predictors and the binary target `HeartDisease`. It was assembled from five historical heart-disease datasets; the source reports 1190 observations before removal of 272 duplicates. The CSV is not committed to this repository: download it from Kaggle and save it as `data/heart.csv`.

Predictors: `Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope`.

## Architecture

```text
data/heart.csv -> data.py -> preprocessing.py -> pipeline.py -> model
                                             |                 |
                                             +-> evaluation <--+
                                                       |
                                      +----------------+----------------+
                                      |                                 |
                                  FastAPI                         Streamlit
```

## Models

Logistic Regression, Random Forest and SVM are intentionally used as interpretable/standard baselines for a small tabular dataset. The project is about software quality and reproducible ML lifecycle rather than leaderboard optimisation.

## Metrics

Accuracy, balanced accuracy, precision, recall, F1, ROC AUC, confusion matrix and ROC curve. In biomedical work, accuracy alone is insufficient; metric choice must reflect the scientific/clinical question.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
# put heart.csv in data/
python -m pytest
uvicorn heart_ai.api:app --reload
streamlit run app/streamlit_app.py
```

API docs: `http://localhost:8000/docs`

## Docker

```bash
docker compose up --build
```

API: `http://localhost:8000/docs`  
Web app: `http://localhost:8501`

## Why this architecture

The important design choice is that preprocessing is inside the scikit-learn pipeline. Imputation, scaling and one-hot encoding are therefore fitted only on the training split, preventing a common form of leakage. The same fitted pipeline is the object used for inference.

The repository deliberately separates data validation, preprocessing, model construction, evaluation, API schemas and UI. This makes each component independently testable and easier to replace.

## Limitations and scientific caveats

The dataset is small and combines multiple historical cohorts. This portfolio project does not establish clinical validity or generalisation to a new hospital. A real translational workflow would require external/site validation, calibration, subgroup analysis, uncertainty estimates, data provenance, privacy/security controls and prospective evaluation.

## Roadmap

Calibration/Brier score, bootstrap confidence intervals, subgroup evaluation, model cards, data contracts (e.g. Pandera), experiment tracking, stronger artifact/version management, privacy-aware local validation, HPC/cloud execution and workflow management (e.g. Nextflow/Snakemake).
