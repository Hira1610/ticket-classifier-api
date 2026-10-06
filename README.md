# Support Ticket Classifier (Python, scikit-learn, FastAPI)

A small ML service that classifies customer support messages into `billing`, `technical`, `account` or `shipping`.

## Features
- TF-IDF (unigrams + bigrams) with Logistic Regression in a scikit-learn pipeline
- FastAPI endpoints: `/predict` (category, confidence, scores for every class), `/predict/batch`, `/health`
- Input validation with Pydantic; clear error when the model has not been trained
- 8 automated tests (pytest), Dockerfile and GitHub Actions CI

## Important note about the data
The dataset is **synthetic** (generated from templates by `scripts/make_dataset.py`). The 100% test accuracy
reflects how simple that data is, not real-world performance. To make this meaningful, replace
`data/tickets.csv` with real labelled messages and compare against a baseline.

## Tech
Python, scikit-learn, pandas, FastAPI, Pydantic, joblib, pytest, Docker

## Run locally
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/make_dataset.py
python scripts/train.py
uvicorn app.main:app --reload       # docs at http://localhost:8000/docs
pytest -q
```

Example:
```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{"text":"I was charged twice for my invoice"}'
```

## Possible next steps
Real dataset, model comparison (Naive Bayes, SVM), confusion matrix, Urdu/Roman Urdu support, model versioning.
