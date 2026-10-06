from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
app = FastAPI(title="Support Ticket Classifier", version="1.0.0")
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, "Model not trained yet. Run: python scripts/train.py")
        _model = joblib.load(MODEL_PATH)
    return _model


class Ticket(BaseModel):
    text: str = Field(min_length=3, max_length=2000, examples=["I was charged twice for my invoice"])


class Prediction(BaseModel):
    category: str
    confidence: float
    scores: dict[str, float]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.post("/predict", response_model=Prediction)
def predict(ticket: Ticket):
    model = get_model()
    probs = model.predict_proba([ticket.text])[0]
    classes = list(model.classes_)
    best = max(range(len(classes)), key=lambda i: probs[i])
    return Prediction(
        category=classes[best],
        confidence=round(float(probs[best]), 3),
        scores={c: round(float(p), 3) for c, p in zip(classes, probs)},
    )


@app.post("/predict/batch")
def predict_batch(tickets: list[Ticket]):
    if not tickets or len(tickets) > 100:
        raise HTTPException(400, "Send between 1 and 100 tickets")
    model = get_model()
    labels = model.predict([t.text for t in tickets])
    return [{"text": t.text, "category": str(l)} for t, l in zip(tickets, labels)]
