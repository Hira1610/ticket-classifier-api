import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    if not (ROOT / "models" / "model.joblib").exists():
        subprocess.run([sys.executable, "scripts/make_dataset.py"], cwd=ROOT, check=True)
        subprocess.run([sys.executable, "scripts/train.py"], cwd=ROOT, check=True)


@pytest.fixture
def client():
    from app.main import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


@pytest.mark.parametrize("text,expected", [
    ("I was charged twice for my invoice", "billing"),
    ("The app keeps crashing with an error", "technical"),
    ("I forgot my password and cannot sign in", "account"),
    ("My package has not arrived yet", "shipping"),
])
def test_predicts_expected_category(client, text, expected):
    r = client.post("/predict", json={"text": text})
    assert r.status_code == 200
    body = r.json()
    assert body["category"] == expected
    assert 0 <= body["confidence"] <= 1
    assert abs(sum(body["scores"].values()) - 1) < 0.05


def test_rejects_empty_text(client):
    assert client.post("/predict", json={"text": ""}).status_code == 422


def test_batch(client):
    r = client.post("/predict/batch", json=[{"text": "refund my payment please"}, {"text": "parcel delivered to wrong address"}])
    assert r.status_code == 200 and len(r.json()) == 2


def test_batch_limit(client):
    assert client.post("/predict/batch", json=[]).status_code == 400
