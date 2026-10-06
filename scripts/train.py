"""Trains a TF-IDF + Logistic Regression text classifier and saves it to models/model.joblib."""
import json

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

df = pd.read_csv("data/tickets.csv")
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
)
model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), lowercase=True, stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000)),
])
model.fit(X_train, y_train)
pred = model.predict(X_test)
report = classification_report(y_test, pred, output_dict=True)
print(classification_report(y_test, pred))
joblib.dump(model, "models/model.joblib")
with open("models/metrics.json", "w") as f:
    json.dump({"accuracy": round(report["accuracy"], 3), "test_size": len(y_test),
               "note": "Synthetic template data: accuracy is optimistic."}, f, indent=2)
print("Saved models/model.joblib")
