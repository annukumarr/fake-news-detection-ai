"""
train_model.py
----------------
Trains the primary (Logistic Regression) and secondary (Naive Bayes)
classifiers on data/news_dataset.csv, evaluates both, and saves the
best-performing model + fitted vectorizer + full metadata (used by the
dashboard) to the models/ directory.

Run directly:
    python -m src.train_model
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import clean_text
from src.feature_extraction import build_vectorizer, save_vectorizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "news_dataset.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def evaluate(name, model, X_test, y_test):
    preds = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
    }
    cm = confusion_matrix(y_test, preds).tolist()
    print(f"\n[{name}]")
    for k, v in metrics.items():
        print(f"  {k:9s}: {v:.4f}")
    print(f"  confusion matrix: {cm}")
    return metrics, cm


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Run `python scripts/generate_dataset.py` first, or add your own "
            "dataset with `text` and `label` columns."
        )

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    n_real = int((df["label"] == 0).sum())
    n_fake = int((df["label"] == 1).sum())
    print(f"Loaded {len(df)} rows. (real={n_real}, fake={n_fake})")

    print("Cleaning text...")
    df["clean_text"] = df["text"].apply(clean_text)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = build_vectorizer()
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    lr_metrics, lr_cm = evaluate("Logistic Regression", log_reg, X_test, y_test)

    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    nb_metrics, nb_cm = evaluate("Naive Bayes", nb, X_test, y_test)

    # Pick the primary model by F1-score; keep both saved for comparison.
    if lr_metrics["f1"] >= nb_metrics["f1"]:
        best_name, best_metrics = "LogisticRegression", lr_metrics
    else:
        best_name, best_metrics = "MultinomialNB", nb_metrics

    save_vectorizer(vectorizer, os.path.join(MODELS_DIR, "vectorizer.joblib"))
    joblib.dump(log_reg, os.path.join(MODELS_DIR, "logistic_regression.joblib"))
    joblib.dump(nb, os.path.join(MODELS_DIR, "naive_bayes.joblib"))

    metadata = {
        "primary_model": best_name,
        "metrics": best_metrics,
        "dataset_size": int(len(df)),
        "class_balance": {"real": n_real, "fake": n_fake},
        "test_size": int(len(y_test)),
        "all_models": {
            "LogisticRegression": {"metrics": lr_metrics, "confusion_matrix": lr_cm},
            "MultinomialNB": {"metrics": nb_metrics, "confusion_matrix": nb_cm},
        },
    }
    joblib.dump(metadata, os.path.join(MODELS_DIR, "metadata.joblib"))

    print(f"\nPrimary model selected: {best_name} (F1={best_metrics['f1']:.4f})")
    print(f"Saved model artifacts to {MODELS_DIR}/")


if __name__ == "__main__":
    main()
