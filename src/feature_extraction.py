"""
feature_extraction.py
-----------------------
TF-IDF feature extraction, with save/load helpers so the fitted vectorizer
can be reused at inference time without retraining.
"""

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )


def save_vectorizer(vectorizer: TfidfVectorizer, path: str) -> None:
    joblib.dump(vectorizer, path)


def load_vectorizer(path: str) -> TfidfVectorizer:
    return joblib.load(path)
