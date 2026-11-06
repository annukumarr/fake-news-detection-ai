"""
predict.py
-----------
Combined inference pipeline:
  1. Preprocess text and run the primary ML model (Logistic Regression /
     Naive Bayes, whichever was selected as primary during training).
  2. If the ML model's confidence is below LOW_CONFIDENCE_THRESHOLD, call
     the LLM fallback for a second opinion + explanation.
  3. Independently query the Google Fact Check Tools API for related,
     previously fact-checked claims (runs in offline demo mode if no key).

If no trained model is found in models/, this module trains one
automatically from data/news_dataset.csv (bundled real dataset) so the
project works immediately after cloning + `pip install -r requirements.txt`.
"""

import os
import sys
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import clean_text, extract_style_signals
from src.llm_fallback import classify_with_llm, llm_available
from src.factcheck_api import search_fact_checks, factcheck_available

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOW_CONFIDENCE_THRESHOLD = 0.65

_vectorizer = None
_model = None
_metadata = None


def _ensure_model_trained():
    """Auto-train on the bundled dataset if no saved model artifacts exist."""
    required = ["vectorizer.joblib", "logistic_regression.joblib", "naive_bayes.joblib", "metadata.joblib"]
    if all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in required):
        return
    print("No trained model found — training on the bundled dataset now (one-time setup)...")
    from src.train_model import main as train_main
    train_main()


def _load_artifacts():
    global _vectorizer, _model, _metadata
    if _vectorizer is not None and _model is not None:
        return

    _ensure_model_trained()

    _vectorizer = joblib.load(os.path.join(MODELS_DIR, "vectorizer.joblib"))
    _metadata = joblib.load(os.path.join(MODELS_DIR, "metadata.joblib"))

    primary_name = _metadata.get("primary_model", "LogisticRegression")
    model_file = "logistic_regression.joblib" if primary_name == "LogisticRegression" else "naive_bayes.joblib"
    _model = joblib.load(os.path.join(MODELS_DIR, model_file))


def predict_news(text: str, use_llm_fallback: bool = True, use_factcheck: bool = True) -> dict:
    """
    Run the full pipeline on a single piece of news text (headline or
    article body) and return a structured result dict.
    """
    _load_artifacts()

    cleaned = clean_text(text)
    style_signals = extract_style_signals(text)

    vector = _vectorizer.transform([cleaned])
    proba = _model.predict_proba(vector)[0]
    pred_index = int(proba.argmax())
    ml_label = "fake" if pred_index == 1 else "real"
    ml_confidence = float(proba[pred_index])

    result = {
        "input_text": text,
        "ml_prediction": {
            "label": ml_label,
            "confidence": round(ml_confidence, 4),
            "model_used": _metadata.get("primary_model", "unknown"),
        },
        "style_signals": style_signals,
        "llm_review": None,
        "fact_check": None,
        "final_label": ml_label,
        "final_confidence": round(ml_confidence, 4),
        "verification_notes": [],
    }

    if use_llm_fallback and ml_confidence < LOW_CONFIDENCE_THRESHOLD:
        if llm_available():
            llm_result = classify_with_llm(text)
            result["llm_review"] = llm_result
            if llm_result and "label" in llm_result:
                result["final_label"] = llm_result["label"]
                result["final_confidence"] = round(llm_result.get("confidence", ml_confidence), 4)
                result["verification_notes"].append(
                    f"ML confidence was low ({ml_confidence:.2f}); LLM fallback consulted for a second opinion."
                )
        else:
            result["verification_notes"].append(
                "ML confidence was low, but no ANTHROPIC_API_KEY is configured, so the LLM fallback was skipped."
            )

    if use_factcheck:
        fc = search_fact_checks(text)
        result["fact_check"] = fc
        if fc.get("mode") == "live" and fc.get("claims"):
            result["verification_notes"].append(
                f"Found {len(fc['claims'])} related fact-check(s) via Google Fact Check Tools API."
            )
        elif fc.get("mode") == "offline_demo":
            result["verification_notes"].append(fc.get("message", ""))

    return result


if __name__ == "__main__":
    sample = "SHOCKING truth about vaccines that doctors don't want you to know!!!"
    import json
    print(json.dumps(predict_news(sample), indent=2))
