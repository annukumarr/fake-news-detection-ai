"""
Basic sanity tests for the preprocessing and prediction pipeline.
Run with: pytest tests/
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import clean_text, extract_style_signals
from src.predict import predict_news


def test_clean_text_lowercases_and_strips_punctuation():
    result = clean_text("BREAKING!!! You WON'T believe this...")
    assert result == result.lower()
    assert "!" not in result
    assert "." not in result


def test_clean_text_handles_empty_and_non_string():
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_style_signals_detect_clickbait():
    signals = extract_style_signals("SHOCKING secret they don't want you to know!!!")
    assert signals["exclamations"] >= 1
    assert signals["clickbait_phrases"] >= 1


def test_predict_news_returns_expected_shape():
    result = predict_news(
        "SHOCKING truth about vaccines that doctors don't want you to know!!!",
        use_llm_fallback=False,
        use_factcheck=False,
    )
    assert "final_label" in result
    assert result["final_label"] in ("real", "fake")
    assert 0.0 <= result["final_confidence"] <= 1.0
    assert "ml_prediction" in result


def test_predict_news_on_neutral_text():
    result = predict_news(
        "Officials confirmed that the policy will be reviewed next quarter.",
        use_llm_fallback=False,
        use_factcheck=False,
    )
    assert result["ml_prediction"]["label"] in ("real", "fake")


def test_batch_runs_on_small_csv():
    from src.batch import run_batch_from_csv_bytes
    csv_bytes = (
        b"text\n"
        b"\"SHOCKING: secret cure doctors are hiding!!!\"\n"
        b"\"Officials confirmed the policy will be reviewed next quarter.\"\n"
    )
    result = run_batch_from_csv_bytes(csv_bytes)
    assert result["total_rows"] == 2
    assert len(result["results"]) == 2
    assert all(r["label"] in ("real", "fake") for r in result["results"])
