"""
batch.py
---------
Batch prediction over a CSV of headlines/articles, used by the /api/batch
endpoint and available for reuse from scripts or notebooks.
"""

import io
import pandas as pd

from src.predict import predict_news

MAX_BATCH_ROWS = 200  # keep demo runs fast and bounded


def _pick_text_column(df: pd.DataFrame) -> str:
    for candidate in ["text", "headline", "title", "article", "news"]:
        for col in df.columns:
            if col.strip().lower() == candidate:
                return col
    return df.columns[0]


def run_batch_from_csv_bytes(file_bytes: bytes) -> dict:
    """
    Parses an uploaded CSV's bytes, runs the ML-only pipeline (LLM/fact-check
    are skipped in batch mode to keep it fast) over each row, and returns a
    dict with per-row results plus aggregate counts.
    """
    df = pd.read_csv(io.BytesIO(file_bytes))
    if df.empty:
        return {"error": "The uploaded CSV has no rows.", "results": []}

    text_col = _pick_text_column(df)
    rows = df[text_col].astype(str).tolist()[:MAX_BATCH_ROWS]
    truncated = len(df) > MAX_BATCH_ROWS

    results = []
    fake_count = 0
    for i, text in enumerate(rows):
        r = predict_news(text, use_llm_fallback=False, use_factcheck=False)
        label = r["final_label"]
        if label == "fake":
            fake_count += 1
        results.append({
            "row": i + 1,
            "text": text[:180] + ("..." if len(text) > 180 else ""),
            "label": label,
            "confidence": r["final_confidence"],
        })

    return {
        "column_used": text_col,
        "total_rows": len(results),
        "fake_count": fake_count,
        "real_count": len(results) - fake_count,
        "truncated": truncated,
        "results": results,
    }


def results_to_csv_bytes(results: list) -> bytes:
    out_df = pd.DataFrame(results)
    buf = io.StringIO()
    out_df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")
