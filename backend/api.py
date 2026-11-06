"""
api.py
-------
Flask backend for the Fake News Detection web app.

Serves the static frontend (frontend/) and exposes:
  GET  /api/status    -> model metrics, dataset stats, LLM/fact-check availability
  POST /api/predict   -> single-text prediction (ML + optional LLM + fact-check)
  POST /api/batch     -> batch prediction from an uploaded CSV
  GET  /api/batch/csv -> download the last batch run as a CSV

Run with:
    python backend/api.py
Then open http://localhost:5000
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from flask import Flask, request, jsonify, send_from_directory, Response
from dotenv import load_dotenv

load_dotenv(os.path.join(BASE_DIR, ".env"))

import joblib
from src.predict import predict_news, _load_artifacts
from src.llm_fallback import llm_available
from src.factcheck_api import factcheck_available
from src.batch import run_batch_from_csv_bytes, results_to_csv_bytes

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")

_last_batch_results = []  # in-memory cache so the CSV download route can re-serve it


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/status")
def status():
    _load_artifacts()
    metadata = joblib.load(os.path.join(MODELS_DIR, "metadata.joblib"))
    return jsonify({
        "llm_available": llm_available(),
        "factcheck_available": factcheck_available(),
        "primary_model": metadata.get("primary_model"),
        "metrics": metadata.get("metrics"),
        "dataset_size": metadata.get("dataset_size"),
        "class_balance": metadata.get("class_balance"),
        "test_size": metadata.get("test_size"),
        "all_models": metadata.get("all_models"),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Please provide non-empty 'text'."}), 400

    use_llm = bool(data.get("use_llm", True))
    use_factcheck = bool(data.get("use_factcheck", True))

    result = predict_news(text, use_llm_fallback=use_llm, use_factcheck=use_factcheck)
    return jsonify(result)


@app.route("/api/batch", methods=["POST"])
def batch():
    global _last_batch_results

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send a CSV under form field 'file'."}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".csv"):
        return jsonify({"error": "Please upload a .csv file."}), 400

    try:
        result = run_batch_from_csv_bytes(file.read())
    except Exception as e:
        return jsonify({"error": f"Could not process CSV: {e}"}), 400

    _last_batch_results = result.get("results", [])
    return jsonify(result)


@app.route("/api/batch/csv")
def batch_csv():
    if not _last_batch_results:
        return jsonify({"error": "No batch results yet. Run a batch analysis first."}), 400

    csv_bytes = results_to_csv_bytes(_last_batch_results)
    return Response(
        csv_bytes,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=batch_results.csv"},
    )


@app.route("/api/sample-batch-csv")
def sample_batch_csv():
    """A tiny sample CSV so users know the expected batch upload format."""
    sample = (
        "text\n"
        "\"SHOCKING: Scientists discover secret cure that Big Pharma is hiding!!!\"\n"
        "\"Officials confirmed the policy will be reviewed next quarter.\"\n"
        "\"You won't believe what happens next \u2014 click to find out!\"\n"
    )
    return Response(
        sample,
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=sample_batch.csv"},
    )


if __name__ == "__main__":
    print("Starting Fake News Detection server at http://localhost:5000")
    app.run(debug=False, port=5000)
