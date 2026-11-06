
from flask import Flask, render_template, request
import joblib
from pathlib import Path
from urllib.parse import quote_plus

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
model = joblib.load(BASE_DIR / "fake_news_model.joblib")

LABELS = {0: "FAKE", 1: "REAL"}


def make_fact_check_links(article):
    # Use a short excerpt to create focused search queries.
    query = " ".join(article.split())[:220]
    encoded = quote_plus(query)

    return [
        {
            "name": "Google News",
            "description": "Find related news coverage",
            "url": f"https://news.google.com/search?q={encoded}"
        },
        {
            "name": "Reuters Fact Check",
            "description": "Search Reuters fact-checking reports",
            "url": (
                "https://www.google.com/search?q="
                + quote_plus("site:reuters.com/fact-check " + query)
            )
        },
        {
            "name": "AP News Verification",
            "description": "Search AP reporting and verification",
            "url": (
                "https://www.google.com/search?q="
                + quote_plus("site:apnews.com fact check " + query)
            )
        },
        {
            "name": "Google Fact Check Explorer",
            "description": "Look for previously investigated claims",
            "url": (
                "https://toolbox.google.com/factcheck/explorer/search/"
                "?query=" + encoded
            )
        }
    ]


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    error = None
    article = ""

    if request.method == "POST":
        article = request.form.get("article", "").strip()

        if len(article) < 30:
            error = "Please enter a news article with at least 30 characters."
        elif len(article) > 50000:
            error = "Article is too long. Please keep it under 50,000 characters."
        else:
            prediction = int(model.predict([article])[0])

            confidence = None
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba([article])[0]
                confidence = round(float(max(probabilities)) * 100, 2)

            result = {
                "label": LABELS.get(prediction, str(prediction)),
                "confidence": confidence,
                "fact_links": make_fact_check_links(article)
            }

    return render_template(
        "index.html",
        result=result,
        error=error,
        article=article
    )


if __name__ == "__main__":
    app.run(debug=True)
