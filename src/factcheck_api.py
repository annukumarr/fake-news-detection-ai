"""
factcheck_api.py
------------------
Live verification layer using Google's Fact Check Tools API
(https://developers.google.com/fact-check/tools/api).

Runs in "offline/demo mode" automatically if GOOGLE_FACTCHECK_API_KEY is not
set, so the project works out of the box on a fresh clone with no API keys
configured. Add a key to `.env` to enable live corroboration.
"""

import os
import requests

GOOGLE_FACTCHECK_API_KEY = os.environ.get("GOOGLE_FACTCHECK_API_KEY", "").strip()
FACTCHECK_ENDPOINT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


def factcheck_available() -> bool:
    return bool(GOOGLE_FACTCHECK_API_KEY)


def search_fact_checks(query: str, max_results: int = 5) -> dict:
    """
    Query the Google Fact Check Tools API for existing fact-checks related
    to the given text/claim. Returns a dict with a `mode` field indicating
    whether this was a live lookup or offline/demo fallback.
    """
    query = (query or "").strip()
    if not query:
        return {"mode": "empty_query", "claims": []}

    if not factcheck_available():
        return {
            "mode": "offline_demo",
            "message": (
                "No GOOGLE_FACTCHECK_API_KEY configured — running in offline demo mode. "
                "Add a key to .env to enable live fact-check corroboration."
            ),
            "claims": [],
        }

    # Fact Check API works best on short claims; trim very long article text.
    short_query = query[:200]

    try:
        resp = requests.get(
            FACTCHECK_ENDPOINT,
            params={"query": short_query, "key": GOOGLE_FACTCHECK_API_KEY, "pageSize": max_results},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        claims = []
        for c in data.get("claims", [])[:max_results]:
            review = (c.get("claimReview") or [{}])[0]
            claims.append({
                "claim_text": c.get("text", ""),
                "claimant": c.get("claimant", "Unknown"),
                "rating": review.get("textualRating", "Unrated"),
                "publisher": review.get("publisher", {}).get("name", "Unknown"),
                "url": review.get("url", ""),
            })
        return {"mode": "live", "claims": claims}

    except requests.exceptions.RequestException as e:
        return {"mode": "error", "message": str(e), "claims": []}
