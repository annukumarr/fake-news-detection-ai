"""
llm_fallback.py
-----------------
Optional second-opinion / explanation layer that calls an LLM (Anthropic's
Claude API by default) when the primary ML model's confidence is low, or
whenever an explanation of the verdict is requested.

This is designed to fail gracefully: if no API key is configured, or the
`anthropic` package isn't installed, the app/CLI will simply skip this step
and rely on the ML model + fact-check API alone.
"""

import os
import json

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()

try:
    import anthropic
    _ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    _ANTHROPIC_SDK_AVAILABLE = False


def llm_available() -> bool:
    return bool(ANTHROPIC_API_KEY) and _ANTHROPIC_SDK_AVAILABLE


def classify_with_llm(text: str, model: str = "claude-sonnet-4-6") -> dict | None:
    """
    Ask an LLM to independently judge whether the given news text looks
    real or fake, and to explain briefly why. Returns None if the LLM
    layer isn't configured, so callers can fall back to ML-only output.
    """
    if not llm_available():
        return None

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    prompt = f"""You are assisting a fake-news detection system as a secondary reviewer.

The primary machine learning model had low confidence on the following news
text. Give your own independent judgement.

Respond with ONLY a JSON object (no markdown, no preamble) in this exact form:
{{"label": "real" or "fake", "confidence": a number between 0 and 1, "explanation": "one or two sentence reason"}}

News text:
\"\"\"{text}\"\"\"
"""

    try:
        response = client.messages.create(
            model=model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(block.text for block in response.content if hasattr(block, "text")).strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(raw)
        return {
            "label": parsed.get("label", "unknown"),
            "confidence": float(parsed.get("confidence", 0.5)),
            "explanation": parsed.get("explanation", ""),
            "source": "llm",
        }
    except Exception as e:
        return {"error": str(e), "source": "llm"}
