"""
preprocessing.py
-----------------
Text cleaning and NLP preprocessing utilities used before feature extraction.
"""

import re
import string

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    def _ensure_nltk_data():
        for pkg, path in [("stopwords", "corpora/stopwords"), ("wordnet", "corpora/wordnet")]:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(pkg, quiet=True)

    _ensure_nltk_data()
    _STOPWORDS = set(stopwords.words("english"))
    _LEMMATIZER = WordNetLemmatizer()
    _NLTK_AVAILABLE = True
except Exception:
    # Graceful fallback if NLTK data can't be downloaded (e.g. no internet).
    _STOPWORDS = {
        "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "in", "on",
        "at", "to", "for", "of", "with", "this", "that", "it", "as", "by", "be", "has",
        "have", "had", "not", "from", "will", "would", "can", "could", "should",
    }
    _LEMMATIZER = None
    _NLTK_AVAILABLE = False

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE = re.compile(r"<.*?>")
_NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
_MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Lowercase, strip URLs/HTML/punctuation/numbers, remove stopwords, lemmatize."""
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = _NON_ALPHA_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()

    tokens = text.split()
    tokens = [t for t in tokens if t not in _STOPWORDS and len(t) > 1]

    if _NLTK_AVAILABLE and _LEMMATIZER is not None:
        tokens = [_LEMMATIZER.lemmatize(t) for t in tokens]

    return " ".join(tokens)


def extract_style_signals(text: str) -> dict:
    """
    Extract simple stylistic signals often associated with sensational / fake
    content (used as auxiliary signals for the CLI/app explanation, not fed
    into the ML model itself).
    """
    if not isinstance(text, str) or not text:
        return {"exclamations": 0, "all_caps_words": 0, "clickbait_phrases": 0}

    exclamations = text.count("!")
    all_caps_words = sum(1 for w in text.split() if w.isupper() and len(w) > 2)

    clickbait_terms = [
        "you won't believe", "shocking", "breaking", "secret", "hoax",
        "exposed", "wake up", "click now", "share before", "leak", "insider",
    ]
    lower = text.lower()
    clickbait_phrases = sum(1 for term in clickbait_terms if term in lower)

    return {
        "exclamations": exclamations,
        "all_caps_words": all_caps_words,
        "clickbait_phrases": clickbait_phrases,
    }
