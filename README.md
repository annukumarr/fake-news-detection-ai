# 📰 The Fact Desk — Fake News Detection with ML, LLM Fallback & Fact-Check API

A working fake news detection system: a **real, trained ML classifier** (93.4% accuracy on a genuine 6,334-article dataset) as the primary decision-maker, an **LLM second opinion** for uncertain cases, and **Google Fact Check API** corroboration — wrapped in a modern web app with single-story analysis, batch CSV upload, and a live model dashboard.

Runs fully out of the box with **zero API keys required**; add keys later to unlock live LLM review and fact-check lookups.

---

## ✅ What's real here (read this first)

- **Dataset:** 6,334 real English news articles (`fake_or_real_news.csv`, a well-known public dataset — ~3,171 real / ~3,163 fake), not synthetic placeholder data.
- **Accuracy:** 93.4% accuracy, 92.3% precision, 94.6% recall, 93.4% F1 — Logistic Regression on TF-IDF features, evaluated on a genuine held-out test split of 1,267 articles. These numbers are reproducible by running `python -m src.train_model` yourself.
- **Known limitation, stated honestly:** the model was trained on full-length article text. Very short one-line headlines sometimes land near the 50% decision boundary rather than confidently one way or the other — this is expected given the training data, and it's exactly the case the LLM fallback layer exists to handle. Longer inputs (a headline plus a paragraph or two) classify more reliably.

---

## 🧠 Architecture

```
News text ──▶ Preprocessing (NLP) ──▶ TF-IDF ──▶ ML Classifier (Logistic Regression)
                                                        │
                                          confidence < 65%?
                                    ┌───────────────────┴───────────────────┐
                                    ▼ yes                                   ▼ no
                        LLM second opinion (Claude API)         use ML result directly
                        + Fact-Check API corroboration
                                    └───────────────────┬───────────────────┘
                                                         ▼
                                          Final verdict + confidence + evidence
```

The ML model runs on every request, instantly and for free. The LLM and Fact-Check layers are optional and only engage for genuinely uncertain cases — so the system is fast and cost-efficient, and works fully offline if no API keys are configured.

---

## ✨ Features

- **Modern web app** (Flask + vanilla HTML/CSS/JS) — no framework build step, just run and open a browser
- **Single-story analysis** — paste a headline/article, get a verdict, confidence score, style signals, LLM opinion, and related fact-checks
- **Batch mode** — upload a CSV of headlines/articles, get a results table + downloadable CSV of verdicts (up to 200 rows per run)
- **Live model dashboard** — accuracy/precision/recall/F1 stat cards, a model comparison chart (Logistic Regression vs. Naive Bayes), and a dataset class-balance chart, all rendered from your actual trained model's metadata
- **CLI tool** for terminal use or scripting/automation
- **Graceful degradation** — works fully with no API keys; each optional layer just reports itself as offline
- **Self-bootstrapping** — auto-trains on first run if no model exists yet
- **Unit tests** (pytest) covering preprocessing, prediction, and batch processing

---

## 📁 Project Structure

```
fake-news-detection/
├── backend/
│   └── api.py                  # Flask app: serves frontend + REST API
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── cli.py                      # Command-line interface
├── data/
│   └── news_dataset.csv        # Real dataset (6,334 articles)
├── models/                     # Trained model artifacts (auto-generated)
├── scripts/
│   └── generate_dataset.py     # Optional: generate a synthetic dataset instead
├── src/
│   ├── preprocessing.py        # Text cleaning + stylistic signal extraction
│   ├── feature_extraction.py   # TF-IDF vectorizer build/save/load
│   ├── train_model.py          # Trains + evaluates + saves the classifier
│   ├── llm_fallback.py         # Claude API second-opinion layer
│   ├── factcheck_api.py        # Google Fact Check Tools API layer
│   ├── batch.py                # Batch CSV processing
│   └── predict.py              # Combined inference pipeline
├── tests/
│   └── test_pipeline.py        # pytest unit tests (6 tests)
└── LICENSE
```

---

## 🚀 Getting Started

### 1. Install
```bash
git clone https://github.com/<your-username>/fake-news-detection.git
cd fake-news-detection
pip install -r requirements.txt
```

### 2. (Optional) Add API keys

⚠️ **Never commit real API keys or paste them into chat tools/screenshots.** If a key is ever exposed anywhere outside your local `.env`, rotate it immediately in the provider's console.

```bash
cp .env.example .env
# then edit .env and add ANTHROPIC_API_KEY and/or GOOGLE_FACTCHECK_API_KEY
```
The app works fully without this step — it just skips the LLM and fact-check layers.

### 3. Run the web app
```bash
python backend/api.py
```
Then open **http://localhost:5000** in your browser. First run auto-trains the model (a few seconds).

### 4. Or run the CLI
```bash
python cli.py --text "SHOCKING: Scientists discover secret cure that Big Pharma is hiding!!!"
```

### 5. Run tests
```bash
pip install pytest
pytest tests/ -v
```

---

## 🖥️ Using the Web App

- **Analyze a story** — paste text, toggle LLM/Fact-Check on or off, click Analyze
- **Batch upload** — upload a CSV with a `text`, `headline`, or `title` column (download the sample CSV from the app if unsure of the format), get a results table, download results as CSV
- **Model dashboard** — live accuracy/precision/recall/F1, a Logistic Regression vs. Naive Bayes comparison chart, and the dataset's real/fake class balance

## 🖥️ CLI Reference

```bash
python cli.py --text "..."                 # full pipeline
python cli.py --file article.txt            # analyze a text file
python cli.py --text "..." --no-llm         # skip LLM fallback
python cli.py --text "..." --no-factcheck   # skip fact-check lookup
python cli.py --text "..." --json           # raw JSON output (for scripting)
```

---

## 🔑 Getting API Keys (optional)

| API | Used for | Get a key |
|---|---|---|
| Anthropic Claude | LLM second-opinion + explanation | https://console.anthropic.com/ |
| Google Fact Check Tools | Live corroboration against published fact-checks | https://console.cloud.google.com/apis/library/factchecktools.googleapis.com |

Both are optional. Neither is required to run or demo the project.

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Logistic Regression** (primary) | 93.4% | 92.3% | 94.6% | 93.4% |
| Naive Bayes | 89.5% | 87.3% | 92.4% | 89.8% |

Trained on 5,067 articles, evaluated on a held-out 1,267-article test set (80/20 split, stratified). Re-run `python -m src.train_model` any time to reproduce these numbers or retrain on updated data.

---

## 🔮 Future Scope

- Transformer-based models (BERT/DistilBERT) for better accuracy on nuanced or short-form text
- Multilingual support
- Browser extension for one-click, in-page checking
- Real-time social media monitoring
- A feedback loop so flagged misclassifications improve future retraining

---

## 👥 Contributors

- Annu Kumar Pal — 25MCI10230

**Guide:** Maajid Bashir (E17205)
**Institution:** Chandigarh University, MCA (2025–2027)

---

## 📄 License

MIT License — see [LICENSE](LICENSE).

---

*⚠️ This tool provides an automated estimate for educational purposes and is not a substitute for professional fact-checking. Always verify important claims through multiple trusted sources.*
