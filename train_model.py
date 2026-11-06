
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load the dataset
df = pd.read_csv("news.csv")

# 2. Clean and validate the data
df = df[["text", "label"]].dropna()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

print("Dataset shape:", df.shape)
print("\nLabel counts:")
print(df["label"].value_counts().sort_index())

# Based on the sample articles:
# 0 = FAKE, 1 = REAL
X = df["text"]
y = df["label"]

# 3. Split before training: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 4. Build and train the model
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        stop_words="english",
        max_features=100000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])

print("\nTraining model. Please wait...")
model.fit(X_train, y_train)

# 5. Evaluate on unseen test data
predictions = model.predict(X_test)

print("\nAccuracy:", round(
    accuracy_score(y_test, predictions) * 100, 2
), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["FAKE", "REAL"],
    digits=4
))

print("\nConfusion Matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, predictions, labels=[0, 1]))

# 6. Save the complete pipeline
joblib.dump(model, "fake_news_model.joblib")
print("\nModel saved as fake_news_model.joblib")

# 7. Test with example articles
examples = [
    "Breaking news: Officials announced a new policy today.",
    "This shocking secret proves everything is a conspiracy!!!"
]

print("\nExample predictions (demonstration only):")
for article in examples:
    prediction = int(model.predict([article])[0])
    probabilities = model.predict_proba([article])[0]
    confidence = max(probabilities) * 100

    label_name = "FAKE" if prediction == 0 else "REAL"

    print(f"\nPrediction: {label_name}")
    print(f"Model confidence: {confidence:.2f}%")
    print("Article:", article)
