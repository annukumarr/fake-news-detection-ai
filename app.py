import streamlit as st
import pickle
import spacy

# Load model
model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

# Load NLP model
nlp = spacy.load("en_core_web_sm")

st.title("AI Fake News Detection System")

st.write(
"This AI model predicts whether a news article is Fake or Real using Machine Learning."
)

news = st.text_area("Enter News Article")

if st.button("Predict"):

    vector = vectorizer.transform([news])

    prediction = model.predict(vector)

    confidence = model.decision_function(vector)

    doc = nlp(news)

    entities = [ent.text for ent in doc.ents]

    if prediction[0] == 0:
        st.error("Fake News")
    else:
        st.success("Real News")

    st.write("Confidence Score:", round(confidence[0], 2))

    if entities:
        st.write("Detected Entities:", entities)