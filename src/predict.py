import pickle

model = pickle.load(open("model/model.pkl", "rb"))
vectorizer = pickle.load(open("model/vectorizer.pkl", "rb"))

news = input("Enter News: ")

vector = vectorizer.transform([news])

prediction = model.predict(vector)

if prediction[0] == 0:
    print("Fake News")
else:
    print("Real News")