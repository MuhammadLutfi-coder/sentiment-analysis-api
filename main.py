import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


df = pd.read_csv("IMDB Dataset.csv")


X = df['review']
y = df['sentiment']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


model = LogisticRegression()
model.fit(X_train_vec, y_train)


y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))


import pickle

# save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model saved!")


while True:
    text = input("Enter review: ")
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)
    print("Sentiment:", prediction[0])

    