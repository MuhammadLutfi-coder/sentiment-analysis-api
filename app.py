from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)


model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route('/')
def home():
    return "Sentiment Analysis API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['text']
        
       
        vec = vectorizer.transform([data])
        
       
        result = model.predict(vec)

        return jsonify({
            "input": data,
            "sentiment": result[0]
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e)
        })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
