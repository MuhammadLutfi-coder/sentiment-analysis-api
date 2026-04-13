# Sentiment Analysis API (Machine Learning Project)

This is an end-to-end Machine Learning project that predicts sentiment (Positive / Negative) from text using Natural Language Processing.

---

## Project Overview
This project uses a trained ML model to analyze movie reviews and classify them as:
- Positive 
- Negative 

The model is deployed as a REST API using Flask and can be tested via Postman or any HTTP client.

---

## Tech Stack
- Python 
- Scikit-learn 
- Pandas 
- Flask 
- TF-IDF Vectorizer
- Render (Cloud Deployment)

---

## How It Works
1. Input text (movie review)
2. Text is converted into numerical features using TF-IDF
3. Machine Learning model predicts sentiment
4. API returns result (positive / negative)

---

##  API Endpoint

### POST `/predict`

**Request Body:**
```json
{
  "text": "this movie is amazing"
}
