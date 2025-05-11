from flask import Flask, request, jsonify
import joblib
import re
import string

# Load model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Flask app
app = Flask(__name__)

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    headline = data.get("headline", "")
    text = data.get("text", "")
    content = clean_text(headline + " " + text)

    vector = vectorizer.transform([content])
    prediction = model.predict(vector)[0]
    confidence = model.predict_proba(vector).max()

    return jsonify({
        "prediction": "real" if prediction == 1 else "fake",
        "confidence": float(confidence)
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
