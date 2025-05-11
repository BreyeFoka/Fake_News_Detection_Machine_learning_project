# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import joblib
# import re
# import string

# # Load model and vectorizer
# model = joblib.load("fake_news_model.pkl")
# vectorizer = joblib.load("vectorizer.pkl")

# # Flask app
# app = Flask(__name__)
# CORS(app)
# # Text cleaning function
# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r"\d+", "", text)
#     text = text.translate(str.maketrans("", "", string.punctuation))
#     text = re.sub(r"\s+", " ", text).strip()
#     return text

# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.json
#     title = data.get("headline", "")
#     text = data.get("text", "")
#     content = clean_text(title + " " + text)

#     vector = vectorizer.transform([content])
#     prediction = model.predict(vector)[0]
#     confidence = model.predict_proba(vector).max()

#     return jsonify({
#         "prediction": "real" if prediction == 1 else "fake",
#         "confidence": float(confidence)
#     })

# if __name__ == "__main__":
#     app.run(debug=True, port=5000)

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import re
import string

# Load tokenizer and model
model = TFBertForSequenceClassification.from_pretrained("./fake_news_model")
tokenizer = BertTokenizer.from_pretrained("./fake_news_model")

# Create Flask app
app = Flask(__name__)
CORS(app)

# Clean text function (optional with BERT but good for consistency)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    headline = data.get("headline", "")
    text = data.get("text", "")
    full_text = clean_text(headline + " " + text)

    # Tokenize input
    inputs = tokenizer(full_text, return_tensors="tf", truncation=True, padding=True, max_length=512)

    # Make prediction
    outputs = model(inputs)
    probs = tf.nn.softmax(outputs.logits, axis=1).numpy()[0]
    predicted_label = int(probs.argmax())
    confidence = float(probs.max())

    return jsonify({
        "prediction": "real" if predicted_label == 1 else "fake",
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
