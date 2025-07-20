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

# from flask import Flask, request, jsonify
# from flask_cors import CORS
# # Load model directly
# from transformers import pipeline


# import re
# import string

# # Load  the model
# classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# # Create Flask app
# app = Flask(__name__)
# CORS(app)

# # Clean text function (optional with BERT but good for consistency)
# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r"\d+", "", text)
#     text = text.translate(str.maketrans("", "", string.punctuation))
#     text = re.sub(r"\s+", " ", text).strip()
#     return text

# # Prediction route
# @app.route("/predict", methods=["POST"])
# def predict():
#     data = request.json
#     headline = data.get("headline", "")
#     text = data.get("text", "")
#     input= clean_text(headline + " " + text)



#     # Make prediction
#     outputs = classifier(input, 
#                          candidate_labels=['Real', 'Fake'])

#     print(f"Sample {input} :")
#     print(f"Prediction: {outputs['labels'][0].upper()} ({outputs['scores'][0]*100:.2f}%)")


#     return jsonify({
#         "prediction": outputs
#         # "confidence": outputs['scores']
#     })

# if __name__ == "__main__":
#     app.run(debug=True, port=5000)

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import re
import string

# Initialize zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Flask setup
app = Flask(__name__)
CORS(app)

# Optional text cleaning function
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
    content = data.get("text", "")
    full_text = clean_text(f"{headline} {content}")

    # Zero-shot classification
    result = classifier(full_text, candidate_labels=["Real", "Fake"])

    prediction = result["labels"][0]  # top label
    confidence = result["scores"][0]  # top score

    return jsonify({
        "prediction": prediction.lower(),
        "confidence": round(confidence * 100, 2),  # % confidence
        "raw": result  # optional for debugging
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
