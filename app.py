from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import re
import string
import os
from datetime import datetime

# Initialize zero-shot classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Flask setup
app = Flask(__name__)
CORS(app, origins=["*"], methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type", "Authorization"])

# Text cleaning function
def clean_text(text):
    """Clean and preprocess text for better model performance"""
    if not text:
        return ""
    
    text = text.lower()
    text = re.sub(r"\d+", "", text)  # Remove numbers
    text = text.translate(str.maketrans("", "", string.punctuation))  # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra whitespace
    return text

@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model": "facebook/bart-large-mnli"
    })

@app.route("/predict", methods=["POST", "OPTIONS"])
def predict():
    """Main prediction endpoint"""
    # Handle preflight OPTIONS request
    if request.method == "OPTIONS":
        response = jsonify({"status": "OK"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
        response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        return response
    
    try:
        data = request.json
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        headline = data.get("headline", "")
        content = data.get("text", "")
        
        if not headline and not content:
            return jsonify({"error": "Please provide either headline or text"}), 400
        
        # Combine and clean text
        full_text = clean_text(f"{headline} {content}")
        
        if len(full_text.strip()) < 5:
            return jsonify({"error": "Text too short for meaningful analysis"}), 400
        
        # Zero-shot classification
        result = classifier(full_text, candidate_labels=["Real", "Fake"])
        
        prediction = result["labels"][0].lower()  # top label
        confidence = round(result["scores"][0] * 100, 2)  # top score as percentage
        
        return jsonify({
            "prediction": prediction,
            "confidence": confidence,
            "analysis": {
                "text_length": len(full_text),
                "headline": headline[:100] + "..." if len(headline) > 100 else headline,
                "content_preview": content[:100] + "..." if len(content) > 100 else content
            }
        })
        
    except Exception as e:
        response = jsonify({"error": f"Internal server error: {str(e)}"})
        response.status_code = 500
        return response

@app.after_request
def after_request(response):
    """Add CORS headers to all responses"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug)
