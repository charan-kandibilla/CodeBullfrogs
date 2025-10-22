import pickle
import os
import numpy as np
import traceback
import requests
from flask import Flask, request, jsonify
from ai_model.content_analysis import extract_text_from_url
from ai_model.ssl_check import check_ssl_expiry
from ai_model.behavioural_analysis import detect_suspicious_behavior

app = Flask(__name__)

# Paths to models
MODEL_PATH = os.path.join("ai_model", "best_model.pkl")
TEXT_MODEL_PATH = os.path.join("ai_model", "text_model.pkl")
VECTORIZER_PATH = os.path.join("ai_model", "vectorizer.pkl")

# Load URL-based model
try:
    with open(MODEL_PATH, "rb") as model_file:
        url_model = pickle.load(model_file)
    print("✅ URL Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading URL model: {e}")
    url_model = None

# Load Text-based phishing detection model
try:
    with open(TEXT_MODEL_PATH, "rb") as text_model_file:
        text_model = pickle.load(text_model_file)
    with open(VECTORIZER_PATH, "rb") as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    print("✅ Text Model and Vectorizer loaded successfully")
except Exception as e:
    print(f"❌ Error loading text model: {e}")
    text_model = None
    vectorizer = None

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"status": 404, "error": "API endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"status": 500, "error": "Internal server error"}), 500

# Function to extract URL features
def extract_features(url):
    features = [
        len(url),
        url.count("."), url.count("/"), url.count("-"),
        sum(c.isdigit() for c in url),
        sum(c.isupper() for c in url),
        url.count("_"), url.count("="), url.count("?"),
        url.count("&"), url.count("%"), url.count(";"),
        url.count("@"), url.count("www"), url.count("https"),
        url.count("http"), int(url.startswith("https")),
        int(url.endswith(".php")), int(url.endswith(".html")),
        int(url.endswith(".exe")), int(url.endswith(".zip")),
        int(url.endswith(".gov")), int(url.endswith(".edu")),
        int(url.endswith(".org")), int("login" in url.lower()),
        int("bank" in url.lower()), int("verify" in url.lower()),
        int("secure" in url.lower()), int("account" in url.lower()),
        int("update" in url.lower()), int("confirm" in url.lower()),
        int("click" in url.lower())
    ]
    return np.array(features).reshape(1, -1)  # Ensure correct shape

@app.route("/analyze-url", methods=["POST"])
def analyze_url():
    try:
        data = request.get_json()
        url = data.get("url")

        if not url:
            return jsonify({"status": 400, "error": "No URL provided"}), 400

        if not url_model:
            return jsonify({"status": 500, "error": "URL Model not loaded"}), 500

        # Extract URL-based features
        url_features = extract_features(url)
        url_prediction = url_model.predict(url_features)[0]  # 0 = safe, 1 = phishing

        # Extract webpage text content
        page_text = extract_text_from_url(url)
        if isinstance(page_text, dict):  # If text extraction failed, return error
            return jsonify({"status": 400, "error": page_text["error"]}), 400

        # Perform text-based phishing detection
        if text_model and vectorizer:
            text_features = vectorizer.transform([page_text])
            text_prediction = text_model.predict(text_features)[0]
        else:
            text_prediction = None  # No valid text model loaded

        # Perform SSL Certificate Expiry Check
        domain = url.replace("http://", "").replace("https://", "").split("/")[0]
        ssl_status = check_ssl_expiry(domain)

        # Final phishing classification
        if text_prediction is None:
            final_prediction = url_prediction  # Use only URL model if text model fails
        else:
            final_prediction = max(url_prediction, text_prediction)  # If either flags phishing, classify as phishing

        return jsonify({
            "status": 200,
            "is_phishing": bool(final_prediction),
            "message": "Phishing site detected!" if final_prediction else "Site is safe.",
            "url_based_prediction": bool(url_prediction),
            "text_based_prediction": bool(text_prediction) if text_prediction is not None else "N/A",
            "ssl_status": ssl_status  # Returns SSL validity and expiry details
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": 500, "error": str(e)}), 500

@app.route("/analyze-behavior", methods=["POST"])
def analyze_behavior():
    try:
        data = request.get_json()
        mouse_movements = data.get("mouseMovements", [])
        clicks = data.get("clicks", [])
        keystrokes = data.get("keystrokes", [])

        if not isinstance(mouse_movements, list) or not isinstance(clicks, list) or not isinstance(keystrokes, list):
            return jsonify({"status": 400, "error": "Invalid input data format"}), 400

        # Detect suspicious behavior
        is_suspicious = detect_suspicious_behavior(mouse_movements, clicks, keystrokes)

        return jsonify({
            "status": 200,
            "is_suspicious": is_suspicious,
            "message": "Suspicious behavior detected!" if is_suspicious else "Behavior seems normal."
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"status": 500, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
