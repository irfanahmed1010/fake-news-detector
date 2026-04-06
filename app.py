from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2

app = Flask(__name__)
CORS(app)

# ========================
# DISABLE IMAGE MODEL (IMPORTANT)
# ========================
image_model_loaded = False

# ========================
# HOME ROUTE
# ========================
@app.route('/')
def home():
    return "Fake News Detector Running 🚀"

# ========================
# TEXT PREDICTION (SMART RULE BASED)
# ========================
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    if not data:
        return jsonify({"error": "No input provided"})

    text = (data.get('text') or data.get('input') or "").lower()
    fake_claims= [
        "earth is flat",
        "aliens landed",
        "moon landing is fake"
        "vaccines cause microchips"
        "world is controlled secretly"
    ]
    for claim in fake_claims:
        if claim in text:
            return jsonify({
                "prediction":"fake",
                "confidence":0.95
            })

    # 🔥 smarter logic
    suspicious_words = [
        "breaking", "shocking", "aliens", "secret",
        "exposed", "conspiracy", "viral", "unbelievable",
        "urgent", "leaked"
    ]


    score = 0.4

    for word in suspicious_words:
        if word in text.lower():
            score += 0.12

    score = min(score, 0.95)

    prediction = "fake" if score > 0.6 else "real"
    return jsonify({
        "prediction" : prediction,
        "confidence" : float(round(score,2))
        
    })
    

# ========================
# IMAGE PREDICTION (DISABLED SAFE)
# ========================
@app.route('/predict-image', methods=['POST'])
def predict_image():
    return jsonify({"error": "Image model disabled to reduce server load"})

# ========================
# VIDEO PREDICTION (DISABLED SAFE)
# ========================
@app.route('/predict-video', methods=['POST'])
def predict_video():
    return jsonify({"error": "Video model disabled to reduce server load"})

# ========================
# FINAL COMBINED RESULT
# ========================
@app.route('/predict-final', methods=['POST'])
def predict_final():
    data = request.json

    text_score = data.get('text', 0)
    image_score = data.get('image', 0)
    video_score = data.get('video', 0)

    score = (text_score + image_score + video_score) / 3

    result = "fake" if score > 0.5 else "real"

    return jsonify({
        "prediction": result,
        "confidence": float(score)
    })

# ========================
# RUN SERVER
# ========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)