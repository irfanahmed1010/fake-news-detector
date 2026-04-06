from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# ========================
# LOAD MODELS
# ========================

# 🔥 BERT (DistilBERT)
classifier = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# OPTIONAL IMAGE MODEL
try:
    image_model = load_model("image_model.h5")
    image_model_loaded = True
except:
    image_model_loaded = False

# ========================
# HOME ROUTE
# ========================
@app.route('/')
def home():
    return "Fake News Detector Running 🚀"

# ========================
# TEXT PREDICTION (BERT)
# ========================
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    if not data:
        return jsonify({"error": "No input provided"})

    text = data.get('text') or data.get('input') or ""

    result = classifier(text)[0]

    label = result['label']   # POSITIVE / NEGATIVE
    score = result['score']

    # Convert sentiment → fake/real
    if label == "NEGATIVE":
        prediction = "fake"
    else:
        prediction = "real"

    return jsonify({
        "prediction": prediction,
        "confidence": float(score)
    })

# ========================
# IMAGE PREDICTION
# ========================
@app.route('/predict-image', methods=['POST'])
def predict_image():
    if not image_model_loaded:
        return jsonify({"error": "Image model not loaded"})

    file = request.files['image']
    file.save("temp.jpg")

    img = cv2.imread("temp.jpg")
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.reshape(img, (1, 64, 64, 3))

    pred = image_model.predict(img)[0][0]

    result = "fake" if pred > 0.5 else "real"

    return jsonify({
        "prediction": result,
        "confidence": float(pred)
    })

# ========================
# VIDEO PREDICTION
# ========================
@app.route('/predict-video', methods=['POST'])
def predict_video():
    if not image_model_loaded:
        return jsonify({"error": "Image model not loaded"})

    file = request.files['video']
    file.save("temp.mp4")

    cap = cv2.VideoCapture("temp.mp4")
    preds = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or count > 10:
            break

        frame = cv2.resize(frame, (64, 64))
        frame = frame / 255.0
        frame = np.reshape(frame, (1, 64, 64, 3))

        p = image_model.predict(frame)[0][0]
        preds.append(p)
        count += 1

    cap.release()

    avg = np.mean(preds) if preds else 0

    result = "fake" if avg > 0.5 else "real"

    return jsonify({
        "prediction": result,
        "confidence": float(avg)
    })

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