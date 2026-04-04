from flask import Flask, request, jsonify
from flask_cors import CORS   # ✅ added
import pickle
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)   # ✅ added

# ========================
# LOAD MODELS
# ========================

# Make sure these files exist
text_model = pickle.load(open("text_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# OPTIONAL (only if you have image_model.h5)
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
# TEXT PREDICTION
# ========================
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # ✅ FIX: support Lovable formats
    if not data:
        return jsonify({"error": "No input provided"})

    text = data.get('text') or data.get('input') or ""

    vec = vectorizer.transform([text])
    prediction = text_model.predict(vec)[0]

    result = "REAL" if prediction == 1 else "FAKE"

    return jsonify({"result": result})

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

    return jsonify({"prediction": float(pred)})

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

    return jsonify({"prediction": float(avg)})

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

    result = "REAL" if score > 0.5 else "FAKE"

    return jsonify({
        "result": result,
        "score": float(score)
    })

# ========================
# RUN SERVER
# ========================
if __name__ == "__main__":
    app.run(debug=True)