from flask import Flask, request, jsonify
from flask_cors import CORS
import re

app = Flask(__name__)
CORS(app)

# ========================
# HOME ROUTE
# ========================
@app.route('/')
def home():
    return "Fake News Detector Running 🚀"

# ========================
# TEXT PREDICTION (IMPROVED)
# ========================
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    if not data:
        return jsonify({"error": "No input provided"})

    text = data.get('text') or data.get('input') or ""
    text_lower = text.lower()

    fake_score = 0
    real_score = 0

    # 🚨 STRONG FAKE PATTERNS (instant detection)
    strong_fake_patterns = [
        "free for all",
        "free laptops to all",
        "all students will get",
        "miracle cure",
        "100% cure",
        "overnight success",
        "share before deleted",
        "secret they don't want you to know"
    ]

    for pattern in strong_fake_patterns:
        if pattern in text_lower:
            return jsonify({
                "prediction": "fake",
                "confidence": 0.95
            })

    # 🔴 FAKE KEYWORDS (weighted)
    fake_keywords = [
        "breaking", "shocking", "secret",
        "exposed", "viral", "unbelievable",
        "urgent", "leaked", "conspiracy",
        "alert", "must watch"
    ]

    for word in fake_keywords:
        if word in text_lower:
            fake_score += 2

    # 🟢 REAL KEYWORDS (more strict)
    real_keywords = [
        "according to", "reported by",
        "research study", "data shows",
        "official report", "press release",
        "confirmed by", "sources said"
    ]

    for word in real_keywords:
        if word in text_lower:
            real_score += 2

    # 🧠 CONTEXT RULES (VERY IMPORTANT)
    if "government" in text_lower and "free" in text_lower:
        fake_score += 3

    if "all students" in text_lower or "everyone" in text_lower:
        fake_score += 2

    if "cure" in text_lower:
        fake_score += 2

    if "click here" in text_lower or "share now" in text_lower:
        fake_score += 2

    # 📢 EXAGGERATION CHECK
    if text.count("!") >= 2:
        fake_score += 1

    if text.isupper():
        fake_score += 2

    # 🧾 LENGTH & STRUCTURE CHECK
    word_count = len(text.split())

    if word_count > 25:
        real_score += 2
    elif word_count < 6:
        fake_score += 1

    # 🔍 NUMBERS / DATA (more realistic)
    if re.search(r'\d+', text):
        real_score += 1

    # ⚖️ FINAL DECISION (FIXED CORE LOGIC)
    total = fake_score + real_score + 1

    if fake_score > real_score:
        prediction = "fake"
        confidence = fake_score / total
    elif real_score > fake_score:
        prediction = "real"
        confidence = real_score / total
    else:
        prediction = "fake"   # safer default
        confidence = 0.5

    return jsonify({
        "prediction": prediction,
        "confidence": round(float(confidence), 2)
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
        "confidence": round(float(score), 2)
    })


# ========================
# RUN SERVER
# ========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)