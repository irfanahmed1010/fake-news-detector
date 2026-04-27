from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import datetime
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from PIL import Image
import cv2
import tempfile
from transformers import pipeline

app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fakenews.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.Text)
    image_result = db.Column(db.String(10))
    video_result = db.Column(db.String(10))
    final_prediction = db.Column(db.String(10))
    confidence = db.Column(db.Float)
    date = db.Column(db.DateTime, default=datetime.datetime.utcnow)

print("Loading dataset...")

df = pd.read_csv("fake_news.csv")

df.columns = df.columns.str.lower().str.strip()

if 'text' not in df.columns:
    if 'content' in df.columns:
        df['text'] = df['content']
    elif 'title' in df.columns:
        df['text'] = df['title']
    else:
        raise Exception("No valid text column found")

df['text'] = df['text'].astype(str)

df['label'] = df['label'].astype(str).str.lower().str.strip()

df['label'] = df['label'].map({
    'fake': 0,
    'real': 1,
    'true': 1,
    'false': 0,
    '0': 0,
    '1': 1
})

df = df.dropna(subset=['text', 'label'])

df['label'] = df['label'].astype(int)

vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X = vectorizer.fit_transform(df['text'])
y = df['label']

text_model = LogisticRegression(max_iter=200)
text_model.fit(X, y)

print("Text model trained!")

print("Loading image model...")
image_model = pipeline("image-classification")
print("Image model ready!")

@app.route('/')
def home():
    return "Multimodal Fake News Detector Running 🚀"

def predict_text_internal(text):
    input_data = vectorizer.transform([text])
    pred = text_model.predict(input_data)[0]
    prob = text_model.predict_proba(input_data)[0]
    result = "real" if pred == 1 else "fake"
    confidence = float(max(prob))
    return result, confidence

@app.route('/predict-text', methods=['POST'])
def predict_text():
    data = request.json
    if not data or not data.get('text'):
        return jsonify({"error": "No text provided"})
    result, confidence = predict_text_internal(data['text'])
    return jsonify({
        "prediction": result,
        "confidence": round(confidence, 2)
    })

def predict_image_internal(file):
    image = Image.open(file)
    results = image_model(image)
    top = results[0]
    confidence = float(top['score'])
    result = "fake" if confidence < 0.6 else "real"
    return result, confidence

@app.route('/predict-image', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({"error": "No image uploaded"})
    result, confidence = predict_image_internal(request.files['file'])
    return jsonify({
        "prediction": result,
        "confidence": round(confidence, 2)
    })

def predict_video_internal(file):
    temp = tempfile.NamedTemporaryFile(delete=False)
    file.save(temp.name)

    cap = cv2.VideoCapture(temp.name)
    scores = []
    count = 0

    while cap.isOpened() and count < 10:
        ret, frame = cap.read()
        if not ret:
            break

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = image_model(image)
        scores.append(results[0]['score'])
        count += 1

    cap.release()
    os.remove(temp.name)

    if not scores:
        return "error", 0.0

    avg_score = sum(scores) / len(scores)
    result = "fake" if avg_score < 0.6 else "real"

    return result, float(avg_score)

@app.route('/predict-video', methods=['POST'])
def predict_video():
    if 'file' not in request.files:
        return jsonify({"error": "No video uploaded"})
    result, confidence = predict_video_internal(request.files['file'])
    return jsonify({
        "prediction": result,
        "confidence": round(confidence, 2)
    })

@app.route('/predict-final', methods=['POST'])
def predict_final():
    text = request.form.get('text')

    text_result, text_conf = ("real", 0)
    image_result, image_conf = ("real", 0)
    video_result, video_conf = ("real", 0)

    if text:
        text_result, text_conf = predict_text_internal(text)

    if 'image' in request.files:
        image_result, image_conf = predict_image_internal(request.files['image'])

    if 'video' in request.files:
        video_result, video_conf = predict_video_internal(request.files['video'])

    final_score = (0.5 * text_conf + 0.3 * image_conf + 0.2 * video_conf)
    final_prediction = "fake" if final_score < 0.5 else "real"

    entry = Prediction(
        text=text,
        image_result=image_result,
        video_result=video_result,
        final_prediction=final_prediction,
        confidence=final_score
    )

    db.session.add(entry)
    db.session.commit()

    return jsonify({
        "prediction": final_prediction,
        "confidence": round(final_score, 2),
        "details": {
            "text": text_result,
            "image": image_result,
            "video": video_result
        }
    })

@app.route('/history', methods=['GET'])
def history():
    data = Prediction.query.all()
    output = []
    for d in data:
        output.append({
            "text": d.text,
            "image": d.image_result,
            "video": d.video_result,
            "final": d.final_prediction,
            "confidence": d.confidence,
            "date": str(d.date)
        })
    return jsonify(output)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)