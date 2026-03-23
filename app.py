from flask import Flask, render_template, request, jsonify
import pickle
import joblib
import numpy as np
from keras.models import load_model
from keras import mixed_precision
import os
import h5py
import cv2
import tensorflow as tf
from werkzeug.utils import secure_filename
import requests as http_requests
from urllib.parse import quote
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

def fix_h5_model(file_path):
    with h5py.File(file_path, 'r+') as f:
        model_config = f.attrs.get('model_config')
        if model_config:
            if isinstance(model_config, bytes):
                config_str = model_config.decode('utf-8')
            else:
                config_str = model_config
            config_str = config_str.replace('batch_shape', 'batch_input_shape')
            if isinstance(model_config, bytes):
                f.attrs['model_config'] = config_str.encode('utf-8')
            else:
                f.attrs['model_config'] = config_str

MODEL_DIR     = os.path.join(os.path.dirname(__file__), "models")
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static", "uploads")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

fix_h5_model(os.path.join(MODEL_DIR, "rumor_classifier_model.h5"))
fix_h5_model(os.path.join(MODEL_DIR, "spam_classifier_model.h5"))
fix_h5_model(os.path.join(MODEL_DIR, "spam_email_model.h5"))
fix_h5_model(os.path.join(MODEL_DIR, "url01_model.h5"))
fix_h5_model(os.path.join(MODEL_DIR, "news_model.h5"))

custom_objects = {"DTypePolicy": mixed_precision.Policy}

deepfake_model     = tf.keras.models.load_model(os.path.join(MODEL_DIR, "deepfake_model.h5"))
email_lstm_model   = tf.keras.models.load_model(os.path.join(MODEL_DIR, "new_email_lstm_model.keras"), compile=False)
email_lstm_max_len = email_lstm_model.input_shape[1]
email_lstm_tok     = pickle.load(open(os.path.join(MODEL_DIR, "new_email_lstm_tokenizer.pkl"), "rb"))

models = {
    "best_model":      load_model(os.path.join(MODEL_DIR, "rumor_classifier_model.h5"), compile=False, custom_objects=custom_objects),
    "xgb_model":       joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl")),
    "spam_classifier": load_model(os.path.join(MODEL_DIR, "spam_classifier_model.h5"), compile=False, custom_objects=custom_objects),
    "spam_email":      load_model(os.path.join(MODEL_DIR, "spam_email_model.h5"), compile=False, custom_objects=custom_objects),
    "url01":           load_model(os.path.join(MODEL_DIR, "url01_model.h5"), compile=False, custom_objects=custom_objects),
    "news":            load_model(os.path.join(MODEL_DIR, "news_model.h5"), compile=False, custom_objects=custom_objects),
}

vectorizers = {
    "best_model":      pickle.load(open(os.path.join(MODEL_DIR, "rumor_vectorizer.pkl"), "rb")),
    "xgb_model":       joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer_xgb.pkl")),
    "spam_classifier": pickle.load(open(os.path.join(MODEL_DIR, "vectorizer_spam.pkl"), "rb")),
    "spam_email":      pickle.load(open(os.path.join(MODEL_DIR, "vectorizer_email.pkl"), "rb")),
    "url01":           pickle.load(open(os.path.join(MODEL_DIR, "vectorizer_url01.pkl"), "rb")),
    "news":            pickle.load(open(os.path.join(MODEL_DIR, "vectorizer_news.pkl"), "rb")),
}

def preprocess_text(text):
    return text.lower().strip()

rumor_labels = {0: "false", 1: "non-rumor", 2: "true", 3: "unverified"}


GOOGLE_API_KEY = "AIzaSyAcBneYGdr5FYDgcokNvEA218wPnwhe_ZU"
GOOGLE_CX      = "07c1b944157d546f4"
NEWS_API_KEY   = "f26fc6abc94246a0a9cbbc5d4609dbce"


@app.route("/", methods=["GET", "POST"])
def home():
    label          = None
    prediction     = None
    selected_model = None
    user_input     = ""

    if request.method == "POST":
        user_input     = request.form.get("user_input")
        selected_model = request.form.get("model_select")

        if user_input and selected_model:
            clean_input = preprocess_text(user_input)

            if selected_model == "email_lstm":
                seq        = email_lstm_tok.texts_to_sequences([clean_input])
                padded     = pad_sequences(seq, maxlen=email_lstm_max_len)
                prob       = float(email_lstm_model.predict(padded)[0][0])
                prediction = int(prob >= 0.5)
                label      = "Spam Email" if prediction == 1 else "Legitimate Email"

            else:
                vect = vectorizers[selected_model].transform([clean_input])

                if selected_model == "xgb_model":
                    pred_proba = models[selected_model].predict_proba(vect)[:, 1]
                    prediction = int(pred_proba[0] >= 0.5)
                    label      = "Positive" if prediction == 1 else "Negative"

                elif selected_model == "best_model":
                    pred       = models[selected_model].predict(vect.toarray())
                    prediction = np.argmax(pred, axis=1)[0]
                    label      = rumor_labels[prediction]

                else:
                    pred       = models[selected_model].predict(vect.toarray())
                    prob       = pred[0][0]
                    prediction = int(prob >= 0.5)

                if selected_model == "spam_classifier":
                    label = "Spam Message" if prediction == 1 else "Likely not spam"
                elif selected_model == "spam_email":
                    label = "Likely not spam" if prediction == 1 else "Spam Email"
                elif selected_model == "url01":
                    label = "Safe Website" if prediction == 1 else "Website is potentially Risky"
                elif selected_model == "news":
                    label = "News is trustable" if prediction == 1 else "The news is likely fake"

    return render_template(
        "index.html",
        prediction=label,
        selected_model=selected_model,
        user_input=user_input,
    )

@app.route("/deepfake", methods=["GET", "POST"])
def deepfake():
    prediction  = None
    confidence  = None
    image_path  = None

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            img = cv2.imread(filepath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224)).astype("float32")
            x   = np.expand_dims(img, axis=0)

            pred       = float(deepfake_model.predict(x)[0][0])
            prediction = "Fake" if pred < 0.5 else "Real"
            confidence = round(pred, 4)
            image_path = f"uploads/{filename}"

    return render_template("deepfake.html", prediction=prediction, confidence=confidence, image_path=image_path)


























































@app.route("/api/google-news")
def google_news():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400
    try:
        url  = f"https://www.googleapis.com/customsearch/v1?q={quote(query + ' news')}&key={GOOGLE_API_KEY}&cx={GOOGLE_CX}&num=6"
        resp = http_requests.get(url, timeout=15)
        data = resp.json()
        print("[Google CSE]", resp.status_code, list(data.keys()))
        if "error" in data:
            return jsonify({"error": data["error"].get("message", "Google API error")}), 502
        results = [
            {
                "title":   item.get("title", ""),
                "snippet": item.get("snippet", ""),
                "url":     item.get("link", ""),
                "source":  item.get("displayLink", ""),
            }
            for item in data.get("items", [])
        ]
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/newsapi-news")
def newsapi_news():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400
    try:
        url  = f"https://newsapi.org/v2/everything?q={quote(query)}&apiKey={NEWS_API_KEY}&pageSize=6&sortBy=relevancy&language=en"
        resp = http_requests.get(url, timeout=15)
        data = resp.json()
        print("[NewsAPI]", resp.status_code, data.get("status"))
        if data.get("status") != "ok":
            return jsonify({"error": data.get("message", "NewsAPI error")}), 502
        results = [
            {
                "title":   a.get("title", ""),
                "snippet": a.get("description") or a.get("content", ""),
                "url":     a.get("url", ""),
                "source":  a.get("source", {}).get("name", ""),
            }
            for a in data.get("articles", [])
            if a.get("title") and a.get("url") and "[Removed]" not in a.get("title", "")
        ]
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route("/api/fact-check")
def fact_check():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400
    try:
        # shorten query to first 5 words for better fact-check matches
        short_query = " ".join(query.split()[:5])
        url  = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={quote(short_query)}&key={GOOGLE_API_KEY}&pageSize=6"
        resp = http_requests.get(url, timeout=15)
        data = resp.json()
        print("[FactCheck]", resp.status_code, list(data.keys()))
        if "error" in data:
            return jsonify({"error": data["error"].get("message", "Fact Check API error")}), 502
        results = []
        for claim in data.get("claims", []):
            for review in claim.get("claimReview", []):
                results.append({
                    "claim":     claim.get("text", ""),
                    "claimant":  claim.get("claimant", "Unknown"),
                    "rating":    review.get("textualRating", ""),
                    "publisher": review.get("publisher", {}).get("name", ""),
                    "url":       review.get("url", ""),
                })
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 502

if __name__ == "__main__":
    app.run(debug=True)
