from flask import Flask, render_template, request, jsonify, session, redirect, url_for
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
import json
from datetime import datetime, date

app = Flask(__name__)
app.secret_key = "truthscope_admin_secret_2024"

# ── Admin credentials ───────────────────────────────────────────
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "truthscope123"

# ── Analytics helpers ───────────────────────────────────────────
ANALYTICS_FILE = os.path.join(os.path.dirname(__file__), "analytics.json")

def load_analytics():
    if not os.path.exists(ANALYTICS_FILE):
        return {
            "total_predictions": 0,
            "total_deepfake_scans": 0,
            "model_usage": {
                "best_model": 0, "xgb_model": 0, "spam_classifier": 0,
                "spam_email": 0, "url01": 0, "news": 0, "email_lstm": 0
            },
            "verdict_counts": {"safe": 0, "danger": 0, "warning": 0},
            "deepfake_verdicts": {"Real": 0, "Fake": 0},
            "daily_activity": {},
            "recent_logs": []
        }
    with open(ANALYTICS_FILE, "r") as f:
        return json.load(f)

def save_analytics(data):
    with open(ANALYTICS_FILE, "w") as f:
        json.dump(data, f, indent=2)

def track_prediction(model, label, user_input):
    data = load_analytics()
    today = str(date.today())

    data["total_predictions"] += 1
    data["model_usage"][model] = data["model_usage"].get(model, 0) + 1
    data["daily_activity"][today] = data["daily_activity"].get(today, 0) + 1

    p = (label or "").lower()
    if any(w in p for w in ["safe", "not spam", "trustable", "legitimate", "non-rumor", "real"]):
        data["verdict_counts"]["safe"] += 1
    elif any(w in p for w in ["spam", "fake", "risky", "false", "positive"]):
        data["verdict_counts"]["danger"] += 1
    else:
        data["verdict_counts"]["warning"] += 1

    log = {
        "time":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type":  "text",
        "model": model,
        "input": user_input[:80] + ("…" if len(user_input) > 80 else ""),
        "label": label
    }
    data["recent_logs"].insert(0, log)
    data["recent_logs"] = data["recent_logs"][:50]
    save_analytics(data)

def track_deepfake(prediction, confidence):
    data = load_analytics()
    today = str(date.today())

    data["total_deepfake_scans"] += 1
    data["deepfake_verdicts"][prediction] = data["deepfake_verdicts"].get(prediction, 0) + 1
    data["daily_activity"][today] = data["daily_activity"].get(today, 0) + 1

    log = {
        "time":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "type":       "deepfake",
        "model":      "deepfake_model",
        "input":      "Image upload",
        "label":      f"{prediction} ({confidence})"
    }
    data["recent_logs"].insert(0, log)
    data["recent_logs"] = data["recent_logs"][:50]
    save_analytics(data)

# ── H5 fix ──────────────────────────────────────────────────────
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

GOOGLE_API_KEY = "your_google_api_key_here"
GOOGLE_CX      = "07c1b944157d546f4"
NEWS_API_KEY   = "f26fc6abc94246a0a9cbbc5d4609dbce"

# ── Main Routes ─────────────────────────────────────────────────
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

            track_prediction(selected_model, label, user_input)

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

            track_deepfake(prediction, confidence)

    return render_template("deepfake.html", prediction=prediction, confidence=confidence, image_path=image_path)

# ── Admin Routes ────────────────────────────────────────────────
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    error = None
    if request.method == "POST":
        if request.form.get("username") == ADMIN_USERNAME and \
           request.form.get("password") == ADMIN_PASSWORD:
            session["admin"] = True
            return redirect(url_for("admin_dashboard"))
        error = "Invalid credentials"
    return render_template("admin_login.html", error=error)

@app.route("/admin/logout")
def admin_logout():
    session.pop("admin", None)
    return redirect(url_for("admin_login"))

@app.route("/admin")
def admin_dashboard():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))

    data = load_analytics()

    # last 7 days for chart
    from datetime import timedelta
    days, counts = [], []
    for i in range(6, -1, -1):
        d = str(date.today() - timedelta(days=i))
        days.append(d[5:])   # MM-DD
        counts.append(data["daily_activity"].get(d, 0))

    # model sizes
    model_files = {
        "rumor_classifier_model.h5": "Tweet Classifier",
        "xgb_model.pkl":             "News XGBoost",
        "spam_classifier_model.h5":  "SMS Classifier",
        "spam_email_model.h5":       "Email Classifier",
        "url01_model.h5":            "URL Analyzer",
        "news_model.h5":             "Article LSTM",
        "new_email_lstm_model.keras":"Email LSTM",
        "deepfake_model.h5":         "Deepfake Model",
    }
    model_status = []
    for fname, display in model_files.items():
        fpath = os.path.join(MODEL_DIR, fname)
        size  = round(os.path.getsize(fpath) / (1024*1024), 1) if os.path.exists(fpath) else 0
        model_status.append({"name": display, "file": fname, "size": size, "loaded": os.path.exists(fpath)})

    total_all = data["total_predictions"] + data["total_deepfake_scans"]

    return render_template(
        "admin.html",
        data=data,
        days=days,
        counts=counts,
        model_status=model_status,
        total_all=total_all,
    )

@app.route("/admin/clear-logs", methods=["POST"])
def admin_clear_logs():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))
    data = load_analytics()
    data["recent_logs"] = []
    save_analytics(data)
    return redirect(url_for("admin_dashboard"))

@app.route("/admin/reset-stats", methods=["POST"])
def admin_reset_stats():
    if not session.get("admin"):
        return redirect(url_for("admin_login"))
    data = load_analytics()
    data["total_predictions"]   = 0
    data["total_deepfake_scans"]= 0
    data["model_usage"]         = {k: 0 for k in data["model_usage"]}
    data["verdict_counts"]      = {"safe": 0, "danger": 0, "warning": 0}
    data["deepfake_verdicts"]   = {"Real": 0, "Fake": 0}
    data["daily_activity"]      = {}
    data["recent_logs"]         = []
    save_analytics(data)
    return redirect(url_for("admin_dashboard"))

# ── API Routes ──────────────────────────────────────────────────
@app.route("/api/google-news")
def google_news():
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "No query provided"}), 400
    try:
        url  = f"https://www.googleapis.com/customsearch/v1?q={quote(query + ' news')}&key={GOOGLE_API_KEY}&cx={GOOGLE_CX}&num=6"
        resp = http_requests.get(url, timeout=15)
        data = resp.json()
        if "error" in data:
            return jsonify({"error": data["error"].get("message", "Google API error")}), 502
        results = [
            {"title": item.get("title",""), "snippet": item.get("snippet",""),
             "url": item.get("link",""), "source": item.get("displayLink","")}
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
        if data.get("status") != "ok":
            return jsonify({"error": data.get("message", "NewsAPI error")}), 502
        results = [
            {"title": a.get("title",""), "snippet": a.get("description") or a.get("content",""),
             "url": a.get("url",""), "source": a.get("source",{}).get("name","")}
            for a in data.get("articles", [])
            if a.get("title") and a.get("url") and "[Removed]" not in a.get("title","")
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
        short_query = " ".join(query.split()[:5])
        url  = f"https://factchecktools.googleapis.com/v1alpha1/claims:search?query={quote(short_query)}&key={GOOGLE_API_KEY}&pageSize=6"
        resp = http_requests.get(url, timeout=15)
        data = resp.json()
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
