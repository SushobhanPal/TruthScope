from flask import Flask, render_template, request
import pickle
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


models = {
    "best_model": load_model(os.path.join(MODEL_DIR, "rumor_classifier_model.h5")),
    "xgb_model": joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl")),
    "spam_classifier": load_model(os.path.join(MODEL_DIR, "spam_classifier_model.h5")),
    "spam_email": load_model(os.path.join(MODEL_DIR, "spam_email_model.h5")),
    "url01": load_model(os.path.join(MODEL_DIR, "url01_model.h5")),
    "news":load_model(os.path.join(MODEL_DIR,"news_model.h5")),
}

vectorizers = {
    "best_model": pickle.load(open(os.path.join(MODEL_DIR, "rumor_vectorizer.pkl"), "rb")),
    "xgb_model": joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer_xgb.pkl")),
    "spam_classifier": pickle.load(open(os.path.join(MODEL_DIR, "vectorizer_spam.pkl"), "rb")),
    "spam_email": pickle.load(open(os.path.join(MODEL_DIR, "vectorizer_email.pkl"), "rb")),
    "url01": pickle.load(open(os.path.join(MODEL_DIR, "vectorizer_url01.pkl"), "rb")),
    "news":pickle.load(open(os.path.join(MODEL_DIR,"vectorizer_news.pkl"),"rb")),
}

def preprocess_text(text):
    return text.lower().strip()

rumor_labels = {0: "false", 1: "non-rumor", 2: "true", 3: "unverified"}

@app.route("/", methods=["GET", "POST"])
def home():
    label = None
    prediction = None
    selected_model = None
    user_input = ""

    if request.method == "POST":
        user_input = request.form.get("user_input")
        selected_model = request.form.get("model_select")

        if user_input and selected_model:
            clean_input = preprocess_text(user_input)
            vect = vectorizers[selected_model].transform([clean_input])

            if selected_model == "xgb_model":
                pred_proba = models[selected_model].predict_proba(vect)[:, 1]
                prediction = int(pred_proba[0] >= 0.5)
                label = "Positive" if prediction == 1 else "Negative"
            elif selected_model == "best_model":
                pred = models[selected_model].predict(vect.toarray()) # rumor classifier (4 classes)
                prediction = np.argmax(pred, axis=1)[0] # get the class index
                label = rumor_labels[prediction] # map index → class name
            else:
                pred = models[selected_model].predict(vect.toarray())
                prob = pred[0][0]
                prediction = int(prob >= 0.5)

           # print(f"User input: {user_input}")
          #  print(f"Cleaned input: {clean_input}")
           # print(f"Vectorized shape: {vect.shape}")
           # print(f"Prediction probability: {pred_proba if selected_model == 'xgb_model' else prob}")
           # print(f"Final prediction: {prediction}")
            if selected_model=="spam_classifier":
                label = 'Spam Message' if prediction == 1 else 'Likely not spam'
            elif selected_model=="spam_email":
                label = 'Likely not spam' if prediction == 1 else 'Spam Email'
            elif selected_model=="url01":
                label = 'Safe Website' if prediction == 1 else 'Website is potentially Risky'
            elif selected_model=="news":
                label = 'News is trustable' if prediction == 1 else 'The news is likely fake'
            
    return render_template(
        "index.html",
        prediction=label,
        selected_model=selected_model,
        user_input=user_input,
    )


if __name__ == "__main__":
    app.run(debug=True)
