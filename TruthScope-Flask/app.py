from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb

app = Flask(__name__)

# Load models
xgb_model = joblib.load('models/xgb_model.pkl')
xgb_vectorizer = joblib.load('models/tfidf_vectorizer_xgb.pkl')

dl_model = tf.keras.models.load_model('models/best_model03_80.h5')
dl_vectorizer = joblib.load('models/tfidf_vectorizer03.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form['input_text']
    model_type = request.form['model_type']

    if model_type == 'xgboost':
        transformed_text = xgb_vectorizer.transform([input_text])
        prediction = xgb_model.predict(transformed_text)
    elif model_type == 'dl':
        # Add dummy 'keyword' and 'location' for transformer
        input_df = pd.DataFrame([{
            'text': input_text,
            'keyword': 'unknown',   # default or placeholder
            'location': 'unknown'   # default or placeholder
        }])
        transformed_text = dl_vectorizer.transform(input_df).toarray()
        prediction = dl_model.predict(transformed_text)
        prediction = (prediction > 0.5).astype("int32")
    else:
        prediction = ['Invalid model']

    label = 'Fake Tweet' if prediction[0] == 1 else 'Real Tweet'
    return render_template('index.html', prediction=label, input_text=input_text)

if __name__ == '__main__':
    app.run(debug=True)
