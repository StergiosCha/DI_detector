from flask import Flask, request, render_template_string, jsonify
import joblib
from pathlib import Path
import os

app = Flask(__name__)

# Load model at startup
try:
    print("Loading model...")
    model = joblib.load('models/dialect_model.joblib')
    vectorizer = joblib.load('models/vectorizer.joblib')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model, vectorizer = None, None

# Your existing HTML_TEMPLATE here...

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not vectorizer:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        text = request.json.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Transform and predict
        X = vectorizer.transform([text])
        dialect = model.predict(X)[0]

        return jsonify({'dialect': dialect})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)