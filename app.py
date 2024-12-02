from flask import Flask, request, render_template_string, jsonify
import joblib
from pathlib import Path

app = Flask(__name__)

# HTML template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Greek Dialect Predictor</title>
    <meta charset="UTF-8">
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        textarea {
            width: 100%;
            height: 150px;
            padding: 12px;
            margin: 10px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 16px;
            resize: vertical;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            margin: 10px 0;
        }
        button:hover {
            background-color: #45a049;
        }
        #result {
            margin-top: 20px;
            padding: 15px;
            border-radius: 4px;
            display: none;
        }
        .success {
            background-color: #dff0d8;
            border: 1px solid #d6e9c6;
            color: #3c763d;
        }
        .error {
            background-color: #f2dede;
            border: 1px solid #ebccd1;
            color: #a94442;
        }
        #loading {
            display: none;
            text-align: center;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Greek Dialect Predictor</h1>
        <div>
            <textarea id="text-input" placeholder="Enter Greek text here..."></textarea>
        </div>
        <button onclick="predict()">Predict Dialect</button>
        <div id="loading">Analyzing...</div>
        <div id="result"></div>
    </div>

    <script>
        function predict() {
            const text = document.getElementById('text-input').value;
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');

            if (!text) {
                showResult('Please enter some text', 'error');
                return;
            }

            // Show loading, hide previous result
            loading.style.display = 'block';
            result.style.display = 'none';

            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text: text })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    showResult(data.error, 'error');
                } else {
                    showResult(`Predicted dialect: ${data.dialect}`, 'success');
                }
            })
            .catch(error => {
                showResult('Error predicting dialect', 'error');
            })
            .finally(() => {
                loading.style.display = 'none';
            });
        }

        function showResult(message, type) {
            const result = document.getElementById('result');
            result.textContent = message;
            result.className = type;
            result.style.display = 'block';
        }
    </script>
</body>
</html>
'''

# Load model at startup
try:
    print("Loading model...")
    model = joblib.load('dialect_model.joblib')
    vectorizer = joblib.load('vectorizer.joblib')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model, vectorizer = None, None

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
    app.run(debug=True, port=5001)