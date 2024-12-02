from flask import Flask, request, render_template_string, jsonify
import joblib
from pathlib import Path
import os

app = Flask(__name__)

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
        .instructions {
            color: #666;
            margin: 20px 0;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
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
        #wordCount {
            color: #666;
            font-size: 0.9em;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Greek Dialect Predictor</h1>
        <div class="instructions">
            Please put more than 20 words of the text you are interested in.<br>
            The model is trained on Standard Greek, Cypriot, Pontic, Cretan and Northern Greek.
        </div>
        <div>
            <textarea id="text-input" placeholder="Enter Greek text here (minimum 20 words)..." oninput="updateWordCount()"></textarea>
            <div id="wordCount">Words: 0</div>
        </div>
        <button onclick="predict()">Predict Dialect</button>
        <div id="loading">Analyzing...</div>
        <div id="result"></div>
    </div>

    <script>
        function updateWordCount() {
            const text = document.getElementById('text-input').value;
            const wordCount = text.trim().split(/\s+/).filter(word => word.length > 0).length;
            document.getElementById('wordCount').textContent = `Words: ${wordCount}`;
        }

        function predict() {
            const text = document.getElementById('text-input').value;
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            
            // Check for minimum word count
            const wordCount = text.trim().split(/\s+/).filter(word => word.length > 0).length;
            if (wordCount < 20) {
                showResult('Please enter at least 20 words for accurate prediction', 'error');
                return;
            }

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
    current_dir = Path(__file__).parent
    print(f"Current directory: {current_dir}")
    
    # List all files in the directory and subdirectories
    print("Files in directory:")
    for file in current_dir.glob('**/*'):
        print(f"Found file: {file}")
    
    model_path = current_dir / 'models' / 'dialect_model.joblib'
    vectorizer_path = current_dir / 'models' / 'vectorizer.joblib'
    
    print(f"Looking for model at: {model_path}")
    print(f"Looking for vectorizer at: {vectorizer_path}")
    
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Current working directory: {Path.cwd()}")
    print(f"Directory contents: {list(Path.cwd().glob('**/*.joblib'))}")
    model, vectorizer = None, None

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not vectorizer:
        return jsonify({'error': 'Model is still loading or failed to load. Please try again later.'}), 503

    try:
        text = request.json.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400

        # Transform and predict
        X = vectorizer.transform([text])
        dialect = model.predict(X)[0]

        return jsonify({'dialect': dialect})

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
