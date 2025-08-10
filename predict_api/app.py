from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging
import sqlite3
from datetime import datetime
import os
from pydantic import BaseModel, validator
import traceback

app = Flask(__name__)

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/predictions.log'),
        logging.StreamHandler()
    ]
)

# Load model
print("Current working directory:", os.getcwd())
print("Checking if model file exists:", os.path.exists('models/best_model.joblib'))

try:
    model = joblib.load('models/best_model.joblib')
    print("Model loaded successfully in try block")
    print("Model type:", type(model))
    print("Model is None:", model is None)
    logging.info("Model loaded successfully")
except Exception as e:
    print("Exception during model loading:", str(e))
    print("Exception type:", type(e))
    logging.error(f"Error loading model: {e}")
    model = None

print("Final model status:", model is not None)

# # Load model
# try:
#     model = joblib.load('models/best_model.joblib')
#     logging.info("Model loaded successfully")
# except Exception as e:
#     logging.error(f"Error loading model: {e}")
#     model = None

# Pydantic model for input validation
class PredictionRequest(BaseModel):
    features: list
    
    @validator('features')
    def validate_features(cls, v):
        if len(v) != 4:
            raise ValueError('Features must contain exactly 4 values for Iris dataset')
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError('All features must be numeric')
        return v

def init_db():
    """Initialize SQLite database for logging"""
    conn = sqlite3.connect('logs/predictions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  features TEXT,
                  prediction INTEGER,
                  confidence REAL)''')
    conn.commit()
    conn.close()

def log_prediction(features, prediction, confidence=None):
    """Log prediction to database"""
    try:
        conn = sqlite3.connect('logs/predictions.db')
        c = conn.cursor()
        c.execute('''INSERT INTO predictions (timestamp, features, prediction, confidence) 
                     VALUES (?, ?, ?, ?)''', 
                  (datetime.now().isoformat(), str(features), int(prediction), confidence))
        conn.commit()
        conn.close()
    except Exception as e:
        logging.error(f"Error logging to database: {e}")

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'message': 'Iris Classification API',
        'version': '1.0',
        'endpoints': {
            '/predict': 'POST - Make predictions',
            '/health': 'GET - Health check',
            '/metrics': 'GET - API metrics'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    model_status = "healthy" if model is not None else "model_not_loaded"
    return jsonify({
        'status': 'healthy',
        'model_status': model_status,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Validate input
        data = request.json
        prediction_request = PredictionRequest(**data)
        features = np.array(prediction_request.features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)
        
        # Get prediction probabilities if available
        confidence = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)
            confidence = float(np.max(proba))
        
        # Map prediction to class names
        class_names = ['setosa', 'versicolor', 'virginica']
        predicted_class = class_names[int(prediction[0])]
        
        # Log the prediction
        logging.info(f"Prediction request: features={prediction_request.features}, prediction={predicted_class}")
        log_prediction(prediction_request.features, prediction[0], confidence)
        
        return jsonify({
            'prediction': int(prediction[0]),
            'predicted_class': predicted_class,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })
        
    except ValueError as ve:
        logging.warning(f"Validation error: {ve}")
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        logging.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/metrics', methods=['GET'])
def metrics():
    try:
        conn = sqlite3.connect('logs/predictions.db')
        c = conn.cursor()
        
        # Get total predictions
        c.execute('SELECT COUNT(*) FROM predictions')
        total_predictions = c.fetchone()[0]
        
        # Get predictions by class
        c.execute('SELECT prediction, COUNT(*) FROM predictions GROUP BY prediction')
        class_counts = dict(c.fetchall())
        
        # Get recent predictions (last 24 hours)
        c.execute('''SELECT COUNT(*) FROM predictions 
                     WHERE datetime(timestamp) > datetime('now', '-1 day')''')
        recent_predictions = c.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'total_predictions': total_predictions,
            'predictions_last_24h': recent_predictions,
            'predictions_by_class': class_counts,
            'model_version': '1.0',
            'status': 'healthy',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logging.error(f"Metrics error: {e}")
        return jsonify({'error': 'Unable to fetch metrics'}), 500

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000, debug=False)