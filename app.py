import os
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up paths
model_path = os.path.dirname(os.path.abspath(__file__))

# Global variables to store loaded models
models = None
metadata = None

def find_latest_model_files():
    """Find the most recent model files"""
    try:
        json_files = [f for f in os.listdir(model_path) if f.startswith('model_metadata_') and f.endswith('.json')]
        
        if not json_files:
            raise FileNotFoundError("Model files not found")
        
        # Get the latest metadata file
        latest_metadata = max(json_files)
        metadata_file = os.path.join(model_path, latest_metadata)
        
        # Extract timestamp from metadata filename
        file_timestamp = latest_metadata.replace('model_metadata_', '').replace('.json', '')
        
        model_files_dict = {
            'overall_score_model': os.path.join(model_path, f"tabnet_score_model_{file_timestamp}.pkl"),
            'label_model': os.path.join(model_path, f"tabnet_label_model_{file_timestamp}.pkl"),
            'metric_scores_model': os.path.join(model_path, f"tabnet_metric_scores_model_{file_timestamp}.pkl"),
            'scaler': os.path.join(model_path, f"feature_scaler_{file_timestamp}.pkl"),
            'label_encoder': os.path.join(model_path, f"label_encoder_{file_timestamp}.pkl"),
            'metadata': metadata_file
        }
        
        return model_files_dict, file_timestamp
    except Exception as e:
        print(f"Error finding model files: {e}")
        return None, None

def load_models():
    """Load all trained models and preprocessors"""
    global models, metadata
    
    try:
        model_files, _ = find_latest_model_files()
        if not model_files:
            return False
        
        models = {}
        
        # Load models
        with open(model_files['overall_score_model'], 'rb') as f:
            models['score_model'] = pickle.load(f)
        
        with open(model_files['label_model'], 'rb') as f:
            models['label_model'] = pickle.load(f)
        
        with open(model_files['metric_scores_model'], 'rb') as f:
            models['metric_model'] = pickle.load(f)
        
        # Load preprocessors
        with open(model_files['scaler'], 'rb') as f:
            models['scaler'] = pickle.load(f)
        
        with open(model_files['label_encoder'], 'rb') as f:
            models['label_encoder'] = pickle.load(f)
        
        # Load metadata
        with open(model_files['metadata'], 'r') as f:
            metadata = json.load(f)
            models['metadata'] = metadata
        
        print("Models loaded successfully")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

def predict_quality(features_dict):
    """Make predictions using the loaded models"""
    try:
        # Get feature order from metadata
        feature_order = metadata['input_features']
        
        # Check if all required features are present
        missing_features = [f for f in feature_order if f not in features_dict]
        if missing_features:
            return {"error": f"Missing features: {missing_features}"}, 400
        
        # Create feature array in correct order
        features_array = np.array([[features_dict[feature] for feature in feature_order]])
        
        # Scale features
        features_scaled = models['scaler'].transform(features_array)
        
        # Make predictions
        predictions = {}
        
        # Overall quality score
        score_pred = models['score_model'].predict(features_scaled)
        # Ensure score is within 0-100 range
        raw_score = float(score_pred[0])
        predictions['overall_score'] = min(100.0, max(0.0, raw_score))
        
        # Add a note if the score was capped
        if raw_score > 100.0:
            predictions['score_note'] = f"Original score ({raw_score:.1f}) was capped at 100.0"
        
        # Quality label
        label_pred_encoded = models['label_model'].predict(features_scaled)
        label_pred_proba = models['label_model'].predict_proba(features_scaled)
        predictions['quality_label'] = models['label_encoder'].inverse_transform(label_pred_encoded)[0]
        predictions['label_probabilities'] = {
            label: float(prob) for label, prob in 
            zip(models['label_encoder'].classes_, label_pred_proba[0])
        }
        
        # Individual metric scores
        metric_scores_pred = models['metric_model'].predict(features_scaled)
        metric_features = metadata['metric_score_features']
        predictions['metric_scores'] = {
            metric.replace('_Score', ''): min(100.0, max(0.0, float(score))) for metric, score in 
            zip(metric_features, metric_scores_pred[0])
        }
        
        # Check if any metric scores were capped
        original_scores = {metric.replace('_Score', ''): float(score) for metric, score in zip(metric_features, metric_scores_pred[0])}
        capped_metrics = {k: v for k, v in original_scores.items() if v > 100.0}
        if capped_metrics:
            predictions['metrics_capped'] = capped_metrics
        
        # Add timestamp for tracking
        predictions['prediction_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Find top 3 worst metric scores for quick insights
        metric_scores = predictions['metric_scores']
        worst_metrics = sorted(metric_scores.items(), key=lambda x: x[1])[:3]
        predictions['top_issues'] = {metric: score for metric, score in worst_metrics}
        
        return predictions, 200
    except Exception as e:
        print(f"Error making predictions: {e}")
        return {"error": f"Prediction error: {str(e)}"}, 500

@app.route('/')
def home():
    """Home page with interactive UI"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    if models is None:
        return jsonify({"status": "error", "message": "Models not loaded"}), 500
    return jsonify({"status": "healthy", "message": "API is running and models are loaded"})

@app.route('/metadata')
def get_metadata():
    """Get model metadata"""
    if metadata is None:
        return jsonify({"error": "Metadata not loaded"}), 500
    
    # Return a simplified version of metadata for API users
    api_metadata = {
        "input_features": metadata["input_features"],
        "metric_score_features": metadata["metric_score_features"],
        "label_classes": metadata["label_classes"],
        "model_version": metadata.get("model_version", "1.0.0"),
        "training_timestamp": metadata.get("training_timestamp", "unknown")
    }
    
    return jsonify(api_metadata)

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint to make predictions"""
    if models is None:
        success = load_models()
        if not success:
            return jsonify({"error": "Failed to load models"}), 500
    
    # Get JSON data from request
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # If features are nested under a key, extract them
        features = data.get('features', data)
        
        # Make predictions
        predictions, status_code = predict_quality(features)
        
        return jsonify(predictions), status_code
    
    except Exception as e:
        return jsonify({"error": f"Request error: {str(e)}"}), 400

@app.route('/sample-input', methods=['GET'])
def sample_input():
    """Return a sample input format for the API"""
    if metadata is None:
        success = load_models()
        if not success:
            return jsonify({"error": "Failed to load models"}), 500
    
    # Create a sample input with default values
    sample = {}
    for feature in metadata['input_features']:
        # Set reasonable default values based on feature name
        if 'Count' in feature:
            sample[feature] = 1000
        elif 'Rate' in feature or 'Pct' in feature:
            sample[feature] = 0.05
        elif 'Size' in feature:
            sample[feature] = 10.0
        else:
            sample[feature] = 0.5
    
    return jsonify({
        "sample_input": sample,
        "usage": "POST this JSON structure to /predict endpoint"
    })

# Create a function to load models at startup
def initialize_app():
    global models, metadata
    if models is None:
        load_models()

# Register the function to run before the first request
@app.route('/initialize', methods=['GET'])
def initialize_route():
    initialize_app()
    return jsonify({"status": "Models initialized successfully"})

# Create a blueprint to run code before first request
with app.app_context():
    initialize_app()

if __name__ == '__main__':
    # Load models at startup
    load_models()
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
