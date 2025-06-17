import os
import json
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from datetime import datetime
from flask_cors import CORS
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


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
        
        # Load models (we only need label_model and metric_scores_model now)
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

def calculate_formula_quality_score(metrics):
    """Calculate data quality score using formula-based approach"""
    
    # Normalization helper functions
    def invert(metric, max_val):
        return max(0, 1 - metric / max_val)
    
    def passthrough(metric):
        return max(0, min(metric, 1))
    
    def bounded(metric, min_val, max_val):
        if max_val <= min_val:
            return 0
        return max(0, min((metric - min_val) / (max_val - min_val), 1))
    
    # Weights for each metric
    weights = {
        "Missing_Values_Pct": 0.08,
        "Data_Density_Completeness": 0.05,
        "Null_vs_NaN_Distribution": 0.03,
        "Domain_Constraint_Violations": 0.05,
        "Data_Type_Mismatch_Rate": 0.05,
        "Range_Violation_Rate": 0.03,
        "Encoding_Coverage_Rate": 0.04,
        "Duplicate_Records_Count": 0.05,
        "Inconsistency_Rate": 0.05,
        "Variance_Threshold_Check": 0.04,
        "Feature_Correlation_Mean": 0.02,
        "Anomaly_Count": 0.03,
        "Outlier_Rate": 0.05,
        "Mean_Median_Drift": 0.04,
        "Class_Overlap_Score": 0.05,
        "Label_Noise_Rate": 0.03,
        "Data_Freshness": 0.05,
        "Target_Imbalance": 0.04,
        "Feature_Importance_Consistency": 0.05,
        "Row_Count": 0.005,
        "Column_Count": 0.01,
        "Cardinality_Categorical": 0.01
    }
    
    # Calculate normalized scores for each metric
    scores = {}
    
    # Only calculate scores for metrics that exist in the input
    if "Missing_Values_Pct" in metrics:
        scores["Missing_Values_Pct"] = invert(metrics["Missing_Values_Pct"], 0.3)
    
    if "Data_Density_Completeness" in metrics:
        scores["Data_Density_Completeness"] = passthrough(metrics["Data_Density_Completeness"])
    
    if "Null_vs_NaN_Distribution" in metrics:
        scores["Null_vs_NaN_Distribution"] = invert(metrics["Null_vs_NaN_Distribution"], 1.0)
    
    if "Domain_Constraint_Violations" in metrics:
        scores["Domain_Constraint_Violations"] = invert(metrics["Domain_Constraint_Violations"], 50)
    
    if "Data_Type_Mismatch_Rate" in metrics:
        scores["Data_Type_Mismatch_Rate"] = invert(metrics["Data_Type_Mismatch_Rate"], 0.5)
    
    if "Range_Violation_Rate" in metrics:
        scores["Range_Violation_Rate"] = invert(metrics["Range_Violation_Rate"], 0.05)
    
    if "Encoding_Coverage_Rate" in metrics:
        scores["Encoding_Coverage_Rate"] = passthrough(metrics["Encoding_Coverage_Rate"])
    
    if "Duplicate_Records_Count" in metrics:
        scores["Duplicate_Records_Count"] = invert(metrics["Duplicate_Records_Count"], 500)
    
    if "Inconsistency_Rate" in metrics:
        scores["Inconsistency_Rate"] = invert(metrics["Inconsistency_Rate"], 0.5)
    
    if "Variance_Threshold_Check" in metrics:
        scores["Variance_Threshold_Check"] = passthrough(metrics["Variance_Threshold_Check"])
    
    if "Feature_Correlation_Mean" in metrics:
        scores["Feature_Correlation_Mean"] = passthrough(metrics["Feature_Correlation_Mean"])
    
    if "Anomaly_Count" in metrics:
        scores["Anomaly_Count"] = invert(metrics["Anomaly_Count"], 100)
    
    if "Outlier_Rate" in metrics:
        scores["Outlier_Rate"] = invert(metrics["Outlier_Rate"], 0.3)
    
    if "Mean_Median_Drift" in metrics:
        scores["Mean_Median_Drift"] = invert(metrics["Mean_Median_Drift"], 0.5)
    
    if "Class_Overlap_Score" in metrics:
        scores["Class_Overlap_Score"] = invert(metrics["Class_Overlap_Score"], 1.0)
    
    if "Label_Noise_Rate" in metrics:
        scores["Label_Noise_Rate"] = invert(metrics["Label_Noise_Rate"], 0.2)
    
    if "Data_Freshness" in metrics:
        scores["Data_Freshness"] = invert(metrics["Data_Freshness"], 365)
    
    if "Target_Imbalance" in metrics:
        scores["Target_Imbalance"] = passthrough(1 - abs(metrics["Target_Imbalance"] - 0.5) * 2)
    
    if "Feature_Importance_Consistency" in metrics:
        scores["Feature_Importance_Consistency"] = passthrough(metrics["Feature_Importance_Consistency"])
    
    if "Row_Count" in metrics:
        scores["Row_Count"] = bounded(metrics["Row_Count"], 1000, 100000)
    
    if "Column_Count" in metrics:
        scores["Column_Count"] = bounded(metrics["Column_Count"], 5, 50)
    
    if "Cardinality_Categorical" in metrics:
        scores["Cardinality_Categorical"] = bounded(metrics["Cardinality_Categorical"], 1, 50)
    
    # Calculate weighted total (only for metrics that exist)
    total_weight = sum(weights[metric] for metric in scores.keys() if metric in weights)
    if total_weight == 0:
        return 0.0
    
    weighted_sum = sum(scores[metric] * weights.get(metric, 0) for metric in scores.keys())
    
    # Normalize by actual total weight and scale to 100
    final_score = (weighted_sum / total_weight) * 100
    
    return round(final_score, 2)

def get_quality_label_from_score(score):
    """Convert formula score to quality label"""
    if score >= 90:
        return "Excellent"
    elif score >= 75:
        return "Good"
    elif score >= 60:
        return "Moderate"
    else:
        return "Poor"

def predict_quality(features_dict):
    """Make predictions using the loaded models and formula"""
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
        
        # Use formula-based scoring for overall quality score
        formula_score = calculate_formula_quality_score(features_dict)
        predictions['overall_score'] = formula_score
        predictions['score_method'] = "formula_based"
        
        # Quality label from TabNet model (keep original model prediction)
        label_pred_encoded = models['label_model'].predict(features_scaled)
        label_pred_proba = models['label_model'].predict_proba(features_scaled)
        predictions['quality_label_model'] = models['label_encoder'].inverse_transform(label_pred_encoded)[0]
        predictions['label_probabilities'] = {
            label: float(prob) for label, prob in 
            zip(models['label_encoder'].classes_, label_pred_proba[0])
        }
        
        # Also provide formula-based label for comparison
        predictions['quality_label_formula'] = get_quality_label_from_score(formula_score)
        
        # Use formula-based label as primary
        predictions['quality_label'] = predictions['quality_label_formula']
        
        # Individual metric scores using TabNet model
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
        
        # Add explanation of scoring method
        predictions['scoring_explanation'] = {
            "overall_score": "Calculated using weighted formula-based approach",
            "metric_scores": "Predicted using TabNet neural network model",
            "quality_label": "Derived from formula-based overall score"
        }
        
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
        "training_timestamp": metadata.get("training_timestamp", "unknown"),
        "scoring_method": "hybrid_formula_and_tabnet"
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
        "usage": "POST this JSON structure to /predict endpoint",
        "scoring_method": "Formula-based overall score + TabNet metric scores"
    })

@app.route('/formula-weights', methods=['GET'])
def get_formula_weights():
    """Return the weights used in formula calculation"""
    weights = {
        "Missing_Values_Pct": 0.08,
        "Data_Density_Completeness": 0.05,
        "Null_vs_NaN_Distribution": 0.03,
        "Domain_Constraint_Violations": 0.05,
        "Data_Type_Mismatch_Rate": 0.05,
        "Range_Violation_Rate": 0.03,
        "Encoding_Coverage_Rate": 0.05,
        "Duplicate_Records_Count": 0.05,
        "Inconsistency_Rate": 0.05,
        "Variance_Threshold_Check": 0.04,
        "Feature_Correlation_Mean": 0.02,
        "Anomaly_Count": 0.05,
        "Outlier_Rate": 0.05,
        "Mean_Median_Drift": 0.04,
        "Class_Overlap_Score": 0.05,
        "Label_Noise_Rate": 0.03,
        "Data_Freshness": 0.05,
        "Target_Imbalance": 0.04,
        "Feature_Importance_Consistency": 0.05,
        "Row_Count": 0.01,
        "Column_Count": 0.01,
        "Cardinality_Categorical": 0.01
    }
    
    return jsonify({
        "formula_weights": weights,
        "total_weight": sum(weights.values()),
        "description": "Weights used in formula-based quality score calculation"
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
    # Run the app on a unique port (2341)
    app.run(host='0.0.0.0', port=2341, debug=False)