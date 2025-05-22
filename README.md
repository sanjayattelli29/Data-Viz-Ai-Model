# Data Quality Prediction API

This API provides data quality prediction services using TabNet models. It analyzes input features and predicts data quality scores, labels, and individual metric scores.

## API Endpoints

### 1. Home
- **URL**: `/`
- **Method**: `GET`
- **Description**: Returns basic API information and available endpoints

### 2. Health Check
- **URL**: `/health`
- **Method**: `GET`
- **Description**: Checks if the API is running and models are loaded properly

### 3. Metadata
- **URL**: `/metadata`
- **Method**: `GET`
- **Description**: Returns model metadata including input features, metric score features, and label classes

### 4. Sample Input
- **URL**: `/sample-input`
- **Method**: `GET`
- **Description**: Returns a sample input format for the prediction endpoint

### 5. Predict
- **URL**: `/predict`
- **Method**: `POST`
- **Description**: Makes data quality predictions based on input features
- **Request Body**: JSON object with input features
- **Response**: JSON object with prediction results

## Example Usage

### Making a Prediction

```python
import requests
import json

# API endpoint
url = "https://your-render-app-url.onrender.com/predict"

# Sample input features
input_features = {
    "Row_Count": 50000,
    "Column_Count": 20,
    "File_Size_MB": 45.2,
    "Numeric_Columns_Count": 14,
    "Categorical_Columns_Count": 4,
    "Date_Columns_Count": 2,
    "Missing_Values_Pct": 1.2,
    "Duplicate_Records_Count": 20,
    "Outlier_Rate": 0.02,
    "Inconsistency_Rate": 0.01,
    "Data_Type_Mismatch_Rate": 0.005,
    "Null_vs_NaN_Distribution": 0.9,
    "Cardinality_Categorical": 50,
    "Target_Imbalance": 0.48,
    "Feature_Importance_Consistency": 0.92,
    "Class_Overlap_Score": 0.1,
    "Label_Noise_Rate": 0.01,
    "Feature_Correlation_Mean": 0.3,
    "Range_Violation_Rate": 0.005,
    "Mean_Median_Drift": 0.05,
    "Data_Freshness": 0.98,
    "Anomaly_Count": 15,
    "Encoding_Coverage_Rate": 0.99,
    "Variance_Threshold_Check": 0.95,
    "Data_Density_Completeness": 0.98,
    "Domain_Constraint_Violations": 5
}

# Make the POST request
response = requests.post(url, json=input_features)

# Print the response
print(json.dumps(response.json(), indent=4))
```

### Response Format

```json
{
    "overall_score": 87.65,
    "quality_label": "Good",
    "label_probabilities": {
        "Excellent": 0.25,
        "Good": 0.65,
        "Moderate": 0.08,
        "Poor": 0.02
    },
    "metric_scores": {
        "Row_Count": 95.0,
        "Column_Count": 85.0,
        "File_Size_MB": 90.0,
        "Numeric_Columns_Count": 88.0,
        ...
    },
    "top_issues": {
        "Missing_Values_Pct": 75.0,
        "Outlier_Rate": 78.0,
        "Data_Type_Mismatch_Rate": 80.0
    },
    "prediction_time": "2025-05-22 23:45:25"
}
```

## Deployment on Render

This API is designed to be deployed on Render. Follow these steps:

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Set the following:
   - **Name**: data-quality-prediction-api
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`

## Input Features

The API requires the following 26 input features:

- Row_Count
- Column_Count
- File_Size_MB
- Numeric_Columns_Count
- Categorical_Columns_Count
- Date_Columns_Count
- Missing_Values_Pct
- Duplicate_Records_Count
- Outlier_Rate
- Inconsistency_Rate
- Data_Type_Mismatch_Rate
- Null_vs_NaN_Distribution
- Cardinality_Categorical
- Target_Imbalance
- Feature_Importance_Consistency
- Class_Overlap_Score
- Label_Noise_Rate
- Feature_Correlation_Mean
- Range_Violation_Rate
- Mean_Median_Drift
- Data_Freshness
- Anomaly_Count
- Encoding_Coverage_Rate
- Variance_Threshold_Check
- Data_Density_Completeness
- Domain_Constraint_Violations
