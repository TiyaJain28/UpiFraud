from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import traceback
import os

# Import utility functions
from utils import preprocess_data, is_valid_upi_id

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables for loaded models
model = None
feature_names = None
preprocessing_objects = {}

def get_model_path(filename):
    """Get the correct path for model files"""
    # Try to find the models directory
    possible_paths = [
        'models/',  # Current directory
        '../models/',  # Parent directory
        '../../models/',  # Grandparent directory
        'backend/models/',  # Backend directory
        '../backend/models/'  # Parent backend directory
    ]
    
    for path in possible_paths:
        full_path = os.path.join(path, filename)
        if os.path.exists(full_path):
            return full_path
    
    # If not found, return the default path
    return os.path.join('models', filename)

def preprocess_data(df, is_training=False):
    """
    Preprocess data for inference - ensure this matches training preprocessing
    """
    df_processed = df.copy()
    
    # Handle missing values
    df_processed['zip'] = df_processed.get('zip', 0)
    
    # Extract time-based features
    df_processed['hour'] = df_processed['trans_hour']
    df_processed['day'] = df_processed['trans_day']
    df_processed['month'] = df_processed['trans_month']
    df_processed['year'] = df_processed['trans_year']
    
    # Create additional features
    df_processed['transaction_amount_log'] = np.log1p(df_processed['trans_amount'])
    
    return df_processed

def load_models():
    """Load the trained models and preprocessing objects"""
    global model, feature_names, preprocessing_objects
    
    try:
        # Check if model files exist using the correct path
        model_path = get_model_path('best_model.joblib')
        features_path = get_model_path('feature_names.joblib')
        scaler_path = get_model_path('scaler.joblib')
        
        # Check if files exist
        required_files = {
            'model': model_path,
            'features': features_path,
            'scaler': scaler_path
        }
        
        for name, path in required_files.items():
            if not os.path.exists(path):
                print(f"ERROR: {name} file not found at {path}. Please train the model first.")
                print("Run: python trainmodel.py")
                return False
                
            if os.path.getsize(path) == 0:
                print(f"ERROR: {name} file is empty. Please train the model again.")
                return False
        
        # Load the models and preprocessing objects
        model = joblib.load(model_path)
        feature_names = joblib.load(features_path)
        preprocessing_objects['scaler'] = joblib.load(scaler_path)
        
        print("Models and preprocessing objects loaded successfully")
        print(f"Model type: {type(model)}")
        print(f"Features: {feature_names}")
        return True
        
    except EOFError:
        print("ERROR: Model file is corrupted. Please delete and retrain.")
        print("Run these commands:")
        print("  rm models/*.joblib")
        print("  python trainmodel.py")
        return False
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        print(traceback.format_exc())
        return False

def ensure_feature_consistency(df, feature_names):
    """Ensure the DataFrame has all the features the model expects"""
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Add any missing features with appropriate values
    for feature in feature_names:
        if feature not in df_copy.columns:
            # For missing features, use 0 as default
            df_copy[feature] = 0
    
    # Ensure the DataFrame has exactly the features the model expects, in the right order
    return df_copy[feature_names]

@app.route('/')
def home():
    return jsonify({
        "message": "UPI Fraud Detection API",
        "status": "active",
        "endpoints": {
            "predict": "/predict (POST)",
            "batch_predict": "/batch_predict (POST)",
            "health": "/health (GET)"
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    models_loaded = model is not None and feature_names is not None
    return jsonify({
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict fraud for a single transaction"""
    try:
        # Check if models are loaded
        if model is None:
            if not load_models():
                return jsonify({"error": "Models not loaded"}), 500
        
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Validate required fields for new dataset
        required_fields = ['trans_amount', 'trans_hour', 'trans_day', 'trans_month', 
                          'trans_year', 'category', 'upi_number', 'state']
        
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Validate UPI ID
        if not is_valid_upi_id(data['upi_number']):
            return jsonify({"error": "Invalid UPI ID format"}), 400
        
        # Create transaction data with proper structure for new dataset
        transaction_data = {
            'trans_hour': int(data['trans_hour']),
            'trans_day': int(data['trans_day']),
            'trans_month': int(data['trans_month']),
            'trans_year': int(data['trans_year']),
            'category': data['category'],
            'upi_number': data['upi_number'],
            'trans_amount': float(data['trans_amount']),
            'state': data['state'],
            'zip': data.get('zip', 0),  # Optional field
        }
        
        # Convert to DataFrame
        transaction_df = pd.DataFrame([transaction_data])
        
        # Preprocess the transaction data
        processed_df = preprocess_data(transaction_df, is_training=False)
        
        # Ensure feature consistency with what the model expects
        X = ensure_feature_consistency(processed_df, feature_names)
        
        # Make prediction
        prediction = model.predict(X)
        probability = model.predict_proba(X)
        
        fraud_probability = float(probability[0][1] * 100)  # Probability of fraud
        
        # Prepare response
        response = {
            "transaction_id": data.get('transaction_id', 'N/A'),
            "prediction": "Fraud" if prediction[0] == 1 else "Safe",
            "fraud_probability": fraud_probability,
            "risk_level": "High" if fraud_probability > 70 else ("Medium" if fraud_probability > 30 else "Low"),
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict fraud for multiple transactions"""
    try:
        if model is None:
            if not load_models():
                return jsonify({"error": "Models not loaded"}), 500
        
        data = request.get_json()
        
        if not data or 'transactions' not in data:
            return jsonify({"error": "No transactions provided"}), 400
        
        transactions = data['transactions']
        results = []
        
        for transaction_data in transactions:
            # Validate required fields
            required_fields = ['trans_amount', 'trans_hour', 'trans_day', 'trans_month', 
                              'trans_year', 'category', 'upi_number', 'state']
            
            for field in required_fields:
                if field not in transaction_data:
                    results.append({
                        "error": f"Missing required field: {field}",
                        "transaction_id": transaction_data.get('transaction_id', 'N/A')
                    })
                    continue
            
            # Validate UPI ID
            if not is_valid_upi_id(transaction_data['upi_number']):
                results.append({
                    "error": "Invalid UPI ID format",
                    "transaction_id": transaction_data.get('transaction_id', 'N/A')
                })
                continue
            
            # Process transaction
            transaction_df = pd.DataFrame([{
                'trans_hour': int(transaction_data['trans_hour']),
                'trans_day': int(transaction_data['trans_day']),
                'trans_month': int(transaction_data['trans_month']),
                'trans_year': int(transaction_data['trans_year']),
                'category': transaction_data['category'],
                'upi_number': transaction_data['upi_number'],
                'trans_amount': float(transaction_data['trans_amount']),
                'state': transaction_data['state'],
                'zip': transaction_data.get('zip', 0),
            }])
            
            processed_df = preprocess_data(transaction_df, is_training=False)
            
            # Ensure feature consistency with what the model expects
            X = ensure_feature_consistency(processed_df, feature_names)
            
            prediction = model.predict(X)
            probability = model.predict_proba(X)
            
            fraud_probability = float(probability[0][1] * 100)
            
            results.append({
                "transaction_id": transaction_data.get('transaction_id', 'N/A'),
                "prediction": "Fraud" if prediction[0] == 1 else "Safe",
                "fraud_probability": fraud_probability,
                "risk_level": "High" if fraud_probability > 70 else ("Medium" if fraud_probability > 30 else "Low")
            })
        
        return jsonify({
            "count": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        print(f"Error in batch prediction: {str(e)}")
        print(traceback.format_exc())
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

# Load models when the application starts
print("Loading models...")
load_models()

if __name__ == '__main__':
    print("Starting UPI Fraud Detection API server...")
    app.run(host='0.0.0.0', port=5000, debug=True)