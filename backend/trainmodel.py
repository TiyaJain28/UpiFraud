import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os
import sys

# Add the current directory to the path so we can import utils
sys.path.append(os.path.dirname(__file__))

def preprocess_data(df, is_training=True):
    """
    Preprocess the new dataset
    """
    df_processed = df.copy()
    
    # Handle missing values
    if 'zip' in df_processed.columns:
        df_processed['zip'] = df_processed['zip'].fillna(0)
    else:
        df_processed['zip'] = 0
    
    # Extract time-based features
    df_processed['hour'] = df_processed['trans_hour']
    df_processed['day'] = df_processed['trans_day']
    df_processed['month'] = df_processed['trans_month']
    df_processed['year'] = df_processed['trans_year']
    
    # Create additional features that might be useful
    df_processed['transaction_amount_log'] = np.log1p(df_processed['trans_amount'])
    
    # For training, use fraud_risk as target, otherwise keep original
    if is_training:
        df_processed['isFraud'] = df_processed['fraud_risk'].astype(int)
    
    return df_processed

def train_models():
    # Load data
    df = pd.read_csv(r"C:\Users\HP\Downloads\upi_fraud_dataset.csv")
    
    # Preprocess data
    df_processed = preprocess_data(df, is_training=True)
    
    # Define features based on new dataset
    features = [
        'trans_amount', 'hour', 'day', 'month', 'year', 
        'transaction_amount_log', 'zip'
    ]
    
    X = df_processed[features]
    y = df_processed['isFraud']
    
    # Check class distribution
    print(f"Class distribution: {y.value_counts()}")
    print(f"Fraud percentage: {y.mean() * 100:.2f}%")
    
    # Handle class imbalance with SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    
    # Train Random Forest
    print("Training Random Forest...")
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_model.fit(X_train_scaled, y_train)
    
    # Evaluate models
    models = {
        'Logistic Regression': lr_model,
        'Random Forest': rf_model
    }
    
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        results[name] = {
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'f1_score': f1_score(y_test, y_pred)
        }
    
    # Print results
    for name, metrics in results.items():
        print(f"\n{name} Results:")
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print("Classification Report:")
        print(metrics['classification_report'])
    
    # Select the best model based on F1 score
    best_model_name = max(results, key=lambda x: results[x]['f1_score'])
    best_model = models[best_model_name]
    print(f"\nBest model: {best_model_name}")
    
    # Save the best model and all preprocessing objects
    joblib.dump(best_model, 'models/best_model.joblib')
    joblib.dump(features, 'models/feature_names.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    # Verify all files were created
    required_files = [
        'models/best_model.joblib',
        'models/feature_names.joblib',
        'models/scaler.joblib'
    ]
    
    print("\nModel files created:")
    for file_path in required_files:
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            print(f"✓ {file_path} ({os.path.getsize(file_path)} bytes)")
        else:
            print(f"✗ ERROR: {file_path} missing or empty")
    
    return best_model_name, results

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Train models
    try:
        best_model, results = train_models()
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()