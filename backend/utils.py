import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import re
import os

def get_model_path(filename):
    """Get the correct path for model files"""

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
    
   
    return os.path.join('models', filename)

def preprocess_data(df, is_training=True):
    """
    Preprocess the new dataset - ensure this matches the training preprocessing
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
    
    return df_processed

def extract_features(df):
    """
    Extract additional features from the transaction data for new dataset
    """
    df = df.copy()
    
    # Extract time features
    df['hour'] = df['trans_hour']
    df['day'] = df['trans_day']
    df['month'] = df['trans_month']
    df['year'] = df['trans_year']
    
    # Create additional features
    df['transaction_amount_log'] = np.log1p(df['trans_amount'])
    
    # Check for unusual transaction times (night transactions: 10pm to 6am)
    df['unusual_time'] = ((df['hour'] >= 22) | (df['hour'] <= 6)).astype(int)
    
    return df

def is_valid_upi_id(upi_id):
    """
    Validate UPI ID format
    """
    if pd.isna(upi_id) or upi_id is None:
        return False
    
    upi_id = str(upi_id).strip()
    pattern = r'^[a-zA-Z0-9.\-_]{2,256}@[a-zA-Z]{2,64}$'
    return re.match(pattern, upi_id) is not None