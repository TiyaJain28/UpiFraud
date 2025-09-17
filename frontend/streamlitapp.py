import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import traceback
import sys
import os

# Add the backend directory to the path so we can import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

# Import utility functions
from utils import is_valid_upi_id

# Set page config
st.set_page_config(
    page_title="UPI Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .fraud-alert {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff4b4b;
    }
    .safe-alert {
        background-color: #ccffcc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4caf50;
    }
    .risk-high {
        color: #ff4b4b;
        font-weight: bold;
    }
    .risk-medium {
        color: #ffa500;
        font-weight: bold;
    }
    .risk-low {
        color: #4caf50;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load models (cached to avoid reloading on every interaction)
@st.cache_resource
def load_models():
    """Load the trained models and preprocessing objects"""
    try:
        model = joblib.load('../models/best_model.joblib')
        feature_names = joblib.load('../models/feature_names.joblib')
        scaler = joblib.load('../models/scaler.joblib')
        print("Models loaded successfully")
        return model, feature_names, scaler
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        print(traceback.format_exc())
        return None, None, None

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

def predict_single_transaction(_model, _feature_names, _scaler, transaction_data):
    """Predict fraud for a single transaction"""
    try:
        # Convert to DataFrame
        transaction_df = pd.DataFrame([transaction_data])
        
        # Preprocess the transaction data
        processed_df = preprocess_data(transaction_df, is_training=False)
        
        # Ensure feature consistency with what the model expects
        X = ensure_feature_consistency(processed_df, _feature_names)
        
        # Scale the data
        X_scaled = _scaler.transform(X)
        
        # Make prediction
        prediction = _model.predict(X_scaled)
        probability = _model.predict_proba(X_scaled)
        
        fraud_probability = float(probability[0][1] * 100)  # Probability of fraud
        
        # Prepare response
        response = {
            "transaction_id": transaction_data.get('transaction_id', 'N/A'),
            "prediction": "Fraud" if prediction[0] == 1 else "Safe",
            "fraud_probability": fraud_probability,
            "risk_level": "High" if fraud_probability > 70 else ("Medium" if fraud_probability > 30 else "Low"),
            "timestamp": datetime.now().isoformat()
        }
        
        return response, None
    
    except Exception as e:
        error_msg = f"Error in prediction: {str(e)}"
        print(traceback.format_exc())
        return None, error_msg

def predict_batch_transactions(_model, _feature_names, _scaler, transactions):
    """Predict fraud for multiple transactions"""
    try:
        # Convert to DataFrame
        transactions_df = pd.DataFrame(transactions)
        
        # Preprocess the transactions data
        processed_df = preprocess_data(transactions_df, is_training=False)
        
        # Ensure feature consistency with what the model expects
        X = ensure_feature_consistency(processed_df, _feature_names)
        
        # Scale the data
        X_scaled = _scaler.transform(X)
        
        # Make predictions
        predictions = _model.predict(X_scaled)
        probabilities = _model.predict_proba(X_scaled)
        
        # Prepare response
        results = []
        for i, (prediction, probability) in enumerate(zip(predictions, probabilities)):
            fraud_probability = float(probability[1] * 100)
            
            results.append({
                "transaction_id": transactions[i].get('transaction_id', f'tx_{i}'),
                "prediction": "Fraud" if prediction == 1 else "Safe",
                "fraud_probability": fraud_probability,
                "risk_level": "High" if fraud_probability > 70 else ("Medium" if fraud_probability > 30 else "Low")
            })
        
        response = {
            "count": len(results),
            "fraud_count": sum(1 for r in results if r['prediction'] == 'Fraud'),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        return response, None
    
    except Exception as e:
        error_msg = f"Error in batch prediction: {str(e)}"
        print(traceback.format_exc())
        return None, error_msg

def main():
    st.markdown('<h1 class="main-header">🛡️ UPI Fraud Detection System</h1>', unsafe_allow_html=True)
    
    # Load models
    model, feature_names, scaler = load_models()
    if model is None or feature_names is None or scaler is None:
        st.error("❌ Failed to load machine learning models. Please make sure:")
        st.info("1. You have trained the models using trainmodel.py")
        st.info("2. The models/ directory contains best_model.joblib, feature_names.joblib, and scaler.joblib")
        return
    
    st.success("✅ Machine Learning Models Loaded Successfully")
    
    # Display feature information for debugging
    with st.expander("Model Information (Debug)"):
        st.write(f"Model type: {type(model)}")
        st.write(f"Number of features: {len(feature_names)}")
        st.write(f"Features: {feature_names}")
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Single Transaction", "Batch Upload"])
    
    with tab1:
        st.header("Single Transaction Analysis")
        
        with st.form("single_transaction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Transaction Details")
                # Use date input instead of individual sliders
                transaction_date = st.date_input(
                    "Transaction Date",
                    value=datetime(2023, 6, 15),
                    min_value=datetime(2020, 1, 1),
                    max_value=datetime(2025, 12, 31)
                )
                transaction_time = st.time_input("Transaction Time", value=datetime.strptime("12:00", "%H:%M").time())
                
                # Extract components from date and time
                trans_year = transaction_date.year
                trans_month = transaction_date.month
                trans_day = transaction_date.day
                trans_hour = transaction_time.hour
                
                category = st.selectbox(
                    "Category",
                    ["Shopping", "Food", "Transport", "Entertainment", "Utilities", "Other"]
                )
                
            with col2:
                st.subheader("Payment Information")
                trans_amount = st.number_input("Amount", min_value=0.0, value=1000.0, step=100.0)
                upi_number = st.text_input("UPI Number", "example@upi")
                state = st.text_input("State", "Maharashtra")
                zip_code = st.number_input("ZIP Code", min_value=0, value=400001, step=1)
            
            submitted = st.form_submit_button("Check for Fraud")
            
            if submitted:
                # Validate UPI ID
                if not is_valid_upi_id(upi_number):
                    st.error("❌ Invalid UPI ID format. Please use format: username@bank")
                    return
                
                transaction_data = {
                    "trans_hour": trans_hour,
                    "trans_day": trans_day,
                    "trans_month": trans_month,
                    "trans_year": trans_year,
                    "category": category,
                    "upi_number": upi_number,
                    "trans_amount": trans_amount,
                    "state": state,
                    "zip": zip_code,
                    "transaction_id": f"tx_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                }
                
                result, error = predict_single_transaction(model, feature_names, scaler, transaction_data)
                
                if error:
                    st.error(f"❌ {error}")
                else:
                    if result["prediction"] == "Fraud":
                        st.markdown(f"""
                        <div class="fraud-alert">
                            <h3>🚨 FRAUD DETECTED!</h3>
                            <p>Probability: {result['fraud_probability']:.2f}%</p>
                            <p>Risk Level: <span class="risk-high">{result['risk_level']}</span></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="safe-alert">
                            <h3>✅ TRANSACTION SAFE</h3>
                            <p>Fraud Probability: {result['fraud_probability']:.2f}%</p>
                            <p>Risk Level: <span class="risk-low">{result['risk_level']}</span></p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show detailed results
                    with st.expander("Detailed Analysis"):
                        st.json(result)
    
    with tab2:
        st.header("Batch Transaction Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload CSV file with transactions",
            type=["csv"],
            help="CSV should contain columns: trans_hour, trans_day, trans_month, trans_year, category, upi_number, trans_amount, state, zip"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write("Preview of uploaded data:")
                st.dataframe(df.head())
                
                # Validate required columns
                required_columns = ['trans_hour', 'trans_day', 'trans_month', 'trans_year', 
                                   'category', 'upi_number', 'trans_amount', 'state']
                
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                else:
                    if st.button("Analyze Batch"):
                        # Validate UPI IDs
                        invalid_upi_ids = []
                        for i, upi_id in enumerate(df['upi_number']):
                            if not is_valid_upi_id(str(upi_id)):
                                invalid_upi_ids.append((i, upi_id))
                        
                        if invalid_upi_ids:
                            st.error("❌ Invalid UPI IDs found:")
                            for i, upi_id in invalid_upi_ids[:5]:  # Show first 5 errors
                                st.write(f"Row {i}: {upi_id}")
                            if len(invalid_upi_ids) > 5:
                                st.write(f"... and {len(invalid_upi_ids) - 5} more")
                            return
                        
                        # Convert DataFrame to list of transactions
                        transactions = []
                        for _, row in df.iterrows():
                            transaction = {
                                "trans_hour": int(row['trans_hour']),
                                "trans_day": int(row['trans_day']),
                                "trans_month": int(row['trans_month']),
                                "trans_year": int(row['trans_year']),
                                "category": str(row['category']),
                                "upi_number": str(row['upi_number']),
                                "trans_amount": float(row['trans_amount']),
                                "state": str(row['state']),
                                "zip": int(row.get('zip', 0))
                            }
                            transactions.append(transaction)
                        
                        with st.spinner("Analyzing transactions..."):
                            result, error = predict_batch_transactions(model, feature_names, scaler, transactions)
                        
                        if error:
                            st.error(f"❌ {error}")
                        else:
                            st.success(f"Analysis complete! Processed {result['count']} transactions.")
                            
                            # Display summary
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Total Transactions", result['count'])
                            col2.metric("Fraudulent Transactions", result['fraud_count'])
                            fraud_rate = (result['fraud_count'] / result['count'] * 100) if result['count'] > 0 else 0
                            col3.metric("Fraud Rate", f"{fraud_rate:.1f}%")
                            
                            # Display results in a table
                            results_df = pd.DataFrame(result['results'])
                            st.dataframe(results_df)
                            
                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name="fraud_analysis_results.csv",
                                mime="text/csv"
                            )
                            
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")

if __name__ == "__main__":
    main()