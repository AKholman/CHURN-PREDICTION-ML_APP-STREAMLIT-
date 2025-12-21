# app.py

import streamlit as st
import joblib
import pandas as pd

# Load model (keep the cache to ensure speed)
@st.cache_resource
def load_model():
    return joblib.load("artifacts/model_pipeline.joblib")

model = load_model()

st.title("Churn Prediction")

with st.form("main_form"):
    # UI Layout
    c1, c2 = st.columns(2)
    m_charges = c1.slider("Monthly Charges", 0.0, 200.0, 50.0)
    t_charges = c1.slider("Total Charges", 0.0, 5000.0, 600.0)
    tenure = c1.slider("Tenure Days", 0, 5000, 365)
    senior = c1.radio("Senior Citizen", [0, 1])
    
    contract = c2.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = c2.selectbox("Internet", ["DSL", "Fiber optic", "No"])
    # ... add other inputs if needed ...

    if st.form_submit_button("Predict"):
        # Create DF directly (Ensure column order matches your X_train)
        df = pd.DataFrame([[contract, 'Yes', 'Electronic check', 'No', 'No', 
                            internet, 'No', 'No', 'No', 'No', 'No', 'No', 'No',
                            m_charges, t_charges, tenure, senior]], 
                          columns=['type', 'paperless_billing', 'payment_method', 'partner', 'dependents', 
                                   'internet_service', 'online_security', 'online_backup', 'device_protection', 
                                   'tech_support', 'streaming_tv', 'streaming_movies', 'multiple_lines',
                                   'monthly_charges', 'total_charges', 'tenure_days', 'senior_citizen'])
        
        proba = model.predict_proba(df)[:, 1][0]
        st.metric("Churn Probability", f"{proba:.2%}")

