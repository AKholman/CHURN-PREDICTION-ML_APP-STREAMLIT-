# app.py
''' import streamlit as st
import joblib
import pandas as pd

# 1. Load the model from the artifacts folder
@st.cache_resource
def load_model():
    # This must match the path where your .ipynb saves the model
    return joblib.load("artifacts/model_pipeline.joblib")

model = load_model()

st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📡")
st.title("📡 Customer Churn Prediction")
st.markdown("Enter customer details below to estimate the probability of churn.")

# 2. User Input Form
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Billing Info")
        m_charges = st.slider("Monthly Charges ($)", 0.0, 200.0, 65.0)
        t_charges = st.number_input("Total Charges ($)", min_value=0.0, value=0.0, help="If unknown, leave as 0.0")
        tenure = st.slider("Tenure (Days)", 0, 2500, 365)
        senior = st.radio("Senior Citizen", [0, 1], index=0, horizontal=True)

    with col2:
        st.subheader("Service Details")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        paperless = st.radio("Paperless Billing", ["Yes", "No"], horizontal=True)

    # Submit button
    submitted = st.form_submit_button("Analyze Risk")

# 3. Prediction Logic
if submitted:
    # Logic fix: If total charges are 0, use monthly charges (matches notebook cleaning)
    if t_charges == 0:
        t_charges = m_charges

    # Dataframe columns MUST match the exact order and names from your X_train in the notebook
    # We use 'No' or 'Yes' for the binary features to match the OneHotEncoder training data
    input_data = pd.DataFrame([[
        contract, paperless, payment, 'No', 'No', 
        internet, 'No', 'No', 'No', 'No', 'No', 'No', 'No',
        m_charges, t_charges, tenure, senior
    ]], columns=[
        'type', 'paperless_billing', 'payment_method', 'partner', 'dependents', 
        'internet_service', 'online_security', 'online_backup', 'device_protection', 
        'tech_support', 'streaming_tv', 'streaming_movies', 'multiple_lines',
        'monthly_charges', 'total_charges', 'tenure_days', 'senior_citizen'
    ])

    # Get probability from XGBoost
    proba = model.predict_proba(input_data)[:, 1][0]
    
    # Display Result
    st.divider()
    if proba > 0.6:
        st.error(f"### High Risk: {proba:.1%}")
    elif proba > 0.3:
        st.warning(f"### Medium Risk: {proba:.1%}")
    else:
        st.success(f"### Low Risk: {proba:.1%}")
    
    st.progress(int(proba * 100))

'''





    import streamlit as st
import joblib
import pandas as pd

@st.cache_resource
def load_model():
    return joblib.load("artifacts/model_pipeline.joblib")

model = load_model()

st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📡")
st.title("📡 Customer Churn Prediction")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Billing Info")
        m_charges = st.slider("Monthly Charges ($)", 0.0, 200.0, 65.0)
        # Changed from number_input to slider
        t_charges = st.slider("Total Charges ($)", 0.0, 9000.0, 0.0, help="Total amount billed to date")
        tenure = st.slider("Tenure (Days)", 0, 2500, 365)
        senior = st.radio("Senior Citizen", [0, 1], index=0, horizontal=True)

    with col2:
        st.subheader("Service Details")
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        paperless = st.radio("Paperless Billing", ["Yes", "No"], horizontal=True)

    submitted = st.form_submit_button("Analyze Risk")

if submitted:
    # Logic: If user leaves Total Charges at 0, default to Monthly Charges
    # This matches your notebook's logic for new customers
    current_total = t_charges if t_charges > 0 else m_charges

    input_data = pd.DataFrame([[
        contract, paperless, payment, 'No', 'No', 
        internet, 'No', 'No', 'No', 'No', 'No', 'No', 'No',
        m_charges, current_total, tenure, senior
    ]], columns=[
        'type', 'paperless_billing', 'payment_method', 'partner', 'dependents', 
        'internet_service', 'online_security', 'online_backup', 'device_protection', 
        'tech_support', 'streaming_tv', 'streaming_movies', 'multiple_lines',
        'monthly_charges', 'total_charges', 'tenure_days', 'senior_citizen'
    ])

    proba = model.predict_proba(input_data)[:, 1][0]
    
    st.divider()
    if proba > 0.6:
        st.error(f"### High Risk: {proba:.1%}")
    elif proba > 0.3:
        st.warning(f"### Medium Risk: {proba:.1%}")
    else:
        st.success(f"### Low Risk: {proba:.1%}")
    
    st.progress(int(proba * 100))