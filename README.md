# Customer Churn Prediction for Telecom Operator Interconnect

## Introduction

In this project, we aim to predict customer churn for the telecom operator Interconnect using machine learning. The company wants to forecast which users are planning to leave and offer them promotional codes and special plan options. 

We start by preprocessing and integrating four datasets, preparing them for exploratory data analysis (EDA). Through feature engineering, we define the key input features and the target variable for a supervised binary classification task.
---

## Data Preprocessing & Feature Engineering

- Combined 4 raw datasets (contract, internet, personal, phone) using `customer_id`.  
- Handled missing values and categorical variables.  
- One-Hot Encoding (OHE) applied to categorical features.  
- StandardScaler applied to numerical features.  

---

To build the churn prediction model, we trained and evaluated several classification algorithms:

- Logistic Regression  
- Random Forest Classifier  
- LightGBM  
- XGBoost  
- CatBoost  

Model performance was evaluated using key metrics, including accuracy, ROC-AUC, and F1 score. Hyperparameter tuning aimed to achieve a ROC-AUC above 0.75. The best-performing classifier (XGBoost) was selected to reliably predict customer churn for Interconnect.

Finally, the selected model was used to build a web application using Streamlit UI and deployoed on Render, following principles of machine learning system design.

---

## Model Training and Evaluation

Models were trained using cross-validation and hyperparameter tuning (GridSearchCV). Performance metrics:

| Model                 | Dataset    | Accuracy | ROC-AUC | F1 Score |
|-----------------------|------------|---------|---------|----------|
| LogisticRegression    | Validation | 0.7360  | 0.8204  | 0.5903   |
| RandomForestClassifier| Validation | 0.7821  | 0.8482  | 0.6119   |
| LightGBMClassifier    | Validation | 0.8034  | 0.8797  | 0.6593   |
| XGBClassifier         | Validation | 0.8474  | 0.8724  | 0.6646   |
| CatboostClassifier    | Validation | 0.8417  | 0.8663  | 0.6443   |
| XGBClassifier         | Test       | 0.8488  | 0.8829  | 0.6816   |

**Best Model:** XGBoost with hyperparameter tuning (ROC-AUC: 0.8829).

---

## Deployment

The XGBoost model is deployed as a **Streamlit web application** on Render. Users can input customer features via an interactive form and receive churn probability in real-time.

## Folder Structure

```bash
churn-prediction-app/
│
├── app.py                      # Streamlit app for deployment
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── notebooks/                  
│   └── churn_modeling.ipynb
│
├── data/                       
│   ├── contract.csv
│   ├── internet.csv
│   ├── personal.csv
│   └── phone.csv
│
├── artifacts/                  
│   ├── model_pipeline.joblib   
│   └── model_metadata.json         
│
└── .gitignore                  
```


![App Screenshot](images/app_dashboard_1.png)
![App Screenshot](images/app_dashboard_2.png)

**Live App URL:** [https://churn-prediction-with-deployment-on.onrender.com]
⚠️ This app is deployed on Render free tier. It may be paused to save resources. Please contact me if you’d like me to reactivate it.





'''
'''import streamlit as st
import joblib
import pandas as pd
import numpy as np
import json

@st.cache_resource
def load_model():
    pipeline = joblib.load("artifacts/model_pipeline.joblib")
    with open("artifacts/model_metadata.json") as f:
        meta = json.load(f)
    return pipeline, meta

pipeline, meta = load_model()
expected_features = meta['features']

st.title("💡 Churn Prediction Dashboard")

# ---- Tabs ----
tab1, tab2, tab3 = st.tabs(["Input", "Prediction", "Feature Info"])

with tab1:
    st.header("Customer Information")
    # Use columns for layout
    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        with col1:
            monthly_charges = st.slider("Monthly Charges", 0.0, 200.0, 50.0)
            total_charges = st.slider("Total Charges", 0.0, 5000.0, 600.0)
            tenure_days = st.slider("Tenure Days", 0, 5000, 365)
            senior_citizen = st.radio("Senior Citizen", ["No", "Yes"])

            type_input = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.radio("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

        with col2:
            partner = st.radio("Partner", ["Yes", "No"])
            dependents = st.radio("Dependents", ["Yes", "No"])
            online_security = st.radio("Online Security", ["Yes", "No"])
            online_backup = st.radio("Online Backup", ["Yes", "No"])
            device_protection = st.radio("Device Protection", ["Yes", "No"])
            tech_support = st.radio("Tech Support", ["Yes", "No"])
            streaming_tv = st.radio("Streaming TV", ["Yes", "No"])
            streaming_movies = st.radio("Streaming Movies", ["Yes", "No"])
            multiple_lines = st.radio("Multiple Lines", ["Yes", "No"])

        submitted = st.form_submit_button("Predict")

with tab2:
    if submitted:
        input_dict = {
            "type": type_input,
            "paperless_billing": paperless_billing,
            "payment_method": payment_method,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "tenure_days": tenure_days,
            "senior_citizen": 1 if senior_citizen=="Yes" else 0,
            "partner": partner,
            "dependents": dependents,
            "internet_service": internet_service,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "multiple_lines": multiple_lines
        }

        df = pd.DataFrame([input_dict])

        # Defaults
        defaults = {
            "type": "Month-to-month",
            "paperless_billing": "Yes",
            "payment_method": "Electronic check",
            "partner": "No",
            "dependents": "No",
            "internet_service": "DSL",
            "online_security": "No",
            "online_backup": "No",
            "device_protection": "No",
            "tech_support": "No",
            "streaming_tv": "No",
            "streaming_movies": "No",
            "multiple_lines": "No",
            "monthly_charges": 0.0,
            "total_charges": 0.0,
            "tenure_days": 0,
            "senior_citizen": 0
        }

        for col in expected_features:
            if col not in df.columns:
                df[col] = defaults[col]

        df = df[expected_features]

        categorical_cols = [
            "type", "paperless_billing", "payment_method", "partner", "dependents",
            "internet_service", "online_security", "online_backup",
            "device_protection", "tech_support", "streaming_tv",
            "streaming_movies", "multiple_lines"
        ]
        df[categorical_cols] = df[categorical_cols].astype(str)

        proba = pipeline.predict_proba(df)[:, 1][0]

        # Color-coded metric
        if proba >= 0.6:
            st.error(f"Churn Probability: {proba:.2%} ⚠️")
        elif proba >= 0.3:
            st.warning(f"Churn Probability: {proba:.2%} ⚠️")
        else:
            st.success(f"Churn Probability: {proba:.2%} ✅")

        st.progress(int(proba*100))

        st.write("Predicted churn:", "Yes" if proba >= 0.5 else "No")

with tab3:
    st.header("Feature Information")
    st.write("The app uses the following features for prediction:")
    st.write(expected_features)

    '''