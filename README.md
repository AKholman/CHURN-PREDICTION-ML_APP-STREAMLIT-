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


| Model                   | Dataset      | Accuracy | ROC-AUC  | F1 Score |
|-------------------------|--------------|----------|----------|----------|
| LogisticRegression      |  validation  |  0.7580  |  0.8508  |  0.6368  |
| RandomForestClassifier  |  validation  |  0.8133  |  0.8751  |  0.6741  |
| LightGBMClassifier      |  validation  |  0.8297  |  0.9078  |  0.7066  |
| XGBClassifier           |  validation  |  0.8623  |  0.9118  |  0.7069  |
| CatboostClassifier      |  validation  |  0.8602  |  0.9106  |  0.7020  |
| XGBClassifier           |     test     |  0.8581  |  0.8953  |  0.7059  |

**Best Model:** XGBoost with hyperparameter tuning (ROC-AUC: 0.8953).

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


**Live App URL:** [https://churn-prediction-with-deployment-on.onrender.com]
⚠️ This app is deployed on Render free tier. It may be paused to save resources. Please contact me if you’d like me to reactivate it.

