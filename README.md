# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to churn based on demographic, service, and account-related information.

## Project Overview

Customer churn refers to customers discontinuing a service.

The goal of this project is to build a supervised machine learning classification model that can identify customers who are more likely to churn.

The project includes data preprocessing, exploratory data analysis, feature encoding, feature scaling, model training, evaluation, and a Streamlit web application for making predictions.

## Dataset

The project uses the IBM Telco Customer Churn dataset.

The dataset contains customer demographic information, subscribed services, contract details, payment information, monthly charges, total charges, and churn status.

## Machine Learning Workflow

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature / Target Separation
   ↓
Train-Test Split
   ↓
Categorical Encoding
   ↓
Numerical Feature Scaling
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Streamlit Prediction App



## Models Used  

### Logistic Regression 
Used as the primary classification model.

Performance on the test set:
Accuracy: 80.55%
Precision: 65.72%
Recall: 55.88%
F1-score: 60.40%
ROC-AUC: 84.21%

### Random Forest 
Used as a second classification model for comparison.

Performance on the test set:

Accuracy: 78.35%
Precision: 61.86%
Recall: 48.13%
F1-score: 54.14%

For this train-test split, Logistic Regression produced higher values across the evaluated metrics.

## Preprocessing 
The project uses a Scikit-learn Pipeline containing:
    - StandardScaler for numerical features
    - OneHotEncoder for categorical features
    - Logistic Regression for classification

Using a pipeline ensures that the same preprocessing steps are applied consistently during training and prediction.

## Application 
A Streamlit web application allows users to enter customer information and receive:
    - Churn prediction
    - Estimated churn probability
The application loads the trained pipeline from:
src/churn_pipeline.pkl

## Technologies Used  
    - Python
    - Pandas
    - NumPy
    - Matplotlib
    - Seaborn
    - Scikit-learn
    - Streamlit
    - Jupyter Notebook
    - Joblib

## How to Run 

1. Clone the repository : 
git clone <your-repository-url>

2. Create and activate a virtual environment :
python -m venv venv
Windows: 
venv\Scripts\activate

3. Install dependencies : 
pip install -r requirements.txt

4. Run the Streamlit application :
streamlit run app.py
The application will open in your browser.

## Project Structure 
Customer-Churn-Prediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── notebooks/
│   ├── python_basics.ipynb
│   └── customer_churn.ipynb
│
├── src/
│   └── churn_pipeline.pkl
│
├── app.py
├── README.md
├── requirements.txt
└── .gitignore

## Disclaimer 
The churn probability shown by the application is a model-generated estimate based on patterns learned from the training dataset. It should not be interpreted as a certainty about an individual customer.


