"""
Configuration file for Churn Prediction System
"""

# File paths
RAW_DATA_PATH = "data/raw/telco_dataset.csv"
PROCESSED_DATA_PATH = "data/processed/"
MODELS_PATH = "models/"
REPORTS_PATH = "reports/"

# Model parameters
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Streamlit configuration
APP_TITLE = "Customer Churn Prediction System"
APP_ICON = "🚨"

# Model configurations
MODELS_CONFIG = {
    "logistic_regression": {
        "random_state": RANDOM_SEED,
        "max_iter": 1000
    },
    "random_forest": {
        "n_estimators": 100,
        "random_state": RANDOM_SEED,
        "n_jobs": -1
    },
    "xgboost": {
        "random_state": RANDOM_SEED,
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6
    }
}

# Feature categories for analysis
CATEGORICAL_FEATURES = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents',
    'PhoneService', 'MultipleLines', 'InternetService',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaperlessBilling', 'PaymentMethod'
]

NUMERICAL_FEATURES = [
    'tenure', 'MonthlyCharges', 'TotalCharges'
]

# Matplotlib style settings
MATPLOTLIB_STYLE = {
    'figure.figsize': (12, 8),
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10
}
