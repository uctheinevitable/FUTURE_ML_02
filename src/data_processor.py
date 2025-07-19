"""
Core data processing module for churn prediction system
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import sys

# Add config to path
sys.path.append("config")
try:
    from config import *
except ImportError:
    # Fallback configuration if config file is not found
    RAW_DATA_PATH = "../data/raw/telco_dataset.csv"
    PROCESSED_DATA_PATH = "../data/processed/"
    MODELS_PATH = "../models/"
    RANDOM_SEED = 42
    TEST_SIZE = 0.2


class ChurnDataProcessor:
    """Main class for data processing operations"""

    def __init__(self):
        self.df = None
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []

    def load_data(self, filepath=RAW_DATA_PATH):
        """Load the raw dataset"""
        try:
            self.df = pd.read_csv(filepath)
            print(f"📊 Data loaded successfully: {self.df.shape}")
            return self.df
        except FileNotFoundError:
            print(f"❌ Data file not found: {filepath}")
            print("   Please place your telco_dataset.csv in data/raw/ folder")
            return None

    def get_data_info(self):
        """Display basic information about the dataset"""
        if self.df is None:
            print("❌ No data loaded")
            return None

        print("\n📋 Dataset Information:")
        print(f"   Shape: {self.df.shape}")
        print(f"   Columns: {len(self.df.columns)}")
        print(f"   Missing values: {self.df.isnull().sum().sum()}")

        # Churn distribution
        if "Churn" in self.df.columns:
            churn_counts = self.df["Churn"].value_counts()
            churn_rate = churn_counts.get("Yes", 0) / len(self.df) * 100
            print(f"   Churn rate: {churn_rate:.1f}%")
            return {
                "shape": self.df.shape,
                "columns": len(self.df.columns),
                "missing_values": self.df.isnull().sum().sum(),
                "churn_rate": churn_rate,
            }

        return {
            "shape": self.df.shape,
            "columns": len(self.df.columns),
            "missing_values": self.df.isnull().sum().sum(),
            "churn_rate": 0,
        }

    def clean_data(self):
        """Clean the dataset"""
        if self.df is None:
            print("❌ No data to clean")
            return None

        print("🧹 Cleaning data...")

        # Fix TotalCharges column (convert from string to numeric)
        self.df["TotalCharges"] = pd.to_numeric(
            self.df["TotalCharges"], errors="coerce"
        )
        missing_total = self.df["TotalCharges"].isnull().sum()
        if missing_total > 0:
            print(f"   Found {missing_total} missing TotalCharges values")
            self.df["TotalCharges"].fillna(0, inplace=True)

        # Convert SeniorCitizen to string for consistency
        self.df["SeniorCitizen"] = self.df["SeniorCitizen"].map({0: "No", 1: "Yes"})

        # Create derived features
        self.df["AvgMonthlyCharges"] = self.df["TotalCharges"] / (self.df["tenure"] + 1)

        # Handle division by zero
        self.df["AvgMonthlyCharges"] = self.df["AvgMonthlyCharges"].replace(
            [np.inf, -np.inf], 0
        )

        # Create tenure groups for analysis
        self.df["TenureGroup"] = pd.cut(
            self.df["tenure"],
            bins=[0, 12, 24, 48, 100],
            labels=["0-1 year", "1-2 years", "2-4 years", "4+ years"],
        )

        print("   ✅ Data cleaning completed")
        return self.df

    def prepare_for_ml(self):
        """Prepare data for machine learning"""
        if self.df is None:
            print("❌ No data to prepare")
            return None, None

        print("🔧 Preparing data for ML...")

        # Remove non-ML columns
        ml_df = self.df.drop(["customerID", "TenureGroup"], axis=1, errors="ignore")

        # Separate features and target
        X = ml_df.drop("Churn", axis=1)
        y = ml_df["Churn"]

        # Store feature names
        self.feature_names = list(X.columns)

        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=["object"]).columns
        print(f"   Encoding {len(categorical_cols)} categorical features...")

        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.encoders[col] = le

        # Encode target
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y)
        self.encoders["target"] = target_encoder

        print(f"   ✅ Data prepared for ML: {X.shape}")
        return X, y_encoded

    def split_and_scale(self, X, y):
        """Split data and apply scaling"""

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Convert back to DataFrames
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=self.feature_names)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=self.feature_names)

        print(
            f"   ✅ Data split and scaled: Train {X_train_scaled.shape}, Test {X_test_scaled.shape}"
        )
        return X_train_scaled, X_test_scaled, y_train, y_test

    def save_processed_data(self, X_train, X_test, y_train, y_test):
        """Save all processed data and objects"""

        # Ensure directories exist
        os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
        os.makedirs(MODELS_PATH, exist_ok=True)

        # Save training and test data
        train_data = X_train.copy()
        train_data["Churn"] = y_train
        train_data.to_csv(f"{PROCESSED_DATA_PATH}train_data.csv", index=False)

        test_data = X_test.copy()
        test_data["Churn"] = y_test
        test_data.to_csv(f"{PROCESSED_DATA_PATH}test_data.csv", index=False)

        # Save cleaned original data
        self.df.to_csv(f"{PROCESSED_DATA_PATH}cleaned_data.csv", index=False)

        # Save encoders and scaler
        joblib.dump(self.encoders, f"{MODELS_PATH}encoders.pkl")
        joblib.dump(self.scaler, f"{MODELS_PATH}scaler.pkl")
        joblib.dump(self.feature_names, f"{MODELS_PATH}feature_names.pkl")

        # Create data summary for dashboard
        data_summary = {
            "total_customers": len(self.df),
            "churned_customers": len(self.df[self.df["Churn"] == "Yes"]),
            "churn_rate": self.df["Churn"].value_counts(normalize=True).get("Yes", 0),
            "avg_monthly_charges": self.df["MonthlyCharges"].mean(),
            "avg_tenure": self.df["tenure"].mean(),
            "contract_types": self.df["Contract"].unique().tolist(),
            "payment_methods": self.df["PaymentMethod"].unique().tolist(),
            "internet_services": self.df["InternetService"].unique().tolist(),
        }
        joblib.dump(data_summary, f"{MODELS_PATH}data_summary.pkl")

        print("💾 All processed data saved successfully!")

    def process_complete_pipeline(self):
        """Run the complete data processing pipeline"""

        print("🔄 Starting complete data processing pipeline...")

        # Load data
        if self.load_data() is None:
            return False

        # Get info
        info = self.get_data_info()
        if info is None:
            return False

        # Clean data
        if self.clean_data() is None:
            return False

        # Prepare for ML
        X, y = self.prepare_for_ml()
        if X is None:
            return False

        # Split and scale
        X_train, X_test, y_train, y_test = self.split_and_scale(X, y)

        # Save everything
        self.save_processed_data(X_train, X_test, y_train, y_test)

        print("\n🎉 Data processing pipeline completed successfully!")
        return True


if __name__ == "__main__":
    processor = ChurnDataProcessor()
    success = processor.process_complete_pipeline()

    if success:
        print("\n✅ Ready for model training and Streamlit app!")
    else:
        print("\n❌ Pipeline failed. Please check your data file.")
