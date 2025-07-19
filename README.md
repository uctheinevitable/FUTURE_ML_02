# Customer Churn Prediction System

A comprehensive machine learning system to predict customer churn using advanced analytics and professional business intelligence dashboards.

## 🎯 Project Overview

This enterprise-grade Customer Churn Prediction System leverages advanced machine learning algorithms to identify customers at risk of churning before they leave. The system provides real-time predictions, comprehensive analytics, and actionable insights through a professional web interface designed for executive decision-making.

### Key Features

- **🤖 Advanced ML Models**: Multiple algorithms including Logistic Regression, Random Forest, and XGBoost
- **📊 Executive Dashboard**: C-level appropriate metrics and business intelligence
- **🔮 Real-time Predictions**: Individual customer risk assessment with actionable recommendations
- **📈 Comprehensive Analytics**: Deep-dive analysis of churn patterns and customer behavior
- **💼 Professional UI**: Modern, clean interface with cohesive color palette and responsive design

## 🛠️ Technology Stack

- **Python** - Core programming language and data processing
- **Streamlit** - Professional web application framework
- **Pandas \& NumPy** - Data manipulation and analysis
- **Scikit-learn** - Machine learning algorithms and evaluation
- **XGBoost** - Advanced gradient boosting for optimal performance
- **Matplotlib \& Seaborn** - Professional data visualization
- **Joblib** - Model serialization and persistence

## 📁 Project Structure

```
churn_prediction_system/
├── data/
│   ├── raw/
│   │   └── telco_dataset.csv              # Original customer dataset
│   └── processed/
│       ├── cleaned_data.csv               # Cleaned and feature-engineered data
│       ├── train_data.csv                 # Training dataset
│       └── test_data.csv                  # Test dataset for evaluation
├── notebooks/
│   ├── 01_data_exploration.ipynb          # Exploratory data analysis
│   ├── 02_model_training.ipynb            # Basic model training
│   └── 03_xgboost_training.ipynb          # Advanced XGBoost optimization
├── src/
│   ├── data_processor.py                  # Complete ETL pipeline
│   └── __init__.py
├── models/
│   ├── logistic_regression.pkl            # Trained Logistic Regression model
│   ├── random_forest.pkl                  # Trained Random Forest model
│   ├── xgboost.pkl                        # Optimized XGBoost model
│   ├── encoders.pkl                       # Categorical variable encoders
│   ├── scaler.pkl                         # Feature scaling transformations
│   ├── feature_names.pkl                  # Model feature specifications
│   ├── data_summary.pkl                   # Dataset statistics for dashboard
│   └── model_summary.pkl                  # Performance metrics and comparisons
├── streamlit_app/
│   └── app.py                             # Professional web application
├── reports/
│   └── figures/
│       └── model_comparison.png           # Performance visualization
├── config/
│   └── config.py                          # System configuration
├── requirements.txt                       # Python dependencies
├── .gitignore                            # Git ignore rules
└── README.md                             # This documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- 8GB+ RAM recommended
- Modern web browser for dashboard access

### Installation

1. **Clone the repository**

```bash
git clone <repository-url>
cd churn_prediction_system
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Add your dataset**

```bash
# Place your telco dataset in the raw data folder
cp your_telco_dataset.csv data/raw/telco_dataset.csv
```

4. **Process the data**

```bash
python src/data_processor.py
```

5. **Launch the application**

```bash
streamlit run streamlit_app/app.py
```

6. **Access the dashboard**
   - Open your browser to `http://localhost:8501`
   - Explore the professional business intelligence interface

## 📊 Dashboard Screenshots

### Executive Summary Dashboard

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/742c2a3f-50f7-483d-8406-461afea849d9" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/f12f71e6-0564-4d8b-8ae9-99efd30a0068" />

### Customer Churn Prediction Interface

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/8815ad77-88d3-45cd-a106-5655ce062b8e" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/2c61cb02-8c6c-4efc-82ca-59ac8b1aa92c" />

### Data Analytics Dashboard

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/cfaff0d6-c395-471b-abc3-59f1a018561e" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/58b824af-6d9b-4835-9dc5-c8915320ffe7" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/b902c70d-45f0-432c-aad5-4e41087b286a" />

### Model Performance Comparison

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/e74f9951-9126-45ef-9f9f-d3f2b261f030" />

## 🎯 Business Value

### Financial Impact

- **Revenue Protection**: Early identification of at-risk customers prevents revenue loss
- **Cost Optimization**: Customer retention is 5-25x cheaper than acquisition
- **ROI Maximization**: Focus retention efforts on highest-value customers
- **Strategic Planning**: Data-driven insights for long-term customer success

### Operational Benefits

- **Proactive Management**: Act before customers churn, not after
- **Resource Efficiency**: Target interventions based on risk scores
- **Performance Monitoring**: Track retention campaign effectiveness
- **Scalable Solution**: Handle growing customer bases with automated insights

## 🤖 Machine Learning Pipeline

### Data Processing

1. **Data Cleaning**: Handle missing values, fix data types, remove duplicates
2. **Feature Engineering**: Create derived features like tenure groups and average charges
3. **Encoding**: Convert categorical variables to numeric format
4. **Scaling**: Standardize numerical features for model compatibility
5. **Splitting**: Stratified train/test split maintaining class distribution

### Model Training

The system implements multiple algorithms for robust predictions:

- **Logistic Regression**: Baseline interpretable model for understanding linear relationships
- **Random Forest**: Ensemble method providing feature importance and handling non-linear patterns
- **XGBoost**: Advanced gradient boosting with hyperparameter optimization for maximum accuracy

### Performance Metrics

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :------------------ | :------- | :-------- | :----- | :------- | :------ |
| Logistic Regression | 85.2%    | 84.7%     | 83.9%  | 84.3%    | 0.847   |
| Random Forest       | 87.1%    | 86.3%     | 85.8%  | 86.0%    | 0.869   |
| XGBoost             | 89.3%    | 88.1%     | 87.6%  | 87.8%    | 0.891   |

## 📈 Key Features Deep Dive

### Executive Dashboard

- **KPI Overview**: Total customers, churn count, churn rate, and average revenue per user
- **Visual Analytics**: Professional charts showing customer distribution and revenue impact
- **Business Insights**: Revenue at risk, customer intelligence, and retention opportunities
- **Strategic Recommendations**: Data-driven action items based on current performance

### Churn Prediction Engine

- **Individual Assessment**: Comprehensive customer risk evaluation
- **Risk Categorization**: High/Medium/Low risk classification with specific thresholds
- **Financial Impact**: Customer lifetime value and retention cost calculations
- **Actionable Recommendations**: Specific intervention strategies for each risk level

### Analytics Platform

- **Contract Analysis**: Impact of different contract types on churn behavior
- **Payment Method Intelligence**: Risk assessment by payment preferences
- **Customer Lifecycle**: Tenure-based churn pattern analysis

### Model Performance Center

- **Multi-Model Comparison**: Side-by-side evaluation of all trained algorithms
- **ROC Curve Analysis**: Visual performance comparison across models
- **Feature Importance**: Understanding key churn drivers
- **Model Recommendations**: Best practices for production deployment

## 👤 Author

**Developed by:**

- Ujjwal Chaurasia
- [LinkedIn](www.linkedin.com/in/ujjwal-chaurasia)
- [GitHub](https://github.com/uctheinevitable)
