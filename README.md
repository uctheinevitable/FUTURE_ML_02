# Customer Churn Prediction System

A comprehensive machine learning system to predict customer churn using telco customer data.

## 🛠️ Tech Stack

- **Python** - Core programming language
- **Pandas & NumPy** - Data manipulation and analysis
- **Scikit-learn** - Machine learning algorithms
- **XGBoost** - Advanced gradient boosting
- **Matplotlib & Seaborn** - Data visualization
- **Streamlit** - Web application framework
- **Jupyter** - Interactive development environment

## 📁 Project Structure

churn_prediction_system/
├── data/ # Data storage
│ ├── raw/ # Original dataset
│ └── processed/ # Cleaned and processed data
├── notebooks/ # Jupyter notebooks for analysis
├── src/ # Source code modules
│ ├── data_processor.py # Data processing pipeline
│ └── init.py
├── models/ # Trained models and encoders
├── streamlit_app/ # Web application
│ ├── components/ # Reusable components
│ ├── app.py # Main application
│ └── init.py
├── reports/ # Generated reports and visualizations
│ └── figures/ # Chart images
├── config/ # Configuration files
│ └── config.py # Main configuration
├── requirements.txt # Python dependencies
└── README.md # This file

text

## 🚀 Quick Start

### 1. Setup Environment
Clone or download the project
cd churn_prediction_system

Run the setup script
python setup_project.py

text

### 2. Add Your Dataset
Place your dataset in the raw data folder
cp your_telco_dataset.csv data/raw/telco_dataset.csv

text

### 3. Process Data
Run the data processing pipeline
python src/data_processor.py

text

### 4. Launch Web Application
Start the Streamlit app
streamlit run streamlit_app/app.py

text

### 5. Development with Jupyter
Start Jupyter for analysis
jupyter notebook

text

## 📊 Features

### Data Processing
- **Automated Cleaning**: Handle missing values and data types
- **Feature Engineering**: Create derived features for better prediction
- **Encoding**: Convert categorical variables for ML algorithms
- **Scaling**: Standardize numerical features
- **Train/Test Split**: Prepare data for model validation

### Machine Learning
- **Multiple Algorithms**: Logistic Regression, Random Forest, XGBoost
- **Model Comparison**: Performance metrics and evaluation
- **Feature Importance**: Understand churn drivers
- **Hyperparameter Tuning**: Optimize model performance

### Web Dashboard
- **Interactive Interface**: User-friendly Streamlit application
- **Real-time Prediction**: Individual customer churn probability
- **Data Visualization**: Charts and graphs using Matplotlib
- **Business Insights**: Actionable recommendations

### Analysis & Reporting
- **Exploratory Data Analysis**: Comprehensive data insights
- **Model Performance**: Detailed evaluation metrics
- **Business Impact**: ROI and cost-benefit analysis
- **Visualization**: Professional charts and reports

## 🎯 Business Impact

### Early Detection
- Identify customers at risk of churning before they leave
- Proactive retention strategies instead of reactive measures
- Higher success rates in customer retention

### Cost Optimization
- Reduce customer acquisition costs (5-25x more expensive than retention)
- Focus retention efforts on high-value customers
- Optimize marketing spend and resource allocation

### Revenue Protection
- Prevent revenue loss from churning customers
- Increase customer lifetime value
- Improve overall business profitability

### Data-Driven Decisions
- Understand key churn drivers in your business
- Make informed decisions about service improvements
- Track retention campaign effectiveness

## 📈 Model Performance

The system includes three main algorithms:

1. **Logistic Regression**
   - Baseline interpretable model
   - Fast training and prediction
   - Good for understanding linear relationships

2. **Random Forest**
   - Ensemble method for improved accuracy
   - Handles non-linear relationships
   - Provides feature importance rankings

3. **XGBoost**
   - State-of-the-art gradient boosting
   - Highest accuracy potential
   - Advanced feature handling

## 🔧 Configuration

All settings are centralized in `config/config.py`:
- File paths and directories
- Model hyperparameters
- Feature definitions
- Visualization settings

## 📱 Streamlit Dashboard

The web application includes:

### 🏠 Dashboard
- Customer overview metrics
- Churn distribution visualization
- Key performance indicators
- Business summary statistics

### 📈 Data Analysis
- Contract type impact analysis
- Payment method churn rates
- Tenure distribution patterns
- Service usage insights

### 🔮 Prediction
- Individual customer churn probability
- Risk assessment (High/Medium/Low)
- Actionable retention recommendations
- Feature contribution analysis

### 📋 About
- System documentation
- Technical specifications
- Business value proposition
- Usage instructions

## 🛠️ Development

### Adding New Features
1. Update configuration in `config/config.py`
2. Modify data processing in `src/data_processor.py`
3. Add new analysis in Jupyter notebooks
4. Update Streamlit app components

### Model Training
1. Process data using the pipeline
2. Create training notebooks
3. Experiment with different algorithms
4. Save best models for production

### Deployment
1. Test locally with Streamlit
2. Containerize with Docker (optional)
3. Deploy to cloud platform
4. Set up monitoring and updates

## 📋 Requirements

- Python 3.7+
- 8GB+ RAM recommended
- 1GB+ disk space
- Internet connection for package installation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the documentation
2. Review configuration settings
3. Verify data file format
4. Check package versions

## 📄 License

This project is open source and available under the MIT License.

---

**Built with ❤️ for better customer retention**
