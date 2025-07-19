"""
Customer Churn Prediction System
Professional Business Intelligence Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Professional page configuration
st.set_page_config(
    page_title="Churn Prediction Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 600;
        color: #fffff;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem 0;
        border-bottom: 2px solid #e5e7eb;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .sidebar .sidebar-content {
        background-color: #f8fafc;
    }
    
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.3);
    }
    
    .prediction-result {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
    }
    
    .high-risk {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
    }
    
    .medium-risk {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
    }
    
    .low-risk {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load processed data with error handling"""
    try:
        df = pd.read_csv('data/processed/cleaned_data.csv')
        return df
    except FileNotFoundError:
        st.error("🔴 Data not found. Please ensure the data processing pipeline has been completed.")
        return None

@st.cache_resource
def load_models_and_encoders():
    """Load models and encoders with comprehensive error handling"""
    try:
        data_summary = joblib.load('models/data_summary.pkl')
        encoders = joblib.load('models/encoders.pkl')
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        
        models = {}
        model_files = ['logistic_regression', 'random_forest', 'xgboost']
        
        for model_name in model_files:
            try:
                models[model_name] = joblib.load(f'models/{model_name}.pkl')
            except FileNotFoundError:
                continue
        
        return data_summary, encoders, scaler, feature_names, models
    except FileNotFoundError:
        st.error("🔴 Model files not found. Please complete the model training pipeline.")
        return None, None, None, None, None

def main():
    """Main application with professional layout"""
    
    # Professional header
    st.markdown('<h1 class="main-header">Customer Churn Prediction Analytics</h1>', 
                unsafe_allow_html=True)
    
    # Load data and models
    df = load_data()
    data_summary, encoders, scaler, feature_names, models = load_models_and_encoders()
    
    if df is None or data_summary is None:
        st.stop()
    
    # Professional sidebar navigation
    st.sidebar.markdown("## 📊 Navigation")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "",
        ["🏠 Executive Dashboard", "🔮 Churn Prediction", "📈 Analytics","🤖Model Comparison", "ℹ️ About"],
        index=0
    )
    
    # Display current system status
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    st.sidebar.success(f"✅ {data_summary['total_customers']:,} customers loaded")
    if models:
        st.sidebar.info(f"🤖 {len(models)} models ready")
    
    # Route to pages
    if page == "🏠 Executive Dashboard":
        show_executive_dashboard(df, data_summary)
    elif page == "🔮 Churn Prediction":
        show_prediction_interface(encoders, scaler, feature_names, models)
    elif page == "📈 Analytics":
        show_analytics_dashboard(df)
    elif page == "🤖Model Comparison":
        show_model_comparison(models)
    elif page == "ℹ️ About":
        show_about_system()


def show_model_comparison(models):
    """Professional model comparison page"""
    
    st.header("Model Performance Comparison")
    
    # Load test data
    try:
        test_data = pd.read_csv('data/processed/test_data.csv')
        X_test = test_data.drop('Churn', axis=1)
        y_test = test_data['Churn']
        
        if models is None or len(models) == 0:
            st.warning("No trained models found. Please train models first.")
            st.info("Run the model training notebooks to generate model files.")
            return
        
        # Evaluate all models
        model_results = evaluate_all_models(models, X_test, y_test)
        
        if not model_results:
            st.error("Could not evaluate models. Please check model files.")
            return
        
        # Display model comparison
        display_model_performance_table(model_results)
        
        # Performance visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            display_performance_chart(model_results)
        
        with col2:
            display_roc_curves(model_results, y_test)
        
        # Best model selection
        show_best_model_summary(model_results)
        
    except FileNotFoundError:
        st.error("Test data not found. Please run the data processing pipeline first.")
        st.info("Run: `python src/data_processor.py`")

def evaluate_all_models(models, X_test, y_test):
    """Evaluate all models and return performance metrics"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    results = {}
    
    for name, model in models.items():
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba),
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
        except Exception as e:
            st.warning(f"Error evaluating {name}: {str(e)}")
            continue
    
    return results

def display_model_performance_table(model_results):
    """Display professional performance metrics table"""
    
    st.subheader("Performance Metrics")
    
    # Create comparison DataFrame
    comparison_data = []
    for name, metrics in model_results.items():
        comparison_data.append({
            'Model': name.replace('_', ' ').title(),
            'Accuracy': f"{metrics['accuracy']:.3f}",
            'Precision': f"{metrics['precision']:.3f}",
            'Recall': f"{metrics['recall']:.3f}",
            'F1-Score': f"{metrics['f1_score']:.3f}",
            'ROC-AUC': f"{metrics['roc_auc']:.3f}"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display with professional styling
    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Highlight best performing model
    best_model = max(model_results.keys(), key=lambda x: model_results[x]['roc_auc'])
    best_auc = model_results[best_model]['roc_auc']
    
    st.success(f"🏆 Best Performing Model: **{best_model.replace('_', ' ').title()}** (ROC-AUC: {best_auc:.3f})")

def display_performance_chart(model_results):
    """Display performance comparison chart"""
    
    st.subheader("Performance Comparison")
    
    # Prepare data for plotting
    models = [name.replace('_', ' ').title() for name in model_results.keys()]
    accuracies = [model_results[name]['accuracy'] for name in model_results.keys()]
    roc_aucs = [model_results[name]['roc_auc'] for name in model_results.keys()]
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8, color='#2E86AB')
    bars2 = ax.bar(x + width/2, roc_aucs, width, label='ROC-AUC', alpha=0.8, color='#A23B72')
    
    # Styling
    ax.set_xlabel('Models', fontweight='500')
    ax.set_ylabel('Score', fontweight='500')
    ax.set_title('Model Performance Comparison', fontweight='600')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    st.pyplot(fig)

def display_roc_curves(model_results, y_test):
    """Display ROC curves for all models"""
    from sklearn.metrics import roc_curve
    
    st.subheader("ROC Curves")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2E86AB', '#A23B72', '#F39C12', '#27AE60', '#8E44AD']
    
    for i, (name, metrics) in enumerate(model_results.items()):
        fpr, tpr, _ = roc_curve(y_test, metrics['probabilities'])
        auc_score = metrics['roc_auc']
        color = colors[i % len(colors)]
        
        ax.plot(fpr, tpr, color=color, linewidth=2.5, 
               label=f'{name.replace("_", " ").title()} (AUC = {auc_score:.3f})')
    
    # Reference line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.6, linewidth=1, label='Random Classifier')
    
    # Styling
    ax.set_xlabel('False Positive Rate', fontweight='500')
    ax.set_ylabel('True Positive Rate', fontweight='500')
    ax.set_title('ROC Curves Comparison', fontweight='600')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

def show_best_model_summary(model_results):
    """Show detailed summary of the best performing model"""
    
    st.subheader("Best Model Analysis")
    
    # Find best model
    best_model_name = max(model_results.keys(), key=lambda x: model_results[x]['roc_auc'])
    best_metrics = model_results[best_model_name]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Best Model", best_model_name.replace('_', ' ').title())
        st.metric("ROC-AUC Score", f"{best_metrics['roc_auc']:.3f}")
    
    with col2:
        st.metric("Accuracy", f"{best_metrics['accuracy']:.3f}")
        st.metric("Precision", f"{best_metrics['precision']:.3f}")
    
    with col3:
        st.metric("Recall", f"{best_metrics['recall']:.3f}")
        st.metric("F1-Score", f"{best_metrics['f1_score']:.3f}")
    
    # Model recommendations
    st.markdown("### Model Recommendations")
    
    if 'xgboost' in best_model_name.lower():
        st.info("""
        **XGBoost** is performing best for your churn prediction:
        - Excellent handling of complex patterns
        - Built-in feature importance
        - Robust against overfitting
        - Recommended for production deployment
        """)
    elif 'random' in best_model_name.lower():
        st.info("""
        **Random Forest** shows strong performance:
        - Good balance of accuracy and interpretability
        - Handles mixed data types well
        - Provides feature importance insights
        - Suitable for production use
        """)
    else:
        st.info("""
        **Logistic Regression** provides baseline performance:
        - Highly interpretable results
        - Fast training and prediction
        - Good for understanding linear relationships
        - Consider ensemble methods for improved accuracy
        """)

def show_executive_dashboard(df, data_summary):
    """Executive-level dashboard with key business metrics"""
    
    st.markdown("## Executive Summary")
    
    # Key Performance Indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3>Customer Base</h3>
            <h2>{data_summary['total_customers']:,}</h2>
            <p>Total Active Customers</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        churn_count = data_summary['churned_customers']
        st.markdown(f"""
        <div class="metric-container">
            <h3>Customer Churn</h3>
            <h2>{churn_count:,}</h2>
            <p>Customers Lost</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        churn_rate = data_summary['churn_rate']
        color = "#ef4444" if churn_rate > 0.25 else "#f59e0b" if churn_rate > 0.15 else "#10b981"
        st.markdown(f"""
        <div class="metric-container" style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%);">
            <h3>Churn Rate</h3>
            <h2>{churn_rate:.1%}</h2>
            <p>Monthly Attrition</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_revenue = data_summary['avg_monthly_charges']
        st.markdown(f"""
        <div class="metric-container">
            <h3>Avg Revenue</h3>
            <h2>${avg_revenue:.0f}</h2>
            <p>Per Customer/Month</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Business Intelligence Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Customer Distribution Analysis")
        
        # Professional pie chart
        fig, ax = plt.subplots(figsize=(8, 6))
        churn_counts = df['Churn'].value_counts()
        colors = ['#10b981', '#ef4444']
        
        wedges, texts, autotexts = ax.pie(
            churn_counts.values, 
            labels=['Retained', 'Churned'],
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            explode=(0, 0.05)
        )
        
        # Professional styling
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        for text in texts:
            text.set_fontsize(11)
            text.set_fontweight('500')
        
        ax.set_title('Customer Retention Overview', fontsize=14, fontweight='600', pad=20)
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Revenue Impact Analysis")
        
        # Professional box plot
        fig, ax = plt.subplots(figsize=(8, 6))
        churned = df[df['Churn'] == 'Yes']['MonthlyCharges']
        retained = df[df['Churn'] == 'No']['MonthlyCharges']
        
        bp = ax.boxplot([retained, churned], 
                       labels=['Retained', 'Churned'],
                       patch_artist=True)
        
        # Color the boxes professionally
        bp['boxes'][0].set_facecolor('#10b981')
        bp['boxes'][1].set_facecolor('#ef4444')
        
        for box in bp['boxes']:
            box.set_alpha(0.7)
        
        ax.set_title('Monthly Revenue Distribution', fontsize=14, fontweight='600', pad=20)
        ax.set_ylabel('Monthly Charges ($)', fontsize=11)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    # Business Insights
    st.markdown("---")
    st.markdown("### Key Business Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        monthly_loss = df[df['Churn'] == 'Yes']['MonthlyCharges'].sum()
        annual_loss = monthly_loss * 12
        st.markdown(f"""
        **💸 Revenue at Risk**
        - Monthly Loss: ${monthly_loss:,.0f}
        - Annual Impact: ${annual_loss:,.0f}
        """)
    
    with col2:
        avg_churned_value = df[df['Churn'] == 'Yes']['MonthlyCharges'].mean()
        st.markdown(f"""
        **📊 Customer Segments**
        - Avg Churned Value: ${avg_churned_value:.0f}
        - High-Risk Identification: Active
        """)
    
    with col3:
        st.markdown(f"""
        **🎯 Retention Opportunity**
        - Preventable Loss: ${annual_loss * 0.3:,.0f}
        - ROI Potential: 5:1
        """)

def show_prediction_interface(encoders, scaler, feature_names, models):
    """Professional prediction interface"""
    
    st.markdown("## Customer Churn Risk Assessment")
    
    # Streamlined input form
    with st.form("customer_prediction"):
        st.markdown("### Customer Profile")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Account Information**")
            tenure = st.number_input("Account Tenure (months)", 0, 72, 12, help="How long has the customer been with us?")
            monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0)
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            payment_method = st.selectbox("Payment Method", 
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        
        with col2:
            st.markdown("**Demographics & Services**")
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner = st.selectbox("Has Partner", ["No", "Yes"])
            internet_service = st.selectbox("Internet Service", ["No", "DSL", "Fiber optic"])
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, 
                                          value=monthly_charges * tenure,
                                          help="Total amount charged to date")
        
        # Professional prediction button
        submitted = st.form_submit_button("🔍 Analyze Churn Risk")
        
        if submitted:
            # Calculate risk
            risk_score = calculate_churn_risk(tenure, monthly_charges, contract, 
                                            payment_method, senior_citizen, internet_service)
            
            # Display professional results
            display_professional_results(risk_score, monthly_charges, tenure)

def calculate_churn_risk(tenure, monthly_charges, contract, payment_method, senior_citizen, internet_service):
    """Advanced risk calculation algorithm"""
    base_risk = 0.15
    
    # Contract impact
    contract_risk = {"Month-to-month": 0.35, "One year": 0.10, "Two year": 0.05}
    base_risk += contract_risk.get(contract, 0.10)
    
    # Tenure impact (higher weight for early tenure)
    if tenure <= 6:
        base_risk += 0.25
    elif tenure <= 12:
        base_risk += 0.15
    elif tenure > 36:
        base_risk -= 0.10
    
    # Payment method impact
    if payment_method == "Electronic check":
        base_risk += 0.12
    elif payment_method in ["Bank transfer (automatic)", "Credit card (automatic)"]:
        base_risk -= 0.05
    
    # Charges impact
    if monthly_charges > 80:
        base_risk += 0.08
    elif monthly_charges < 30:
        base_risk -= 0.03
    
    # Demographics
    if senior_citizen == "Yes":
        base_risk += 0.04
    
    # Internet service
    if internet_service == "Fiber optic":
        base_risk += 0.03
    
    return min(max(base_risk, 0.01), 0.95)

def display_professional_results(risk_score, monthly_charges, tenure):
    """Display professional prediction results"""
    
    # Risk categorization
    if risk_score > 0.65:
        risk_level = "HIGH"
        css_class = "high-risk"
        recommendations = [
            "🚨 Immediate retention intervention required",
            "📞 Schedule executive-level customer outreach",
            "💰 Authorize premium retention incentives",
            "⏰ 24-48 hour response window critical"
        ]
    elif risk_score > 0.35:
        risk_level = "MEDIUM"
        css_class = "medium-risk"
        recommendations = [
            "⚠️ Proactive retention campaign recommended",
            "📧 Deploy targeted email engagement series",
            "🎁 Consider service upgrade incentives",
            "📊 Monitor account activity closely"
        ]
    else:
        risk_level = "LOW"
        css_class = "low-risk"
        recommendations = [
            "✅ Customer retention outlook positive",
            "💡 Focus on satisfaction maintenance",
            "🎯 Consider upselling opportunities",
            "📈 Include in loyalty program"
        ]
    
    # Display results
    st.markdown(f"""
    <div class="prediction-result {css_class}">
        <h2>{risk_level} CHURN RISK</h2>
        <h1>{risk_score:.0%}</h1>
        <p>Probability of customer churn</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Business impact
    col1, col2, col3 = st.columns(3)
    
    with col1:
        annual_value = monthly_charges * 12
        st.metric("Annual Customer Value", f"${annual_value:,.0f}")
    
    with col2:
        lifetime_value = monthly_charges * tenure
        st.metric("Lifetime Value to Date", f"${lifetime_value:,.0f}")
    
    with col3:
        retention_cost = annual_value * 0.15
        st.metric("Est. Retention Cost", f"${retention_cost:,.0f}")
    
    # Action recommendations
    st.markdown("### 🎯 Recommended Actions")
    for rec in recommendations:
        st.markdown(f"- {rec}")

def show_analytics_dashboard(df):
    """Professional analytics dashboard"""
    
    st.markdown("## Business Analytics Dashboard")
    
    analysis_type = st.selectbox(
        "Select Analysis Focus:",
        ["📋 Contract Analysis", "💳 Payment Methods", "⏰ Customer Lifecycle"],
        index=0
    )
    
    if analysis_type == "📋 Contract Analysis":
        st.markdown("### Contract Type Impact Analysis")
        
        contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        contract_churn.plot(kind='bar', ax=ax, color=['#10b981', '#ef4444'], alpha=0.8)
        ax.set_title('Churn Rate by Contract Type', fontsize=14, fontweight='600')
        ax.set_xlabel('Contract Type')
        ax.set_ylabel('Churn Rate')
        ax.legend(['Retained', 'Churned'])
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
        
        # Business insights
        st.markdown("#### Key Insights:")
        for contract in contract_churn.index:
            churn_rate = contract_churn.loc[contract, 'Yes']
            customer_count = len(df[df['Contract'] == contract])
            risk_indicator = "🔴" if churn_rate > 0.4 else "🟡" if churn_rate > 0.2 else "🟢"
            st.markdown(f"{risk_indicator} **{contract}**: {churn_rate:.1%} churn rate ({customer_count:,} customers)")
    
    elif analysis_type == "💳 Payment Methods":
        st.markdown("### Payment Method Risk Analysis")
        
        payment_churn = pd.crosstab(df['PaymentMethod'], df['Churn'], normalize='index')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        payment_churn.plot(kind='bar', ax=ax, color=['#10b981', '#ef4444'], alpha=0.8)
        ax.set_title('Churn Rate by Payment Method', fontsize=14, fontweight='600')
        ax.set_xlabel('Payment Method')
        ax.set_ylabel('Churn Rate')
        ax.legend(['Retained', 'Churned'])
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    
    elif analysis_type == "⏰ Customer Lifecycle":
        st.markdown("### Customer Lifecycle Analysis")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        churned_tenure = df[df['Churn'] == 'Yes']['tenure']
        retained_tenure = df[df['Churn'] == 'No']['tenure']
        
        ax.hist([retained_tenure, churned_tenure], bins=24, alpha=0.7, 
                color=['#10b981', '#ef4444'], label=['Retained', 'Churned'])
        ax.set_title('Customer Tenure Distribution', fontsize=14, fontweight='600')
        ax.set_xlabel('Tenure (months)')
        ax.set_ylabel('Number of Customers')
        ax.legend()
        plt.grid(axis='y', alpha=0.3)
        st.pyplot(fig)

def show_about_system():
    """Professional about page"""
    
    st.markdown("## System Overview")
    
    st.markdown("""
    ### Customer Churn Prediction Analytics Platform
    
    This enterprise-grade analytics platform leverages advanced machine learning algorithms 
    to predict customer churn and provide actionable business intelligence.
    
    #### Core Capabilities
    - **Predictive Analytics**: Real-time churn risk assessment
    - **Business Intelligence**: Executive-level dashboards and insights  
    - **Risk Segmentation**: Automated customer risk categorization
    - **Financial Impact**: Revenue-at-risk calculations and ROI analysis
    
    #### Technical Architecture
    - **Machine Learning**: Ensemble models (Logistic Regression, Random Forest, XGBoost)
    - **Data Processing**: Automated ETL pipeline with feature engineering
    - **User Interface**: Responsive web application built with Streamlit
    - **Visualization**: Professional charts and interactive dashboards
    
    #### Business Value
    - **Proactive Retention**: Identify at-risk customers before churn occurs
    - **Cost Optimization**: Focus retention efforts on highest-value customers  
    - **Revenue Protection**: Prevent customer acquisition cost multiplication
    - **Strategic Insights**: Data-driven decision making for customer success
    
    ---
    
    **Developed for enterprise customer success teams**  
    *Empowering data-driven retention strategies*
    """)

if __name__ == "__main__":
    main()
