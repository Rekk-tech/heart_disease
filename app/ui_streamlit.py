"""
Streamlit UI for heart disease prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #e74c3c;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .high-risk {
        background-color: #ffebee;
        border-left: 5px solid #e74c3c;
    }
    .low-risk {
        background-color: #e8f5e8;
        border-left: 5px solid #27ae60;
    }
    .medium-risk {
        background-color: #fff3e0;
        border-left: 5px solid #f39c12;
    }
</style>
""", unsafe_allow_html=True)

# API configuration
API_BASE_URL = "http://localhost:8000"


def check_api_health():
    """Check if API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_model_info():
    """Get model information from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/model-info")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def get_feature_descriptions():
    """Get feature descriptions from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/features-description")
        if response.status_code == 200:
            return response.json()
        return {}
    except:
        return {}


def make_prediction(features):
    """Make prediction using API."""
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=features)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Connection Error: {str(e)}")
        return None


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction System</h1>', 
                unsafe_allow_html=True)
    
    # Check API health
    if not check_api_health():
        st.error("⚠️ API is not running. Please start the API server first.")
        st.info("Run: `python app/api.py` or `make api`")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Model info
        model_info = get_model_info()
        if model_info:
            st.subheader("Model Information")
            st.write(f"**Type:** {model_info['model_type']}")
            st.write(f"**Version:** {model_info['version']}")
            st.write(f"**Accuracy:** {model_info['accuracy']:.1%}")
        
        # Feature descriptions
        st.subheader("📖 Feature Guide")
        feature_descriptions = get_feature_descriptions()
        
        with st.expander("Feature Descriptions"):
            for feature, description in feature_descriptions.items():
                st.write(f"**{feature}:** {description}")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Prediction", "📊 Analysis", "📈 Model Info", "ℹ️ About"])
    
    with tab1:
        st.header("Heart Disease Risk Prediction")
        
        # Create two columns for input form
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Patient Information")
            age = st.slider("Age (years)", min_value=20, max_value=120, value=50)
            sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
            cp = st.selectbox("Chest Pain Type", 
                            options=[0, 1, 2, 3],
                            format_func=lambda x: {
                                0: "Typical Angina",
                                1: "Atypical Angina", 
                                2: "Non-anginal Pain",
                                3: "Asymptomatic"
                            }[x])
            trestbps = st.slider("Resting Blood Pressure (mm Hg)", 
                               min_value=80, max_value=250, value=120)
            chol = st.slider("Serum Cholesterol (mg/dl)", 
                           min_value=100, max_value=600, value=200)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", 
                             options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        
        with col2:
            st.subheader("Medical Test Results")
            restecg = st.selectbox("Resting ECG Results",
                                 options=[0, 1, 2],
                                 format_func=lambda x: {
                                     0: "Normal",
                                     1: "ST-T Wave Abnormality",
                                     2: "Left Ventricular Hypertrophy"
                                 }[x])
            thalach = st.slider("Maximum Heart Rate Achieved (bpm)", 
                              min_value=60, max_value=220, value=150)
            exang = st.selectbox("Exercise Induced Angina",
                               options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
            oldpeak = st.slider("ST Depression (Old Peak)", 
                              min_value=0.0, max_value=6.2, value=1.0, step=0.1)
            slope = st.selectbox("Slope of Peak Exercise ST Segment",
                               options=[0, 1, 2],
                               format_func=lambda x: {
                                   0: "Upsloping",
                                   1: "Flat", 
                                   2: "Downsloping"
                               }[x])
            ca = st.slider("Number of Major Vessels", min_value=0, max_value=4, value=0)
            thal = st.selectbox("Thalassemia Type",
                              options=[0, 1, 2, 3],
                              format_func=lambda x: {
                                  0: "Normal",
                                  1: "Fixed Defect",
                                  2: "Reversible Defect",
                                  3: "Other"
                              }[x])
        
        # Prediction button
        if st.button("🔍 Predict Heart Disease Risk", type="primary", use_container_width=True):
            
            # Prepare features
            features = {
                "age": age,
                "sex": sex,
                "cp": cp,
                "trestbps": trestbps,
                "chol": chol,
                "fbs": fbs,
                "restecg": restecg,
                "thalach": thalach,
                "exang": exang,
                "oldpeak": oldpeak,
                "slope": slope,
                "ca": ca,
                "thal": thal
            }
            
            # Make prediction
            with st.spinner("Analyzing patient data..."):
                result = make_prediction(features)
            
            if result:
                # Display prediction results
                st.subheader("🎯 Prediction Results")
                
                # Risk level and styling
                risk_level = result['confidence']
                probability = result['probability']
                prediction = result['prediction']
                
                if prediction == 1:
                    risk_text = "HIGH RISK"
                    risk_class = "high-risk"
                    risk_color = "#e74c3c"
                    risk_icon = "🚨"
                else:
                    risk_text = "LOW RISK"
                    risk_class = "low-risk"
                    risk_color = "#27ae60"
                    risk_icon = "✅"
                
                # Risk display
                st.markdown(f"""
                <div class="prediction-box {risk_class}">
                    <h3>{risk_icon} {risk_text}</h3>
                    <p><strong>Probability:</strong> {probability:.1%}</p>
                    <p><strong>Confidence:</strong> {risk_level}</p>
                    <p><strong>Timestamp:</strong> {result['timestamp']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Visualization
                col1, col2 = st.columns(2)
                
                with col1:
                    # Risk gauge
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number+delta",
                        value = probability * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Heart Disease Risk %"},
                        delta = {'reference': 50},
                        gauge = {
                            'axis': {'range': [None, 100]},
                            'bar': {'color': risk_color},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgray"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    
                    fig_gauge.update_layout(height=300)
                    st.plotly_chart(fig_gauge, use_container_width=True)
                
                with col2:
                    # Risk distribution
                    risk_data = pd.DataFrame({
                        'Risk Level': ['Low Risk', 'High Risk'],
                        'Probability': [1 - probability, probability]
                    })
                    
                    fig_bar = px.bar(risk_data, x='Risk Level', y='Probability',
                                    color='Risk Level',
                                    color_discrete_map={
                                        'Low Risk': '#27ae60',
                                        'High Risk': '#e74c3c'
                                    })
                    fig_bar.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab2:
        st.header("Data Analysis")
        
        # Feature importance
        try:
            response = requests.get(f"{API_BASE_URL}/feature-importance")
            if response.status_code == 200:
                importance_data = response.json()
                
                st.subheader("Feature Importance")
                
                # Create DataFrame for visualization
                importance_df = pd.DataFrame([
                    {'Feature': feature, 'Importance': importance}
                    for feature, importance in importance_data.items()
                ])
                
                # Bar chart
                fig_importance = px.bar(importance_df, x='Importance', y='Feature',
                                      orientation='h', title="Feature Importance Scores")
                fig_importance.update_layout(height=400)
                st.plotly_chart(fig_importance, use_container_width=True)
                
                # Display as table
                st.subheader("Feature Importance Table")
                st.dataframe(importance_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Could not load feature importance: {str(e)}")
    
    with tab3:
        st.header("Model Information")
        
        if model_info:
            # Model metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Model Type", model_info['model_type'])
            
            with col2:
                st.metric("Accuracy", f"{model_info['accuracy']:.1%}")
            
            with col3:
                st.metric("Version", model_info['version'])
            
            # Features used
            st.subheader("Features Used")
            features_list = model_info['features_used']
            
            # Create columns for features
            num_cols = 3
            cols = st.columns(num_cols)
            
            for i, feature in enumerate(features_list):
                with cols[i % num_cols]:
                    st.write(f"• {feature}")
        
        else:
            st.error("Could not load model information")
    
    with tab4:
        st.header("About This Application")
        
        st.markdown("""
        ### Heart Disease Prediction System
        
        This application uses machine learning to predict the risk of heart disease based on various medical and demographic factors.
        
        #### Features:
        - **Real-time Prediction**: Get instant heart disease risk assessment
        - **Interactive Interface**: Easy-to-use form for inputting patient data
        - **Visual Analytics**: Charts and graphs to understand predictions
        - **Feature Importance**: See which factors most influence predictions
        
        #### Model Information:
        - **Algorithm**: Ensemble of multiple machine learning models
        - **Training Data**: Cleveland Heart Disease Dataset
        - **Features**: 13 medical and demographic features
        - **Target**: Binary classification (Disease/No Disease)
        
        #### Disclaimer:
        This tool is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.
        
        #### Technical Details:
        - **Backend**: FastAPI with scikit-learn models
        - **Frontend**: Streamlit with Plotly visualizations
        - **Deployment**: Local development environment
        
        For more information, please refer to the project documentation.
        """)


if __name__ == "__main__":
    main()
