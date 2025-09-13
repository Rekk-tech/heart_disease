"""
FastAPI application for heart disease prediction.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import yaml
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Heart Disease Prediction API",
    description="API for predicting heart disease risk using machine learning",
    version="1.0.0"
)

# Global variables for model and preprocessor
model = None
preprocessor = None
feature_names = None
config = None


class HeartDiseaseFeatures(BaseModel):
    """Pydantic model for heart disease prediction input."""
    age: int = Field(..., ge=20, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="Sex (0: female, 1: male)")
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: int = Field(..., ge=80, le=250, description="Resting blood pressure")
    chol: int = Field(..., ge=100, le=600, description="Serum cholesterol")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results")
    thalach: int = Field(..., ge=60, le=220, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0.0, le=6.2, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy")
    thal: int = Field(..., ge=0, le=3, description="Thalassemia type")


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: int = Field(..., description="Predicted class (0: No disease, 1: Disease)")
    probability: float = Field(..., description="Probability of heart disease")
    confidence: str = Field(..., description="Confidence level")
    timestamp: str = Field(..., description="Prediction timestamp")


class ModelInfo(BaseModel):
    """Model information response."""
    model_type: str
    training_date: str
    version: str
    accuracy: float
    features_used: List[str]


def load_model_artifacts():
    """Load model and preprocessor from files."""
    global model, preprocessor, feature_names, config
    
    try:
        # Load configuration
        with open("conf/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Load model
        model_data = joblib.load("models/heart_model.pkl")
        if isinstance(model_data, dict):
            model = model_data['model']
        else:
            model = model_data
        
        # Load preprocessor
        preprocessor_data = joblib.load("models/preprocessor.pkl")
        if isinstance(preprocessor_data, dict):
            preprocessor = preprocessor_data['preprocessor']
            feature_names = preprocessor_data.get('feature_names')
        else:
            preprocessor = preprocessor_data
            feature_names = None
        
        logger.info("Model artifacts loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model artifacts: {e}")
        raise


def predict_heart_disease(features: HeartDiseaseFeatures) -> Dict[str, Any]:
    """
    Predict heart disease risk for given features.
    
    Args:
        features: Heart disease features
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Convert features to DataFrame
        feature_dict = features.dict()
        feature_df = pd.DataFrame([feature_dict])
        
        # Apply preprocessing
        if preprocessor is not None:
            processed_features = preprocessor.transform(feature_df)
            if feature_names:
                feature_df = pd.DataFrame(processed_features, columns=feature_names)
            else:
                feature_df = pd.DataFrame(processed_features)
        
        # Make prediction
        prediction = model.predict(feature_df)[0]
        
        # Get prediction probability if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(feature_df)[0]
            probability = probabilities[1]  # Probability of class 1 (disease)
        else:
            probability = float(prediction)
        
        # Determine confidence level
        if probability > 0.8 or probability < 0.2:
            confidence = "High"
        elif probability > 0.6 or probability < 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return {
            "prediction": int(prediction),
            "probability": round(probability, 4),
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Load model artifacts on startup."""
    try:
        load_model_artifacts()
        logger.info("API startup completed successfully")
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Heart Disease Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint."""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: HeartDiseaseFeatures):
    """
    Predict heart disease risk.
    
    Args:
        features: Heart disease features
        
    Returns:
        Prediction results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    result = predict_heart_disease(features)
    return PredictionResponse(**result)


@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model."""
    if model is None or config is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Get model type
    model_type = type(model).__name__
    
    # Get features used
    numeric_features = config.get('features', {}).get('numeric', [])
    categorical_features = config.get('features', {}).get('categorical', [])
    all_features = numeric_features + categorical_features
    
    return ModelInfo(
        model_type=model_type,
        training_date="2024-01-01",  # This would come from model metadata
        version="1.0.0",
        accuracy=0.85,  # This would come from model evaluation
        features_used=all_features
    )


@app.get("/feature-importance", response_model=Dict[str, float])
async def get_feature_importance():
    """Get feature importance scores."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not hasattr(model, 'feature_importances_'):
        raise HTTPException(status_code=400, detail="Model does not support feature importance")
    
    if feature_names is None:
        # Try to get feature names from config
        if config is not None:
            numeric_features = config.get('features', {}).get('numeric', [])
            categorical_features = config.get('features', {}).get('categorical', [])
            feature_names = numeric_features + categorical_features
        else:
            raise HTTPException(status_code=500, detail="Feature names not available")
    
    importance_dict = dict(zip(feature_names, model.feature_importances_))
    
    # Sort by importance
    sorted_importance = dict(sorted(importance_dict.items(), 
                                  key=lambda x: x[1], reverse=True))
    
    return sorted_importance


@app.post("/batch-predict", response_model=List[PredictionResponse])
async def batch_predict(features_list: List[HeartDiseaseFeatures]):
    """
    Predict heart disease risk for multiple instances.
    
    Args:
        features_list: List of heart disease features
        
    Returns:
        List of prediction results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(features_list) > 100:
        raise HTTPException(status_code=400, detail="Batch size too large (max 100)")
    
    results = []
    for features in features_list:
        result = predict_heart_disease(features)
        results.append(PredictionResponse(**result))
    
    return results


@app.get("/features-description", response_model=Dict[str, str])
async def get_features_description():
    """Get description of all features."""
    descriptions = {
        "age": "Age in years",
        "sex": "Sex (0: female, 1: male)",
        "cp": "Chest pain type (0: typical angina, 1: atypical angina, 2: non-anginal pain, 3: asymptomatic)",
        "trestbps": "Resting blood pressure (mm Hg)",
        "chol": "Serum cholesterol (mg/dl)",
        "fbs": "Fasting blood sugar > 120 mg/dl (0: false, 1: true)",
        "restecg": "Resting ECG results (0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy)",
        "thalach": "Maximum heart rate achieved (bpm)",
        "exang": "Exercise induced angina (0: no, 1: yes)",
        "oldpeak": "ST depression induced by exercise relative to rest",
        "slope": "Slope of peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping)",
        "ca": "Number of major vessels colored by fluoroscopy (0-4)",
        "thal": "Thalassemia type (0: normal, 1: fixed defect, 2: reversible defect, 3: other)"
    }
    
    return descriptions


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
