"""
Heart Disease Prediction Project

A comprehensive machine learning project for predicting heart disease
using various algorithms and advanced techniques.
"""

__version__ = "1.0.0"
__author__ = "Heart Disease Prediction Team"

# Import main modules for easy access
from .data_loader import load_data, validate_schema
from .preprocessing import preprocess_data, split_data
from .feature_engineering import create_features, select_features
from .modeling import train_models, evaluate_models, create_ensemble
from .evaluation import calculate_metrics, plot_confusion_matrix, plot_roc_curve
from .persistence import save_model, load_model
from .explain import explain_model, plot_feature_importance
from .utils import setup_logging, set_seed

__all__ = [
    "load_data",
    "validate_schema", 
    "preprocess_data",
    "split_data",
    "create_features",
    "select_features",
    "train_models",
    "evaluate_models",
    "create_ensemble",
    "calculate_metrics",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "save_model",
    "load_model",
    "explain_model",
    "plot_feature_importance",
    "setup_logging",
    "set_seed"
]
