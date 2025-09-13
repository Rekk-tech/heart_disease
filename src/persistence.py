"""
Model persistence utilities for heart disease prediction.
"""

import pickle
import joblib
import json
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


def save_model(model: Any, 
               file_path: str = "models/heart_model.pkl",
               metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save trained model with metadata.
    
    Args:
        model: Trained model to save
        file_path: Path to save the model
        metadata: Optional metadata dictionary
    """
    try:
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare save data
        save_data = {
            'model': model,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        # Save model
        joblib.dump(save_data, file_path)
        logger.info(f"Model saved to {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise


def load_model(file_path: str = "models/heart_model.pkl") -> Dict[str, Any]:
    """
    Load saved model with metadata.
    
    Args:
        file_path: Path to load the model from
        
    Returns:
        Dictionary containing model and metadata
    """
    try:
        save_data = joblib.load(file_path)
        logger.info(f"Model loaded from {file_path}")
        return save_data
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def save_preprocessor(preprocessor: Any, 
                     file_path: str = "models/preprocessor.pkl",
                     feature_names: Optional[list] = None) -> None:
    """
    Save preprocessing pipeline.
    
    Args:
        preprocessor: Preprocessing pipeline
        file_path: Path to save the preprocessor
        feature_names: List of feature names after preprocessing
    """
    try:
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare save data
        save_data = {
            'preprocessor': preprocessor,
            'feature_names': feature_names,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save preprocessor
        joblib.dump(save_data, file_path)
        logger.info(f"Preprocessor saved to {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving preprocessor: {e}")
        raise


def load_preprocessor(file_path: str = "models/preprocessor.pkl") -> Dict[str, Any]:
    """
    Load preprocessing pipeline.
    
    Args:
        file_path: Path to load the preprocessor from
        
    Returns:
        Dictionary containing preprocessor and feature names
    """
    try:
        save_data = joblib.load(file_path)
        logger.info(f"Preprocessor loaded from {file_path}")
        return save_data
        
    except Exception as e:
        logger.error(f"Error loading preprocessor: {e}")
        raise


def save_quantiles(quantiles: Dict[str, Dict[str, float]], 
                  file_path: str = "models/quantiles.json") -> None:
    """
    Save quantiles for numeric features.
    
    Args:
        quantiles: Quantiles dictionary
        file_path: Path to save quantiles
    """
    try:
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        save_data = {
            'quantiles': quantiles,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save quantiles
        with open(file_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"Quantiles saved to {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving quantiles: {e}")
        raise


def load_quantiles(file_path: str = "models/quantiles.json") -> Dict[str, Any]:
    """
    Load quantiles for numeric features.
    
    Args:
        file_path: Path to load quantiles from
        
    Returns:
        Dictionary containing quantiles and metadata
    """
    try:
        with open(file_path, 'r') as f:
            save_data = json.load(f)
        
        logger.info(f"Quantiles loaded from {file_path}")
        return save_data
        
    except Exception as e:
        logger.error(f"Error loading quantiles: {e}")
        raise


def save_model_card(model_card: Dict[str, Any], 
                   file_path: str = "models/model_card.md") -> None:
    """
    Save model card as Markdown file.
    
    Args:
        model_card: Model card dictionary
        file_path: Path to save model card
    """
    try:
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Generate Markdown content
        md_content = generate_model_card_markdown(model_card)
        
        # Save model card
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"Model card saved to {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving model card: {e}")
        raise


def generate_model_card_markdown(model_card: Dict[str, Any]) -> str:
    """
    Generate Markdown content for model card.
    
    Args:
        model_card: Model card dictionary
        
    Returns:
        Markdown content as string
    """
    try:
        md_lines = [
            "# Heart Disease Prediction Model Card",
            "",
            f"**Created:** {model_card.get('timestamp', 'Unknown')}",
            f"**Version:** {model_card.get('version', '1.0.0')}",
            "",
            "## Model Overview",
            "",
            f"**Purpose:** {model_card.get('purpose', 'Predict heart disease risk')}",
            f"**Algorithm:** {model_card.get('algorithm', 'Ensemble')}",
            f"**Training Data:** {model_card.get('training_data_size', 'Unknown')} samples",
            "",
            "## Performance Metrics",
            ""
        ]
        
        # Add performance metrics
        metrics = model_card.get('performance', {})
        for metric, value in metrics.items():
            md_lines.append(f"- **{metric.upper()}:** {value:.4f}")
        
        md_lines.extend([
            "",
            "## Data Information",
            "",
            f"**Features:** {model_card.get('n_features', 'Unknown')}",
            f"**Target Classes:** {model_card.get('target_classes', '0: No Disease, 1: Disease')}",
            f"**Class Distribution:** {model_card.get('class_distribution', 'Unknown')}",
            "",
            "## Model Limitations",
            ""
        ])
        
        # Add limitations
        limitations = model_card.get('limitations', [
            "Model trained on specific dataset and may not generalize to other populations",
            "Clinical decisions should not be based solely on model predictions",
            "Model requires complete feature information for accurate predictions"
        ])
        
        for limitation in limitations:
            md_lines.append(f"- {limitation}")
        
        md_lines.extend([
            "",
            "## Usage Guidelines",
            "",
            "1. Ensure all required features are provided",
            "2. Preprocess data using the same pipeline used during training",
            "3. Interpret predictions as risk scores, not definitive diagnoses",
            "4. Consult healthcare professionals for medical decisions",
            "",
            "## Technical Details",
            "",
            f"**Preprocessing:** {model_card.get('preprocessing', 'Standard scaling and label encoding')}",
            f"**Feature Engineering:** {model_card.get('feature_engineering', 'None')}",
            f"**Cross-validation:** {model_card.get('cv_folds', 5)} folds",
            "",
            "---",
            f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        ])
        
        return "\n".join(md_lines)
        
    except Exception as e:
        logger.error(f"Error generating model card markdown: {e}")
        raise


def save_evaluation_results(results: Dict[str, Any], 
                           file_path: str = "evaluation_results.json") -> None:
    """
    Save evaluation results.
    
    Args:
        results: Evaluation results dictionary
        file_path: Path to save results
    """
    try:
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        save_data = {
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        save_data = convert_numpy(save_data)
        
        # Save results
        with open(file_path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"Evaluation results saved to {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving evaluation results: {e}")
        raise


def load_evaluation_results(file_path: str = "evaluation_results.json") -> Dict[str, Any]:
    """
    Load evaluation results.
    
    Args:
        file_path: Path to load results from
        
    Returns:
        Dictionary containing evaluation results
    """
    try:
        with open(file_path, 'r') as f:
            save_data = json.load(f)
        
        logger.info(f"Evaluation results loaded from {file_path}")
        return save_data
        
    except Exception as e:
        logger.error(f"Error loading evaluation results: {e}")
        raise


def save_config(config: Dict[str, Any], 
               file_path: str = "models/config_snapshot.yaml") -> None:
    """
    Save configuration snapshot used for training.
    
    Args:
        config: Configuration dictionary
        file_path: Path to save config
    """
    try:
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        save_data = {
            'config': config,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save config
        with open(file_path, 'w') as f:
            yaml.dump(save_data, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration snapshot saved to {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving configuration: {e}")
        raise


def load_config(file_path: str = "models/config_snapshot.yaml") -> Dict[str, Any]:
    """
    Load configuration snapshot.
    
    Args:
        file_path: Path to load config from
        
    Returns:
        Dictionary containing configuration
    """
    try:
        with open(file_path, 'r') as f:
            save_data = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from {file_path}")
        return save_data
        
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def create_model_package(model: Any,
                        preprocessor: Any,
                        config: Dict[str, Any],
                        evaluation_results: Dict[str, Any],
                        package_dir: str = "model_package") -> None:
    """
    Create a complete model package with all necessary components.
    
    Args:
        model: Trained model
        preprocessor: Preprocessing pipeline
        config: Configuration used for training
        evaluation_results: Model evaluation results
        package_dir: Directory to create the package
    """
    try:
        # Create package directory
        package_path = Path(package_dir)
        package_path.mkdir(parents=True, exist_ok=True)
        
        # Save all components
        save_model(model, f"{package_dir}/model.pkl")
        save_preprocessor(preprocessor, f"{package_dir}/preprocessor.pkl")
        save_config(config, f"{package_dir}/config.yaml")
        save_evaluation_results(evaluation_results, f"{package_dir}/evaluation.json")
        
        # Create model card
        model_card = {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'algorithm': type(model).__name__,
            'performance': evaluation_results.get('metrics', {}),
            'n_features': len(config.get('features', {}).get('numeric', []) + 
                             config.get('features', {}).get('categorical', [])),
            'training_data_size': evaluation_results.get('training_size', 'Unknown'),
            'cv_folds': config.get('evaluation', {}).get('cv_folds', 5)
        }
        
        save_model_card(model_card, f"{package_dir}/model_card.md")
        
        # Create requirements file
        requirements = [
            "scikit-learn>=1.0.0",
            "pandas>=1.3.0",
            "numpy>=1.21.0",
            "joblib>=1.0.0"
        ]
        
        with open(f"{package_dir}/requirements.txt", 'w') as f:
            f.write("\n".join(requirements))
        
        logger.info(f"Model package created in {package_dir}")
        
    except Exception as e:
        logger.error(f"Error creating model package: {e}")
        raise
