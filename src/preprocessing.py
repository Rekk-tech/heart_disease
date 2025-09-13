"""
Data preprocessing utilities for heart disease prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from typing import Tuple, Dict, Any, Optional
import yaml
import pickle
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def split_data(df: pd.DataFrame, 
               config: Dict[str, Any], 
               target_col: str = 'target') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        target_col: Name of target column
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    try:
        splitting_config = config['splitting']
        
        # First split: separate test set
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=splitting_config['test_size'],
            random_state=splitting_config['random_state'],
            stratify=y if splitting_config.get('stratify', True) else None
        )
        
        # Second split: separate train and validation sets
        val_size_adjusted = splitting_config['val_size'] / (1 - splitting_config['test_size'])
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=splitting_config['random_state'],
            stratify=y_temp if splitting_config.get('stratify', True) else None
        )
        
        # Combine features and target
        train_df = pd.concat([X_train, y_train], axis=1)
        val_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)
        
        logger.info(f"Data split completed:")
        logger.info(f"Train: {train_df.shape}")
        logger.info(f"Validation: {val_df.shape}")
        logger.info(f"Test: {test_df.shape}")
        
        return train_df, val_df, test_df
        
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        raise


def create_preprocessor(config: Dict[str, Any]) -> ColumnTransformer:
    """
    Create preprocessing pipeline using ColumnTransformer.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Fitted ColumnTransformer
    """
    try:
        numeric_features = config['features']['numeric']
        categorical_features = config['features']['categorical']
        preprocessing_config = config['preprocessing']
        
        # Numeric pipeline
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy=preprocessing_config['numeric_strategy'])),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(
                strategy=preprocessing_config['categorical_strategy']
            )),
            ('encoder', LabelEncoder())
        ])
        
        # Combine pipelines
        preprocessor = ColumnTransformer([
            ('numeric', numeric_pipeline, numeric_features),
            ('categorical', categorical_pipeline, categorical_features)
        ])
        
        logger.info("Preprocessor created successfully")
        return preprocessor
        
    except Exception as e:
        logger.error(f"Error creating preprocessor: {e}")
        raise


def preprocess_data(df: pd.DataFrame, 
                   preprocessor: ColumnTransformer, 
                   fit: bool = True) -> pd.DataFrame:
    """
    Apply preprocessing to the dataset.
    
    Args:
        df: Input DataFrame
        preprocessor: ColumnTransformer object
        fit: Whether to fit the preprocessor
        
    Returns:
        Preprocessed DataFrame
    """
    try:
        # Separate features and target
        target_col = 'target'
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Apply preprocessing
        if fit:
            X_transformed = preprocessor.fit_transform(X)
        else:
            X_transformed = preprocessor.transform(X)
        
        # Get feature names after transformation
        feature_names = get_feature_names(preprocessor)
        
        # Create DataFrame with transformed features
        X_df = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)
        
        # Combine with target
        result_df = pd.concat([X_df, y], axis=1)
        
        logger.info(f"Data preprocessing completed. Shape: {result_df.shape}")
        return result_df
        
    except Exception as e:
        logger.error(f"Error preprocessing data: {e}")
        raise


def get_feature_names(column_transformer: ColumnTransformer) -> list:
    """
    Get feature names after ColumnTransformer transformation.
    
    Args:
        column_transformer: Fitted ColumnTransformer
        
    Returns:
        List of feature names
    """
    feature_names = []
    
    for name, transformer, columns in column_transformer.transformers_:
        if transformer == 'drop':
            continue
            
        if hasattr(transformer, 'get_feature_names_out'):
            # For transformers with get_feature_names_out method
            feature_names.extend(transformer.get_feature_names_out(columns))
        else:
            # For other transformers, use original column names
            feature_names.extend(columns)
    
    return feature_names


def save_preprocessor(preprocessor: ColumnTransformer, 
                     file_path: str = "models/preprocessor.pkl") -> None:
    """
    Save preprocessor to file.
    
    Args:
        preprocessor: ColumnTransformer to save
        file_path: Path to save the preprocessor
    """
    try:
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(preprocessor, f)
        
        logger.info(f"Preprocessor saved to {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving preprocessor: {e}")
        raise


def load_preprocessor(file_path: str = "models/preprocessor.pkl") -> ColumnTransformer:
    """
    Load preprocessor from file.
    
    Args:
        file_path: Path to load the preprocessor from
        
    Returns:
        ColumnTransformer object
    """
    try:
        with open(file_path, 'rb') as f:
            preprocessor = pickle.load(f)
        
        logger.info(f"Preprocessor loaded from {file_path}")
        return preprocessor
        
    except Exception as e:
        logger.error(f"Error loading preprocessor: {e}")
        raise


def calculate_quantiles(df: pd.DataFrame, 
                       numeric_columns: list, 
                       quantiles: list = [0.25, 0.5, 0.75]) -> Dict[str, Dict[str, float]]:
    """
    Calculate quantiles for numeric columns.
    
    Args:
        df: Input DataFrame
        numeric_columns: List of numeric column names
        quantiles: List of quantiles to calculate
        
    Returns:
        Dictionary with quantiles for each column
    """
    quantile_dict = {}
    
    for col in numeric_columns:
        if col in df.columns:
            quantile_dict[col] = {
                f'q{int(q*100)}': float(df[col].quantile(q)) 
                for q in quantiles
            }
    
    return quantile_dict


def save_quantiles(quantiles: Dict[str, Dict[str, float]], 
                  file_path: str = "models/quantiles.json") -> None:
    """
    Save quantiles to JSON file.
    
    Args:
        quantiles: Quantiles dictionary
        file_path: Path to save quantiles
    """
    try:
        import json
        
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(quantiles, f, indent=2)
        
        logger.info(f"Quantiles saved to {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving quantiles: {e}")
        raise
