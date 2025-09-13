"""
Data loading and validation utilities for heart disease prediction.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


def load_data(file_path: str, config_path: str = "conf/config.yaml") -> pd.DataFrame:
    """
    Load heart disease dataset from CSV file.
    
    Args:
        file_path: Path to the CSV file
        config_path: Path to configuration file
        
    Returns:
        DataFrame containing the heart disease data
    """
    try:
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load data
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded data from {file_path}")
        logger.info(f"Data shape: {df.shape}")
        
        return df
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def validate_schema(df: pd.DataFrame, config_path: str = "conf/config.yaml") -> bool:
    """
    Validate dataset schema against expected columns and data types.
    
    Args:
        df: DataFrame to validate
        config_path: Path to configuration file
        
    Returns:
        True if validation passes, False otherwise
    """
    try:
        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Expected columns
        expected_columns = set(config['features']['numeric'] + 
                             config['features']['categorical'] + 
                             [config['features']['target']])
        
        # Check if all expected columns exist
        actual_columns = set(df.columns)
        missing_columns = expected_columns - actual_columns
        extra_columns = actual_columns - expected_columns
        
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
            return False
            
        if extra_columns:
            logger.info(f"Extra columns found: {extra_columns}")
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.sum() > 0:
            logger.warning(f"Missing values found:\n{missing_counts[missing_counts > 0]}")
        
        # Check target variable distribution
        target_col = config['features']['target']
        if target_col in df.columns:
            target_dist = df[target_col].value_counts()
            logger.info(f"Target distribution:\n{target_dist}")
            
            # Check for class imbalance
            min_class_count = target_dist.min()
            max_class_count = target_dist.max()
            imbalance_ratio = max_class_count / min_class_count
            
            if imbalance_ratio > 3:
                logger.warning(f"Class imbalance detected. Ratio: {imbalance_ratio:.2f}")
        
        logger.info("Schema validation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Schema validation failed: {e}")
        return False


def get_data_info(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get comprehensive information about the dataset.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary containing dataset information
    """
    info = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'memory_usage': df.memory_usage(deep=True).sum(),
        'duplicate_rows': df.duplicated().sum()
    }
    
    # Add target distribution if target column exists
    target_col = 'target'
    if target_col in df.columns:
        info['target_distribution'] = df[target_col].value_counts().to_dict()
        info['target_percentage'] = (df[target_col].value_counts(normalize=True) * 100).to_dict()
    
    return info


def detect_outliers(df: pd.DataFrame, numeric_columns: list, method: str = 'iqr') -> Dict[str, list]:
    """
    Detect outliers in numeric columns.
    
    Args:
        df: DataFrame to analyze
        numeric_columns: List of numeric column names
        method: Method for outlier detection ('iqr' or 'zscore')
        
    Returns:
        Dictionary with outlier indices for each column
    """
    outliers = {}
    
    for col in numeric_columns:
        if col not in df.columns:
            continue
            
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outlier_mask = z_scores > 3
            
        outliers[col] = df[outlier_mask].index.tolist()
    
    return outliers


def load_config(config_path: str = "conf/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        raise
