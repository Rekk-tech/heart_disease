"""
Feature engineering utilities for heart disease prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


def create_age_bins(df: pd.DataFrame, 
                   age_col: str = 'age', 
                   bins: List[float] = [0, 35, 50, 65, 100],
                   labels: List[str] = ["young", "middle", "senior", "elderly"]) -> pd.DataFrame:
    """
    Create age bins from continuous age variable.
    
    Args:
        df: Input DataFrame
        age_col: Name of age column
        bins: Bin edges
        labels: Bin labels
        
    Returns:
        DataFrame with age bins
    """
    try:
        df = df.copy()
        df[f'{age_col}_bin'] = pd.cut(df[age_col], bins=bins, labels=labels, include_lowest=True)
        
        logger.info(f"Created age bins: {labels}")
        return df
        
    except Exception as e:
        logger.error(f"Error creating age bins: {e}")
        raise


def create_ratio_features(df: pd.DataFrame, 
                         ratio_pairs: List[List[str]]) -> pd.DataFrame:
    """
    Create ratio features from pairs of columns.
    
    Args:
        df: Input DataFrame
        ratio_pairs: List of column pairs to create ratios
        
    Returns:
        DataFrame with ratio features
    """
    try:
        df = df.copy()
        
        for col1, col2 in ratio_pairs:
            if col1 in df.columns and col2 in df.columns:
                # Avoid division by zero
                df[f'{col1}_{col2}_ratio'] = df[col1] / (df[col2] + 1e-8)
                logger.info(f"Created ratio feature: {col1}_{col2}_ratio")
        
        return df
        
    except Exception as e:
        logger.error(f"Error creating ratio features: {e}")
        raise


def create_polynomial_features(df: pd.DataFrame, 
                             numeric_columns: List[str], 
                             degree: int = 2,
                             include_bias: bool = False) -> pd.DataFrame:
    """
    Create polynomial features for numeric columns.
    
    Args:
        df: Input DataFrame
        numeric_columns: List of numeric column names
        degree: Degree of polynomial features
        include_bias: Whether to include bias term
        
    Returns:
        DataFrame with polynomial features
    """
    try:
        df = df.copy()
        
        # Select numeric columns that exist in DataFrame
        existing_numeric_cols = [col for col in numeric_columns if col in df.columns]
        
        if not existing_numeric_cols:
            logger.warning("No numeric columns found for polynomial features")
            return df
        
        # Create polynomial features
        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        poly_features = poly.fit_transform(df[existing_numeric_cols])
        
        # Get feature names
        feature_names = poly.get_feature_names_out(existing_numeric_cols)
        
        # Create DataFrame with polynomial features
        poly_df = pd.DataFrame(poly_features, columns=feature_names, index=df.index)
        
        # Add polynomial features to original DataFrame
        df = pd.concat([df, poly_df], axis=1)
        
        logger.info(f"Created polynomial features with degree {degree}")
        return df
        
    except Exception as e:
        logger.error(f"Error creating polynomial features: {e}")
        raise


def create_interaction_features(df: pd.DataFrame, 
                              feature_pairs: List[List[str]]) -> pd.DataFrame:
    """
    Create interaction features between pairs of columns.
    
    Args:
        df: Input DataFrame
        feature_pairs: List of column pairs for interaction
        
    Returns:
        DataFrame with interaction features
    """
    try:
        df = df.copy()
        
        for col1, col2 in feature_pairs:
            if col1 in df.columns and col2 in df.columns:
                df[f'{col1}_{col2}_interaction'] = df[col1] * df[col2]
                logger.info(f"Created interaction feature: {col1}_{col2}_interaction")
        
        return df
        
    except Exception as e:
        logger.error(f"Error creating interaction features: {e}")
        raise


def discretize_features(df: pd.DataFrame, 
                       numeric_columns: List[str], 
                       n_bins: int = 5,
                       strategy: str = 'quantile') -> pd.DataFrame:
    """
    Discretize numeric features into bins.
    
    Args:
        df: Input DataFrame
        numeric_columns: List of numeric column names
        n_bins: Number of bins
        strategy: Discretization strategy ('uniform', 'quantile', 'kmeans')
        
    Returns:
        DataFrame with discretized features
    """
    try:
        df = df.copy()
        
        # Select numeric columns that exist in DataFrame
        existing_numeric_cols = [col for col in numeric_columns if col in df.columns]
        
        if not existing_numeric_cols:
            logger.warning("No numeric columns found for discretization")
            return df
        
        # Create discretizer
        discretizer = KBinsDiscretizer(n_bins=n_bins, strategy=strategy, encode='ordinal')
        
        # Fit and transform
        discretized_features = discretizer.fit_transform(df[existing_numeric_cols])
        
        # Create new column names
        new_column_names = [f'{col}_bin' for col in existing_numeric_cols]
        
        # Add discretized features to DataFrame
        discretized_df = pd.DataFrame(discretized_features, 
                                    columns=new_column_names, 
                                    index=df.index)
        
        df = pd.concat([df, discretized_df], axis=1)
        
        logger.info(f"Discretized {len(existing_numeric_cols)} features into {n_bins} bins")
        return df
        
    except Exception as e:
        logger.error(f"Error discretizing features: {e}")
        raise


def select_features_univariate(X: pd.DataFrame, 
                              y: pd.Series, 
                              k: int = 10,
                              score_func=f_classif) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select features using univariate statistical tests.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        k: Number of features to select
        score_func: Scoring function
        
    Returns:
        Tuple of (selected_features_df, selected_feature_names)
    """
    try:
        # Select k best features
        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Create DataFrame with selected features
        X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        logger.info(f"Selected {len(selected_features)} features using univariate selection")
        return X_selected_df, selected_features
        
    except Exception as e:
        logger.error(f"Error in univariate feature selection: {e}")
        raise


def select_features_recursive(X: pd.DataFrame, 
                             y: pd.Series, 
                             estimator=None,
                             n_features_to_select: int = 10,
                             step: int = 1) -> Tuple[pd.DataFrame, List[str]]:
    """
    Select features using recursive feature elimination.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        estimator: Base estimator for RFE
        n_features_to_select: Number of features to select
        step: Step size for RFE
        
    Returns:
        Tuple of (selected_features_df, selected_feature_names)
    """
    try:
        if estimator is None:
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Recursive feature elimination
        selector = RFE(estimator=estimator, 
                      n_features_to_select=n_features_to_select, 
                      step=step)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        
        # Create DataFrame with selected features
        X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        logger.info(f"Selected {len(selected_features)} features using recursive elimination")
        return X_selected_df, selected_features
        
    except Exception as e:
        logger.error(f"Error in recursive feature selection: {e}")
        raise


def create_features(df: pd.DataFrame, 
                   config: Dict[str, Any]) -> pd.DataFrame:
    """
    Create all engineered features based on configuration.
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        
    Returns:
        DataFrame with engineered features
    """
    try:
        df = df.copy()
        feature_config = config.get('feature_engineering', {})
        
        # Create age bins if specified
        if feature_config.get('age_bins'):
            df = create_age_bins(df)
        
        # Create ratio features if specified
        if feature_config.get('create_ratios', False):
            ratio_pairs = [
                ['chol', 'age'],  # cholesterol to age ratio
                ['trestbps', 'thalach'],  # blood pressure to max heart rate ratio
                ['oldpeak', 'age']  # ST depression to age ratio
            ]
            df = create_ratio_features(df, ratio_pairs)
        
        # Create polynomial features if specified
        if feature_config.get('polynomial_features', {}).get('degree'):
            degree = feature_config['polynomial_features']['degree']
            numeric_cols = config['features']['numeric']
            df = create_polynomial_features(df, numeric_cols, degree=degree)
        
        # Create interaction features if specified
        if feature_config.get('interaction_features'):
            interaction_pairs = [
                ['age', 'chol'],
                ['trestbps', 'thalach'],
                ['age', 'oldpeak']
            ]
            df = create_interaction_features(df, interaction_pairs)
        
        logger.info(f"Feature engineering completed. New shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        raise


def select_features(df: pd.DataFrame, 
                   config: Dict[str, Any]) -> pd.DataFrame:
    """
    Select features based on configuration.
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        
    Returns:
        DataFrame with selected features
    """
    try:
        if not config.get('feature_engineering', {}).get('feature_selection', False):
            return df
        
        # Separate features and target
        target_col = config['features']['target']
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        feature_selection_config = config['feature_selection']
        method = feature_selection_config.get('method', 'univariate')
        
        if method == 'univariate':
            k = feature_selection_config.get('n_features_to_select', 10)
            X_selected, selected_features = select_features_univariate(X, y, k=k)
        elif method == 'recursive':
            n_features = feature_selection_config.get('n_features_to_select', 10)
            X_selected, selected_features = select_features_recursive(X, y, n_features_to_select=n_features)
        else:
            logger.warning(f"Unknown feature selection method: {method}")
            return df
        
        # Combine selected features with target
        result_df = pd.concat([X_selected, y], axis=1)
        
        logger.info(f"Feature selection completed. Selected {len(selected_features)} features")
        return result_df
        
    except Exception as e:
        logger.error(f"Error in feature selection: {e}")
        raise
