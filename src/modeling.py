"""
Modeling utilities for heart disease prediction.
"""

import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, Any, List, Tuple, Optional
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def train_naive_bayes(X_train: pd.DataFrame, 
                     y_train: pd.Series, 
                     params: Dict[str, Any] = None) -> GaussianNB:
    """
    Train Naive Bayes classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        params: Hyperparameters
        
    Returns:
        Trained Naive Bayes model
    """
    try:
        if params is None:
            params = {'var_smoothing': 1e-9}
        
        model = GaussianNB(**params)
        model.fit(X_train, y_train)
        
        logger.info("Naive Bayes model trained successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error training Naive Bayes: {e}")
        raise


def train_knn(X_train: pd.DataFrame, 
              y_train: pd.Series, 
              params: Dict[str, Any] = None) -> KNeighborsClassifier:
    """
    Train K-Nearest Neighbors classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        params: Hyperparameters
        
    Returns:
        Trained KNN model
    """
    try:
        if params is None:
            params = {
                'n_neighbors': 5,
                'weights': 'distance',
                'metric': 'euclidean'
            }
        
        model = KNeighborsClassifier(**params)
        model.fit(X_train, y_train)
        
        logger.info("KNN model trained successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error training KNN: {e}")
        raise


def train_decision_tree(X_train: pd.DataFrame, 
                       y_train: pd.Series, 
                       params: Dict[str, Any] = None) -> DecisionTreeClassifier:
    """
    Train Decision Tree classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        params: Hyperparameters
        
    Returns:
        Trained Decision Tree model
    """
    try:
        if params is None:
            params = {
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }
        
        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)
        
        logger.info("Decision Tree model trained successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error training Decision Tree: {e}")
        raise


def train_random_forest(X_train: pd.DataFrame, 
                       y_train: pd.Series, 
                       params: Dict[str, Any] = None) -> RandomForestClassifier:
    """
    Train Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        params: Hyperparameters
        
    Returns:
        Trained Random Forest model
    """
    try:
        if params is None:
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }
        
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        logger.info("Random Forest model trained successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error training Random Forest: {e}")
        raise


def train_gradient_boosting(X_train: pd.DataFrame, 
                           y_train: pd.Series, 
                           params: Dict[str, Any] = None) -> GradientBoostingClassifier:
    """
    Train Gradient Boosting classifier.
    
    Args:
        X_train: Training features
        y_train: Training target
        params: Hyperparameters
        
    Returns:
        Trained Gradient Boosting model
    """
    try:
        if params is None:
            params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3,
                'random_state': 42
            }
        
        model = GradientBoostingClassifier(**params)
        model.fit(X_train, y_train)
        
        logger.info("Gradient Boosting model trained successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error training Gradient Boosting: {e}")
        raise


def create_ensemble(models: List[Tuple[str, Any]], 
                   voting: str = 'soft') -> VotingClassifier:
    """
    Create ensemble model using VotingClassifier.
    
    Args:
        models: List of (name, model) tuples
        voting: Voting strategy ('hard' or 'soft')
        
    Returns:
        Ensemble VotingClassifier
    """
    try:
        ensemble = VotingClassifier(
            estimators=models,
            voting=voting,
            n_jobs=-1
        )
        
        logger.info(f"Ensemble model created with {len(models)} base models")
        return ensemble
        
    except Exception as e:
        logger.error(f"Error creating ensemble: {e}")
        raise


def train_models(X_train: pd.DataFrame, 
                y_train: pd.Series, 
                config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train multiple models based on configuration.
    
    Args:
        X_train: Training features
        y_train: Training target
        config: Configuration dictionary
        
    Returns:
        Dictionary of trained models
    """
    try:
        models = {}
        model_configs = config.get('models', {})
        
        # Train Naive Bayes
        if 'naive_bayes' in model_configs:
            nb_params = model_configs['naive_bayes']
            models['naive_bayes'] = train_naive_bayes(X_train, y_train, nb_params)
        
        # Train KNN
        if 'knn' in model_configs:
            knn_params = model_configs['knn']
            models['knn'] = train_knn(X_train, y_train, knn_params)
        
        # Train Decision Tree
        if 'decision_tree' in model_configs:
            dt_params = model_configs['decision_tree']
            models['decision_tree'] = train_decision_tree(X_train, y_train, dt_params)
        
        # Train Random Forest
        if 'random_forest' in model_configs:
            rf_params = model_configs['random_forest']
            models['random_forest'] = train_random_forest(X_train, y_train, rf_params)
        
        # Train Gradient Boosting
        if 'gradient_boosting' in model_configs:
            gb_params = model_configs['gradient_boosting']
            models['gradient_boosting'] = train_gradient_boosting(X_train, y_train, gb_params)
        
        logger.info(f"Trained {len(models)} models successfully")
        return models
        
    except Exception as e:
        logger.error(f"Error training models: {e}")
        raise


def hyperparameter_tuning(model, 
                         param_grid: Dict[str, List], 
                         X_train: pd.DataFrame, 
                         y_train: pd.Series,
                         cv: int = 5,
                         scoring: str = 'roc_auc') -> Any:
    """
    Perform hyperparameter tuning using GridSearchCV.
    
    Args:
        model: Base model for tuning
        param_grid: Parameter grid
        X_train: Training features
        y_train: Training target
        cv: Number of CV folds
        scoring: Scoring metric
        
    Returns:
        Best model from grid search
    """
    try:
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
        
    except Exception as e:
        logger.error(f"Error in hyperparameter tuning: {e}")
        raise


def evaluate_models(models: Dict[str, Any], 
                   X_val: pd.DataFrame, 
                   y_val: pd.Series) -> Dict[str, Dict[str, float]]:
    """
    Evaluate multiple models on validation set.
    
    Args:
        models: Dictionary of trained models
        X_val: Validation features
        y_val: Validation target
        
    Returns:
        Dictionary of evaluation metrics for each model
    """
    try:
        results = {}
        
        for name, model in models.items():
            # Make predictions
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_val, y_pred),
                'precision': precision_score(y_val, y_pred),
                'recall': recall_score(y_val, y_pred),
                'f1': f1_score(y_val, y_pred)
            }
            
            # Add ROC-AUC if probabilities are available
            if y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_val, y_pred_proba)
            
            results[name] = metrics
            
            logger.info(f"{name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating models: {e}")
        raise


def cross_validate_model(model, 
                        X: pd.DataFrame, 
                        y: pd.Series, 
                        cv: int = 5,
                        scoring: str = 'roc_auc') -> Dict[str, float]:
    """
    Perform cross-validation on a model.
    
    Args:
        model: Model to validate
        X: Features
        y: Target
        cv: Number of CV folds
        scoring: Scoring metric
        
    Returns:
        Dictionary with CV results
    """
    try:
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        results = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores.tolist()
        }
        
        logger.info(f"CV {scoring}: {results['mean_score']:.4f} (+/- {results['std_score']*2:.4f})")
        return results
        
    except Exception as e:
        logger.error(f"Error in cross-validation: {e}")
        raise


def save_model(model: Any, 
               file_path: str = "models/heart_model.pkl") -> None:
    """
    Save trained model to file.
    
    Args:
        model: Trained model
        file_path: Path to save model
    """
    try:
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model, file_path)
        logger.info(f"Model saved to {file_path}")
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise


def load_model(file_path: str = "models/heart_model.pkl") -> Any:
    """
    Load model from file.
    
    Args:
        file_path: Path to load model from
        
    Returns:
        Loaded model
    """
    try:
        model = joblib.load(file_path)
        logger.info(f"Model loaded from {file_path}")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
