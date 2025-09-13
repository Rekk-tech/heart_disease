"""
Main pipeline script for heart disease prediction project.

This script orchestrates the complete machine learning pipeline:
1. Data loading and validation
2. Data preprocessing and feature engineering
3. Model training and evaluation
4. Model explanation and deployment preparation
"""

import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import project modules
from data_loader import load_data, validate_schema, get_data_info
from preprocessing import split_data, create_preprocessor, preprocess_data, save_preprocessor, calculate_quantiles, save_quantiles
from feature_engineering import create_features, select_features
from modeling import train_models, evaluate_models, create_ensemble, save_model, cross_validate_model
from evaluation import calculate_metrics, evaluate_model_comprehensive, create_evaluation_summary
from explain import explain_model
from persistence import save_model, save_preprocessor, save_quantiles, save_model_card, create_model_package
from utils import setup_logging, set_seed, load_config, print_section_header, print_separator

# Setup logging
logger = setup_logging(log_file="logs/pipeline.log")
set_seed(42)


def load_and_validate_data(config_path: str = "conf/config.yaml"):
    """
    Load and validate the heart disease dataset.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        DataFrame: Loaded and validated dataset
    """
    print_section_header("Loading and Validating Data")
    
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Load data
        data_path = config['data']['raw']
        logger.info(f"Loading data from {data_path}")
        df = load_data(data_path, config_path)
        
        # Validate schema
        logger.info("Validating data schema...")
        is_valid = validate_schema(df, config_path)
        if not is_valid:
            logger.warning("Data validation failed, but continuing...")
        else:
            logger.info("Data validation passed")
        
        # Get data information
        data_info = get_data_info(df)
        logger.info(f"Dataset shape: {data_info['shape']}")
        logger.info(f"Memory usage: {data_info['memory_usage'] / 1024**2:.2f} MB")
        
        print_separator()
        return df, config
        
    except Exception as e:
        logger.error(f"Error loading and validating data: {e}")
        raise


def preprocess_data_pipeline(df: pd.DataFrame, config: dict):
    """
    Preprocess data and create train/validation/test splits.
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_df, val_df, test_df, preprocessor)
    """
    print_section_header("Data Preprocessing")
    
    try:
        # Create feature engineering
        logger.info("Creating engineered features...")
        df_engineered = create_features(df, config)
        
        # Feature selection (if enabled)
        if config.get('feature_engineering', {}).get('feature_selection', False):
            logger.info("Selecting features...")
            df_engineered = select_features(df_engineered, config)
        
        # Split data
        logger.info("Splitting data into train/validation/test sets...")
        train_df, val_df, test_df = split_data(df_engineered, config)
        
        # Create preprocessor
        logger.info("Creating preprocessing pipeline...")
        preprocessor = create_preprocessor(config)
        
        # Fit preprocessor on training data
        logger.info("Fitting preprocessor on training data...")
        train_processed = preprocess_data(train_df, preprocessor, fit=True)
        
        # Transform validation and test data
        logger.info("Transforming validation and test data...")
        val_processed = preprocess_data(val_df, preprocessor, fit=False)
        test_processed = preprocess_data(test_df, preprocessor, fit=False)
        
        # Calculate quantiles for numeric features
        logger.info("Calculating quantiles for numeric features...")
        numeric_features = config['features']['numeric']
        quantiles = calculate_quantiles(train_processed, numeric_features)
        
        # Save preprocessor and quantiles
        logger.info("Saving preprocessor and quantiles...")
        save_preprocessor(preprocessor, config['models']['preprocessor'])
        save_quantiles(quantiles, config['models']['quantiles'])
        
        logger.info("Data preprocessing completed successfully")
        print_separator()
        
        return train_processed, val_processed, test_processed, preprocessor
        
    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}")
        raise


def train_models_pipeline(train_df: pd.DataFrame, val_df: pd.DataFrame, config: dict):
    """
    Train multiple models and create ensemble.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        config: Configuration dictionary
        
    Returns:
        Dictionary of trained models and ensemble
    """
    print_section_header("Model Training")
    
    try:
        # Separate features and target
        target_col = config['features']['target']
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_val = val_df.drop(columns=[target_col])
        y_val = val_df[target_col]
        
        # Train individual models
        logger.info("Training individual models...")
        models = train_models(X_train, y_train, config)
        
        # Evaluate models on validation set
        logger.info("Evaluating models on validation set...")
        validation_results = evaluate_models(models, X_val, y_val)
        
        # Print validation results
        logger.info("Validation Results:")
        for model_name, metrics in validation_results.items():
            logger.info(f"{model_name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")
        
        # Create ensemble from best models
        logger.info("Creating ensemble model...")
        ensemble_models = [
            ('naive_bayes', models['naive_bayes']),
            ('knn', models['knn']),
            ('decision_tree', models['decision_tree'])
        ]
        ensemble = create_ensemble(ensemble_models, voting='soft')
        ensemble.fit(X_train, y_train)
        
        # Evaluate ensemble
        ensemble_results = evaluate_models({'ensemble': ensemble}, X_val, y_val)
        logger.info(f"Ensemble Results: {ensemble_results['ensemble']}")
        
        # Save models
        logger.info("Saving trained models...")
        save_model(ensemble, config['models']['model'])
        
        # Add ensemble to models dictionary
        models['ensemble'] = ensemble
        
        logger.info("Model training completed successfully")
        print_separator()
        
        return models, validation_results, ensemble_results
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise


def evaluate_models_pipeline(models: dict, test_df: pd.DataFrame, config: dict):
    """
    Evaluate models on test set and generate comprehensive reports.
    
    Args:
        models: Dictionary of trained models
        test_df: Test DataFrame
        config: Configuration dictionary
        
    Returns:
        Dictionary of evaluation results
    """
    print_section_header("Model Evaluation")
    
    try:
        # Separate features and target
        target_col = config['features']['target']
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        
        # Evaluate all models
        logger.info("Evaluating models on test set...")
        test_results = {}
        
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}...")
            results = evaluate_model_comprehensive(
                model, X_test, y_test, 
                model_name=model_name,
                save_plots=True,
                plots_dir="plots"
            )
            test_results[model_name] = results
        
        # Create evaluation summary
        logger.info("Creating evaluation summary...")
        create_evaluation_summary(test_results, "evaluation_summary.txt")
        
        # Print summary results
        logger.info("Test Set Results Summary:")
        for model_name, results in test_results.items():
            metrics = results['metrics']
            logger.info(f"{model_name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, ROC-AUC={metrics['roc_auc']:.4f}")
        
        logger.info("Model evaluation completed successfully")
        print_separator()
        
        return test_results
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        raise


def explain_models_pipeline(models: dict, test_df: pd.DataFrame, config: dict):
    """
    Generate model explanations and interpretability reports.
    
    Args:
        models: Dictionary of trained models
        test_df: Test DataFrame
        config: Configuration dictionary
        
    Returns:
        Dictionary of explanation results
    """
    print_section_header("Model Explanation")
    
    try:
        # Separate features and target
        target_col = config['features']['target']
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        
        # Explain best performing model (ensemble)
        best_model = models['ensemble']
        logger.info("Generating explanations for ensemble model...")
        
        explanation_results = explain_model(
            best_model, X_test, y_test,
            explanation_type='all',
            save_plots=True,
            plots_dir="plots"
        )
        
        # Create explanation report
        logger.info("Creating explanation report...")
        from explain import create_explanation_report
        create_explanation_report(explanation_results, "ensemble_model", "explanation_report.md")
        
        logger.info("Model explanation completed successfully")
        print_separator()
        
        return explanation_results
        
    except Exception as e:
        logger.error(f"Error in model explanation: {e}")
        raise


def create_model_package_pipeline(models: dict, preprocessor, config: dict, evaluation_results: dict):
    """
    Create a complete model package for deployment.
    
    Args:
        models: Dictionary of trained models
        preprocessor: Fitted preprocessor
        config: Configuration dictionary
        evaluation_results: Model evaluation results
        
    Returns:
        None
    """
    print_section_header("Creating Model Package")
    
    try:
        # Get best model (ensemble)
        best_model = models['ensemble']
        
        # Create model card
        model_card = {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'purpose': 'Predict heart disease risk from medical features',
            'algorithm': type(best_model).__name__,
            'performance': evaluation_results.get('ensemble', {}).get('metrics', {}),
            'training_data_size': len(models),
            'n_features': len(config['features']['numeric']) + len(config['features']['categorical']),
            'target_classes': '0: No Disease, 1: Disease',
            'cv_folds': config.get('evaluation', {}).get('cv_folds', 5),
            'limitations': [
                "Model trained on specific dataset and may not generalize to other populations",
                "Clinical decisions should not be based solely on model predictions",
                "Model requires complete feature information for accurate predictions"
            ]
        }
        
        # Create model package
        logger.info("Creating model package...")
        create_model_package(
            model=best_model,
            preprocessor=preprocessor,
            config=config,
            evaluation_results=evaluation_results,
            package_dir="model_package"
        )
        
        logger.info("Model package created successfully")
        print_separator()
        
    except Exception as e:
        logger.error(f"Error creating model package: {e}")
        raise


def main():
    """
    Main pipeline execution function.
    """
    try:
        print_section_header("Heart Disease Prediction Pipeline")
        logger.info("Starting Heart Disease Prediction Pipeline")
        
        # Step 1: Load and validate data
        df, config = load_and_validate_data()
        
        # Step 2: Preprocess data
        train_df, val_df, test_df, preprocessor = preprocess_data_pipeline(df, config)
        
        # Step 3: Train models
        models, validation_results, ensemble_results = train_models_pipeline(train_df, val_df, config)
        
        # Step 4: Evaluate models
        test_results = evaluate_models_pipeline(models, test_df, config)
        
        # Step 5: Explain models
        explanation_results = explain_models_pipeline(models, test_df, config)
        
        # Step 6: Create model package
        create_model_package_pipeline(models, preprocessor, config, test_results)
        
        # Final summary
        print_section_header("Pipeline Completed Successfully")
        logger.info("Heart Disease Prediction Pipeline completed successfully!")
        
        # Print final results
        best_model_name = 'ensemble'
        if best_model_name in test_results:
            best_metrics = test_results[best_model_name]['metrics']
            logger.info(f"Best Model ({best_model_name}):")
            logger.info(f"  Accuracy: {best_metrics['accuracy']:.4f}")
            logger.info(f"  F1 Score: {best_metrics['f1']:.4f}")
            logger.info(f"  ROC-AUC: {best_metrics['roc_auc']:.4f}")
        
        print_separator()
        logger.info("Pipeline execution completed!")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
