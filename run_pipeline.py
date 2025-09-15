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
import os

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Import project modules
try:
    from data_loader import load_data, validate_schema, get_data_info
    from preprocessing import (split_data, create_preprocessor, preprocess_data, 
                              save_preprocessor, calculate_quantiles, save_quantiles)
    from feature_engineering import create_features, select_features
    from modeling import train_models, evaluate_models, create_ensemble, save_model, cross_validate_model
    from evaluation import calculate_metrics, evaluate_model_comprehensive, create_evaluation_summary
    from explain import explain_model, create_explanation_report
    from persistence import save_model, save_preprocessor, save_quantiles, save_model_card, create_model_package
    from utils import setup_logging, set_seed, load_config, print_section_header, print_separator
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all required modules are available in the src directory")
    sys.exit(1)

# Setup logging and seed
logger = None  # Will be initialized in main()


def initialize_directories():
    """Create necessary directories if they don't exist."""
    directories = ["logs", "plots", "models", "model_package", "data/processed"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def load_and_validate_data(config_path: str = "conf/config.yaml"):
    """
    Load and validate the heart disease dataset.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Tuple: (DataFrame, config_dict)
    """
    print_section_header("Loading and Validating Data")
    
    try:
        # Load configuration
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        config = load_config(config_path)
        
        # Validate config structure
        required_keys = ['data', 'features', 'models']
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Missing required config key: {key}")
        
        # Load data
        data_path = config['data']['raw']
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        logger.info(f"Loading data from {data_path}")
        df = load_data(data_path, config_path)
        
        if df.empty:
            raise ValueError("Loaded dataset is empty")
        
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
        
        # Check for target column
        target_col = config['features']['target']
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
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
        Tuple: (train_df, val_df, test_df, preprocessor)
    """
    print_section_header("Data Preprocessing")
    
    try:
        # Create feature engineering
        logger.info("Creating engineered features...")
        df_engineered = create_features(df, config)
        
        # Validate engineered features
        if df_engineered.empty:
            raise ValueError("Feature engineering resulted in empty dataset")
        
        # Split data
        logger.info("Splitting data into train/validation/test sets...")
        train_df, val_df, test_df = split_data(df_engineered, config)
        
        # Validate splits
        for name, split_df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
            if split_df.empty:
                raise ValueError(f"{name} split is empty")
        
        # Create preprocessor
        logger.info("Creating preprocessing pipeline...")
        preprocessor = create_preprocessor(config)
        
        # Separate features and target
        target_col = config['features']['target']
        
        # Fit preprocessor on training data
        logger.info("Fitting preprocessor on training data...")
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        
        # Fit and transform training data
        X_train_processed = preprocessor.fit_transform(X_train)
        
        # Transform validation and test data
        logger.info("Transforming validation and test data...")
        X_val = val_df.drop(columns=[target_col])
        X_test = test_df.drop(columns=[target_col])
        
        X_val_processed = preprocessor.transform(X_val)
        X_test_processed = preprocessor.transform(X_test)
        
        # Convert back to DataFrames with proper column names
        feature_names = preprocessor.get_feature_names_out() if hasattr(preprocessor, 'get_feature_names_out') else X_train.columns
        
        train_processed = pd.DataFrame(X_train_processed, columns=feature_names, index=train_df.index)
        val_processed = pd.DataFrame(X_val_processed, columns=feature_names, index=val_df.index)
        test_processed = pd.DataFrame(X_test_processed, columns=feature_names, index=test_df.index)
        
        # Add target back
        train_processed[target_col] = y_train
        val_processed[target_col] = val_df[target_col]
        test_processed[target_col] = test_df[target_col]
        
        # Feature selection (if enabled)
        if config.get('feature_engineering', {}).get('feature_selection', False):
            logger.info("Selecting features...")
            train_processed = select_features(train_processed, config)
            # Align val and test with selected features
            selected_features = train_processed.columns.tolist()
            val_processed = val_processed[selected_features]
            test_processed = test_processed[selected_features]
        
        # Calculate quantiles for numeric features (if specified)
        if 'numeric' in config['features']:
            logger.info("Calculating quantiles for numeric features...")
            numeric_features = [col for col in config['features']['numeric'] if col in train_processed.columns]
            if numeric_features:
                quantiles = calculate_quantiles(train_processed[numeric_features])
                
                # Save quantiles
                quantiles_path = config['models'].get('quantiles', 'models/quantiles.pkl')
                save_quantiles(quantiles, quantiles_path)
        
        # Save preprocessor
        logger.info("Saving preprocessor...")
        preprocessor_path = config['models'].get('preprocessor', 'models/preprocessor.pkl')
        save_preprocessor(preprocessor, preprocessor_path)
        
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
        Tuple: (models_dict, validation_results, ensemble_results)
    """
    print_section_header("Model Training")
    
    try:
        # Separate features and target
        target_col = config['features']['target']
        X_train = train_df.drop(columns=[target_col])
        y_train = train_df[target_col]
        X_val = val_df.drop(columns=[target_col])
        y_val = val_df[target_col]
        
        # Ensure consistent feature columns
        common_features = X_train.columns.intersection(X_val.columns)
        X_train = X_train[common_features]
        X_val = X_val[common_features]
        
        logger.info(f"Training with {len(common_features)} features")
        
        # Train individual models
        logger.info("Training individual models...")
        models = train_models(X_train, y_train, config)
        
        if not models:
            raise ValueError("No models were trained successfully")
        
        # Evaluate models on validation set
        logger.info("Evaluating models on validation set...")
        validation_results = {}
        
        for model_name, model in models.items():
            try:
                result = evaluate_models({model_name: model}, X_val, y_val)
                validation_results[model_name] = result[model_name]
                logger.info(f"{model_name}: Accuracy={result[model_name]['accuracy']:.4f}, "
                          f"F1={result[model_name]['f1']:.4f}, ROC-AUC={result[model_name]['roc_auc']:.4f}")
            except Exception as e:
                logger.warning(f"Failed to evaluate {model_name}: {e}")
                continue
        
        # Create ensemble from available models
        logger.info("Creating ensemble model...")
        ensemble_models = [(name, model) for name, model in models.items() 
                          if name in validation_results]
        
        if len(ensemble_models) < 2:
            logger.warning("Not enough models for ensemble, using best single model")
            best_model_name = max(validation_results.keys(), 
                                key=lambda x: validation_results[x]['accuracy'])
            ensemble = models[best_model_name]
        else:
            ensemble = create_ensemble(ensemble_models, voting='soft')
            ensemble.fit(X_train, y_train)
        
        # Evaluate ensemble
        ensemble_results = evaluate_models({'ensemble': ensemble}, X_val, y_val)
        logger.info(f"Ensemble Results: {ensemble_results['ensemble']}")
        
        # Save best model
        logger.info("Saving trained models...")
        model_path = config['models'].get('model', 'models/best_model.pkl')
        save_model(ensemble, model_path)
        
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
        Dict: evaluation results
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
            try:
                logger.info(f"Evaluating {model_name}...")
                results = evaluate_model_comprehensive(
                    model, X_test, y_test, 
                    model_name=model_name,
                    save_plots=True,
                    plots_dir="plots"
                )
                test_results[model_name] = results
            except Exception as e:
                logger.warning(f"Failed to evaluate {model_name}: {e}")
                continue
        
        if not test_results:
            raise ValueError("No models were evaluated successfully")
        
        # Create evaluation summary
        logger.info("Creating evaluation summary...")
        create_evaluation_summary(test_results, "evaluation_summary.txt")
        
        # Print summary results
        logger.info("Test Set Results Summary:")
        for model_name, results in test_results.items():
            metrics = results.get('metrics', {})
            logger.info(f"{model_name}: Accuracy={metrics.get('accuracy', 'N/A'):.4f}, "
                      f"F1={metrics.get('f1', 'N/A'):.4f}, ROC-AUC={metrics.get('roc_auc', 'N/A'):.4f}")
        
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
        Dict: explanation results
    """
    print_section_header("Model Explanation")
    
    try:
        # Separate features and target
        target_col = config['features']['target']
        X_test = test_df.drop(columns=[target_col])
        y_test = test_df[target_col]
        
        # Use best available model
        best_model_name = 'ensemble' if 'ensemble' in models else list(models.keys())[0]
        best_model = models[best_model_name]
        
        logger.info(f"Generating explanations for {best_model_name} model...")
        
        explanation_results = explain_model(
            best_model, X_test, y_test,
            explanation_type='all',
            save_plots=True,
            plots_dir="plots"
        )
        
        # Create explanation report
        logger.info("Creating explanation report...")
        create_explanation_report(explanation_results, best_model_name, "explanation_report.md")
        
        logger.info("Model explanation completed successfully")
        print_separator()
        
        return explanation_results
        
    except Exception as e:
        logger.error(f"Error in model explanation: {e}")
        # Don't raise here, explanation is optional
        logger.warning("Continuing pipeline without explanations")
        return {}


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
        # Get best model
        best_model_name = 'ensemble' if 'ensemble' in models else list(models.keys())[0]
        best_model = models[best_model_name]
        
        # Create model card
        best_metrics = evaluation_results.get(best_model_name, {}).get('metrics', {})
        
        model_card = {
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'purpose': 'Predict heart disease risk from medical features',
            'algorithm': type(best_model).__name__,
            'performance': best_metrics,
            'training_data_size': len(models),
            'n_features': len(config.get('features', {}).get('numeric', [])) + 
                         len(config.get('features', {}).get('categorical', [])),
            'target_classes': '0: No Disease, 1: Disease',
            'cv_folds': config.get('evaluation', {}).get('cv_folds', 5),
            'limitations': [
                "Model trained on specific dataset and may not generalize to other populations",
                "Clinical decisions should not be based solely on model predictions",
                "Model requires complete feature information for accurate predictions"
            ]
        }
        
        # Save model card
        save_model_card(model_card, "model_package/model_card.json")
        
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
        # Don't raise here, packaging is optional
        logger.warning("Continuing without creating model package")


def main():
    """
    Main pipeline execution function.
    """
    global logger
    
    try:
        # Initialize directories
        initialize_directories()
        
        # Setup logging
        logger = setup_logging(log_file="logs/pipeline.log")
        set_seed(42)
        
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
        
        # Step 5: Explain models (optional)
        try:
            explanation_results = explain_models_pipeline(models, test_df, config)
        except Exception as e:
            logger.warning(f"Model explanation failed: {e}")
            explanation_results = {}
        
        # Step 6: Create model package (optional)
        try:
            create_model_package_pipeline(models, preprocessor, config, test_results)
        except Exception as e:
            logger.warning(f"Model packaging failed: {e}")
        
        # Final summary
        print_section_header("Pipeline Completed Successfully")
        logger.info("Heart Disease Prediction Pipeline completed successfully!")
        
        # Print final results
        best_model_name = 'ensemble' if 'ensemble' in test_results else list(test_results.keys())[0]
        if best_model_name in test_results:
            best_metrics = test_results[best_model_name].get('metrics', {})
            logger.info(f"Best Model ({best_model_name}):")
            logger.info(f"  Accuracy: {best_metrics.get('accuracy', 'N/A'):.4f}")
            logger.info(f"  F1 Score: {best_metrics.get('f1', 'N/A'):.4f}")
            logger.info(f"  ROC-AUC: {best_metrics.get('roc_auc', 'N/A'):.4f}")
        
        print_separator()
        logger.info("Pipeline execution completed!")
        
        return True
        
    except Exception as e:
        if logger:
            logger.error(f"Pipeline execution failed: {e}")
        else:
            print(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Pipeline failed with error: {e}")
        sys.exit(1)