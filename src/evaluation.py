"""
Evaluation utilities for heart disease prediction models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score
)
from typing import Dict, Any, List, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def calculate_metrics(y_true: np.ndarray, 
                     y_pred: np.ndarray, 
                     y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        
    Returns:
        Dictionary of evaluation metrics
    """
    try:
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }
        
        # Add ROC-AUC if probabilities are available
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        raise


def plot_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         labels: List[str] = None,
                         title: str = "Confusion Matrix",
                         save_path: Optional[str] = None) -> None:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        title: Plot title
        save_path: Path to save the plot
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {e}")
        raise


def plot_roc_curve(y_true: np.ndarray, 
                  y_pred_proba: np.ndarray,
                  title: str = "ROC Curve",
                  save_path: Optional[str] = None) -> None:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        title: Plot title
        save_path: Path to save the plot
    """
    try:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve saved to {save_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Error plotting ROC curve: {e}")
        raise


def plot_precision_recall_curve(y_true: np.ndarray, 
                               y_pred_proba: np.ndarray,
                               title: str = "Precision-Recall Curve",
                               save_path: Optional[str] = None) -> None:
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        title: Plot title
        save_path: Path to save the plot
    """
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.2f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title)
        plt.legend(loc="lower left")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curve saved to {save_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Error plotting precision-recall curve: {e}")
        raise


def plot_model_comparison(results: Dict[str, Dict[str, float]], 
                         metric: str = 'f1',
                         title: str = "Model Comparison",
                         save_path: Optional[str] = None) -> None:
    """
    Plot model comparison chart.
    
    Args:
        results: Dictionary of model results
        metric: Metric to compare
        title: Plot title
        save_path: Path to save the plot
    """
    try:
        models = list(results.keys())
        scores = [results[model][metric] for model in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, scores, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.title(title)
        plt.xlabel('Models')
        plt.ylabel(metric.upper())
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Error plotting model comparison: {e}")
        raise


def plot_metrics_heatmap(results: Dict[str, Dict[str, float]],
                        title: str = "Model Metrics Heatmap",
                        save_path: Optional[str] = None) -> None:
    """
    Plot metrics heatmap for all models.
    
    Args:
        results: Dictionary of model results
        title: Plot title
        save_path: Path to save the plot
    """
    try:
        # Convert results to DataFrame
        df_results = pd.DataFrame(results).T
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(df_results, annot=True, cmap='YlOrRd', 
                   fmt='.3f', cbar_kws={'label': 'Score'})
        plt.title(title)
        plt.xlabel('Metrics')
        plt.ylabel('Models')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics heatmap saved to {save_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Error plotting metrics heatmap: {e}")
        raise


def generate_classification_report(y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 labels: List[str] = None) -> str:
    """
    Generate detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Class labels
        
    Returns:
        Classification report as string
    """
    try:
        report = classification_report(y_true, y_pred, target_names=labels)
        logger.info("Classification report generated")
        return report
        
    except Exception as e:
        logger.error(f"Error generating classification report: {e}")
        raise


def plot_feature_importance_plot(importance_scores: Dict[str, float],
                               title: str = "Feature Importance",
                               top_n: int = 15,
                               save_path: Optional[str] = None) -> None:
    """
    Plot feature importance.
    
    Args:
        importance_scores: Dictionary of feature importance scores
        title: Plot title
        top_n: Number of top features to show
        save_path: Path to save the plot
    """
    try:
        # Sort features by importance
        sorted_features = sorted(importance_scores.items(), 
                               key=lambda x: x[1], reverse=True)[:top_n]
        
        features, scores = zip(*sorted_features)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(features)), scores, color='lightcoral', alpha=0.7)
        
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title(title)
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', va='center')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Error plotting feature importance: {e}")
        raise


def evaluate_model_comprehensive(model: Any, 
                               X_test: pd.DataFrame, 
                               y_test: pd.Series,
                               model_name: str = "Model",
                               save_plots: bool = True,
                               plots_dir: str = "plots") -> Dict[str, Any]:
    """
    Perform comprehensive model evaluation.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name: Name of the model
        save_plots: Whether to save plots
        plots_dir: Directory to save plots
        
    Returns:
        Dictionary with evaluation results
    """
    try:
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Generate classification report
        report = generate_classification_report(y_test, y_pred)
        
        # Create plots directory if saving plots
        if save_plots:
            Path(plots_dir).mkdir(parents=True, exist_ok=True)
        
        # Plot confusion matrix
        if save_plots:
            cm_path = f"{plots_dir}/{model_name.lower()}_confusion_matrix.png"
            plot_confusion_matrix(y_test, y_pred, save_path=cm_path)
        
        # Plot ROC curve
        if y_pred_proba is not None and save_plots:
            roc_path = f"{plots_dir}/{model_name.lower()}_roc_curve.png"
            plot_roc_curve(y_test, y_pred_proba, save_path=roc_path)
            
            pr_path = f"{plots_dir}/{model_name.lower()}_precision_recall.png"
            plot_precision_recall_curve(y_test, y_pred_proba, save_path=pr_path)
        
        results = {
            'metrics': metrics,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        logger.info(f"Comprehensive evaluation completed for {model_name}")
        return results
        
    except Exception as e:
        logger.error(f"Error in comprehensive evaluation: {e}")
        raise


def create_evaluation_summary(all_results: Dict[str, Dict[str, Any]],
                            save_path: str = "evaluation_summary.txt") -> None:
    """
    Create a comprehensive evaluation summary.
    
    Args:
        all_results: Dictionary with results for all models
        save_path: Path to save the summary
    """
    try:
        with open(save_path, 'w') as f:
            f.write("HEART DISEASE PREDICTION - MODEL EVALUATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            for model_name, results in all_results.items():
                f.write(f"MODEL: {model_name.upper()}\n")
                f.write("-" * 30 + "\n")
                
                # Write metrics
                metrics = results['metrics']
                for metric, value in metrics.items():
                    f.write(f"{metric.upper()}: {value:.4f}\n")
                
                f.write("\nCLASSIFICATION REPORT:\n")
                f.write(results['classification_report'])
                f.write("\n" + "=" * 60 + "\n\n")
        
        logger.info(f"Evaluation summary saved to {save_path}")
        
    except Exception as e:
        logger.error(f"Error creating evaluation summary: {e}")
        raise
