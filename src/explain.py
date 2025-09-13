"""
Model explainability utilities for heart disease prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.tree import export_text
import shap
import lime
import lime.lime_tabular
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def plot_feature_importance(model: Any, 
                           feature_names: List[str],
                           title: str = "Feature Importance",
                           top_n: int = 15,
                           save_path: Optional[str] = None) -> None:
    """
    Plot feature importance from tree-based models.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        title: Plot title
        top_n: Number of top features to show
        save_path: Path to save the plot
    """
    try:
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        # Create plot
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(indices)), importances[indices], 
                       color='lightcoral', alpha=0.7)
        
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance Score')
        plt.title(title)
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, idx) in enumerate(zip(bars, indices)):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{importances[idx]:.3f}', va='center')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Error plotting feature importance: {e}")
        raise


def calculate_permutation_importance(model: Any, 
                                   X: pd.DataFrame, 
                                   y: pd.Series,
                                   n_repeats: int = 10,
                                   random_state: int = 42) -> Dict[str, float]:
    """
    Calculate permutation importance for model features.
    
    Args:
        model: Trained model
        X: Feature DataFrame
        y: Target Series
        n_repeats: Number of permutation repeats
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with feature importance scores
    """
    try:
        # Calculate permutation importance
        perm_importance = permutation_importance(
            model, X, y, 
            n_repeats=n_repeats, 
            random_state=random_state,
            scoring='roc_auc'
        )
        
        # Create importance dictionary
        importance_dict = {
            feature: score for feature, score in 
            zip(X.columns, perm_importance.importances_mean)
        }
        
        # Sort by importance
        importance_dict = dict(sorted(importance_dict.items(), 
                                    key=lambda x: x[1], reverse=True))
        
        logger.info(f"Permutation importance calculated for {len(importance_dict)} features")
        return importance_dict
        
    except Exception as e:
        logger.error(f"Error calculating permutation importance: {e}")
        raise


def plot_permutation_importance(importance_dict: Dict[str, float],
                               title: str = "Permutation Importance",
                               top_n: int = 15,
                               save_path: Optional[str] = None) -> None:
    """
    Plot permutation importance scores.
    
    Args:
        importance_dict: Dictionary of feature importance scores
        title: Plot title
        top_n: Number of top features to show
        save_path: Path to save the plot
    """
    try:
        # Get top N features
        top_features = list(importance_dict.items())[:top_n]
        features, scores = zip(*top_features)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(features)), scores, color='skyblue', alpha=0.7)
        
        plt.yticks(range(len(features)), features)
        plt.xlabel('Permutation Importance Score')
        plt.title(title)
        plt.gca().invert_yaxis()
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', va='center')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Permutation importance plot saved to {save_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Error plotting permutation importance: {e}")
        raise


def plot_partial_dependence(model: Any,
                           X: pd.DataFrame,
                           features: List[str],
                           feature_names: List[str] = None,
                           save_path: Optional[str] = None) -> None:
    """
    Plot partial dependence for selected features.
    
    Args:
        model: Trained model
        X: Feature DataFrame
        features: List of feature indices or names
        feature_names: List of feature names (if features are indices)
        save_path: Path to save the plot
    """
    try:
        n_features = len(features)
        
        # Create subplots
        fig, axes = plt.subplots(2, (n_features + 1) // 2, figsize=(15, 10))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for i, feature in enumerate(features):
            if i >= len(axes):
                break
                
            # Calculate partial dependence
            pd_result = partial_dependence(model, X, [feature])
            
            # Plot partial dependence
            axes[i].plot(pd_result['values'][0], pd_result['average'][0], 
                        linewidth=2, marker='o')
            axes[i].set_xlabel(feature_names[i] if feature_names else feature)
            axes[i].set_ylabel('Partial Dependence')
            axes[i].set_title(f'Partial Dependence: {feature}')
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Partial dependence plot saved to {save_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Error plotting partial dependence: {e}")
        raise


def explain_with_shap(model: Any,
                     X: pd.DataFrame,
                     feature_names: List[str] = None,
                     max_display: int = 10,
                     save_path: Optional[str] = None) -> None:
    """
    Explain model predictions using SHAP values.
    
    Args:
        model: Trained model
        X: Feature DataFrame
        feature_names: List of feature names
        max_display: Maximum number of features to display
        save_path: Path to save the plot
    """
    try:
        # Create SHAP explainer
        if hasattr(model, 'predict_proba'):
            explainer = shap.Explainer(model, X)
            shap_values = explainer(X.iloc[:100])  # Use subset for efficiency
            
            # Plot SHAP summary
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X.iloc[:100], 
                            feature_names=feature_names,
                            max_display=max_display, show=False)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"SHAP summary plot saved to {save_path}")
            
            plt.show()
            
        else:
            logger.warning("Model does not support SHAP explanation (no predict_proba method)")
            
    except Exception as e:
        logger.error(f"Error with SHAP explanation: {e}")
        raise


def explain_with_lime(model: Any,
                     X: pd.DataFrame,
                     y: pd.Series,
                     instance_idx: int,
                     feature_names: List[str] = None,
                     num_features: int = 10) -> Dict[str, Any]:
    """
    Explain individual prediction using LIME.
    
    Args:
        model: Trained model
        X: Feature DataFrame
        y: Target Series
        instance_idx: Index of instance to explain
        feature_names: List of feature names
        num_features: Number of features to include in explanation
        
    Returns:
        Dictionary with LIME explanation
    """
    try:
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X.values,
            feature_names=feature_names or X.columns.tolist(),
            class_names=['No Disease', 'Disease'],
            mode='classification'
        )
        
        # Explain instance
        instance = X.iloc[instance_idx]
        explanation = explainer.explain_instance(
            instance.values, 
            model.predict_proba, 
            num_features=num_features
        )
        
        # Get explanation data
        explanation_data = {
            'prediction': model.predict([instance])[0],
            'probability': model.predict_proba([instance])[0],
            'explanation': explanation.as_list(),
            'instance': instance.to_dict()
        }
        
        logger.info(f"LIME explanation generated for instance {instance_idx}")
        return explanation_data
        
    except Exception as e:
        logger.error(f"Error with LIME explanation: {e}")
        raise


def plot_lime_explanation(explanation_data: Dict[str, Any],
                         title: str = "LIME Explanation",
                         save_path: Optional[str] = None) -> None:
    """
    Plot LIME explanation for an instance.
    
    Args:
        explanation_data: LIME explanation data
        title: Plot title
        save_path: Path to save the plot
    """
    try:
        # Extract explanation data
        explanation_list = explanation_data['explanation']
        features, scores = zip(*explanation_list)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        colors = ['red' if score < 0 else 'green' for score in scores]
        bars = plt.barh(range(len(features)), scores, color=colors, alpha=0.7)
        
        plt.yticks(range(len(features)), features)
        plt.xlabel('LIME Score')
        plt.title(title)
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            plt.text(bar.get_width() + (0.01 if score > 0 else -0.01), 
                    bar.get_y() + bar.get_height()/2,
                    f'{score:.3f}', 
                    va='center',
                    ha='left' if score > 0 else 'right')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"LIME explanation plot saved to {save_path}")
        
        plt.show()
        
    except Exception as e:
        logger.error(f"Error plotting LIME explanation: {e}")
        raise


def explain_model(model: Any,
                 X: pd.DataFrame,
                 y: pd.Series,
                 explanation_type: str = 'all',
                 save_plots: bool = True,
                 plots_dir: str = "plots") -> Dict[str, Any]:
    """
    Comprehensive model explanation.
    
    Args:
        model: Trained model
        X: Feature DataFrame
        y: Target Series
        explanation_type: Type of explanation ('importance', 'shap', 'lime', 'all')
        save_plots: Whether to save plots
        plots_dir: Directory to save plots
        
    Returns:
        Dictionary with explanation results
    """
    try:
        results = {}
        
        # Create plots directory if saving plots
        if save_plots:
            Path(plots_dir).mkdir(parents=True, exist_ok=True)
        
        # Feature importance (for tree-based models)
        if explanation_type in ['importance', 'all'] and hasattr(model, 'feature_importances_'):
            feature_names = X.columns.tolist()
            importance_path = f"{plots_dir}/feature_importance.png" if save_plots else None
            plot_feature_importance(model, feature_names, save_path=importance_path)
            
            results['feature_importance'] = dict(zip(feature_names, model.feature_importances_))
        
        # Permutation importance
        if explanation_type in ['permutation', 'all']:
            perm_importance = calculate_permutation_importance(model, X, y)
            perm_path = f"{plots_dir}/permutation_importance.png" if save_plots else None
            plot_permutation_importance(perm_importance, save_path=perm_path)
            
            results['permutation_importance'] = perm_importance
        
        # SHAP explanation
        if explanation_type in ['shap', 'all'] and hasattr(model, 'predict_proba'):
            shap_path = f"{plots_dir}/shap_summary.png" if save_plots else None
            explain_with_shap(model, X, feature_names=X.columns.tolist(), save_path=shap_path)
        
        # LIME explanation (for a sample instance)
        if explanation_type in ['lime', 'all'] and hasattr(model, 'predict_proba'):
            instance_idx = 0  # Explain first instance
            lime_explanation = explain_with_lime(model, X, y, instance_idx, 
                                               feature_names=X.columns.tolist())
            lime_path = f"{plots_dir}/lime_explanation.png" if save_plots else None
            plot_lime_explanation(lime_explanation, save_path=lime_path)
            
            results['lime_explanation'] = lime_explanation
        
        logger.info(f"Model explanation completed: {explanation_type}")
        return results
        
    except Exception as e:
        logger.error(f"Error in model explanation: {e}")
        raise


def create_explanation_report(explanation_results: Dict[str, Any],
                            model_name: str = "Model",
                            save_path: str = "explanation_report.md") -> None:
    """
    Create a comprehensive explanation report.
    
    Args:
        explanation_results: Results from explain_model function
        model_name: Name of the model
        save_path: Path to save the report
    """
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(f"# Model Explanation Report: {model_name}\n\n")
            
            # Feature importance section
            if 'feature_importance' in explanation_results:
                f.write("## Feature Importance\n\n")
                importance = explanation_results['feature_importance']
                sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                
                f.write("| Feature | Importance |\n")
                f.write("|---------|------------|\n")
                for feature, score in sorted_features[:10]:
                    f.write(f"| {feature} | {score:.4f} |\n")
                f.write("\n")
            
            # Permutation importance section
            if 'permutation_importance' in explanation_results:
                f.write("## Permutation Importance\n\n")
                perm_importance = explanation_results['permutation_importance']
                sorted_features = sorted(perm_importance.items(), key=lambda x: x[1], reverse=True)
                
                f.write("| Feature | Permutation Importance |\n")
                f.write("|---------|------------------------|\n")
                for feature, score in sorted_features[:10]:
                    f.write(f"| {feature} | {score:.4f} |\n")
                f.write("\n")
            
            # LIME explanation section
            if 'lime_explanation' in explanation_results:
                f.write("## LIME Explanation (Sample Instance)\n\n")
                lime_data = explanation_results['lime_explanation']
                f.write(f"**Prediction:** {lime_data['prediction']}\n")
                f.write(f"**Probability:** {lime_data['probability'][1]:.4f}\n\n")
                
                f.write("| Feature | LIME Score |\n")
                f.write("|---------|------------|\n")
                for feature, score in lime_data['explanation']:
                    f.write(f"| {feature} | {score:.4f} |\n")
                f.write("\n")
            
            f.write("---\n")
            f.write("*Generated by Heart Disease Prediction System*\n")
        
        logger.info(f"Explanation report saved to {save_path}")
        
    except Exception as e:
        logger.error(f"Error creating explanation report: {e}")
        raise
