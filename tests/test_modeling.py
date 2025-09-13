"""
Tests for modeling module.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from modeling import (
    train_naive_bayes, train_knn, train_decision_tree,
    create_ensemble, train_models, evaluate_models,
    cross_validate_model, hyperparameter_tuning
)


class TestModelTraining:
    """Test individual model training functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        np.random.seed(42)
        n_samples = 100
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'feature3': np.random.normal(0, 1, n_samples)
        })
        y = pd.Series(np.random.randint(0, 2, n_samples))
        return X, y
    
    def test_train_naive_bayes(self, sample_data):
        """Test Naive Bayes training."""
        X, y = sample_data
        model = train_naive_bayes(X, y)
        
        assert isinstance(model, GaussianNB)
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    def test_train_knn(self, sample_data):
        """Test KNN training."""
        X, y = sample_data
        model = train_knn(X, y)
        
        assert isinstance(model, KNeighborsClassifier)
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    def test_train_decision_tree(self, sample_data):
        """Test Decision Tree training."""
        X, y = sample_data
        model = train_decision_tree(X, y)
        
        assert isinstance(model, DecisionTreeClassifier)
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
        assert hasattr(model, 'feature_importances_')
    
    def test_train_models(self, sample_data):
        """Test training multiple models."""
        X, y = sample_data
        
        config = {
            'models': {
                'naive_bayes': {'var_smoothing': 1e-9},
                'knn': {'n_neighbors': 5},
                'decision_tree': {'max_depth': 5}
            }
        }
        
        models = train_models(X, y, config)
        
        assert len(models) == 3
        assert 'naive_bayes' in models
        assert 'knn' in models
        assert 'decision_tree' in models
        
        # Test that all models can make predictions
        for name, model in models.items():
            predictions = model.predict(X)
            assert len(predictions) == len(X)
            assert all(pred in [0, 1] for pred in predictions)


class TestEnsemble:
    """Test ensemble model creation."""
    
    @pytest.fixture
    def sample_models(self):
        """Create sample models."""
        models = [
            ('nb', GaussianNB()),
            ('knn', KNeighborsClassifier()),
            ('dt', DecisionTreeClassifier())
        ]
        return models
    
    def test_create_ensemble(self, sample_models):
        """Test ensemble creation."""
        ensemble = create_ensemble(sample_models)
        
        assert isinstance(ensemble, VotingClassifier)
        assert len(ensemble.estimators) == 3
    
    def test_ensemble_predictions(self, sample_models, sample_data):
        """Test ensemble predictions."""
        X, y = sample_data
        ensemble = create_ensemble(sample_models)
        
        # Fit ensemble
        ensemble.fit(X, y)
        
        # Test predictions
        predictions = ensemble.predict(X)
        probabilities = ensemble.predict_proba(X)
        
        assert len(predictions) == len(X)
        assert probabilities.shape == (len(X), 2)
        assert all(pred in [0, 1] for pred in predictions)


class TestModelEvaluation:
    """Test model evaluation functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset."""
        np.random.seed(42)
        n_samples = 100
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples)
        })
        y = pd.Series(np.random.randint(0, 2, n_samples))
        return X, y
    
    @pytest.fixture
    def sample_models(self, sample_data):
        """Create fitted sample models."""
        X, y = sample_data
        
        models = {
            'naive_bayes': train_naive_bayes(X, y),
            'knn': train_knn(X, y),
            'decision_tree': train_decision_tree(X, y)
        }
        return models
    
    def test_evaluate_models(self, sample_models, sample_data):
        """Test model evaluation."""
        X, y = sample_data
        
        # Split data for evaluation
        X_val = X.iloc[:20]
        y_val = y.iloc[:20]
        
        results = evaluate_models(sample_models, X_val, y_val)
        
        assert len(results) == len(sample_models)
        
        for model_name, metrics in results.items():
            assert 'accuracy' in metrics
            assert 'precision' in metrics
            assert 'recall' in metrics
            assert 'f1' in metrics
            assert 'roc_auc' in metrics
            
            # Check that metrics are valid
            assert 0 <= metrics['accuracy'] <= 1
            assert 0 <= metrics['precision'] <= 1
            assert 0 <= metrics['recall'] <= 1
            assert 0 <= metrics['f1'] <= 1
            assert 0 <= metrics['roc_auc'] <= 1
    
    def test_cross_validate_model(self, sample_data):
        """Test cross-validation."""
        X, y = sample_data
        model = train_naive_bayes(X, y)
        
        cv_results = cross_validate_model(model, X, y, cv=3)
        
        assert 'mean_score' in cv_results
        assert 'std_score' in cv_results
        assert 'scores' in cv_results
        assert len(cv_results['scores']) == 3
        assert 0 <= cv_results['mean_score'] <= 1


class TestHyperparameterTuning:
    """Test hyperparameter tuning."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset."""
        np.random.seed(42)
        n_samples = 50  # Smaller dataset for faster testing
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples)
        })
        y = pd.Series(np.random.randint(0, 2, n_samples))
        return X, y
    
    def test_hyperparameter_tuning(self, sample_data):
        """Test hyperparameter tuning."""
        X, y = sample_data
        
        # Create base model
        base_model = KNeighborsClassifier()
        
        # Define parameter grid
        param_grid = {
            'n_neighbors': [3, 5],
            'weights': ['uniform', 'distance']
        }
        
        # Perform tuning
        best_model = hyperparameter_tuning(
            base_model, param_grid, X, y, cv=2, scoring='accuracy'
        )
        
        assert best_model is not None
        assert hasattr(best_model, 'predict')
        
        # Test that best model can make predictions
        predictions = best_model.predict(X)
        assert len(predictions) == len(X)


class TestModelPersistence:
    """Test model saving and loading."""
    
    @pytest.fixture
    def sample_model(self):
        """Create a sample trained model."""
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 50),
            'feature2': np.random.normal(0, 1, 50)
        })
        y = pd.Series(np.random.randint(0, 2, 50))
        
        return train_naive_bayes(X, y)
    
    def test_save_load_model(self, sample_model, tmp_path):
        """Test saving and loading model."""
        file_path = tmp_path / "test_model.pkl"
        
        # Import save_model and load_model
        from modeling import save_model, load_model
        
        # Save model
        save_model(sample_model, str(file_path))
        assert file_path.exists()
        
        # Load model
        loaded_model = load_model(str(file_path))
        assert loaded_model is not None
        
        # Test that loaded model works
        X_test = pd.DataFrame({
            'feature1': [0.5, -0.5],
            'feature2': [1.0, -1.0]
        })
        
        original_predictions = sample_model.predict(X_test)
        loaded_predictions = loaded_model.predict(X_test)
        
        assert np.array_equal(original_predictions, loaded_predictions)


if __name__ == "__main__":
    pytest.main([__file__])
