"""
Tests for preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from preprocessing import (
    split_data, create_preprocessor, preprocess_data,
    save_preprocessor, load_preprocessor, calculate_quantiles
)


class TestDataSplitting:
    """Test data splitting functionality."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        np.random.seed(42)
        n_samples = 100
        data = {
            'age': np.random.randint(20, 80, n_samples),
            'trestbps': np.random.randint(90, 200, n_samples),
            'chol': np.random.randint(150, 400, n_samples),
            'target': np.random.randint(0, 2, n_samples)
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def config(self):
        """Create sample configuration."""
        return {
            'splitting': {
                'test_size': 0.2,
                'val_size': 0.2,
                'random_state': 42,
                'stratify': True
            }
        }
    
    def test_split_data_shape(self, sample_data, config):
        """Test that data splitting produces correct shapes."""
        train_df, val_df, test_df = split_data(sample_data, config)
        
        total_samples = len(sample_data)
        expected_test_size = int(total_samples * config['splitting']['test_size'])
        expected_val_size = int(total_samples * config['splitting']['val_size'] * (1 - config['splitting']['test_size']))
        expected_train_size = total_samples - expected_test_size - expected_val_size
        
        assert len(train_df) == expected_train_size
        assert len(val_df) == expected_val_size
        assert len(test_df) == expected_test_size
    
    def test_split_data_stratification(self, sample_data, config):
        """Test that stratification maintains class distribution."""
        train_df, val_df, test_df = split_data(sample_data, config)
        
        original_dist = sample_data['target'].value_counts(normalize=True)
        
        # Check that distributions are similar (within 10%)
        for df in [train_df, val_df, test_df]:
            df_dist = df['target'].value_counts(normalize=True)
            for class_label in [0, 1]:
                if class_label in original_dist.index and class_label in df_dist.index:
                    assert abs(original_dist[class_label] - df_dist[class_label]) < 0.1


class TestPreprocessor:
    """Test preprocessor creation and usage."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset with missing values."""
        data = {
            'age': [25, 30, np.nan, 40, 50],
            'trestbps': [120, np.nan, 140, 150, 160],
            'sex': ['M', 'F', 'M', np.nan, 'F'],
            'cp': ['A', 'B', 'A', 'B', 'A'],
            'target': [0, 1, 0, 1, 0]
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def config(self):
        """Create sample configuration."""
        return {
            'features': {
                'numeric': ['age', 'trestbps'],
                'categorical': ['sex', 'cp']
            },
            'preprocessing': {
                'numeric_strategy': 'median',
                'categorical_strategy': 'most_frequent'
            }
        }
    
    def test_create_preprocessor(self, config):
        """Test preprocessor creation."""
        preprocessor = create_preprocessor(config)
        assert preprocessor is not None
        assert hasattr(preprocessor, 'transformers_')
    
    def test_preprocess_data_fit(self, sample_data, config):
        """Test preprocessing with fitting."""
        preprocessor = create_preprocessor(config)
        processed_df = preprocess_data(sample_data, preprocessor, fit=True)
        
        # Check that missing values are handled
        assert processed_df.isnull().sum().sum() == 0
        
        # Check that target is preserved
        assert 'target' in processed_df.columns
    
    def test_preprocess_data_transform(self, sample_data, config):
        """Test preprocessing without fitting."""
        preprocessor = create_preprocessor(config)
        
        # First fit the preprocessor
        preprocess_data(sample_data, preprocessor, fit=True)
        
        # Then transform new data
        new_data = sample_data.copy()
        processed_df = preprocess_data(new_data, preprocessor, fit=False)
        
        assert processed_df.isnull().sum().sum() == 0
        assert 'target' in processed_df.columns


class TestQuantiles:
    """Test quantile calculation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample numeric data."""
        np.random.seed(42)
        data = {
            'age': np.random.normal(50, 15, 100),
            'trestbps': np.random.normal(120, 20, 100),
            'chol': np.random.normal(200, 50, 100)
        }
        return pd.DataFrame(data)
    
    def test_calculate_quantiles(self, sample_data):
        """Test quantile calculation."""
        numeric_columns = ['age', 'trestbps', 'chol']
        quantiles = calculate_quantiles(sample_data, numeric_columns)
        
        # Check that all columns are present
        assert all(col in quantiles for col in numeric_columns)
        
        # Check that quantiles are calculated correctly
        for col in numeric_columns:
            assert 'q25' in quantiles[col]
            assert 'q50' in quantiles[col]
            assert 'q75' in quantiles[col]
            
            # Check that q50 (median) matches pandas median
            assert abs(quantiles[col]['q50'] - sample_data[col].median()) < 0.01


class TestPersistence:
    """Test saving and loading preprocessor."""
    
    @pytest.fixture
    def sample_preprocessor(self):
        """Create a sample preprocessor."""
        config = {
            'features': {
                'numeric': ['age', 'trestbps'],
                'categorical': ['sex']
            },
            'preprocessing': {
                'numeric_strategy': 'median',
                'categorical_strategy': 'most_frequent'
            }
        }
        return create_preprocessor(config)
    
    def test_save_load_preprocessor(self, sample_preprocessor, tmp_path):
        """Test saving and loading preprocessor."""
        file_path = tmp_path / "test_preprocessor.pkl"
        
        # Save preprocessor
        save_preprocessor(sample_preprocessor, str(file_path))
        assert file_path.exists()
        
        # Load preprocessor
        loaded_preprocessor = load_preprocessor(str(file_path))
        assert loaded_preprocessor is not None
        
        # Check that transformers are preserved
        assert len(sample_preprocessor.transformers_) == len(loaded_preprocessor.transformers_)


if __name__ == "__main__":
    pytest.main([__file__])
