"""
Tests for FastAPI application.
"""

import pytest
import requests
import json
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add app to path
sys.path.append(str(Path(__file__).parent.parent / "app"))

from api import app


class TestAPIEndpoints:
    """Test API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_features(self):
        """Create sample features for testing."""
        return {
            "age": 50,
            "sex": 1,
            "cp": 0,
            "trestbps": 120,
            "chol": 200,
            "fbs": 0,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 1.0,
            "slope": 1,
            "ca": 0,
            "thal": 0
        }
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["version"] == "1.0.0"
    
    def test_health_endpoint_without_model(self, client):
        """Test health endpoint when model is not loaded."""
        # Mock that model is not loaded
        from api import model
        original_model = model
        from api import model
        model = None
        
        try:
            response = client.get("/health")
            assert response.status_code == 503
        finally:
            # Restore original model
            from api import model
            model = original_model
    
    def test_features_description_endpoint(self, client):
        """Test features description endpoint."""
        response = client.get("/features-description")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, dict)
        assert "age" in data
        assert "sex" in data
        assert "target" not in data  # Target should not be in feature descriptions
    
    def test_predict_endpoint_validation(self, client):
        """Test prediction endpoint with invalid data."""
        # Test with missing required fields
        invalid_data = {"age": 50}  # Missing other required fields
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_predict_endpoint_out_of_range(self, client):
        """Test prediction endpoint with out-of-range values."""
        invalid_data = {
            "age": 150,  # Out of range (max 120)
            "sex": 1,
            "cp": 0,
            "trestbps": 120,
            "chol": 200,
            "fbs": 0,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 1.0,
            "slope": 1,
            "ca": 0,
            "thal": 0
        }
        
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == 422  # Validation error
    
    def test_batch_predict_endpoint(self, client):
        """Test batch prediction endpoint."""
        batch_data = [
            {
                "age": 50,
                "sex": 1,
                "cp": 0,
                "trestbps": 120,
                "chol": 200,
                "fbs": 0,
                "restecg": 0,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 1.0,
                "slope": 1,
                "ca": 0,
                "thal": 0
            },
            {
                "age": 60,
                "sex": 0,
                "cp": 1,
                "trestbps": 140,
                "chol": 250,
                "fbs": 1,
                "restecg": 1,
                "thalach": 130,
                "exang": 1,
                "oldpeak": 2.0,
                "slope": 2,
                "ca": 1,
                "thal": 1
            }
        ]
        
        response = client.post("/batch-predict", json=batch_data)
        
        if response.status_code == 503:
            pytest.skip("Model not loaded - skipping prediction test")
        
        assert response.status_code == 200
        
        data = response.json()
        assert len(data) == 2
        
        for result in data:
            assert "prediction" in result
            assert "probability" in result
            assert "confidence" in result
            assert "timestamp" in result
            assert result["prediction"] in [0, 1]
            assert 0 <= result["probability"] <= 1
    
    def test_batch_predict_too_large(self, client):
        """Test batch prediction with too many samples."""
        # Create 101 samples (over the limit)
        large_batch = []
        for i in range(101):
            large_batch.append({
                "age": 50 + i % 30,
                "sex": i % 2,
                "cp": i % 4,
                "trestbps": 120 + i % 50,
                "chol": 200 + i % 100,
                "fbs": i % 2,
                "restecg": i % 3,
                "thalach": 150 + i % 50,
                "exang": i % 2,
                "oldpeak": 1.0 + (i % 20) / 10,
                "slope": i % 3,
                "ca": i % 5,
                "thal": i % 4
            })
        
        response = client.post("/batch-predict", json=large_batch)
        assert response.status_code == 400


class TestDataValidation:
    """Test data validation in API."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_valid_feature_ranges(self, client):
        """Test that all feature ranges are properly validated."""
        # Test age bounds
        data_min_age = {
            "age": 19,  # Below minimum
            "sex": 1, "cp": 0, "trestbps": 120, "chol": 200,
            "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
            "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 0
        }
        
        response = client.post("/predict", json=data_min_age)
        assert response.status_code == 422
        
        data_max_age = {
            "age": 121,  # Above maximum
            "sex": 1, "cp": 0, "trestbps": 120, "chol": 200,
            "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
            "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 0
        }
        
        response = client.post("/predict", json=data_max_age)
        assert response.status_code == 422
    
    def test_categorical_feature_validation(self, client):
        """Test categorical feature validation."""
        # Test sex validation
        data_invalid_sex = {
            "age": 50, "sex": 2, "cp": 0, "trestbps": 120, "chol": 200,
            "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
            "oldpeak": 1.0, "slope": 1, "ca": 0, "thal": 0
        }
        
        response = client.post("/predict", json=data_invalid_sex)
        assert response.status_code == 422
    
    def test_numeric_feature_validation(self, client):
        """Test numeric feature validation."""
        # Test oldpeak validation (should be float)
        data_invalid_oldpeak = {
            "age": 50, "sex": 1, "cp": 0, "trestbps": 120, "chol": 200,
            "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
            "oldpeak": "invalid", "slope": 1, "ca": 0, "thal": 0
        }
        
        response = client.post("/predict", json=data_invalid_oldpeak)
        assert response.status_code == 422


class TestAPIDocumentation:
    """Test API documentation endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    def test_openapi_schema(self, client):
        """Test OpenAPI schema endpoint."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        
        # Check that our endpoints are documented
        assert "/predict" in schema["paths"]
        assert "/health" in schema["paths"]
        assert "/features-description" in schema["paths"]
    
    def test_docs_endpoint(self, client):
        """Test Swagger docs endpoint."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]


if __name__ == "__main__":
    pytest.main([__file__])
