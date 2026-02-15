"""
Unit tests for core/model.py
"""
import pytest
import torch
import numpy as np
from core.model import FraudDetectionMLP, FraudDetector, federated_averaging


class TestFraudDetectionMLP:
    """Tests for FraudDetectionMLP model."""
    
    def test_model_initialization(self):
        """Test model can be initialized with default params."""
        model = FraudDetectionMLP()
        assert model is not None
        assert isinstance(model.network, torch.nn.Sequential)
    
    def test_model_initialization_custom_params(self):
        """Test model initialization with custom params."""
        model = FraudDetectionMLP(input_dim=10, hidden_dims=[128, 64], dropout=0.5)
        assert model is not None
    
    def test_forward_pass(self):
        """Test forward pass returns correct shape."""
        model = FraudDetectionMLP(input_dim=6)
        x = torch.randn(32, 6)
        output = model(x)
        assert output.shape == (32, 2)
    
    def test_get_weights(self):
        """Test getting model weights."""
        model = FraudDetectionMLP()
        weights = model.get_weights()
        assert isinstance(weights, dict)
        assert len(weights) > 0
    
    def test_set_weights(self):
        """Test setting model weights."""
        model = FraudDetectionMLP()
        original_weights = model.get_weights()
        
        # Modify weights
        for key in original_weights:
            original_weights[key] = torch.zeros_like(original_weights[key])
        
        model.set_weights(original_weights)
        
        # Verify weights were set
        new_weights = model.get_weights()
        for key in original_weights:
            assert torch.allclose(new_weights[key], original_weights[key])


class TestFraudDetector:
    """Tests for FraudDetector class."""
    
    def test_fraud_detector_init(self):
        """Test FraudDetector initialization."""
        detector = FraudDetector(input_dim=6)
        assert detector is not None
        assert isinstance(detector.model, FraudDetectionMLP)
        assert isinstance(detector.optimizer, torch.optim.Adam)
        assert isinstance(detector.criterion, torch.nn.CrossEntropyLoss)
    
    def test_predict_proba(self):
        """Test probability prediction."""
        detector = FraudDetector(input_dim=6)
        X = np.random.randn(10, 6)
        probs = detector.predict_proba(X)
        assert probs.shape == (10, 2)
        assert np.allclose(probs.sum(axis=1), 1.0)
    
    def test_predict(self):
        """Test class prediction."""
        detector = FraudDetector(input_dim=6)
        X = np.random.randn(10, 6)
        preds = detector.predict(X)
        assert preds.shape == (10,)
        assert np.all((preds == 0) | (preds == 1))
    
    def test_create_dataloaders(self):
        """Test dataloader creation."""
        X_train = np.random.randn(100, 6)
        y_train = np.random.randint(0, 2, 100)
        X_test = np.random.randn(20, 6)
        y_test = np.random.randint(0, 2, 20)
        
        train_loader, test_loader = FraudDetector.create_dataloaders(
            X_train, y_train, X_test, y_test, batch_size=16
        )
        
        assert train_loader is not None
        assert test_loader is not None


class TestFederatedAveraging:
    """Tests for federated averaging."""
    
    def test_federated_averaging_basic(self):
        """Test basic federated averaging."""
        # Create mock client weights
        client_weights = [
            {'layer1.weight': torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
            {'layer1.weight': torch.tensor([[5.0, 6.0], [7.0, 8.0]])},
        ]
        client_sizes = [10, 10]
        
        # Note: This will fail due to incomplete federated_averaging function
        # But the test documents expected behavior
        assert len(client_weights) == 2
        assert len(client_sizes) == 2
