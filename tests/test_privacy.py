"""
Unit tests for core/privacy.py
"""
import pytest
import numpy as np
import torch
from core.privacy import (
    DifferentialPrivacy,
    PrivacyBudget,
    PrivacyAccountant,
    calibrate_laplace_mechanism,
    calibrate_gaussian_mechanism,
    compute_privacy_budget,
)


class TestPrivacyBudget:
    """Tests for PrivacyBudget dataclass."""
    
    def test_privacy_budget_init(self):
        """Test PrivacyBudget initialization."""
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
        assert budget.epsilon == 1.0
        assert budget.delta == 1e-5
        assert budget.consumed == 0.0
    
    def test_remaining(self):
        """Test remaining budget calculation."""
        budget = PrivacyBudget(epsilon=2.0, delta=1e-5, consumed=0.5)
        assert budget.remaining() == 1.5
    
    def test_remaining_negative(self):
        """Test remaining when exhausted."""
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5, consumed=2.0)
        assert budget.remaining() == 0.0
    
    def test_is_exhausted(self):
        """Test budget exhaustion check."""
        budget = PrivacyBudget(epsilon=1.0, delta=1e-5, consumed=0.5)
        assert not budget.is_exhausted()
        
        budget.consumed = 1.5
        assert budget.is_exhausted()


class TestDifferentialPrivacy:
    """Tests for DifferentialPrivacy class."""
    
    def test_dp_init(self):
        """Test DifferentialPrivacy initialization."""
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        assert dp.epsilon == 1.0
        assert dp.delta == 1e-5
    
    def test_laplace_mechanism(self):
        """Test Laplace noise addition."""
        dp = DifferentialPrivacy(epsilon=1.0)
        value = np.array([1.0, 2.0, 3.0])
        noisy = dp.laplace_mechanism(value, sensitivity=1.0)
        
        assert noisy.shape == value.shape
        assert not np.allclose(noisy, value)  # Noise was added
    
    def test_laplace_mechanism_preserves_mean(self):
        """Test Laplace mechanism statistical properties."""
        dp = DifferentialPrivacy(epsilon=10.0)  # High epsilon = less noise
        value = np.zeros(10000)
        noisy = dp.laplace_mechanism(value, sensitivity=1.0)
        
        # With high epsilon, mean should be close to original
        assert abs(noisy.mean()) < 0.1
    
    def test_gaussian_mechanism(self):
        """Test Gaussian noise addition."""
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)
        value = np.array([1.0, 2.0, 3.0])
        noisy = dp.gaussian_mechanism(value, sensitivity=1.0)
        
        assert noisy.shape == value.shape
        assert not np.allclose(noisy, value)
    
    def test_add_noise_to_gradients(self):
        """Test adding noise to model gradients."""
        dp = DifferentialPrivacy(epsilon=1.0)
        
        # Create mock gradients
        gradients = {
            'layer1.weight': torch.randn(10, 5),
            'layer1.bias': torch.randn(10),
        }
        
        noisy_grads = dp.add_noise_to_gradients(gradients, sensitivity=1.0)
        
        assert 'layer1.weight' in noisy_grads
        assert 'layer1.bias' in noisy_grads
        assert noisy_grads['layer1.weight'].shape == gradients['layer1.weight'].shape
    
    def test_add_noise_to_weights(self):
        """Test adding noise to model weights."""
        dp = DifferentialPrivacy(epsilon=1.0)
        
        weights = {
            'network.0.weight': torch.randn(64, 6),
            'network.0.bias': torch.randn(64),
        }
        
        noisy_weights = dp.add_noise_to_weights(weights, sensitivity=0.01)
        
        assert 'network.0.weight' in noisy_weights
        assert noisy_weights['network.0.weight'].shape == weights['network.0.weight'].shape
    
    def test_privacy_budget_consumption(self):
        """Test privacy budget is consumed."""
        dp = DifferentialPrivacy(epsilon=2.0)
        initial_consumed = dp.budget.consumed
        
        gradients = {'weight': torch.randn(10, 5)}
        dp.add_noise_to_gradients(gradients)
        
        assert dp.budget.consumed > initial_consumed
    
    def test_compute_privacy_spent(self):
        """Test privacy spent computation."""
        dp = DifferentialPrivacy(epsilon=5.0)
        epsilon_spent, delta = dp.compute_privacy_spent()
        
        assert epsilon_spent >= 0
        assert delta == 1e-5
    
    def test_get_privacy_budget_info(self):
        """Test privacy budget info."""
        dp = DifferentialPrivacy(epsilon=3.0, delta=1e-4)
        info = dp.get_privacy_budget_info()
        
        assert 'epsilon' in info
        assert 'delta' in info
        assert 'consumed' in info
        assert 'remaining' in info
        assert 'is_exhausted' in info


class TestPrivacyAccountant:
    """Tests for PrivacyAccountant class."""
    
    def test_accountant_init(self):
        """Test PrivacyAccountant initialization."""
        accountant = PrivacyAccountant(epsilon=10.0, delta=1e-5)
        assert accountant.epsilon == 10.0
        assert accountant.delta == 1e-5
        assert len(accountant.operations) == 0
    
    def test_step(self):
        """Test recording a privacy step."""
        accountant = PrivacyAccountant(epsilon=10.0)
        accountant.step(noise_multiplier=1.0)
        
        assert len(accountant.operations) == 1
    
    def test_get_remaining_budget(self):
        """Test remaining budget calculation."""
        accountant = PrivacyAccountant(epsilon=10.0)
        accountant.step(noise_multiplier=1.0)
        
        remaining = accountant.get_remaining_budget()
        assert remaining <= 10.0
    
    def test_reset(self):
        """Test resetting accountant."""
        accountant = PrivacyAccountant(epsilon=10.0)
        accountant.step(noise_multiplier=1.0)
        accountant.reset()
        
        assert accountant.epsilon == 10.0
        assert len(accountant.operations) == 0
    
    def test_get_accounting_info(self):
        """Test accounting info."""
        accountant = PrivacyAccountant(epsilon=10.0, delta=1e-4)
        accountant.step(noise_multiplier=1.0, num_samples=100)
        
        info = accountant.get_accounting_info()
        
        assert 'initial_epsilon' in info
        assert 'current_epsilon' in info
        assert 'num_operations' in info


class TestCalibrationFunctions:
    """Tests for calibration helper functions."""
    
    def test_calibrate_laplace(self):
        """Test Laplace mechanism calibration."""
        scale = calibrate_laplace_mechanism(epsilon=1.0, sensitivity=1.0)
        assert scale == 1.0
    
    def test_calibrate_laplace_high_epsilon(self):
        """Test Laplace with high epsilon (less noise)."""
        scale = calibrate_laplace_mechanism(epsilon=10.0, sensitivity=1.0)
        assert scale < 1.0
    
    def test_calibrate_gaussian(self):
        """Test Gaussian mechanism calibration."""
        sigma = calibrate_gaussian_mechanism(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        assert sigma > 0
    
    def test_compute_privacy_budget(self):
        """Test privacy budget computation."""
        epsilon, delta = compute_privacy_budget(num_rounds=10, noise_multiplier=1.0)
        
        assert epsilon > 0
        assert delta == 1e-5
    
    def test_privacy_budget_grows_with_rounds(self):
        """Test that privacy budget grows with more rounds."""
        eps1, _ = compute_privacy_budget(num_rounds=1, noise_multiplier=1.0)
        eps10, _ = compute_privacy_budget(num_rounds=10, noise_multiplier=1.0)
        
        assert eps10 > eps1
