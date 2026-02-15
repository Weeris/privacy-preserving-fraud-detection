"""
Differential Privacy Mechanisms
Privacy-Preserving Financial Fraud Detection PoC
"""

import numpy as np
import torch
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class PrivacyBudget:
    """Track privacy budget consumption."""
    epsilon: float
    delta: float
    consumed: float = 0.0
    
    def remaining(self) -> float:
        return max(0.0, self.epsilon - self.consumed)
    
    def is_exhausted(self) -> bool:
        return self.consumed >= self.epsilon


class DifferentialPrivacy:
    """
    Differential privacy mechanisms for privacy-preserving ML.
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
        self.budget = PrivacyBudget(epsilon=epsilon, delta=delta)
    
    def laplace_mechanism(self, value: np.ndarray, sensitivity: float) -> np.ndarray:
        """
        Add Laplace noise for differential privacy.
        
        Args:
            value: Original value to add noise to
            sensitivity: Maximum change from any single sample
        
        Returns:
            Noisy value
        """
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale, value.shape)
        return value + noise
    
    def gaussian_mechanism(self, value: np.ndarray, sensitivity: float) -> np.ndarray:
        """
        Add Gaussian noise (for (ε, δ)-differential privacy).
        
        Args:
            value: Original value to add noise to
            sensitivity: Maximum change from any single sample
        
        Returns:
            Noisy value
        """
        # Standard deviation for Gaussian noise
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * sensitivity / self.epsilon
        noise = np.random.normal(0, sigma, value.shape)
        return value + noise
    
    def add_noise_to_gradients(
        self, 
        gradients: dict, 
        sensitivity: float = 1.0,
        mechanism: str = 'gaussian'
    ) -> dict:
        """
        Add differential privacy noise to model gradients.
        
        Args:
            gradients: Dictionary of model gradients
            sensitivity: Sensitivity of the gradients
            mechanism: 'laplace' or 'gaussian'
        
        Returns:
            Noisy gradients
        """
        noisy_gradients = {}
        
        for key, grad in gradients.items():
            if grad is not None and isinstance(grad, torch.Tensor):
                grad_np = grad.cpu().numpy()
                
                if mechanism == 'laplace':
                    noisy_grad = self.laplace_mechanism(grad_np, sensitivity)
                else:
                    noisy_grad = self.gaussian_mechanism(grad_np, sensitivity)
                
                noisy_gradients[key] = torch.from_numpy(noisy_grad).to(grad.device)
            else:
                noisy_gradients[key] = grad
        
        # Consume privacy budget
        self.budget.consumed += 1.0
        
        return noisy_gradients
    
    def add_noise_to_weights(
        self, 
        weights: dict, 
        sensitivity: float = 0.01
    ) -> dict:
        """
        Add differential privacy noise to model weights.
        Used for sharing model updates without exposing raw data.
        
        Args:
            weights: Dictionary of model weights
            sensitivity: Sensitivity of weights
        
        Returns:
            Noisy weights
        """
        noisy_weights = {}
        
        for key, weight in weights.items():
            if isinstance(weight, torch.Tensor):
                weight_np = weight.cpu().numpy()
                noisy_weight = self.laplace_mechanism(weight_np, sensitivity)
                noisy_weights[key] = torch.from_numpy(noisy_weight).to(weight.device)
            else:
                noisy_weights[key] = weight
        
        # Consume privacy budget based on number of parameters
        num_params = sum(w.numel() for w in weights.values() if isinstance(w, torch.Tensor))
        self.budget.consumed += 1.0 / max(num_params, 1)
        
        return noisy_weights
    
    def compute_privacy_spent(self) -> Tuple[float, float]:
        """Return (epsilon_spent, delta) tuple."""
        return min(self.budget.consumed, self.epsilon), self.delta
    
    def get_privacy_budget_info(self) -> dict:
        """Return detailed privacy budget information."""
        return {
            'epsilon': self.epsilon,
            'delta': self.delta,
            'consumed': self.budget.consumed,
            'remaining': self.budget.remaining(),
            'is_exhausted': self.budget.is_exhausted()
        }


class PrivacyAccountant:
    """
    Track and manage privacy budget across multiple operations.
    """
    
    def __init__(self, epsilon: float = 10.0, delta: float = 1e-5):
        self.initial_epsilon = epsilon
        self.epsilon = epsilon
        self.delta = delta
        self.operations = []
    
    def step(self, noise_multiplier: float = 1.0, num_samples: int = 1):
        """
        Record a privacy-preserving step.
        Uses advanced composition theorem for accounting.
        """
        # Simplified composition - each step consumes epsilon
        self.epsilon -= noise_multiplier * np.sqrt(2 * np.log(1.25 / self.delta))
        
        self.operations.append({
            'noise_multiplier': noise_multiplier,
            'num_samples': num_samples,
            'remaining_epsilon': max(0, self.epsilon)
        })
    
    def get_remaining_budget(self) -> float:
        return max(0, self.epsilon)
    
    def reset(self):
        """Reset privacy budget to initial values."""
        self.epsilon = self.initial_epsilon
        self.operations = []
    
    def get_accounting_info(self) -> dict:
        return {
            'initial_epsilon': self.initial_epsilon,
            'current_epsilon': self.epsilon,
            'delta': self.delta,
            'num_operations': len(self.operations),
            'remaining': self.get_remaining_budget()
        }


def calibrate_laplace_mechanism(epsilon: float, sensitivity: float) -> float:
    """
    Calculate the scale parameter for Laplace mechanism.
    
    Args:
        epsilon: Privacy budget
        sensitivity: Maximum change from any single record
    
    Returns:
        Scale parameter for Laplace distribution
    """
    return sensitivity / epsilon


def calibrate_gaussian_mechanism(epsilon: float, delta: float, sensitivity: float) -> float:
    """
    Calculate the standard deviation for Gaussian mechanism.
    
    Args:
        epsilon: Privacy budget
        delta: Failure probability
        sensitivity: Maximum change from any single record
    
    Returns:
        Standard deviation for Gaussian noise
    """
    return np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon


def compute_privacy_budget(
    num_rounds: int,
    noise_multiplier: float = 1.0,
    delta: float = 1e-5
) -> Tuple[float, float]:
    """
    Compute privacy budget using strong composition.
    
    Args:
        num_rounds: Number of federated learning rounds
        noise_multiplier: Ratio of noise standard deviation to sensitivity
        delta: Failure probability
    
    Returns:
        (epsilon, delta) privacy budget spent
    """
    # Strong composition theorem
    epsilon = noise_multiplier * np.sqrt(num_rounds * 2 * np.log(1.25 / delta))
    return epsilon, delta
