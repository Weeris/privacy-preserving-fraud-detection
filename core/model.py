"""
Fraud Detection Neural Network Model
Privacy-Preserving Financial Fraud Detection PoC
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Optional


class FraudDetectionMLP(nn.Module):
    """
    Multi-layer Perceptron for fraud detection.
    Privacy-preserving neural network classifier.
    """
    
    def __init__(self, input_dim: int = 6, hidden_dims: list = None, dropout: float = 0.3):
        super(FraudDetectionMLP, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 32, 16]
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
    
    def get_weights(self) -> dict:
        """Return model weights as dictionary."""
        return self.state_dict()
    
    def set_weights(self, weights: dict):
        """Set model weights from dictionary."""
        self.load_state_dict(weights)


class FraudDetector:
    """
    High-level interface for fraud detection model training and inference.
    """
    
    def __init__(self, input_dim: int = 6, learning_rate: float = 0.001, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = FraudDetectionMLP(input_dim=input_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch. Returns (loss, accuracy)."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels in train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * features.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return total_loss / total, 100.0 * correct / total
    
    def evaluate(self, test_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model. Returns (loss, accuracy)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * features.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return total_loss / total, 100.0 * correct / total
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict fraud probability. Returns array of probabilities."""
        self.model.eval()
        tensor = torch.FloatTensor(features).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
        
        return probs.cpu().numpy()
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(features)
        return np.argmax(probs, axis=1)
    
    def get_weights(self) -> bytes:
        """Serialize model weights."""
        return self.model.get_weights()
    
    def set_weights(self, weights):
        """Deserialize and set model weights."""
        self.model.set_weights(weights)
    
    @staticmethod
    def create_dataloaders(
        X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray,
        batch_size: int = 32
    ) -> Tuple[DataLoader, DataLoader]:
        """Create PyTorch DataLoaders from numpy arrays."""
        
        train_tensor = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        test_tensor = TensorDataset(
            torch.FloatTensor(X_test),
            torch.LongTensor(y_test)
        )
        
        train_loader = DataLoader(train_tensor, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_tensor, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader


def federated_averaging(client_weights: list, client_sizes: list) -> dict:
    """
    Perform Federated Averaging (FedAvg).
    
    Args:
        client_weights: List of model state_dicts from clients
        client_sizes: Number of samples each client has
    
    Returns:
        Aggregated model weights
    """
    total_size = sum(client_sizes)
    
    # Initialize aggregated weights
    aggregated_weights = {}
    
    # Get keys from first client
    keys = client_weights[0].keys()
    
    for key in keys:
        # Weighted average of each parameter
        weighted_sum = torch.zeros_like(client_weights[0][key], dtype=torch.float32)
        
        for weights, size in zip(client_weights, client_sizes):
            weighted_sum += weights[key].float() * (size / total_size)
        
        aggregated_weights[key] = weighted_sum
    
    return aggregated_weights
