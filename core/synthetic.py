"""
Synthetic transaction data generator for fraud detection PoC.
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional


def generate_transactions(region: str, n_samples: int = 1000, fraud_rate: float = 0.02) -> pd.DataFrame:
    """Generate synthetic transaction data for a region.
    
    Args:
        region: Region code (TH, SG, HK)
        n_samples: Number of transactions to generate
        fraud_rate: Base fraud rate (0.0 to 1.0)
    
    Returns:
        DataFrame with transaction features
    """
    np.random.seed(hash(region) % 2**32)
    
    # Region-specific parameters
    region_params = {
        'TH': {'currency': 'THB', 'max_amount': 500000, 'avg_amount': 5000},
        'SG': {'currency': 'SGD', 'max_amount': 100000, 'avg_amount': 2000},
        'HK': {'currency': 'HKD', 'max_amount': 800000, 'avg_amount': 8000},
    }
    params = region_params.get(region, region_params['TH'])
    
    # Generate base features
    data = {
        'transaction_id': [f'TXN_{region}_{i:06d}' for i in range(n_samples)],
        'amount': np.random.lognormal(
            mean=np.log(params['avg_amount']),
            sigma=1.0,
            size=n_samples
        ).clip(10, params['max_amount']),
        'frequency': np.random.poisson(5, n_samples),
        'merchant_category': np.random.choice(
            ['retail', 'dining', 'travel', 'online', 'atm', 'transfer'],
            n_samples,
            p=[0.3, 0.2, 0.1, 0.25, 0.1, 0.05]
        ),
        'time_hour': np.random.randint(0, 24, n_samples),
        'is_weekend': np.random.randint(0, 2, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Add derived features
    df['avg_amount_7d'] = df['amount'] * np.random.uniform(0.8, 1.2, n_samples)
    df['transaction_count_7d'] = np.random.poisson(10, n_samples)
    df['amount_to_avg_ratio'] = df['amount'] / (df['avg_amount_7d'] + 1)
    
    # Generate fraud labels
    fraud_prob = fraud_rate * np.ones(n_samples)
    
    # Increase fraud probability for suspicious patterns
    high_risk_mask = (
        (df['amount'] > params['avg_amount'] * 10) |
        (df['time_hour'].isin([2, 3, 4, 5])) |
        (df['merchant_category'] == 'online')
    )
    fraud_prob[high_risk_mask] *= 3
    
    # Normalize probabilities
    fraud_prob = np.clip(fraud_prob, 0, 1)
    df['is_fraud'] = (np.random.random(n_samples) < fraud_prob).astype(int)
    
    return df


def add_fraud_patterns(df: pd.DataFrame, pattern_type: str = 'high_amount') -> pd.DataFrame:
    """Add specific fraud patterns to transaction data.
    
    Args:
        df: Input DataFrame
        pattern_type: Type of fraud pattern to add
    
    Returns:
        DataFrame with enhanced fraud patterns
    """
    df = df.copy()
    
    if pattern_type == 'high_amount':
        # Unusually high transactions
        fraud_mask = df['is_fraud'] == 1
        df.loc[fraud_mask, 'amount'] *= np.random.uniform(5, 20, fraud_mask.sum())
        
    elif pattern_type == 'unusual_time':
        # Transactions at unusual hours
        fraud_mask = df['is_fraud'] == 1
        df.loc[fraud_mask, 'time_hour'] = np.random.choice([2, 3, 4, 5], fraud_mask.sum())
        
    elif pattern_type == 'velocity':
        # High frequency in short time
        fraud_mask = df['is_fraud'] == 1
        df.loc[fraud_mask, 'transaction_count_7d'] *= np.random.uniform(5, 15, fraud_mask.sum())
    
    return df


def get_aggregated_patterns(df: pd.DataFrame) -> Dict:
    """Calculate aggregated fraud patterns (without revealing individual data).
    
    Args:
        df: Transaction DataFrame
    
    Returns:
        Dictionary of aggregated statistics
    """
    fraud_df = df[df['is_fraud'] == 1]
    
    patterns = {
        'total_transactions': len(df),
        'fraud_count': len(fraud_df),
        'fraud_rate': len(fraud_df) / len(df) if len(df) > 0 else 0,
        'avg_fraud_amount': fraud_df['amount'].mean() if len(fraud_df) > 0 else 0,
        'fraud_by_hour': fraud_df.groupby('time_hour').size().to_dict(),
        'fraud_by_merchant': fraud_df.groupby('merchant_category').size().to_dict(),
    }
    
    return patterns
