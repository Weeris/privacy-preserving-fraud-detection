"""
Unit tests for core/synthetic.py
"""
import pytest
import pandas as pd
import numpy as np
from core.synthetic import (
    generate_transactions,
    add_fraud_patterns,
    get_aggregated_patterns,
)


class TestGenerateTransactions:
    """Tests for transaction data generation."""
    
    def test_generate_transactions_basic(self):
        """Test basic transaction generation."""
        df = generate_transactions('TH', n_samples=100)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 100
    
    def test_generate_transactions_columns(self):
        """Test generated data has required columns."""
        df = generate_transactions('TH', n_samples=50)
        
        required_cols = [
            'transaction_id', 'amount', 'frequency', 
            'merchant_category', 'time_hour', 'is_weekend',
            'avg_amount_7d', 'transaction_count_7d', 'amount_to_avg_ratio',
            'is_fraud'
        ]
        
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"
    
    def test_generate_transactions_thailand(self):
        """Test Thailand region generation."""
        df = generate_transactions('TH', n_samples=100)
        
        # Check transaction IDs
        assert df['transaction_id'].str.startswith('TXN_TH_').all()
    
    def test_generate_transactions_singapore(self):
        """Test Singapore region generation."""
        df = generate_transactions('SG', n_samples=100)
        assert df['transaction_id'].str.startswith('TXN_SG_').all()
    
    def test_generate_transactions_hongkong(self):
        """Test Hong Kong region generation."""
        df = generate_transactions('HK', n_samples=100)
        assert df['transaction_id'].str.startswith('TXN_HK_').all()
    
    def test_generate_transactions_amount_range(self):
        """Test transaction amounts are within expected range."""
        df = generate_transactions('TH', n_samples=100)
        
        assert df['amount'].min() >= 10  # Min amount
        assert df['amount'].max() <= 500000  # Max for TH
    
    def test_generate_transactions_time_hour(self):
        """Test time hour is valid."""
        df = generate_transactions('TH', n_samples=100)
        
        assert df['time_hour'].min() >= 0
        assert df['time_hour'].max() <= 23
    
    def test_generate_transactions_merchant_categories(self):
        """Test merchant categories are valid."""
        df = generate_transactions('TH', n_samples=100)
        valid_cats = ['retail', 'dining', 'travel', 'online', 'atm', 'transfer']
        
        assert df['merchant_category'].isin(valid_cats).all()
    
    def test_generate_transactions_fraud_labels(self):
        """Test fraud labels are binary."""
        df = generate_transactions('TH', n_samples=100)
        
        assert df['is_fraud'].isin([0, 1]).all()
    
    def test_generate_transactions_custom_fraud_rate(self):
        """Test custom fraud rate."""
        df_high = generate_transactions('TH', n_samples=1000, fraud_rate=0.5)
        df_low = generate_transactions('TH', n_samples=1000, fraud_rate=0.01)
        
        assert df_high['is_fraud'].mean() > df_low['is_fraud'].mean()
    
    def test_generate_transactions_reproducibility(self):
        """Test reproducibility with same seed."""
        # Note: Different regions have different seeds
        df1 = generate_transactions('TH', n_samples=50)
        df2 = generate_transactions('TH', n_samples=50)
        
        # Should be same (same region = same seed)
        assert df1['amount'].equals(df2['amount'])


class TestAddFraudPatterns:
    """Tests for adding fraud patterns."""
    
    def test_add_fraud_patterns_high_amount(self):
        """Test high amount fraud pattern."""
        df = generate_transactions('TH', n_samples=100, fraud_rate=0.5)
        df = add_fraud_patterns(df, pattern_type='high_amount')
        
        # Fraud transactions should have higher amounts
        fraud_df = df[df['is_fraud'] == 1]
        assert len(fraud_df) > 0
    
    def test_add_fraud_patterns_unusual_time(self):
        """Test unusual time fraud pattern."""
        df = generate_transactions('TH', n_samples=100, fraud_rate=0.5)
        df = add_fraud_patterns(df, pattern_type='unusual_time')
        
        assert 'time_hour' in df.columns
    
    def test_add_fraud_patterns_velocity(self):
        """Test velocity fraud pattern."""
        df = generate_transactions('TH', n_samples=100, fraud_rate=0.5)
        df = add_fraud_patterns(df, pattern_type='velocity')
        
        assert 'transaction_count_7d' in df.columns
    
    def test_add_fraud_patterns_preserves_dataframe(self):
        """Test pattern addition returns DataFrame."""
        df = generate_transactions('TH', n_samples=50)
        result = add_fraud_patterns(df)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)


class TestGetAggregatedPatterns:
    """Tests for aggregated pattern calculation."""
    
    def test_get_aggregated_patterns_basic(self):
        """Test basic aggregated patterns."""
        df = generate_transactions('TH', n_samples=100)
        patterns = get_aggregated_patterns(df)
        
        assert isinstance(patterns, dict)
        assert 'total_transactions' in patterns
        assert 'fraud_count' in patterns
        assert 'fraud_rate' in patterns
    
    def test_get_aggregated_patterns_count(self):
        """Test transaction counts."""
        df = generate_transactions('TH', n_samples=200)
        patterns = get_aggregated_patterns(df)
        
        assert patterns['total_transactions'] == 200
    
    def test_get_aggregated_patterns_fraud_count(self):
        """Test fraud count calculation."""
        df = generate_transactions('TH', n_samples=100, fraud_rate=0.1)
        patterns = get_aggregated_patterns(df)
        
        assert patterns['fraud_count'] >= 0
        assert patterns['fraud_count'] <= patterns['total_transactions']
    
    def test_get_aggregated_patterns_fraud_rate(self):
        """Test fraud rate calculation."""
        df = generate_transactions('TH', n_samples=1000, fraud_rate=0.05)
        patterns = get_aggregated_patterns(df)
        
        assert 0 <= patterns['fraud_rate'] <= 1
    
    def test_get_aggregated_patterns_hourly(self):
        """Test hourly fraud breakdown."""
        df = generate_transactions('TH', n_samples=100)
        patterns = get_aggregated_patterns(df)
        
        assert 'fraud_by_hour' in patterns
        assert isinstance(patterns['fraud_by_hour'], dict)
    
    def test_get_aggregated_patterns_merchant(self):
        """Test merchant category breakdown."""
        df = generate_transactions('TH', n_samples=100)
        patterns = get_aggregated_patterns(df)
        
        assert 'fraud_by_merchant' in patterns
        assert isinstance(patterns['fraud_by_merchant'], dict)
    
    def test_get_aggregated_patterns_empty(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame(columns=['amount', 'is_fraud', 'time_hour', 'merchant_category'])
        patterns = get_aggregated_patterns(df)
        
        assert patterns['total_transactions'] == 0
        assert patterns['fraud_count'] == 0
        assert patterns['fraud_rate'] == 0
