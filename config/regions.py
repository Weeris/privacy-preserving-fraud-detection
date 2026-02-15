"""Region configurations for fraud detection PoC."""

REGIONS = {
    'TH': {
        'name': 'Thailand',
        'currency': 'THB',
        'central_bank': 'Bank of Thailand',
        'max_amount': 500000,
        'avg_amount': 5000,
    },
    'SG': {
        'name': 'Singapore',
        'currency': 'SGD',
        'central_bank': 'Monetary Authority of Singapore',
        'max_amount': 100000,
        'avg_amount': 2000,
    },
    'HK': {
        'name': 'Hong Kong',
        'currency': 'HKD',
        'central_bank': 'Hong Kong Monetary Authority',
        'max_amount': 800000,
        'avg_amount': 8000,
    },
}
