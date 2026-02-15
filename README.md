# Privacy-Preserving Fraud Detection

A proof-of-concept (PoC) for privacy-preserving fraud detection across Southeast Asian financial institutions using Federated Learning.

## Description

This project demonstrates a privacy-preserving approach to detecting financial fraud across multiple regions (Thailand, Singapore, Hong Kong) without sharing raw customer data. It uses Federated Learning (via Flower/FLWR) to train a collaborative fraud detection model while keeping each bank's data local and private.

## Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Bank (TH)      │     │  Bank (SG)      │     │  Bank (HK)      │
│  Local Data     │     │  Local Data     │     │  Local Data     │
│       ↓         │     │       ↓         │     │       ↓         │
│  Local Model    │     │  Local Model    │     │  Local Model    │
│       ↓         │     │       ↓         │     │       ↓         │
│  ┌───────┐      │     │  ┌───────┐      │     │  ┌───────┐      │
│  │ FLWR  │◄─────┼─────┼──│ FLWR  │◄─────┼─────┼──│ FLWR  │      │
│  │Client │      │     │  │Client │      │     │  │Client │      │
│  └───────┘      │     │  └───────┘      │     │  └───────┘      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         ↑                      ↑                       ↑
         └──────────────────────┼───────────────────────┘
                                ↓
                    ┌─────────────────────┐
                    │   Aggregation       │
                    │   Server (FLWR)    │
                    └─────────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/privacy-preserving-fraud-detection.git
cd privacy-preserving-fraud-detection

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

1. **Start the aggregation server:**
   ```bash
   python server.py
   ```

2. **Start client simulations (in separate terminals):**
   ```bash
   python client.py --region TH
   python client.py --region SG
   python client.py --region HK
   ```

3. **Launch the Streamlit dashboard:**
   ```bash
   streamlit run app.py
   ```

4. Open your browser at `http://localhost:8501`

## Demo Screenshots

The Streamlit dashboard provides:
- Real-time federated learning training progress
- Fraud detection metrics per region
- Model performance comparisons
- Data distribution visualization
- Synthetic data generation controls (using SDV)

## Tech Stack

- **Machine Learning:** PyTorch, scikit-learn
- **Federated Learning:** Flower (FLWR)
- **Visualization:** Plotly, Altair
- **Web App:** Streamlit
- **Data Synthesis:** SDV (Synthetic Data Vault)
- **Configuration:** PyYAML

## License

MIT License - see [LICENSE](LICENSE) file for details.
