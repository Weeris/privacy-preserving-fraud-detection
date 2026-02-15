# Privacy-Preserving Financial Fraud Detection PoC

## 1. Project Overview

**Project Name:** Privacy-Preserving Fraud Detection with Federated Learning  
**Type:** Proof-of-Concept for Regulators  
**Core Functionality:** Cross-border fraud pattern sharing and federated learning system enabling banks across TH, SG, and HK to collaboratively detect fraud while preserving data privacy through differential privacy and synthetic data generation.  
**Target Users:** Financial regulators, compliance officers, and banking institutions in Asia-Pacific.

---

## 2. UI/UX Specification

### Layout Structure

**Page Sections:**
- **Header:** Project title, navigation tabs (Dashboard, FL Training, Pattern Sharing, Inference Demo)
- **Main Content Area:** Dynamic based on selected tab
- **Sidebar:** Configuration controls (region selection, epsilon slider, model parameters)
- **Footer:** Status indicators, version info

**Responsive Breakpoints:**
- Desktop: 1200px+ (full dashboard with all panels)
- Tablet: 768px-1199px (stacked panels)
- Mobile: <768px (single column, collapsible sections)

### Visual Design

**Color Palette:**
- Primary: `#1E3A5F` (Deep Navy - trust, security)
- Secondary: `#2D5A87` (Ocean Blue)
- Accent: `#00D4AA` (Teal - privacy/safety indicator)
- Background: `#0F1419` (Dark charcoal)
- Surface: `#1A2332` (Card backgrounds)
- Text Primary: `#E8EEF4`
- Text Secondary: `#8B9AAB`
- Success: `#10B981`
- Warning: `#F59E0B`
- Error: `#EF4444`
- Region Colors: TH `#F59E0B`, SG `#10B981`, HK `#EF4444`

**Typography:**
- Headings: Inter Bold, 24px/20px/16px (h1/h2/h3)
- Body: Inter Regular, 14px
- Code/Data: JetBrains Mono, 13px
- Line height: 1.5

**Spacing System:**
- Base unit: 8px
- Card padding: 24px
- Section gaps: 32px
- Element gaps: 16px

**Visual Effects:**
- Card shadows: `0 4px 24px rgba(0,0,0,0.3)`
- Border radius: 12px (cards), 8px (buttons), 4px (inputs)
- Hover transitions: 200ms ease-out
- Gradient accents: linear-gradient(135deg, #1E3A5F, #2D5A87)

### Components

**Navigation Tabs:**
- States: default (text secondary), hover (text primary + underline), active (accent color + bold)
- Icons: Streamlit icons (material icons)

**Cards:**
- Region cards with colored top border
- Metric cards with large numbers
- Training status cards with progress indicators

**Controls:**
- Sliders for epsilon (0.1 to 10.0)
- Dropdown for region selection
- Buttons for training actions

**Charts:**
- Line charts for training metrics
- Bar charts for region comparisons
- Heatmaps for pattern visualization

---

## 3. Functionality Specification

### Core Features

#### 3.1 Dashboard Tab
- Display 3 region cards (TH, SG, HK) showing:
  - Local fraud rate
  - Transactions processed
  - Model accuracy
  - Privacy budget remaining
- Real-time metrics updates
- Region comparison visualization

#### 3.2 Federated Learning Tab
- **Local Training Simulation:**
  - Button to trigger local model training per region
  - Training progress visualization
  - Loss/accuracy curves
  
- **Federated Aggregation:**
  - Display model weights from each client
  - Show FedAvg aggregation process
  - Display aggregated model metrics
  - Number of rounds slider (1-10)

#### 3.3 Pattern Sharing Tab
- **Differential Privacy Demo:**
  - Adjustable epsilon slider (0.1 - 10.0)
  - Show noise addition visualization
  - Display privacy budget consumption
  
- **Synthetic Data Generation:**
  - Generate synthetic fraud patterns
  - Compare original vs synthetic distributions
  - Privacy utility tradeoff visualization

#### 3.4 Inference Demo Tab
- **Transaction Input Form:**
  - Amount (numeric)
  - Frequency (transactions/day)
  - Time pattern (hour of day)
  - Merchant category (dropdown)
  - Geographic risk score
  - Account age
  
- **Prediction Output:**
  - Fraud probability (0-100%)
  - Risk level indicator (Low/Medium/High)
  - Model confidence
  - Explanation of key factors

### Data Handling

**Region Data Schemas:**

```python
# Thailand (TH)
{
    "transaction_id": str,
    "amount": float,  # THB
    "timestamp": datetime,
    "merchant_category": str,  # 1-8 (mcc codes simplified)
    "account_age_days": int,
    "transaction_frequency_daily": int,
    "hour_of_transaction": int,  # 0-23
    "is_fraud": bool
}

# Singapore (SG)
{
    "transaction_id": str,
    "amount": float,  # SGD
    "timestamp": datetime,
    "merchant_category": str,
    "account_age_days": int,
    "transaction_frequency_daily": int,
    "hour_of_transaction": int,
    "is_fraud": bool
}

# Hong Kong (HK)
{
    "transaction_id": str,
    "amount": float,  # HKD
    "timestamp": datetime,
    "merchant_category": str,
    "account_age_days": int,
    "transaction_frequency_daily": int,
    "hour_of_transaction": int,
    "is_fraud": bool
}
```

### Edge Cases
- Handle epsilon <= 0 (set minimum 0.1)
- Handle empty training data (show warning)
- Handle model convergence failure (fallback to baseline)
- Handle very small datasets (minimum 100 samples)

---

## 4. Technical Architecture

### Core Modules

#### 4.1 core/model.py
- `FraudDetectionMLP`: PyTorch neural network
  - Input: 6 features
  - Hidden layers: [64, 32, 16]
  - Output: 2 classes (fraud/not fraud)
  - ReLU activation, dropout 0.3

#### 4.2 core/federated_client.py
- `FLClient`: Flower client wrapper
  - Local data loading
  - Local model training
  - Weight serialization for server

#### 4.3 core/federated_server.py
- `FLServer`: Flower server with FedAvg
  - Client management
  - Weight aggregation
  - Global model distribution

#### 4.4 core/privacy.py
- `add_differential_noise()`: Laplace mechanism
- `compute_privacy_budget()`: epsilon tracking
- `apply_dp_to_gradients()`: gradient perturbation

#### 4.5 core/synthetic.py
- `SyntheticDataGenerator`: CTGAN-style generation
- Feature distribution learning
- Privacy-preserving sample generation

### Configuration

**config/regions.yaml:**
```yaml
thailand:
  currency: THB
  timezone: Asia/Bangkok
  mcc_categories: ["retail", "dining", "travel", "utility", "entertainment", "healthcare", "education", "other"]
  risk_weights: {high_risk_merchants: ["entertainment", "travel"], avg_transaction: 500}

singapore:
  currency: SGD
  timezone: Asia/Singapore
  mcc_categories: ["retail", "dining", "travel", "utility", "entertainment", "healthcare", "education", "other"]
  risk_weights: {high_risk_merchants: ["travel"], avg_transaction: 100}

hong_kong:
  currency: HKD
  timezone: Asia/Hong_Kong
  mcc_categories: ["retail", "dining", "travel", "utility", "entertainment", "healthcare", "education", "other"]
  risk_weights: {high_risk_merchants: ["retail", "entertainment"], avg_transaction: 800}
```

---

## 5. Acceptance Criteria

### Visual Checkpoints
- [ ] Dashboard shows 3 distinct region cards with correct colors
- [ ] Navigation tabs switch content smoothly
- [ ] Sliders and controls are responsive
- [ ] Charts render correctly with data

### Functional Checkpoints
- [ ] Local training runs and shows metrics
- [ ] Federated aggregation produces global model
- [ ] Differential privacy adds visible noise at low epsilon
- [ ] Synthetic data preserves pattern distribution
- [ ] Inference returns fraud probability
- [ ] Epsilon slider affects privacy/utility tradeoff

### Compliance Features
- [ ] Privacy budget tracking visible
- [ ] No raw data exposed in UI
- [ ] Clear privacy indicators

---

## 6. GitHub Repository

**Repository Name:** `privacy-preserving-fraud-detection`  
**License:** MIT  
**Initial Commit:** All project files  
**Description:** Privacy-preserving financial fraud detection using federated learning and differential privacy for cross-border regulatory compliance in Asia-Pacific.
