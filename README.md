# RELAY Module 6 – Two-Wheeler Fair Price Engine & ML Anomaly Detection

Production-ready implementation of **RELAY Module 6**, delivering valuation and machine-learning-based service history anomaly detection focused entirely on **Indian Two-Wheelers (Motorcycles and Scooters)**.

---

## 1. Architecture Overview

RELAY Module 6 consists of two complementary layers designed for high reliability, clean separation of concerns, and ease of backend integration:

1. **Python Flask Microservice (Port 5001)**
   - Houses the mathematical **Two-Wheeler Fair Price Engine** (`ai-module/price_engine.py`).
   - Houses the Scikit-learn **Isolation Forest Anomaly Detection Engine** (`ai-module/serve.py`, `ai-module/train.py`).
   - Exposes internal endpoints for price estimation and anomaly evaluation.

2. **Node.js Express API Gateway (Port 3000)**
   - Exposes public REST endpoints (`server.js`).
   - Provides input validation, sanitization, and upstream proxying with connection timeouts (5000ms) to the Python microservice.
   - Designed as a modular Express application that can be easily integrated into the main RELAY backend.

```
                           [Client / Consumer]
                                    |
                                    v (HTTP Port 3000)
                     +-------------------------------+
                     |  Node.js Express API Gateway  |
                     +---------------+---------------+
                                     |
                                     v (HTTP Port 5001, Timeout 5s)
                     +-------------------------------+
                     |   Python Flask Microservice   |
                     |  - Two-Wheeler Price Engine   |
                     |  - ML Anomaly Detector (IF)   |
                     +-------------------------------+
```

---

## 2. Technology Stack

- **Python**: 3.10+ (tested on Python 3.11.9)
- **Node.js**: 18+ (tested on Node.js v22.20.0)
- **Python Libraries**: `Flask==3.1.0`, `scikit-learn>=1.5`, `pandas>=2.2`, `numpy>=1.26`, `joblib>=1.4`, `pytest>=8.0`
- **Node.js Libraries**: `express==4.21.2` (native Fetch API)

---

## 3. Two-Wheeler Dataset Status & Market Representation

> [!IMPORTANT]
> **Synthetic Development Dataset**: The official 500-vehicle dataset containing 75 known anomalies has **not yet been provided**. In strict compliance with project guidelines:
> - A synthetic development dataset of **500 two-wheelers** (364 motorcycles, 136 scooters) and **75 seeded anomalies** was generated (`ai-module/generate_data.py`).
> - Performance metrics reported below reflect evaluation on a **held-out untouched synthetic test set** and must not be represented as validation against the official seed dataset.
> - The dataset strictly contains **two-wheelers only** (zero cars or SUVs).

### Realistic Used Two-Wheeler Market Distribution
Reflecting the Indian pre-owned two-wheeler market where older vehicles dominate:
- **0–1 year**: 5.0% (25 vehicles)
- **1–3 years**: 10.0% (50 vehicles)
- **3–5 years**: 20.0% (100 vehicles)
- **5–8 years**: 30.0% (150 vehicles)
- **8–12 years**: 25.0% (125 vehicles)
- **>12 years**: 10.0% (50 vehicles)
- **Total vehicles older than 3 years**: **85.0% (425 of 500 vehicles)**.
- **Older models**: 91.8% (459 vehicles) vs. **Newer models**: 8.2% (41 vehicles).
- **Valid Launch Years**: Every vehicle is verified against its historical launch year; no model is assigned to a year prior to its market introduction.

---

## 4. Feature Summary

### Feature A: Fair Price Engine (`price_engine.py`)
Calculates estimated fair market value range based on:
- **Benchmark Catalog**: Reference showroom pricing for **28 two-wheeler models across 7 major Indian OEMs**:
  - *Hero MotoCorp*: Splendor Plus, HF Deluxe, Passion Pro, Maestro, Xtreme 125R
  - *Bajaj Auto*: Pulsar 150, Platina 100, Discover 125, Pulsar N160, Pulsar NS200
  - *TVS Motor*: Apache RTR 160, Star City Plus, Jupiter, Scooty Pep+, Apache RTR 200 4V, Ntorq 125
  - *Honda (HMSI)*: Activa, CB Shine 125, Unicorn, SP 125
  - *Royal Enfield*: Classic 350, Bullet 350, Hunter 350, Meteor 350
  - *Suzuki*: Access 125, Avenis 125
  - *Yamaha*: R15 V4, MT-15 V2
- **Age Depreciation Bands**:
  - `0 – 6 months`: 95% remaining value
  - `6 – 12 months`: 85% remaining value
  - `1 – 2 years`: 80% remaining value
  - `2 – 3 years`: 70% remaining value
  - `3 – 4 years`: 60% remaining value
  - `4 – 5 years`: 50% remaining value
  - `> 5 years`: Tiered fallback assumption (*5–8y: 40%, 8–12y: 30%, >12y: 20%*) with explicit documentation.
- **Condition Adjustment**:
  - Excellent: `+5%`
  - Good: `+2%`
  - Average: `0%`
  - Poor: `-8%`
- **Odometer Integrity Adjustment**:
  - Normal: `0%`
  - Flagged: `-10%`
- **Fair Price Range**: `±8%` tolerance around the final adjusted price (`fairPriceLow` to `fairPriceHigh`).

### Feature B: ML Anomaly Detection (`train.py`, `serve.py`)
Uses Scikit-learn's `IsolationForest` to analyze 8 critical service history features:
1. `service_interval_mean`: Mean days between consecutive service dates.
2. `service_interval_std`: Standard deviation of service intervals.
3. `odometer_increment_mean`: Average kilometers driven between services.
4. `odometer_increment_std`: Standard deviation of odometer increments.
5. `confirmed_ratio`: Proportion of confirmed/verified service visits.
6. `unique_dealers`: Count of distinct servicing dealerships.
7. `gap_count`: Number of service gaps exceeding 365 days.
8. `service_regularity_score`: Normalized score reflecting interval regularity.

#### 3-Way Split & Evaluation Methodology:
- **Vehicle-Level Split**: 60% Training (300 vehicles), 20% Validation (100 vehicles), 20% Untouched Test (100 vehicles).
- **Preprocessing**: `StandardScaler` fitted strictly on Training data only (no data leakage).
- **Validation Tuning**: Contamination parameter tuned on Validation set (`0.20` selected for achieving 80.0% validation recall).
- **Untouched Test Evaluation (100 vehicles / 15 anomalies)**:
  - **Anomaly Recall**: **80.0%** (12 of 15 synthetic anomalies detected, surpassing ≥70% threshold)
  - **Anomaly Precision**: **52.2%** (12 TP / 23 total predicted anomalies)
  - **Anomaly F1-Score**: **0.6316**
  - **Overall Accuracy**: **86.0%**
  - **Confusion Matrix**: `[[74, 11], [3, 12]]` (74 True Normals, 12 True Anomalies, 11 False Positives, 3 False Negatives)

---

## 5. Quick Start Instructions (VS Code / Windows Terminal)

### Step 1: Open Terminal & Navigate to Project
```powershell
cd d:\relay-module6
```

### Step 2: Set Up Python Virtual Environment
```powershell
cd ai-module
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd ..
```

### Step 3: Generate Two-Wheeler Data & Train Model
```powershell
python ai-module\generate_data.py
python ai-module\train.py
```

### Step 4: Install Node.js Dependencies
In project root:
```powershell
cd d:\relay-module6
npm install
```

### Step 5: Run Automated Tests
Execute the complete test suites:

**Python Unit & Integration Tests (51 tests):**
```powershell
.\ai-module\.venv\Scripts\pytest.exe ai-module -v
```

**Node.js Gateway & End-to-End Tests (13 tests):**
```powershell
npm test
```

### Step 6: Start Services

**Terminal 1 (Flask Microservice):**
```powershell
cd d:\relay-module6\ai-module
.\.venv\Scripts\Activate.ps1
python serve.py
```
*Service starts on `http://127.0.0.1:5001`.*

**Terminal 2 (Node.js API Gateway):**
```powershell
cd d:\relay-module6
npm start
```
*Gateway starts on `http://localhost:3000`.*

---

## 6. API Reference

### 1. Health Check
`GET http://localhost:3000/health`
```json
{
  "gateway": "relay-node-gateway",
  "status": "ok",
  "upstreamStatus": "connected",
  "aiService": {
    "dataset": "synthetic_demo",
    "modelLoaded": true,
    "scalerLoaded": true,
    "service": "relay-ai-module",
    "status": "ok"
  }
}
```

### 2. Fair Price Calculation
`GET http://localhost:3000/api/v1/vehicles/:id/price`
Example: `GET http://localhost:3000/api/v1/vehicles/RELAY_TW_001/price`
```json
{
  "basis": "Two-wheeler showroom benchmark + Age depreciation + Condition adjustment + Odometer status adjustment (±8% range)",
  "currency": "INR",
  "details": {
    "adjustedFairPrice": 30600,
    "ageYears": 4.5,
    "assumptions": "Showroom prices are illustrative reference benchmarks. For vehicles >5 years, a documented tiered fallback is applied (5-8y: 40%, 8-12y: 30%, >12y: 20%), as the assignment defines depreciation bands only up to 5 years.",
    "basePriceAfterDepreciation": 30000,
    "condition": "Good",
    "conditionAdjustmentPercent": 2.0,
    "depreciationBand": "4–5 years",
    "isExtendedAgeFallback": false,
    "odometerAdjustmentPercent": 0.0,
    "odometerStatus": "Normal",
    "remainingValuePercent": 50.0,
    "showroomPrice": 60000
  },
  "fairPriceHigh": 33048,
  "fairPriceLow": 28152,
  "make": "Bajaj",
  "model": "Discover 125",
  "vehicleId": "RELAY_TW_001",
  "vehicleType": "Motorcycle"
}
```

### 3. Anomaly Detection
`GET http://localhost:3000/api/v1/vehicles/:id/anomaly-score`
Example: `GET http://localhost:3000/api/v1/vehicles/RELAY_TW_001/anomaly-score`
```json
{
  "anomalyScore": 0.0218,
  "dataNote": "Prediction computed against synthetic development dataset.",
  "extractedFeatures": {
    "confirmed_ratio": 0.2,
    "gap_count": 0,
    "odometer_increment_mean": 2998.4,
    "odometer_increment_std": 4.8,
    "service_interval_mean": 120.2,
    "service_interval_std": 1.1,
    "service_regularity_score": 0.99,
    "unique_dealers": 1
  },
  "reasons": [
    "Service intervals are unnaturally consistent (standard deviation < 8 days)",
    "Odometer increments are unnaturally uniform (standard deviation < 100 km)",
    "Low proportion of confirmed service records (20.0%)",
    "Single dealer appears across entire service history",
    "Service regularity score is abnormally rigid (potential synthetic records)"
  ],
  "scoreNote": "Negative decision function output from Scikit-Learn IsolationForest. Higher values indicate higher anomaly likelihood.",
  "vehicleId": "RELAY_TW_001",
  "verdict": "Suspicious"
}
```

---

## 7. Project File Structure

```text
relay-module6/
├── .gitignore                    # Git ignore file for Python/Node/OS artifacts
├── package.json                  # Node.js dependencies and scripts (Express)
├── package-lock.json             # Locked npm dependencies
├── README.md                     # Comprehensive project documentation
├── server.js                     # Express API Gateway (Port 3000)
├── test_gateway.js               # Node.js automated test suite & E2E runner (13 tests)
└── ai-module/                    # Python Flask AI/ML microservice (Port 5001)
    ├── AI_MODULE.md              # Detailed technical specification of AI microservice
    ├── generate_data.py          # Synthetic dataset generator (500 two-wheelers, 75 anomalies)
    ├── price_engine.py           # Reusable Two-Wheeler Fair Price engine (28 models, 7 OEMs)
    ├── requirements.txt          # Python package requirements
    ├── serve.py                  # Flask microservice implementation
    ├── train.py                  # Model training and 3-way split validation pipeline
    ├── test_flask_api.py         # Pytest suite for Flask endpoints (7 tests)
    ├── test_ml_anomaly.py        # Pytest suite for ML features & dataset integrity (16 tests)
    ├── test_price_engine.py      # Pytest suite for Two-Wheeler Fair Price engine (28 tests)
    ├── model.pkl                 # Persisted Scikit-learn IsolationForest model
    ├── scaler.pkl                # Persisted StandardScaler
    ├── service_history.csv       # Synthetic two-wheeler service history records
    ├── training_metrics.json     # Saved training metrics & confusion matrix
    ├── vehicle_features.csv      # Extracted 8 features per vehicle
    └── vehicles.json             # Two-wheeler metadata catalog
```

---

## 8. Integration & Production Deployment

To integrate Module 6 into the main RELAY backend:
1. **Mount Express Gateway Router**: Import `createApp` from `server.js` or mount its route handlers directly into your primary Express routing tree.
2. **Containerize Services**: Dockerize the Flask AI microservice and deploy behind an internal load balancer or private VPC.
3. **Configuration**: Use the `AI_URL` environment variable in Node.js to configure the private endpoint of the Flask service.
4. **Authentication**: Add JWT or API Key authentication middleware to the gateway routes before deploying to public staging/production.
