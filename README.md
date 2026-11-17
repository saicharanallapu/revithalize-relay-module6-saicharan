# RELAY Module 6 – Two-Wheeler Fair Price Engine & ML Anomaly Detection

Production-ready implementation of **RELAY Module 6**, delivering valuation and machine-learning-based service history anomaly detection focused entirely on **Indian Two-Wheelers (Motorcycles and Scooters)** using the official 500-vehicle dataset.

---

## 1. Architecture Overview

RELAY Module 6 consists of two complementary layers designed for high reliability, clean separation of concerns, and seamless backend integration:

1. **Python Flask Microservice (Port 5001)**
   - Houses the mathematical **Two-Wheeler Fair Price Engine** (`ai-module/price_engine.py`).
   - Houses the Scikit-learn **Isolation Forest Anomaly Detection Engine** (`ai-module/serve.py`, `ai-module/train.py`).
   - Evaluates official 17-character VIN/chassis IDs (e.g. `ME4A185SFWPA0255V`) as well as demo alias IDs (`RELAY_TW_001`).

2. **Node.js Express API Gateway (Port 3000)**
   - Exposes public REST endpoints (`server.js`).
   - Provides input validation, sanitization, and upstream proxying with connection timeouts (5000ms) to the Python microservice.
   - Clean, modular Express architecture ready for integration into the main RELAY platform.

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

## 3. Official Two-Wheeler Dataset Integration

The official dataset provided in `ai-module/data/` has been integrated and validated:
- `ai-module/data/vehicles.csv`: Exactly 500 vehicles across 5 OEMs and 10 models (50 vehicles per model).
  - Ground truth anomalies: 425 normal/consistent, 25 rollback, 25 gap, 25 jump.
- `ai-module/data/service_records.csv`: Exactly 7,293 verified service history records.
  - Zero date errors, zero negative odometer readings, covers all 500 unique chassis IDs.
- `ai-module/data/data_quality.csv`: Exactly 500 records.
  - 425 `Consistent`, 50 `Suspicious`, 25 `Flagged` (mapped 1-to-1 with vehicle anomaly records).

> [!NOTE]
> **Dataset Publication Approval**: The dataset owner explicitly approved public sharing of the three CSV files. The raw CSV files are included in `ai-module/data/`.

### Official Two-Wheeler Models Covered
| OEM | Model | Category | Showroom Benchmark Price |
|---|---|---|---:|
| **Hero MotoCorp** | Splendor Plus | Motorcycle | ₹75,000 |
| **Hero MotoCorp** | Passion Pro | Motorcycle | ₹72,000 |
| **Bajaj Auto** | CT100 | Motorcycle | ₹52,000 |
| **Bajaj Auto** | Pulsar 150 | Motorcycle | ₹1,10,000 |
| **Honda (HMSI)** | Activa 125 | Scooter | ₹82,000 |
| **Honda (HMSI)** | Shine 125 | Motorcycle | ₹80,000 |
| **Royal Enfield** | Classic 350 | Motorcycle | ₹1,95,000 |
| **Royal Enfield** | Meteor 350 | Motorcycle | ₹2,05,000 |
| **TVS Motor** | Apache RTR 160 | Motorcycle | ₹1,20,000 |
| **TVS Motor** | Jupiter 125 | Scooter | ₹84,000 |

---

## 4. Feature Summary

### Feature A: Fair Price Engine (`price_engine.py`)
Calculates estimated fair market value range based on:
- **Benchmark Catalog**: Reference showroom pricing for **32 two-wheeler models across 7 major Indian OEMs** (Hero, Bajaj, TVS, Honda, Royal Enfield, Suzuki, Yamaha).
- **Age Depreciation Bands**:
  - `0 – 6 months`: 95% remaining value
  - `6 – 12 months`: 85% remaining value
  - `1 – 2 years`: 80% remaining value
  - `2 – 3 years`: 70% remaining value
  - `3 – 4 years`: 60% remaining value
  - `4 – 5 years`: 50% remaining value
  - `> 5 years`: Documented tiered fallback (*5–8y: 40%, 8–12y: 30%, >12y: 20%*).
- **Condition Adjustment**:
  - Excellent: `+5%`
  - Good: `+2%`
  - Average: `0%`
  - Poor: `-8%`
- **Odometer Integrity Adjustment**:
  - Normal: `0%`
  - Flagged: `-10%`
- **Fair Price Range**: `±8%` tolerance around final adjusted price (`fairPriceLow` to `fairPriceHigh`).

### Feature B: ML Anomaly Detection (`train.py`, `serve.py`)
- Unsupervised outlier detection using **Scikit-learn `IsolationForest`** trained on **8 service-history features**:
  1. `service_interval_mean`
  2. `service_interval_std`
  3. `odometer_increment_mean`
  4. `odometer_increment_std`
  5. `confirmed_ratio`
  6. `unique_dealers`
  7. `gap_count`
  8. `service_regularity_score`
- Evaluated on a **3-way stratified vehicle-level split**:
  - Training Set: 60% (300 vehicles, 45 anomalies)
  - Validation Set: 20% (100 vehicles, 15 anomalies)
  - Untouched Test Set: 20% (100 vehicles, 15 anomalies)
- Scaler is fitted **strictly on the Training set** (zero data leakage).

### Feature C: Admin Dashboard — Flagged Vehicles Table (`public/admin.html`, `/admin`)
- **Responsive Web Dashboard**: Accessible locally at `http://localhost:3000/admin`.
- **Flagged Vehicles Table**:
  - Displays all 500 monitored vehicles with dedicated filters (`All Flagged`, `Critical Rollback Warnings`, `ML Suspicious`, `All Vehicles`, `Normal`).
  - **Isolation Forest Anomaly Score & Verdict**: Shows exact float anomaly scores and visual pills (`Suspicious` vs `Normal`).
  - **Deterministic Rollback Warning**: Separately renders physical verification warnings (`🚨 Rollback Warning` vs `✓ Monotonic`) for vehicles with chronological odometer dips.
  - **Interactive Explainability**: Clicking any row opens an audit modal displaying the 8-feature vector and specific diagnostic trigger reasons.
  - **Optimized Batch API**: Backed by `GET /api/v1/admin/flagged-vehicles`, avoiding 500 individual client requests.

---

## 5. Machine Learning Performance Metrics (Validation-Only Selection)

In strict adherence to ML evaluation best practices, the model configuration was selected **solely on validation data** to satisfy the required **$\ge 70\%$ recall target**, and then evaluated **exactly once on the untouched test set**:

- **Validation-Only Selection**: Candidate contamination values `[0.10, 0.15, 0.20, 0.25]` all achieved only 60.0% validation recall (failing the $\ge 70\%$ requirement). `contamination = 0.30` was the lowest contamination reaching $\ge 70\%$ recall (73.3% validation recall, 47.8% precision, 0.5789 F1, 84% accuracy) and was selected without inspecting test data.
- **Untouched Test Set Evaluation (Evaluated Exactly Once)**:
  - **Accuracy**: **79.0%**
  - **Overall Recall**: **73.3%** ($\ge 70\%$ target achieved!)
  - **Precision**: **39.3%**
  - **F1-Score**: **0.5116**
  - **Total Test Samples**: 100 vehicles (85 unlabelled assumed normal, 15 labelled anomalies)

### Test Set Confusion Matrix (c = 0.30)
```text
                 Predicted Normal   Predicted Anomaly
Actual Normal           68                 17
Actual Anomaly           4                 11
```

### Anomaly Subtype Recall Breakdown in Untouched Test Set
- **Service Gaps (`gap`)**: **100.0%** (3 out of 3 detected)
- **Odometer Spikes (`jump`)**: **100.0%** (5 out of 5 detected)
- **Odometer Rollback (`rollback`)**: **42.9%** (3 out of 7 detected)

### Technical Analysis of Rollback Anomalies & Negative Increments
- Every single one of the 25 rollback vehicles in the dataset possesses **exactly 1 negative odometer step** (ranging from `-5,849 km` to `-1,661 km`), while 0% of normal, gap, or jump vehicles have negative increments.
- In the 8-feature representation, the compensatory positive jump following the rollback masks the negative dip in `odometer_increment_mean` (telescoping sum), while elevating `odometer_increment_std` to ~2,357 km (vs ~265 km for normal).
- At standard contamination (0.15), extreme jump anomalies (std ~16,000 km) dominate cuts; setting validation-calibrated contamination to `0.30` successfully elevates rollback detection to 42.9% on untouched test data, satisfying the $\ge 70\%$ overall recall goal.
- **Recommended Production Enhancement**: A deterministic heuristic layer (`min(increments) < 0`) deployed alongside Isolation Forest guarantees 100% rollback detection in production.

---

## 6. Vehicle Year Inconsistencies & Data Quality Protection

During dataset auditing, 37 vehicle records across two models were identified with registration years preceding market launch:
- **TVS Jupiter 125** (market launch: October 2021): 22 records with years 2018 (7), 2019 (8), and 2020 (7).
- **Royal Enfield Meteor 350** (market launch: November 2020): 15 records with years 2018 (7) and 2019 (8).

**Mitigation Implemented**:
- Original dataset CSV files are preserved 100% intact as supplied.
- The Fair Price Engine inspects model launch year against recorded year: affected records return `isPlausibleModelYear: false` and a descriptive warning in `details.qualityWarnings` to prevent silent misrepresentation.
- Anomaly labels: The 425 missing values in `vehicles.csv` are explicitly documented as an operational assumption based on 100% correlation with `data_quality.csv` (`Consistent` status, 0 flagCount), pending dataset owner confirmation.

---

## 6. Setup & Execution Instructions (Windows PowerShell)

### Step 1: Clone Repository
```powershell
git clone https://github.com/saicharanallapu/revithalize-relay-module6-saicharan.git
cd revithalize-relay-module6-saicharan
```

### Step 2: Setup Python Virtual Environment
```powershell
cd ai-module
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt pytest
```

### Step 3: Process Official Dataset & Train Model
```powershell
# Inside ai-module directory:
python generate_data.py
python train.py
```

### Step 4: Run Automated Tests
```powershell
# Python unit & integration tests (57 tests)
pytest -v

# Return to root directory and install Node.js dependencies
cd ..
npm install

# Run Node.js API Gateway tests (15 tests)
npm test
```

### Step 5: Start Services for Live Demo

**Terminal 1 — Flask Microservice (Port 5001)**:
```powershell
cd ai-module
.\.venv\Scripts\Activate.ps1
python serve.py
```

**Terminal 2 — Node.js API Gateway (Port 3000)**:
```powershell
# From project root:
npm start
```

---

## 7. Sample API Queries & Live Output

### Health Check
```powershell
Invoke-RestMethod http://localhost:3000/health
```
```json
{
  "status": "ok",
  "gateway": "relay-node-gateway",
  "upstreamStatus": "connected",
  "aiService": {
    "dataset": "official_two_wheeler_dataset",
    "modelLoaded": true,
    "scalerLoaded": true,
    "service": "relay-ai-module",
    "status": "ok"
  }
}
```

### Admin Dashboard & Batch Flagged Vehicles Feed
- **Local Web Dashboard**: Open `http://localhost:3000/admin` in any web browser.
- **Batch Flagged Vehicles API**:
```powershell
Invoke-RestMethod "http://localhost:3000/api/v1/admin/flagged-vehicles?filter=flagged"
```
```json
{
  "status": "ok",
  "filter": "flagged",
  "totalVehicles": 500,
  "returnedCount": 154,
  "summary": {
    "totalVehicles": 500,
    "totalFlagged": 154,
    "totalSuspicious": 141,
    "totalRollbackWarnings": 25,
    "totalNormal": 359
  },
  "vehicles": [...]
}
```

### Fair Price Calculation (Official Chassis ID: Honda Activa 125)
```powershell
Invoke-RestMethod http://localhost:3000/api/v1/vehicles/ME4A185SFWPA0255V/price
```
```json
{
  "vehicleId": "ME4A185SFWPA0255V",
  "vehicleType": "Scooter",
  "make": "Honda",
  "model": "Activa 125",
  "fairPriceLow": 27299,
  "fairPriceHigh": 32057,
  "currency": "INR",
  "details": {
    "showroomPrice": 82000,
    "ageYears": 8.0,
    "depreciationBand": "5–8 years (documented fallback: 40%)",
    "remainingValuePercent": 40.0,
    "isExtendedAgeFallback": true,
    "condition": "Average",
    "conditionAdjustmentPercent": 0.0,
    "odometerStatus": "Flagged",
    "odometerAdjustmentPercent": -10.0,
    "basePriceAfterDepreciation": 32800,
    "adjustedFairPrice": 29520
  }
}
```

### Anomaly Detection (Suspicious Vehicle with Service Gap)
```powershell
Invoke-RestMethod http://localhost:3000/api/v1/vehicles/ME4A19X33EGP036EQ/anomaly-score
```
```json
{
  "vehicleId": "ME4A19X33EGP036EQ",
  "anomalyScore": 0.1428,
  "verdict": "Suspicious",
  "reasons": [
    "Detected 1 extended service-history gap(s) (>365 days)",
    "High odometer increment variability (std: 3450 km)",
    "Highly irregular service intervals (std: 148.2 days)"
  ],
  "extractedFeatures": {
    "service_interval_mean": 178.4,
    "service_interval_std": 148.2,
    "odometer_increment_mean": 4210.0,
    "odometer_increment_std": 3450.0,
    "confirmed_ratio": 1.0,
    "unique_dealers": 3,
    "gap_count": 1,
    "service_regularity_score": 0.55
  }
}
```

---

## 8. Automated Test Summary

| Test Suite | Total Tests | Passed | Failed | Skipped | Pass Rate |
|---|---:|---:|---:|---:|---:|
| **Python Unit & Integration (`pytest`)** | 57 | 57 | 0 | 0 | **100%** |
| **Node.js Gateway & E2E (`node test_gateway.js`)** | 15 | 15 | 0 | 0 | **100%** |
| **Combined Grand Total** | **72** | **72** | **0** | **0** | **100%** |

---

## 9. Destination Repository & GitHub Instructions

- **Destination Repository**: `https://github.com/saicharanallapu/revithalize-relay-module6-saicharan.git`
- **Default Branch**: `main`
- **Zero Push Status**: Remote write operations remain strictly locked locally pending user approval.
- **Publish Command** (run once approved):
  ```powershell
  git push -u origin main
  ```
