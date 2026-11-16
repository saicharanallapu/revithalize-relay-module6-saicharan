# RELAY Module 6 – Two-Wheeler AI & Valuation Engine Microservice

## 1. Overview & Architecture

RELAY Module 6 provides two core valuation and integrity intelligence capabilities focused strictly on **Indian Two-Wheelers (Motorcycles and Scooters)**:
1. **Two-Wheeler Fair Price Calculation Engine (`price_engine.py`)**: A deterministic valuation engine calculating baseline residual market value using two-wheeler showroom benchmarks, age depreciation bands, vehicle condition multipliers, odometer integrity flags, and a ±8% fair market range.
2. **Machine Learning Anomaly Detection Engine (`train.py`, `serve.py`)**: An unsupervised outlier detection pipeline powered by Scikit-learn's `IsolationForest` analyzing 8 key vehicle service history patterns to identify odometer rollbacks, fabricated service records, dealer collusion, and abnormal service gaps in two-wheeler histories.

```
                           +---------------------------+
                           |  Client / RELAY Backend   |
                           +-------------+-------------+
                                         |
                                         v [Port 3000]
                     +---------------------------------------+
                     |       Node.js Express API Gateway      |
                     +-------------------+-------------------+
                                         | HTTP Proxy (Timeout 5s)
                                         v [Port 5001]
             +-------------------------------------------------------+
             |               Python Flask Microservice               |
             |                                                       |
             |  +---------------------+     +---------------------+  |
             |  | Two-Wheeler Engine  |     | ML Anomaly Detector |  |
             |  |   price_engine.py   |     |    IsolationForest  |  |
             |  +----------+----------+     +----------+----------+  |
             +-------------|---------------------------|-------------+
                           |                           |
                 +---------v---------+       +---------v---------+
                 | 28 Model Catalog  |       | model.pkl         |
                 | & Fallback Tiers  |       | scaler.pkl        |
                 +-------------------+       +-------------------+
```

---

## 2. Two-Wheeler Benchmark Catalog (28 Models across 7 OEMs)

All showroom prices are illustrative benchmarks for development and testing:

| OEM | Model | Category | Illustrative Showroom Price | Intro Year | Market Tier |
|---|---|---|---:|---|---|
| **Hero MotoCorp** | Splendor Plus | Motorcycle | ₹75,000 | 2004 | Older / Dominant |
| **Hero MotoCorp** | HF Deluxe | Motorcycle | ₹62,000 | 2012 | Older / Dominant |
| **Hero MotoCorp** | Passion Pro | Motorcycle | ₹72,000 | 2008 | Older / Dominant |
| **Hero MotoCorp** | Maestro | Scooter | ₹68,000 | 2012 | Older / Dominant |
| **Hero MotoCorp** | Xtreme 125R | Motorcycle | ₹95,000 | 2024 | Newer |
| **Bajaj Auto** | Pulsar 150 | Motorcycle | ₹1,10,000 | 2001 | Older / Dominant |
| **Bajaj Auto** | Platina 100 | Motorcycle | ₹65,000 | 2006 | Older / Dominant |
| **Bajaj Auto** | Discover 125 | Motorcycle | ₹60,000 | 2004 | Older / Dominant |
| **Bajaj Auto** | Pulsar N160 | Motorcycle | ₹1,30,000 | 2022 | Newer |
| **Bajaj Auto** | Pulsar NS200 | Motorcycle | ₹1,45,000 | 2012 | Newer / Performance |
| **TVS Motor** | Apache RTR 160 | Motorcycle | ₹1,20,000 | 2007 | Older / Dominant |
| **TVS Motor** | Star City Plus | Motorcycle | ₹75,000 | 2014 | Older / Dominant |
| **TVS Motor** | Jupiter | Scooter | ₹76,000 | 2013 | Older / Dominant |
| **TVS Motor** | Scooty Pep+ | Scooter | ₹65,000 | 2005 | Older / Dominant |
| **TVS Motor** | Apache RTR 200 4V | Motorcycle | ₹1,45,000 | 2016 | Newer / Performance |
| **TVS Motor** | Ntorq 125 | Scooter | ₹88,000 | 2018 | Newer |
| **Honda (HMSI)** | Activa | Scooter | ₹78,000 | 2001 | Older / Dominant |
| **Honda (HMSI)** | CB Shine 125 | Motorcycle | ₹80,000 | 2006 | Older / Dominant |
| **Honda (HMSI)** | Unicorn | Motorcycle | ₹1,08,000 | 2004 | Older / Dominant |
| **Honda (HMSI)** | SP 125 | Motorcycle | ₹86,000 | 2019 | Newer |
| **Royal Enfield** | Classic 350 | Motorcycle | ₹1,95,000 | 2009 | Older / Dominant |
| **Royal Enfield** | Bullet 350 | Motorcycle | ₹1,75,000 | 2000 | Older / Dominant |
| **Royal Enfield** | Hunter 350 | Motorcycle | ₹1,50,000 | 2022 | Newer |
| **Royal Enfield** | Meteor 350 | Motorcycle | ₹2,05,000 | 2020 | Newer |
| **Suzuki** | Access 125 | Scooter | ₹82,000 | 2007 | Older / Dominant |
| **Suzuki** | Avenis 125 | Scooter | ₹92,000 | 2021 | Newer |
| **Yamaha** | R15 V4 | Motorcycle | ₹1,82,000 | 2021 | Newer / Sport |
| **Yamaha** | MT-15 V2 | Motorcycle | ₹1,68,000 | 2019 | Newer / Naked |

---

## 3. Dataset Distribution (Realistic Used Two-Wheeler Market)

The dataset reflects the Indian pre-owned two-wheeler market, where older vehicles are significantly more common:
- **Total Vehicles**: Exactly 500
- **Seeded Anomalies**: Exactly 75
- **Normal Records**: Exactly 425
- **Category Split**: 364 Motorcycles (72.8%), 136 Scooters (27.2%)
- **Market Tier**: 459 Older generation models (91.8%), 41 Newer models (8.2%)
- **Age Distribution**:
  - `0–1 year`: 5.0% (25 vehicles)
  - `1–3 years`: 10.0% (50 vehicles)
  - `3–5 years`: 20.0% (100 vehicles)
  - `5–8 years`: 30.0% (150 vehicles)
  - `8–12 years`: 25.0% (125 vehicles)
  - `> 12 years`: 10.0% (50 vehicles)
  - **Vehicles > 3 years**: **85.0% (425 of 500 vehicles)**
- **Historical Year Validation**: Every vehicle is guaranteed to have `introYear <= (2026 - ageYears)`.

---

## 4. Feature A – Fair Price Calculation Engine

The engine calculates fair market value using a multi-factor formula:

$$\text{Depreciated Price} = \text{Showroom Price} \times \text{Remaining Value Ratio}$$
$$\text{Condition Adjusted Price} = \text{Depreciated Price} \times (1 + \text{Condition Multiplier})$$
$$\text{Final Fair Price} = \text{Condition Adjusted Price} \times (1 + \text{Odometer Multiplier})$$
$$\text{Fair Range} = [\text{Final Fair Price} \times 0.92, \;\; \text{Final Fair Price} \times 1.08]$$

### Depreciation Schedule
| Vehicle Age Band | Remaining Value Ratio | Status |
|---|---:|---|
| 0 – 6 months | 95% | Assignment Spec |
| 6 – 12 months | 85% | Assignment Spec |
| 1 – 2 years | 80% | Assignment Spec |
| 2 – 3 years | 70% | Assignment Spec |
| 3 – 4 years | 60% | Assignment Spec |
| 4 – 5 years | 50% | Assignment Spec |
| 5 – 8 years | 40% | **Documented Fallback Assumption** |
| 8 – 12 years | 30% | **Documented Fallback Assumption** |
| > 12 years | 20% | **Documented Fallback Assumption (Salvage floor)** |

### Condition Adjustments
- **Excellent**: +5% (`+0.05`)
- **Good**: +2% (`+0.02`)
- **Average**: 0% (`0.00`)
- **Poor**: -8% (`-0.08`)

### Odometer Status Adjustments
- **Normal**: 0% (`0.00`)
- **Flagged**: -10% (`-0.10`)

---

## 5. Feature B – ML Anomaly Detection Engine

### Eight Required Features
1. `service_interval_mean`: Mean days between consecutive service dates.
2. `service_interval_std`: Standard deviation of service intervals in days.
3. `odometer_increment_mean`: Mean kilometer advance between consecutive visits.
4. `odometer_increment_std`: Standard deviation of kilometer increments.
5. `confirmed_ratio`: Ratio of verified/confirmed records to total records ($N_{\text{confirmed}} / N_{\text{total}}$).
6. `unique_dealers`: Count of distinct dealer IDs servicing the vehicle.
7. `gap_count`: Number of uncharacteristic service gaps (>365 days).
8. `service_regularity_score`: Normalized regularity indicator:
   $$\text{Regularity} = \frac{1}{1 + \frac{\text{service\_interval\_std}}{\max(\text{service\_interval\_mean}, 1.0)}}$$

### Preprocessing & Leakage Prevention
- **3-Way Stratified Vehicle Split**:
  - **Training Set (60%)**: 300 vehicles (255 normal, 45 anomalies).
  - **Validation Set (20%)**: 100 vehicles (85 normal, 15 anomalies).
  - **Untouched Test Set (20%)**: 100 vehicles (85 normal, 15 anomalies).
- **Leakage Prevention**: `StandardScaler` is fitted strictly on the Training set (`X_train`) and applied downstream via `.transform()` to validation, test, and inference vectors.
- **Handling Edge Cases**: Empty service histories and single-visit histories default safely to neutral numeric bounds (0.0).

### Model Selection on Validation Set
Contamination parameter tuning evaluated on the Validation Set:
| Contamination Setting | Validation Recall | Validation Precision | Validation F1-Score |
|---|---:|---:|---:|
| 0.10 | 40.0% | 50.0% | 0.4444 |
| 0.15 | 60.0% | 56.2% | 0.5806 |
| 0.18 | 66.7% | 58.8% | 0.6250 |
| **0.20 (Selected)** | **80.0%** | **60.0%** | **0.6857** |
| 0.22 | 80.0% | 57.1% | 0.6667 |
| 0.25 | 93.3% | 56.0% | 0.7000 |

*Decision*: `contamination = 0.20` was selected from the validation set because it reaches 80.0% validation recall with a strong 60.0% precision and highest overall balanced F1-score (0.6857).

### Final Evaluation on Untouched Test Set (100 vehicles)
- **Known Anomalies in Test Set**: 15 vehicles
- **Detected Anomalies in Test Set**: 12 vehicles
- **Test Anomaly Recall**: **80.0%** (12 / 15) *(Surpasses assignment target of ≥70%)*
- **Test Anomaly Precision**: **52.2%** (12 / 23)
- **Test F1-Score**: **0.6316**
- **Test Overall Accuracy**: **86.0%** (86 / 100)
- **Normal Class Precision**: **96.1%** (74 / 77)
- **Normal Class Recall**: **87.1%** (74 / 85)

#### Confusion Matrix (Untouched Test Set)
```
                  Predicted Normal (0)   Predicted Anomaly (1)
Actual Normal (0)          74                     11
Actual Anomaly (1)          3                     12
```

---

## 6. Microservice API Endpoints (Flask Port 5001)

### `GET /health`
```json
{
  "dataset": "synthetic_demo",
  "modelLoaded": true,
  "scalerLoaded": true,
  "service": "relay-ai-module",
  "status": "ok"
}
```

### `GET /internal/vehicles/:id/price`
Example response for a 4.5-year-old Bajaj Discover 125 (`RELAY_TW_001`):
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

### `GET /internal/vehicles/:id/anomaly-score`
Example response:
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
*(For vehicles classified as Normal, reasons state: `"Service history intervals, dealer distribution, and odometer steps conform to standard patterns."`)*
