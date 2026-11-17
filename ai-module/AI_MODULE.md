# RELAY Module 6 – Two-Wheeler AI & Valuation Engine Microservice

## 1. Overview & Architecture

RELAY Module 6 provides two core valuation and integrity intelligence capabilities focused strictly on **Indian Two-Wheelers (Motorcycles and Scooters)** using the official 500-vehicle dataset:
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
                 | 32 Model Catalog  |       | model.pkl         |
                 | & Fallback Tiers  |       | scaler.pkl        |
                 +-------------------+       +-------------------+
```

---

## 2. Two-Wheeler Benchmark Catalog (32 Models across 7 OEMs)

All showroom prices are illustrative benchmarks for development, testing, and evaluation:

| OEM | Model | Category | Illustrative Showroom Price | Intro Year | Market Tier |
|---|---|---|---:|---|---|
| **Hero MotoCorp** | Splendor Plus | Motorcycle | ₹75,000 | 2004 | Older / Dominant |
| **Hero MotoCorp** | HF Deluxe | Motorcycle | ₹62,000 | 2012 | Older / Dominant |
| **Hero MotoCorp** | Passion Pro | Motorcycle | ₹72,000 | 2008 | Older / Dominant |
| **Hero MotoCorp** | Maestro | Scooter | ₹68,000 | 2012 | Older / Dominant |
| **Hero MotoCorp** | Xtreme 125R | Motorcycle | ₹95,000 | 2024 | Newer |
| **Bajaj Auto** | CT100 | Motorcycle | ₹52,000 | 2001 | Older / Dominant |
| **Bajaj Auto** | Pulsar 150 | Motorcycle | ₹1,10,000 | 2001 | Older / Dominant |
| **Bajaj Auto** | Platina 100 | Motorcycle | ₹65,000 | 2006 | Older / Dominant |
| **Bajaj Auto** | Discover 125 | Motorcycle | ₹60,000 | 2004 | Older / Dominant |
| **Bajaj Auto** | Pulsar N160 | Motorcycle | ₹1,30,000 | 2022 | Newer |
| **Bajaj Auto** | Pulsar NS200 | Motorcycle | ₹1,45,000 | 2012 | Newer / Performance |
| **TVS Motor** | Apache RTR 160 | Motorcycle | ₹1,20,000 | 2007 | Older / Dominant |
| **TVS Motor** | Star City Plus | Motorcycle | ₹75,000 | 2014 | Older / Dominant |
| **TVS Motor** | Jupiter | Scooter | ₹76,000 | 2013 | Older / Dominant |
| **TVS Motor** | Jupiter 125 | Scooter | ₹84,000 | 2021 | Newer |
| **TVS Motor** | Scooty Pep+ | Scooter | ₹65,000 | 2005 | Older / Dominant |
| **TVS Motor** | Apache RTR 200 4V | Motorcycle | ₹1,45,000 | 2016 | Newer / Performance |
| **TVS Motor** | Ntorq 125 | Scooter | ₹88,000 | 2018 | Newer |
| **Honda (HMSI)** | Activa | Scooter | ₹78,000 | 2001 | Older / Dominant |
| **Honda (HMSI)** | Activa 125 | Scooter | ₹82,000 | 2014 | Older / Dominant |
| **Honda (HMSI)** | CB Shine 125 | Motorcycle | ₹80,000 | 2006 | Older / Dominant |
| **Honda (HMSI)** | Shine 125 | Motorcycle | ₹80,000 | 2006 | Older / Dominant |
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

## 3. Official Dataset Structure & Empirical Mapping

The official dataset located in `ai-module/data/` includes 3 primary files:
1. `vehicles.csv` (500 rows, 6 columns): `chassis`, `oem`, `model`, `year`, `registration`, `anomaly`.
2. `service_records.csv` (7,293 rows, 6 columns): `chassis`, `serviceDate`, `odometerKm`, `serviceType`, `dealerCode`, `confirmed`.
3. `data_quality.csv` (500 rows, 7 columns): `chassis`, `oem`, `model`, `year`, `careScore`, `odometerStatus`, `flagCount`.

### Anomaly Label Interpretation (Handling the 425 Missing Values)
In `vehicles.csv`, 425 records have `anomaly == NaN`, while 75 records have explicit anomaly strings (`rollback: 25`, `gap: 25`, `jump: 25`).
- Cross-tabulation with `data_quality.csv` reveals that **all 425 records with `anomaly == NaN` have `odometerStatus == "Consistent"`, `flagCount == 0`, and a mean `careScore` of 99.2 (median 100)**.
- Conversely, all 25 `gap` records have `odometerStatus == "Flagged"`, and all 50 `jump` and `rollback` records have `odometerStatus == "Suspicious"`.
- Therefore, the 425 missing values empirically represent verified consistent normal vehicles (`label = 0`), while preserving the 3 specific subtype labels (`rollback`, `gap`, `jump`) for diagnostic evaluation.

---

## 4. Feature A – Fair Price Calculation Engine

The engine calculates fair market value using the multi-factor formulation:

$$\text{Depreciated Price} = \text{Showroom Price} \times \text{Remaining Value Ratio}$$
$$\text{Condition Adjusted Price} = \text{Depreciated Price} \times (1 + \text{Condition Multiplier})$$
$$\text{Final Fair Price} = \text{Condition Adjusted Price} \times (1 + \text{Odometer Multiplier})$$
$$\text{Fair Range} = [\text{Final Fair Price} \times 0.92, \text{Final Fair Price} \times 1.08]$$

### Official Depreciation Bands (0 to 5 Years)
- **0–6 months**: 95%
- **6–12 months**: 85%
- **1–2 years**: 80%
- **2–3 years**: 70%
- **3–4 years**: 60%
- **4–5 years**: 50%

### Documented Fallback Tiers (>5 Years)
Because the official assignment defines depreciation bands only up to 5 years, RELAY Module 6 implements transparent, configurable fallback tiers for older vehicles:
- **5–8 years**: 40% remaining value
- **8–12 years**: 30% remaining value
- **>12 years**: 20% remaining value

### Condition Adjustments
- **Excellent**: `+5%` (`careScore >= 98`)
- **Good**: `+2%` (`88 <= careScore < 98`)
- **Average**: `0%` (`75 <= careScore < 88`)
- **Poor**: `-8%` (`careScore < 75`)

### Odometer Integrity Flags
- **Normal**: `0%` (`odometerStatus == "Consistent"`)
- **Flagged**: `-10%` (`odometerStatus` in `["Suspicious", "Flagged"]`)

---

## 5. Feature B – ML Anomaly Detection

### 8 Service-History Features
1. `service_interval_mean`: Mean days between consecutive service events.
2. `service_interval_std`: Standard deviation of service intervals (low = clockwork fake logs, high = irregular).
3. `odometer_increment_mean`: Mean odometer km accumulated per service interval.
4. `odometer_increment_std`: Standard deviation of odometer increments (high = sudden odometer jumps/spikes).
5. `confirmed_ratio`: Ratio of authorized service dealer confirmations (0.0 to 1.0).
6. `unique_dealers`: Count of distinct authorized service centers visited.
7. `gap_count`: Number of inter-service gaps exceeding 365 calendar days.
8. `service_regularity_score`: Normalized smoothness indicator: $\frac{1}{1 + \frac{\text{std}}{\max(\text{mean}, 1)}}$.

### 3-Way Vehicle-Level Split (Zero Leakage)
To avoid vehicle-level data leakage, service records are strictly aggregated at the vehicle level prior to splitting:
- **Training Set (60%)**: 300 vehicles (255 normal, 45 anomalies)
- **Validation Set (20%)**: 100 vehicles (85 normal, 15 anomalies)
- **Untouched Test Set (20%)**: 100 vehicles (85 normal, 15 anomalies)

`StandardScaler` is fitted **strictly on the Training set** and transformed across Validation and Test sets.

### Contamination Tuning & Validation-Only Selection
Contamination was tuned **strictly on validation data** (`X_val`, `y_val`) across candidate parameters `[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]`:

| Contamination Candidate | Val Recall | Val Precision | Val F1 | Val Accuracy | Rollback Detection (Val) | Status / Decision |
|---|---:|---:|---:|---:|---:|---|
| **0.10** | 60.0% | 100.0% | 0.7500 | 94.0% | 0.0% | Fails $\ge 70\%$ recall threshold |
| **0.15** | 60.0% | 69.2% | 0.6429 | 90.0% | 0.0% | Fails $\ge 70\%$ recall threshold |
| **0.20** | 60.0% | 56.2% | 0.5806 | 87.0% | 0.0% | Fails $\ge 70\%$ recall threshold |
| **0.25** | 60.0% | 52.9% | 0.5625 | 86.0% | 0.0% | Fails $\ge 70\%$ recall threshold |
| **0.30** | **73.3%** | **47.8%** | **0.5789** | **84.0%** | **33.3%** | **SELECTED**: Lowest contamination meeting $\ge 70\%$ target |
| **0.35** | 93.3% | 43.8% | 0.5957 | 81.0% | 83.3% | Alternative: highest recall |

**Selection Rule**: `contamination = 0.30` was selected solely based on validation metrics because it is the most conservative parameter satisfying the $\ge 70\%$ recall mandate. The model configuration was frozen prior to test set evaluation.

### Untouched Test Set Evaluation (Evaluated Exactly Once)
- **Accuracy**: **79.0%**
- **Overall Recall**: **73.3%** (11 out of 15 detected $\ge 70\%$)
- **Precision**: **39.3%**
- **F1-Score**: **0.5116**
- **Confusion Matrix**: `[[68, 17], [4, 11]]` (68 TN, 17 FP, 4 FN, 11 TP)

### Test Set Subtype Recall Breakdown
- **Service Gaps (`gap`)**: **100.0% Recall** (3/3 detected)
- **Odometer Spikes (`jump`)**: **100.0% Recall** (5/5 detected)
- **Odometer Rollbacks (`rollback`)**: **42.9% Recall** (3/7 detected)

### Technical Analysis of Signed Negative Increments & Rollback Detection
- **Signed Increments**: In `generate_data.py`, `np.diff(odos)` retains negative values without truncation or conversion to absolute values. All 25 rollback vehicles exhibit exactly 1 negative odometer step between `-5,849 km` and `-1,661 km`, while all 475 other vehicles have zero negative steps.
- **Feature Masking**: Because the rollback event is followed by a compensatory upward service jump, the telescoping sum $(x_n - x_0)$ leaves `odometer_increment_mean` (~3,177 km) indistinguishable from normal (~3,161 km). The anomaly signal is concentrated in `odometer_increment_std` (~2,357 km vs ~265 km normal).
- **Ensemble Partitioning**: Because extreme jump anomalies have `odometer_increment_std` ~16,000 km, lower contamination settings cut jump anomalies first. Calibrating contamination to `0.30` on validation data successfully isolates rollback records, achieving 42.9% rollback recall on the held-out test set.
- **Recommended Architectural Improvement**: In production systems, pairing Isolation Forest with a deterministic heuristic check (`any(increment < 0)`) achieves 100% rollback detection with zero false alarms.

---

## 6. Known Dataset Plausibility Observations & Data Quality Safeguards

During deep integrity auditing of the official dataset, two historical timeline anomalies were identified:
1. **Model Launch Year Discrepancies**:
   - `TVS Jupiter 125` was introduced to the Indian market in **October 2021**, but 22 records in `vehicles.csv` have years 2018 (7), 2019 (8), and 2020 (7).
   - `Royal Enfield Meteor 350` was launched in **November 2020**, but 15 records in `vehicles.csv` have years 2018 (7) and 2019 (8).
   - In total, **37 vehicle records** feature registration years and simulated service histories that predate the real-world market debut of the model.
   - *Safeguard Implemented*: The original dataset CSVs remain untouched. The Fair Price Engine flags affected vehicles by returning `isPlausibleModelYear: false` and adds an explicit warning in `details.qualityWarnings` so consumers are alerted.
2. **Anomaly Label Assumption**:
   - The 425 missing anomaly values in `vehicles.csv` are explicitly documented as an operational working assumption (supported by 100% correlation with `Consistent` status and 0 flag count in `data_quality.csv`), pending formal confirmation from the dataset author.
2. **Deterministic Uniform Model Assignment**:
   - Exactly 50 vehicles were generated for each of the 10 models (7 or 8 vehicles per calendar year from 2018 to 2024).

---

## 7. Admin Dashboard Integration (`/admin`)

To fulfill the assignment requirement to *"show the anomaly score and verdict in the admin dashboard's flagged vehicles table"*, RELAY Module 6 provides:
1. **Interactive Admin Web Dashboard (`public/admin.html`)**: Served on `http://127.0.0.1:3000/admin`.
2. **Batch Flagged Vehicles API**: `GET /api/v1/admin/flagged-vehicles?filter={flagged|rollback|suspicious|all|normal}`.
3. **Data Displayed**:
   - Vehicle Chassis ID & Alias
   - Make, Model, Year, Category
   - Care Score & Odometer Integrity Status
   - Isolation Forest numeric anomaly score (`+0.0821`)
   - ML verdict (`Suspicious` vs `Normal`)
   - Deterministic Rollback Warning (`🚨 Rollback Warning` vs `✓ Monotonic`)
   - Diagnostic reasons & 8-feature inspection drawer.

---

## 8. Retraining & Operational Commands

```powershell
cd ai-module
# Process official dataset
python generate_data.py

# Train Isolation Forest with 3-way evaluation
python train.py

# Run test suite
pytest -v
```
