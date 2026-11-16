"""
Official Dataset Feature Processor and Vehicle Metadata Generator for RELAY Module 6.

Processes the official two-wheeler dataset supplied in `ai-module/data/`:
- vehicles.csv (500 vehicles: 425 normal, 25 rollback, 25 gap, 25 jump)
- service_records.csv (7,293 service events)
- data_quality.csv (500 data quality records with careScore and odometerStatus)

Generates:
1. vehicle_features.csv (500 rows, 8 mandatory ML features + ground truth label)
2. vehicles.json (500 vehicle records with showroom price, age, condition, odometerStatus)
3. service_history.csv (7,293 service history records with standard column names)
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from price_engine import TWO_WHEELER_CATALOG, lookup_two_wheeler

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
CURRENT_YEAR = 2026


def extract_features(history):
    """
    Extract 8 mandatory service-history features safely.
    Handles empty and single-record histories without divide-by-zero errors.
    """
    if not history:
        return {
            "service_interval_mean": 0.0,
            "service_interval_std": 0.0,
            "odometer_increment_mean": 0.0,
            "odometer_increment_std": 0.0,
            "confirmed_ratio": 0.0,
            "unique_dealers": 0,
            "gap_count": 0,
            "service_regularity_score": 0.0,
        }

    h = sorted(history, key=lambda x: str(x.get("serviceDate", "")))
    dates = pd.to_datetime([x["serviceDate"] for x in h], errors="coerce")
    odos = np.array([float(x.get("odometerKm", 0)) for x in h], dtype=float)

    if len(dates) > 1:
        intervals = np.diff(dates.values).astype("timedelta64[D]").astype(float)
        increments = np.diff(odos)
        interval_mean = float(np.mean(intervals))
        interval_std = float(np.std(intervals))
        increment_mean = float(np.mean(increments))
        increment_std = float(np.std(increments))
        gap_count = int(np.sum(intervals > 365))
        regularity = float(1.0 / (1.0 + (interval_std / max(interval_mean, 1.0))))
    else:
        interval_mean = 0.0
        interval_std = 0.0
        increment_mean = 0.0
        increment_std = 0.0
        gap_count = 0
        regularity = 0.0

    confirmed_ratio = float(np.mean([bool(x.get("confirmed", False)) for x in h]))
    unique_dealers = int(len(set(str(x.get("dealerId") or x.get("dealerCode") or "") for x in h)))

    return {
        "service_interval_mean": round(interval_mean, 4),
        "service_interval_std": round(interval_std, 4),
        "odometer_increment_mean": round(increment_mean, 4),
        "odometer_increment_std": round(increment_std, 4),
        "confirmed_ratio": round(confirmed_ratio, 4),
        "unique_dealers": unique_dealers,
        "gap_count": gap_count,
        "service_regularity_score": round(regularity, 4),
    }


def map_condition_from_care_score(care_score: float) -> str:
    """
    Map continuous careScore (0-100) to the 4 discrete condition tiers
    defined by the Fair Price Engine:
    - >= 98: Excellent (+5%)
    - 88 - 97: Good (+2%)
    - 75 - 87: Average (0%)
    - < 75: Poor (-8%)
    """
    if care_score >= 98.0:
        return "Excellent"
    elif care_score >= 88.0:
        return "Good"
    elif care_score >= 75.0:
        return "Average"
    else:
        return "Poor"


def process_official_data():
    vehicles_path = DATA_DIR / "vehicles.csv"
    services_path = DATA_DIR / "service_records.csv"
    quality_path = DATA_DIR / "data_quality.csv"

    if not (vehicles_path.exists() and services_path.exists() and quality_path.exists()):
        raise FileNotFoundError(
            f"Official dataset CSVs missing in {DATA_DIR}. "
            "Expected: vehicles.csv, service_records.csv, data_quality.csv"
        )

    v_df = pd.read_csv(vehicles_path)
    s_df = pd.read_csv(services_path)
    q_df = pd.read_csv(quality_path)

    # Validate dataset integrity
    assert len(v_df) == 500, f"Expected 500 vehicles, got {len(v_df)}"
    assert len(s_df) == 7293, f"Expected 7,293 service records, got {len(s_df)}"
    assert len(q_df) == 500, f"Expected 500 quality records, got {len(q_df)}"

    # Map quality info by chassis
    quality_by_chassis = q_df.set_index("chassis").to_dict(orient="index")

    # Group service records by chassis
    services_by_chassis = {}
    for row in s_df.to_dict(orient="records"):
        c = row["chassis"]
        if c not in services_by_chassis:
            services_by_chassis[c] = []
        services_by_chassis[c].append({
            "vehicleId": c,
            "serviceDate": row["serviceDate"],
            "odometerKm": int(row["odometerKm"]),
            "serviceType": row["serviceType"],
            "dealerId": row["dealerCode"],
            "confirmed": bool(row["confirmed"]),
        })

    vehicles_list = []
    feature_rows = []
    normalized_service_records = []

    for idx, v_row in v_df.iterrows():
        chassis = v_row["chassis"]
        oem = v_row["oem"]
        model = v_row["model"]
        year = int(v_row["year"])
        registration = v_row["registration"]
        anomaly_val = v_row["anomaly"]
        has_anomaly = pd.notnull(anomaly_val)
        anomaly_type = str(anomaly_val) if has_anomaly else "normal"

        # Alias ID for compatibility with test suites and legacy queries
        alias_id = f"RELAY_TW_{idx + 1:03d}"

        # Quality metrics lookup
        q_info = quality_by_chassis.get(chassis, {})
        care_score = float(q_info.get("careScore", 95.0))
        raw_odo_status = str(q_info.get("odometerStatus", "Consistent"))
        flag_count = int(q_info.get("flagCount", 0))

        # Catalog lookup
        catalog_info = lookup_two_wheeler(oem, model)
        if catalog_info:
            showroom_price = catalog_info["showroomPrice"]
            category = catalog_info["category"]
            tier = catalog_info["tier"]
            intro_year = catalog_info["introYear"]
        else:
            showroom_price = 80000
            category = "Two-Wheeler"
            tier = "older"
            intro_year = 2005

        age_years = max(0.0, float(CURRENT_YEAR - year))
        condition = map_condition_from_care_score(care_score)

        # Map odometer status: Consistent -> Normal, Suspicious/Flagged -> Flagged
        odometer_status = "Normal" if raw_odo_status == "Consistent" else "Flagged"

        # Extract features for this vehicle's service history
        v_services = services_by_chassis.get(chassis, [])
        v_features = extract_features(v_services)

        for s in v_services:
            normalized_service_records.append({
                "vehicleId": chassis,
                "aliasId": alias_id,
                **s
            })

        feature_row = {
            "vehicle_id": chassis,
            "alias_id": alias_id,
            **v_features,
            "label": 1 if has_anomaly else 0,
            "anomaly_type": anomaly_type,
            "odometer_status": odometer_status,
        }
        feature_rows.append(feature_row)

        vehicle_dict = {
            "id": chassis,
            "chassis": chassis,
            "vehicleId": chassis,
            "aliasId": alias_id,
            "make": oem,
            "oem": oem,
            "model": model,
            "year": year,
            "registration": registration,
            "category": category,
            "vehicleType": category,
            "tier": tier,
            "introYear": intro_year,
            "showroomPrice": showroom_price,
            "ageYears": age_years,
            "condition": condition,
            "odometerStatus": odometer_status,
            "rawOdometerStatus": raw_odo_status,
            "careScore": care_score,
            "flagCount": flag_count,
            "anomaly": anomaly_type if has_anomaly else None,
            "isAnomaly": has_anomaly,
        }
        vehicles_list.append(vehicle_dict)

    # Save output artifacts
    feat_df = pd.DataFrame(feature_rows)
    feat_df.to_csv(BASE / "vehicle_features.csv", index=False)

    (BASE / "vehicles.json").write_text(
        json.dumps(vehicles_list, indent=2), encoding="utf-8"
    )

    pd.DataFrame(normalized_service_records).to_csv(
        BASE / "service_history.csv", index=False
    )

    print("=== Official Two-Wheeler Dataset Processed ===")
    print(f"Total Vehicles: {len(vehicles_list)}")
    print(f"Total Service Records: {len(normalized_service_records)}")
    print(f"Seeded Anomalies: {sum(1 for v in vehicles_list if v['isAnomaly'])}")
    print(f"  - Rollback: {sum(1 for v in vehicles_list if v['anomaly'] == 'rollback')}")
    print(f"  - Gap: {sum(1 for v in vehicles_list if v['anomaly'] == 'gap')}")
    print(f"  - Jump: {sum(1 for v in vehicles_list if v['anomaly'] == 'jump')}")
    print(f"Normal / Consistent: {sum(1 for v in vehicles_list if not v['isAnomaly'])}")
    print("Files created: vehicle_features.csv, vehicles.json, service_history.csv")


if __name__ == "__main__":
    process_official_data()
