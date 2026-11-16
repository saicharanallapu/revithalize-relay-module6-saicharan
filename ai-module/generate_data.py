"""
Generate synthetic Indian Two-Wheeler dataset (500 vehicles, 75 seeded anomalies).

Distribution specifications:
- Exactly 500 two-wheeler vehicle records (motorcycles and scooters)
- Exactly 75 seeded anomalies, 425 normal vehicles
- Age distribution:
  - 0–1 year: ~5% (25 vehicles)
  - 1–3 years: ~10% (50 vehicles)
  - 3–5 years: ~20% (100 vehicles)
  - 5–8 years: ~30% (150 vehicles)
  - 8–12 years: ~25% (125 vehicles)
  - > 12 years: ~10% (50 vehicles)
  => 85% of vehicles are older than 3 years.
- Valid model/year combinations: Models are never assigned to years before their introduction.
- Older models dominate representation.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from price_engine import TWO_WHEELER_CATALOG

BASE = Path(__file__).resolve().parent
CURRENT_YEAR = 2026
RNG = np.random.default_rng(42)

# Build list of two-wheeler model metadata from catalog
CATALOG_MODELS = []
for (make, model), meta in TWO_WHEELER_CATALOG.items():
    CATALOG_MODELS.append({
        "make": make,
        "model": model,
        "category": meta["category"],
        "showroomPrice": meta["showroomPrice"],
        "introYear": meta["introYear"],
        "tier": meta["tier"],  # "older" vs "newer"
    })


def select_model_for_age(age_years: float, manufacturer_counts: dict) -> dict:
    """
    Select an appropriate model that was already introduced when the vehicle was built.
    Ensures:
    1. introYear <= (CURRENT_YEAR - age_years)
    2. Older models receive significantly higher sampling weight.
    3. Balanced OEM representation across the dataset.
    """
    mfg_year = CURRENT_YEAR - int(age_years)
    eligible = [m for m in CATALOG_MODELS if m["introYear"] <= mfg_year]
    if not eligible:
        # Fallback to the oldest model (e.g. Bullet 350 / Activa)
        eligible = [m for m in CATALOG_MODELS if m["introYear"] <= 2001]

    # Compute sampling weights:
    # - Older tier gets 4x higher baseline weight than newer tier.
    # - Penalize manufacturers that have been selected too frequently to maintain diversity.
    weights = []
    for m in eligible:
        w = 4.0 if m["tier"] == "older" else 1.0
        # Diversity balancing: slightly downweight over-sampled OEMs
        current_oem_count = manufacturer_counts.get(m["make"], 0)
        w = w / (1.0 + (current_oem_count / 30.0))
        weights.append(w)

    weights = np.array(weights) / np.sum(weights)
    idx = int(RNG.choice(len(eligible), p=weights))
    chosen = eligible[idx]
    manufacturer_counts[chosen["make"]] = manufacturer_counts.get(chosen["make"], 0) + 1
    return chosen


def generate_service_history(vehicle_id: str, age_years: float, is_anomaly: bool, index: int):
    """
    Generate realistic two-wheeler service history.
    - Two-wheelers typically run 3,000 - 8,000 km per year.
    - Service intervals are typically 90 - 180 days.
    - Older vehicles naturally have more service visits, but may have some missing/unconfirmed records.
    """
    # Number of services scaled with age: ~2 services per year of age
    base_services = max(3, int(round(age_years * 2.2)))
    num_services = int(np.clip(base_services + RNG.integers(-2, 3), 3, 28))

    # Starting date roughly (age_years) ago
    days_ago = int(round(age_years * 365.25))
    start_date = pd.Timestamp("2026-01-01") - pd.Timedelta(days=days_ago)
    current_date = start_date
    current_odo = int(RNG.integers(500, 2500))

    dates, odos, dealers, confirmed = [], [], [], []

    if is_anomaly:
        # 4 distinct synthetic anomaly tampering profiles
        profile = index % 4
        if profile == 0:
            # Profile 0: Clockwork artificial intervals and steps (fake logbook)
            interval = int(RNG.choice([90, 120, 150]))
            odo_step = int(RNG.choice([2500, 3000, 3500]))
            dealer_pool = [f"DLR_TW_{(index % 8) + 1:03d}"]
            conf_prob = 0.20
            interval_noise, odo_noise = 1.0, 5.0
        elif profile == 1:
            # Profile 1: Odometer rollback / tampered low increments
            interval = int(RNG.integers(120, 200))
            odo_step = int(RNG.choice([400, 600, 800]))  # Unnaturally low for two-wheelers
            dealer_pool = [f"DLR_TW_{int(RNG.integers(1, 15)):03d}" for _ in range(2)]
            conf_prob = 0.30
            interval_noise, odo_noise = 40.0, 30.0
        elif profile == 2:
            # Profile 2: Long service gaps with rigid regularity in remaining records
            interval = int(RNG.choice([100, 180]))
            odo_step = int(RNG.choice([2000, 4000]))
            dealer_pool = [f"DLR_TW_{(index % 5) + 1:03d}"]
            conf_prob = 0.25
            interval_noise, odo_noise = 2.0, 10.0
        else:
            # Profile 3: Single unverified dealer across long history with low confirmation
            interval = int(RNG.integers(100, 180))
            odo_step = int(RNG.integers(2000, 4000))
            dealer_pool = [f"DLR_LOCAL_{index % 3:02d}"]
            conf_prob = 0.15
            interval_noise, odo_noise = 3.0, 25.0
    else:
        # Realistic Normal Indian Two-Wheeler usage:
        interval = int(RNG.integers(110, 190))
        odo_step = int(RNG.integers(1800, 4500))
        dealer_pool = [f"DLR_AUTH_{int(RNG.integers(1, 40)):03d}" for _ in range(5)]
        conf_prob = 0.85
        interval_noise, odo_noise = 32.0, 750.0

    for j in range(num_services):
        if j > 0:
            step_days = max(30, int(RNG.normal(interval, interval_noise)))
            step_km = max(300, int(RNG.normal(odo_step, odo_noise)))
            current_date += pd.Timedelta(days=step_days)
            current_odo += step_km

            # If date surpasses current time, stop
            if current_date > pd.Timestamp("2026-03-01"):
                break

        dates.append(current_date)
        odos.append(current_odo)
        dealers.append(str(RNG.choice(dealer_pool)))
        confirmed.append(bool(RNG.random() < conf_prob))

    # Add a genuine service gap to a small fraction of older normal vehicles
    if not is_anomaly and age_years > 5.0 and index % 12 == 0 and len(dates) > 4:
        dates[-1] += pd.Timedelta(days=390)
    elif is_anomaly and profile == 2 and len(dates) > 3:
        dates[-1] += pd.Timedelta(days=420)

    return [
        {
            "vehicleId": vehicle_id,
            "serviceDate": d.date().isoformat(),
            "odometerKm": int(o),
            "dealerId": dealer,
            "confirmed": conf,
        }
        for d, o, dealer, conf in zip(dates, odos, dealers, confirmed)
    ]


def extract_features(history):
    """Extract 8 mandatory service-history features safely."""
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
    h = sorted(history, key=lambda x: x["serviceDate"])
    dates = pd.to_datetime([x["serviceDate"] for x in h])
    odos = np.array([x["odometerKm"] for x in h], dtype=float)
    intervals = np.diff(dates.values).astype("timedelta64[D]").astype(float)
    increments = np.diff(odos)

    interval_mean = float(np.mean(intervals)) if len(intervals) else 0.0
    interval_std = float(np.std(intervals)) if len(intervals) else 0.0
    increment_mean = float(np.mean(increments)) if len(increments) else 0.0
    increment_std = float(np.std(increments)) if len(increments) else 0.0
    confirmed_ratio = float(np.mean([bool(x.get("confirmed", False)) for x in h])) if len(h) else 0.0
    unique_dealers = int(len(set(x.get("dealerId", "") for x in h)))
    gap_count = int(np.sum(intervals > 365)) if len(intervals) else 0
    regularity = float(1.0 / (1.0 + (interval_std / max(interval_mean, 1.0)))) if len(intervals) else 0.0

    return {
        "service_interval_mean": interval_mean,
        "service_interval_std": interval_std,
        "odometer_increment_mean": increment_mean,
        "odometer_increment_std": increment_std,
        "confirmed_ratio": confirmed_ratio,
        "unique_dealers": unique_dealers,
        "gap_count": gap_count,
        "service_regularity_score": regularity,
    }


def main():
    # Age distribution specification (Exactly 500 vehicles):
    # - 0–1 year:  5% (25 vehicles)
    # - 1–3 years: 10% (50 vehicles)
    # - 3–5 years: 20% (100 vehicles)
    # - 5–8 years: 30% (150 vehicles)
    # - 8–12 years: 25% (125 vehicles)
    # - >12 years: 10% (50 vehicles)
    age_brackets = [
        (0.2, 1.0, 25),    # 0–1 year
        (1.0, 3.0, 50),    # 1–3 years
        (3.0, 5.0, 100),   # 3–5 years
        (5.0, 8.0, 150),   # 5–8 years
        (8.0, 12.0, 125),  # 8–12 years
        (12.0, 18.0, 50),  # >12 years
    ]

    all_ages = []
    for low, high, count in age_brackets:
        ages = RNG.uniform(low, high, size=count)
        all_ages.extend(ages)

    # Shuffle ages to distribute them across IDs
    RNG.shuffle(all_ages)
    assert len(all_ages) == 500, f"Expected 500 ages, got {len(all_ages)}"

    # Ground truth anomalies: exactly 75 anomalies
    # Distribute anomalies across indices 1 to 500 (e.g., indices chosen randomly or seeded)
    anomaly_indices = set(RNG.choice(500, size=75, replace=False))
    assert len(anomaly_indices) == 75, "Expected exactly 75 anomalies"

    histories, feature_rows, vehicles = [], [], []
    manufacturer_counts = {}

    for i in range(500):
        vid = f"RELAY_TW_{i+1:03d}"
        age = round(float(all_ages[i]), 2)
        is_anomaly = (i in anomaly_indices)

        # Select valid model for vehicle age
        model_info = select_model_for_age(age, manufacturer_counts)

        # Condition distribution
        if age <= 2.0:
            condition = str(RNG.choice(["Excellent", "Good", "Average"], p=[0.40, 0.45, 0.15]))
        elif age <= 6.0:
            condition = str(RNG.choice(["Excellent", "Good", "Average", "Poor"], p=[0.15, 0.45, 0.30, 0.10]))
        else:
            condition = str(RNG.choice(["Good", "Average", "Poor"], p=[0.20, 0.50, 0.30]))

        # Odometer flag: only flagged if anomalous or high mileage tampered
        odometer_status = "Flagged" if (is_anomaly and (i % 3 == 0)) else "Normal"

        vehicles.append({
            "id": vid,
            "make": model_info["make"],
            "model": model_info["model"],
            "category": model_info["category"],
            "showroomPrice": model_info["showroomPrice"],
            "introYear": model_info["introYear"],
            "tier": model_info["tier"],
            "ageYears": age,
            "condition": condition,
            "odometerStatus": odometer_status,
        })

        history = generate_service_history(vid, age, is_anomaly, i)
        histories.extend(history)
        feats = extract_features(history)
        feature_rows.append({"vehicle_id": vid, **feats, "label": int(is_anomaly)})

    # Persist dataset files
    pd.DataFrame(histories).to_csv(BASE / "service_history.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(BASE / "vehicle_features.csv", index=False)
    (BASE / "vehicles.json").write_text(json.dumps(vehicles, indent=2), encoding="utf-8")

    # Print summary statistics
    df_v = pd.DataFrame(vehicles)
    older_count = sum(df_v["tier"] == "older")
    newer_count = sum(df_v["tier"] == "newer")
    mc_count = sum(df_v["category"] == "Motorcycle")
    sc_count = sum(df_v["category"] == "Scooter")
    over_3y = sum(df_v["ageYears"] > 3.0)

    print("==================================================")
    print("Generated SYNTHETIC TWO-WHEELER Dataset Summary:")
    print(f"  Total vehicles: {len(vehicles)}")
    print(f"  Seeded anomalies: {len(anomaly_indices)}")
    print(f"  Normal vehicles: {len(vehicles) - len(anomaly_indices)}")
    print(f"  Total service records: {len(histories)}")
    print(f"  Motorcycles: {mc_count} ({mc_count/len(vehicles):.1%})")
    print(f"  Scooters: {sc_count} ({sc_count/len(vehicles):.1%})")
    print(f"  Older models: {older_count} ({older_count/len(vehicles):.1%})")
    print(f"  Newer models: {newer_count} ({newer_count/len(vehicles):.1%})")
    print(f"  Vehicles >3 years: {over_3y} ({over_3y/len(vehicles):.1%})")
    print(f"  Unique OEMs ({len(df_v['make'].unique())}): {dict(df_v['make'].value_counts())}")
    print("==================================================")


if __name__ == "__main__":
    main()
