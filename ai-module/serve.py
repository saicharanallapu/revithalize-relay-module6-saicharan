"""
Flask Microservice for RELAY Module 6 (Port 5001).

Exposes internal endpoints for Fair Price Calculation and ML Anomaly Detection
using the official Indian two-wheeler dataset.
The Node.js API Gateway proxies public requests to these internal routes.
"""

import sys
from pathlib import Path
import json
import logging
import joblib
import pandas as pd
import sklearn
import sklearn.ensemble
from flask import Flask, jsonify, request

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from price_engine import calculate_fair_price

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("relay-ai-service")

FEATURES = [
    "service_interval_mean", "service_interval_std",
    "odometer_increment_mean", "odometer_increment_std",
    "confirmed_ratio", "unique_dealers", "gap_count",
    "service_regularity_score"
]

app = Flask(__name__)

# Cached assets
model = None
scaler = None
vehicles_cache = {}
alias_cache = {}
features_df = None
features_by_alias = {}


def load_assets():
    """Load pre-trained model, scaler, and vehicle dataset."""
    global model, scaler, vehicles_cache, alias_cache, features_df, features_by_alias
    model_path = BASE / "model.pkl"
    scaler_path = BASE / "scaler.pkl"
    vehicles_path = BASE / "vehicles.json"
    features_path = BASE / "vehicle_features.csv"

    if not model_path.exists() or not scaler_path.exists():
        raise RuntimeError(f"Missing model or scaler in {BASE}. Run train.py first.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    vehicles_cache = {}
    alias_cache = {}
    if vehicles_path.exists():
        vehicles_list = json.loads(vehicles_path.read_text(encoding="utf-8"))
        for v in vehicles_list:
            chassis = v.get("chassis") or v.get("id")
            if chassis:
                vehicles_cache[chassis] = v
                vehicles_cache[str(chassis).upper()] = v
            alias = v.get("aliasId")
            if alias:
                alias_cache[alias] = v
                alias_cache[str(alias).upper()] = v

    if features_path.exists():
        f_df = pd.read_csv(features_path)
        features_df = f_df.set_index("vehicle_id")
        if "alias_id" in f_df.columns:
            features_by_alias = f_df.set_index("alias_id")
        else:
            features_by_alias = {}
    else:
        features_df = pd.DataFrame(columns=FEATURES)
        features_by_alias = {}

    logger.info(
        "Assets loaded successfully: %d vehicles, %d feature rows",
        len(vehicles_cache),
        len(features_df) if features_df is not None else 0,
    )


# Load assets on module import
load_assets()


def resolve_vehicle(vid: str):
    """Resolve vehicle record by chassis ID or alias ID."""
    if not vid:
        return None
    key = str(vid).strip()
    return vehicles_cache.get(key) or vehicles_cache.get(key.upper()) or alias_cache.get(key) or alias_cache.get(key.upper())


def resolve_features_row(vid: str):
    """Resolve feature series by chassis ID or alias ID."""
    if not vid or features_df is None:
        return None
    key = str(vid).strip()
    if key in features_df.index:
        return features_df.loc[key]
    if key.upper() in features_df.index:
        return features_df.loc[key.upper()]
    if isinstance(features_by_alias, pd.DataFrame) and key in features_by_alias.index:
        return features_by_alias.loc[key]
    if isinstance(features_by_alias, pd.DataFrame) and key.upper() in features_by_alias.index:
        return features_by_alias.loc[key.upper()]
    return None


def get_anomaly_evaluation(vehicle_id: str) -> dict:
    """Evaluate anomaly score and generate explainable verdict."""
    row = resolve_features_row(vehicle_id)
    if row is None:
        return None

    # Handle Series if single match
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    feature_vector = pd.DataFrame([[float(row[col]) for col in FEATURES]], columns=FEATURES)
    feature_scaled = scaler.transform(feature_vector)

    prediction = int(model.predict(feature_scaled)[0])
    raw_decision_score = float(model.decision_function(feature_scaled)[0])

    is_anomaly = (prediction == -1)
    reasons = []

    if is_anomaly:
        if row.get("gap_count", 0) > 0:
            reasons.append(f"Detected {int(row['gap_count'])} extended service-history gap(s) (>365 days)")
        if row.get("odometer_increment_std", 0) > 3000.0:
            reasons.append(f"High odometer increment variability (std: {row['odometer_increment_std']:.0f} km)")
        if row.get("odometer_increment_mean", 0) > 5000.0:
            reasons.append(f"Unusually large average odometer jump ({row['odometer_increment_mean']:.0f} km/service)")
        if row.get("service_interval_std", 0) > 40.0:
            reasons.append(f"Highly irregular service intervals (std: {row['service_interval_std']:.1f} days)")
        if row.get("confirmed_ratio", 1.0) < 0.70:
            reasons.append(f"Low proportion of confirmed service records ({row['confirmed_ratio'] * 100:.1f}%)")
        if row.get("unique_dealers", 0) <= 1:
            reasons.append("Single dealer appears across entire service history")

        if not reasons:
            reasons.append("Multi-feature statistical anomaly detected across service intervals and odometer steps")
    else:
        reasons.append("Service history intervals, dealer distribution, and odometer steps conform to standard patterns")

    vehicle_info = resolve_vehicle(vehicle_id)
    rollback_check = vehicle_info.get("deterministicRollbackCheck") if vehicle_info else None
    if not rollback_check:
        rollback_check = {
            "hasNegativeIncrement": False,
            "negativeIncrementCount": 0,
            "minIncrementKm": 0.0,
            "rollbackWarning": False,
            "warningMessage": None,
        }

    return {
        "vehicleId": vehicle_id,
        "anomalyScore": round(-raw_decision_score, 4),
        "verdict": "Suspicious" if is_anomaly else "Normal",
        "reasons": reasons,
        "extractedFeatures": {col: round(float(row[col]), 2) for col in FEATURES},
        "mlEvaluation": {
            "model": "IsolationForest",
            "anomalyScore": round(-raw_decision_score, 4),
            "verdict": "Suspicious" if is_anomaly else "Normal",
            "scoreNote": "Negative decision function output from Scikit-Learn IsolationForest. Higher values indicate higher anomaly likelihood."
        },
        "deterministicChecks": {
            "hasNegativeIncrement": rollback_check.get("hasNegativeIncrement", False),
            "negativeIncrementCount": rollback_check.get("negativeIncrementCount", 0),
            "minIncrementKm": rollback_check.get("minIncrementKm", 0.0),
            "rollbackWarning": rollback_check.get("rollbackWarning", False),
            "warningMessage": rollback_check.get("warningMessage"),
            "note": "Deterministic physical check for chronological negative odometer changes. A negative change flags a rollback warning requiring document verification."
        },
        "scoreNote": "Negative decision function output from Scikit-Learn IsolationForest. Higher values indicate higher anomaly likelihood.",
        "dataNote": "Prediction computed using Isolation Forest trained on official two-wheeler dataset."
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "service": "relay-ai-module",
        "modelLoaded": model is not None,
        "scalerLoaded": scaler is not None,
        "dataset": "official_two_wheeler_dataset"
    }), 200


@app.get("/internal/vehicles/<vehicle_id>/price")
def internal_price(vehicle_id: str):
    """Calculate fair price for a registered vehicle."""
    vid = (vehicle_id or "").strip()
    vehicle = resolve_vehicle(vid)
    if not vehicle:
        return jsonify({"error": "Vehicle not found", "vehicleId": vid}), 404

    try:
        result = calculate_fair_price(vehicle)
        # Ensure returned ID matches requested ID for consistency
        result["vehicleId"] = vid
        return jsonify(result), 200
    except ValueError as err:
        return jsonify({"error": str(err), "vehicleId": vid}), 400
    except Exception as err:
        logger.error("Error calculating price for %s: %s", vid, err)
        return jsonify({"error": "Internal pricing error", "details": str(err)}), 500


@app.get("/internal/vehicles/<vehicle_id>/anomaly-score")
def internal_anomaly(vehicle_id: str):
    """Detect anomalies in vehicle service history."""
    vid = (vehicle_id or "").strip()
    if not vid:
        return jsonify({"error": "Invalid vehicle ID"}), 400

    evaluation = get_anomaly_evaluation(vid)
    if evaluation is None:
        return jsonify({"error": "Vehicle not found", "vehicleId": vid}), 404

    return jsonify(evaluation), 200


@app.get("/internal/vehicles")
def list_vehicles():
    """List summary of available vehicles."""
    # Count unique vehicles (since cache stores uppercase and alias aliases)
    unique_chassis = set(v["chassis"] for v in vehicles_cache.values() if isinstance(v, dict) and "chassis" in v)
    return jsonify({
        "totalVehicles": len(unique_chassis),
        "sampleIds": list(unique_chassis)[:10]
    }), 200


@app.errorhandler(404)
def not_found(_err):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(err):
    return jsonify({"error": "Internal server error", "details": str(err)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
