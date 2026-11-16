"""
Flask Microservice for RELAY Module 6 (Port 5001).

Exposes internal endpoints for Fair Price Calculation and ML Anomaly Detection.
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

BASE = Path(__file__).resolve().parent
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
features_df = None


def load_assets():
    """Load pre-trained model, scaler, and vehicle dataset."""
    global model, scaler, vehicles_cache, features_df
    model_path = BASE / "model.pkl"
    scaler_path = BASE / "scaler.pkl"
    vehicles_path = BASE / "vehicles.json"
    features_path = BASE / "vehicle_features.csv"

    if not model_path.exists() or not scaler_path.exists():
        raise RuntimeError(f"Missing model or scaler in {BASE}. Run train.py first.")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    if vehicles_path.exists():
        vehicles_list = json.loads(vehicles_path.read_text(encoding="utf-8"))
        vehicles_cache = {v["id"]: v for v in vehicles_list}
    else:
        vehicles_cache = {}

    if features_path.exists():
        features_df = pd.read_csv(features_path).set_index("vehicle_id")
    else:
        features_df = pd.DataFrame(columns=FEATURES)

    logger.info("Assets loaded successfully: %d vehicles, %d feature rows", len(vehicles_cache), len(features_df))


# Load assets on module import
load_assets()


def get_anomaly_evaluation(vehicle_id: str) -> dict:
    """Evaluate anomaly score and generate explainable verdict."""
    if features_df is None or vehicle_id not in features_df.index:
        return None

    row = features_df.loc[vehicle_id]
    feature_vector = pd.DataFrame([[float(row[col]) for col in FEATURES]], columns=FEATURES)
    feature_scaled = scaler.transform(feature_vector)

    prediction = int(model.predict(feature_scaled)[0])
    raw_decision_score = float(model.decision_function(feature_scaled)[0])

    is_anomaly = (prediction == -1)
    reasons = []

    if is_anomaly:
        if row["service_interval_std"] < 8.0:
            reasons.append("Service intervals are unnaturally consistent (standard deviation < 8 days)")
        if row["odometer_increment_std"] < 100.0:
            reasons.append("Odometer increments are unnaturally uniform (standard deviation < 100 km)")
        if row["confirmed_ratio"] < 0.5:
            reasons.append(f"Low proportion of confirmed service records ({row['confirmed_ratio'] * 100:.1f}%)")
        if row["unique_dealers"] <= 1:
            reasons.append("Single dealer appears across entire service history")
        if row["gap_count"] > 0:
            reasons.append(f"Detected {int(row['gap_count'])} uncharacteristic service gap(s) (>365 days)")
        if row.get("service_regularity_score", 0) > 0.95:
            reasons.append("Service regularity score is abnormally rigid (potential synthetic records)")

        if not reasons:
            reasons.append("Multi-feature anomaly detected across service intervals and odometer steps")
    else:
        reasons.append("Service history intervals, dealer distribution, and odometer steps conform to standard patterns")

    return {
        "vehicleId": vehicle_id,
        "anomalyScore": round(-raw_decision_score, 4),
        "verdict": "Suspicious" if is_anomaly else "Normal",
        "reasons": reasons,
        "extractedFeatures": {col: round(float(row[col]), 2) for col in FEATURES},
        "scoreNote": "Negative decision function output from Scikit-Learn IsolationForest. Higher values indicate higher anomaly likelihood.",
        "dataNote": "Prediction computed against synthetic development dataset."
    }


@app.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "service": "relay-ai-module",
        "modelLoaded": model is not None,
        "scalerLoaded": scaler is not None,
        "dataset": "synthetic_demo"
    }), 200


@app.get("/internal/vehicles/<vehicle_id>/price")
def internal_price(vehicle_id: str):
    """Calculate fair price for a registered vehicle."""
    vid = (vehicle_id or "").strip()
    if not vid or vid not in vehicles_cache:
        return jsonify({"error": "Vehicle not found", "vehicleId": vid}), 404

    vehicle = vehicles_cache[vid]
    try:
        result = calculate_fair_price(vehicle)
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
    return jsonify({
        "totalVehicles": len(vehicles_cache),
        "sampleIds": list(vehicles_cache.keys())[:10]
    }), 200


@app.errorhandler(404)
def not_found(_err):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(err):
    return jsonify({"error": "Internal server error", "details": str(err)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
