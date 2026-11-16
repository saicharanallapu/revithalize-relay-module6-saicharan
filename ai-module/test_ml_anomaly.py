"""
Automated unit tests for Two-Wheeler ML Anomaly Detection (RELAY Module 6).
Tests:
- Dataset integrity: 500 two-wheelers, 75 anomalies, 0 cars/SUVs
- Age distribution: >= 85% older than 3 years
- Model distribution: older models significantly exceed newer models
- Valid model launch years vs manufacturing year
- 8 required features extraction from two-wheeler histories
- Edge cases: empty history, 1 record
- Model & scaler compatibility and persistence
- Inference & explainable anomaly reasons
- 3-way split training metrics verification
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

from generate_data import extract_features
from serve import get_anomaly_evaluation, FEATURES

BASE = Path(__file__).resolve().parent


class TestTwoWheelerDatasetIntegrity:
    @pytest.fixture(autouse=True)
    def load_data(self):
        vehicles_path = BASE / "vehicles.json"
        features_path = BASE / "vehicle_features.csv"
        assert vehicles_path.exists(), "vehicles.json missing"
        assert features_path.exists(), "vehicle_features.csv missing"
        self.vehicles = json.loads(vehicles_path.read_text(encoding="utf-8"))
        self.features_df = pd.read_csv(features_path)

    def test_total_count_and_anomalies(self):
        """Verify exactly 500 vehicles and exactly 75 seeded anomalies."""
        assert len(self.vehicles) == 500
        assert len(self.features_df) == 500
        anomalies_count = int(self.features_df["label"].sum())
        assert anomalies_count == 75, f"Expected exactly 75 seeded anomalies, got {anomalies_count}"
        normal_count = 500 - anomalies_count
        assert normal_count == 425

    def test_no_cars_or_suvs_only_two_wheelers(self):
        """Verify 100% of vehicles are motorcycles or scooters, no cars or SUVs."""
        categories = set(v.get("category") for v in self.vehicles)
        assert categories == {"Motorcycle", "Scooter"}
        for v in self.vehicles:
            assert v.get("category") in ["Motorcycle", "Scooter"], f"Non-two-wheeler found: {v}"

    def test_age_distribution_over_85_percent_older_than_3_years(self):
        """Verify at least 85% of records are older than 3 years."""
        ages = [v["ageYears"] for v in self.vehicles]
        over_3y = sum(1 for a in ages if a > 3.0)
        share_over_3y = over_3y / len(ages)
        assert share_over_3y >= 0.85, f"Expected >= 85% older than 3y, got {share_over_3y:.1%}"

    def test_older_models_dominate_newer_models(self):
        """Verify older generation models significantly outnumber newer models."""
        tiers = [v["tier"] for v in self.vehicles]
        older_count = sum(1 for t in tiers if t == "older")
        newer_count = sum(1 for t in tiers if t == "newer")
        assert older_count > newer_count * 5, f"Expected older >> newer, got {older_count} vs {newer_count}"

    def test_valid_model_launch_years(self):
        """Verify no vehicle is assigned to a year before its introduction."""
        for v in self.vehicles:
            mfg_year = 2026 - int(v["ageYears"])
            intro_year = v.get("introYear")
            assert intro_year is not None
            assert intro_year <= mfg_year, (
                f"Invalid model/year combination: {v['make']} {v['model']} "
                f"introduced in {intro_year} but assigned mfg year {mfg_year}"
            )


class TestFeatureExtraction:
    def test_extract_features_all_eight_present(self):
        """Verify all 8 mandatory features are computed."""
        history = [
            {"serviceDate": "2023-01-10", "odometerKm": 3000, "dealerId": "DLR01", "confirmed": True},
            {"serviceDate": "2023-05-15", "odometerKm": 6200, "dealerId": "DLR02", "confirmed": True},
            {"serviceDate": "2023-10-20", "odometerKm": 9800, "dealerId": "DLR03", "confirmed": False},
        ]
        features = extract_features(history)
        expected_keys = [
            "service_interval_mean",
            "service_interval_std",
            "odometer_increment_mean",
            "odometer_increment_std",
            "confirmed_ratio",
            "unique_dealers",
            "gap_count",
            "service_regularity_score",
        ]
        for key in expected_keys:
            assert key in features, f"Missing feature: {key}"

        assert features["unique_dealers"] == 3
        assert features["confirmed_ratio"] == pytest.approx(2 / 3, abs=1e-4)
        assert features["odometer_increment_mean"] == pytest.approx((3200 + 3600) / 2, abs=1.0)
        assert features["gap_count"] == 0

    def test_extract_features_detects_service_gap(self):
        """Service gap > 365 days increases gap_count."""
        history = [
            {"serviceDate": "2021-01-01", "odometerKm": 2000, "dealerId": "DLR01", "confirmed": True},
            {"serviceDate": "2022-03-01", "odometerKm": 6500, "dealerId": "DLR01", "confirmed": True},  # 424 days
        ]
        features = extract_features(history)
        assert features["gap_count"] == 1

    def test_extract_features_empty_history(self):
        features = extract_features([])
        assert features["unique_dealers"] == 0
        assert features["confirmed_ratio"] == 0.0
        assert features["service_interval_mean"] == 0.0

    def test_extract_features_single_record(self):
        history = [{"serviceDate": "2023-01-01", "odometerKm": 2500, "dealerId": "DLR01", "confirmed": True}]
        features = extract_features(history)
        assert features["unique_dealers"] == 1
        assert features["confirmed_ratio"] == 1.0
        assert features["service_interval_mean"] == 0.0
        assert features["service_interval_std"] == 0.0


class TestModelAndInference:
    def test_saved_model_and_scaler_exist(self):
        assert (BASE / "model.pkl").exists(), "model.pkl not found"
        assert (BASE / "scaler.pkl").exists(), "scaler.pkl not found"

    def test_model_inference_on_scaled_vector(self):
        model = joblib.load(BASE / "model.pkl")
        scaler = joblib.load(BASE / "scaler.pkl")

        normal_vec = pd.DataFrame([{
            "service_interval_mean": 150.0,
            "service_interval_std": 30.0,
            "odometer_increment_mean": 3000.0,
            "odometer_increment_std": 700.0,
            "confirmed_ratio": 0.85,
            "unique_dealers": 4,
            "gap_count": 0,
            "service_regularity_score": 0.83,
        }])[FEATURES]

        scaled = scaler.transform(normal_vec)
        pred = model.predict(scaled)[0]
        assert pred in [1, -1]

    def test_evaluation_reasons_and_verdict(self):
        # Test first vehicle
        eval_res = get_anomaly_evaluation("RELAY_TW_001")
        assert eval_res is not None
        assert eval_res["vehicleId"] == "RELAY_TW_001"
        assert eval_res["verdict"] in ["Suspicious", "Normal"]
        assert isinstance(eval_res["reasons"], list)
        assert len(eval_res["reasons"]) > 0
        assert "anomalyScore" in eval_res

    def test_training_metrics_file_has_evaluation(self):
        metrics_path = BASE / "training_metrics.json"
        assert metrics_path.exists(), "training_metrics.json missing"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert metrics["dataset"] == "synthetic_demo_only"
        assert "methodology" in metrics
        assert metrics["train_rows"] == 300
        assert metrics["validation_rows"] == 100
        assert metrics["test_rows"] == 100
        assert "validation_tuning_results" in metrics
        test_m = metrics["test_metrics"]
        assert "recall" in test_m
        assert "confusion_matrix_labels_0_normal_1_anomaly" in test_m
        assert "classification_report" in test_m
        # Synthetic test recall on untouched test set satisfies >= 70%
        assert test_m["recall"] >= 0.70
