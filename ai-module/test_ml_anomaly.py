"""
Automated unit tests for Two-Wheeler ML Anomaly Detection (RELAY Module 6).
Tests:
- Official dataset integrity: 500 two-wheelers, 75 anomalies, 0 cars/SUVs
- Age distribution: 72% older than 3 years (2018-2022)
- Model distribution: 80% older models (400) vs 20% newer models (100)
- Manufacturing years range: 2018 to 2024
- 8 required features extraction from two-wheeler histories
- Edge cases: empty history, 1 record
- Model & scaler compatibility and persistence
- Inference & explainable anomaly reasons (chassis & alias)
- 3-way split training metrics verification on official dataset
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
DATA_DIR = BASE / "data"


class TestOfficialDatasetFiles:
    def test_official_csv_files_exist(self):
        """Verify the 3 official dataset files exist and are non-empty."""
        for filename in ["vehicles.csv", "service_records.csv", "data_quality.csv"]:
            p = DATA_DIR / filename
            assert p.exists(), f"Official file {filename} missing in {DATA_DIR}"
            assert p.stat().st_size > 0, f"Official file {filename} is empty"

    def test_official_record_counts(self):
        """Verify official dataset row counts."""
        v_df = pd.read_csv(DATA_DIR / "vehicles.csv")
        s_df = pd.read_csv(DATA_DIR / "service_records.csv")
        q_df = pd.read_csv(DATA_DIR / "data_quality.csv")
        assert len(v_df) == 500, f"Expected 500 vehicles, got {len(v_df)}"
        assert len(s_df) == 7293, f"Expected 7,293 service records, got {len(s_df)}"
        assert len(q_df) == 500, f"Expected 500 quality records, got {len(q_df)}"
        assert set(v_df["chassis"]) == set(s_df["chassis"]) == set(q_df["chassis"])


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

    def test_anomaly_subtypes_present(self):
        """Verify all three seeded anomaly subtypes are present (25 each)."""
        counts = self.features_df["anomaly_type"].value_counts()
        assert counts["rollback"] == 25
        assert counts["gap"] == 25
        assert counts["jump"] == 25
        assert counts["normal"] == 425

    def test_no_cars_or_suvs_only_two_wheelers(self):
        """Verify 100% of vehicles are motorcycles or scooters, no cars or SUVs."""
        categories = set(v.get("category") for v in self.vehicles)
        assert categories == {"Motorcycle", "Scooter"}
        for v in self.vehicles:
            assert v.get("category") in ["Motorcycle", "Scooter"], f"Non-two-wheeler found: {v}"

    def test_age_distribution_over_70_percent_older_than_3_years(self):
        """Verify at least 70% of records are older than 3 years (2018-2022)."""
        ages = [v["ageYears"] for v in self.vehicles]
        over_3y = sum(1 for a in ages if a > 3.0)
        share_over_3y = over_3y / len(ages)
        assert share_over_3y >= 0.70, f"Expected >= 70% older than 3y, got {share_over_3y:.1%}"
        assert share_over_3y == pytest.approx(0.72, abs=0.01)

    def test_older_models_dominate_newer_models(self):
        """Verify older generation models (400, 80%) outnumber newer models (100, 20%)."""
        tiers = [v["tier"] for v in self.vehicles]
        older_count = sum(1 for t in tiers if t == "older")
        newer_count = sum(1 for t in tiers if t == "newer")
        assert older_count == 400
        assert newer_count == 100
        assert older_count >= newer_count * 3

    def test_manufacturing_years_range(self):
        """Verify all vehicles were built between 2018 and 2024."""
        years = set(v["year"] for v in self.vehicles)
        assert years == {2018, 2019, 2020, 2021, 2022, 2023, 2024}


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
            "service_interval_mean": 136.0,
            "service_interval_std": 12.0,
            "odometer_increment_mean": 3160.0,
            "odometer_increment_std": 265.0,
            "confirmed_ratio": 0.97,
            "unique_dealers": 3,
            "gap_count": 0,
            "service_regularity_score": 0.92,
        }])[FEATURES]

        scaled = scaler.transform(normal_vec)
        pred = model.predict(scaled)[0]
        assert pred in [1, -1]

    def test_evaluation_reasons_and_verdict_by_chassis(self):
        # Test first vehicle using official chassis ID
        chassis_id = "ME4A185SFWPA0255V"
        eval_res = get_anomaly_evaluation(chassis_id)
        assert eval_res is not None
        assert eval_res["vehicleId"] == chassis_id
        assert eval_res["verdict"] in ["Suspicious", "Normal"]
        assert isinstance(eval_res["reasons"], list)
        assert len(eval_res["reasons"]) > 0
        assert "anomalyScore" in eval_res

    def test_evaluation_reasons_and_verdict_by_alias(self):
        # Test alias ID for backward compatibility
        eval_res = get_anomaly_evaluation("RELAY_TW_001")
        assert eval_res is not None
        assert eval_res["vehicleId"] == "RELAY_TW_001"
        assert eval_res["verdict"] in ["Suspicious", "Normal"]
        assert isinstance(eval_res["reasons"], list)

    def test_training_metrics_file_has_evaluation(self):
        metrics_path = BASE / "training_metrics.json"
        assert metrics_path.exists(), "training_metrics.json missing"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert metrics["dataset"] == "official_two_wheeler_dataset"
        assert "methodology" in metrics
        assert metrics["train_rows"] == 300
        assert metrics["validation_rows"] == 100
        assert metrics["test_rows"] == 100
        assert "validation_tuning_results" in metrics
        assert "contamination_comparison" in metrics
        test_m = metrics["test_metrics"]
        assert "recall" in test_m
        assert "precision" in test_m
        assert "f1_score" in test_m
        assert "confusion_matrix_labels_0_normal_1_anomaly" in test_m
        assert "classification_report" in test_m
        assert "subtype_breakdown" in test_m
