"""
Automated integration tests for Flask Microservice endpoints (Official Two-Wheeler Scope).
Tests:
- GET /health
- GET /internal/vehicles/:id/price (chassis ID and alias ID, 404 handling)
- GET /internal/vehicles/:id/anomaly-score (chassis ID and alias ID, 404 handling)
- GET /internal/vehicles (vehicle listing: 500 total vehicles)
- Error handling on non-existent routes
"""

import pytest
from serve import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


class TestFlaskEndpoints:
    def test_health_endpoint(self, client):
        res = client.get("/health")
        assert res.status_code == 200
        data = res.get_json()
        assert data["status"] == "ok"
        assert data["service"] == "relay-ai-module"
        assert data["modelLoaded"] is True
        assert data["scalerLoaded"] is True
        assert data["dataset"] == "official_two_wheeler_dataset"

    def test_internal_price_official_chassis(self, client):
        """Test pricing using official chassis ID (Honda Activa 125)."""
        res = client.get("/internal/vehicles/ME4A185SFWPA0255V/price")
        assert res.status_code == 200
        data = res.get_json()
        assert data["vehicleId"] == "ME4A185SFWPA0255V"
        assert "fairPriceLow" in data
        assert "fairPriceHigh" in data
        assert data["fairPriceLow"] < data["fairPriceHigh"]
        assert data["currency"] == "INR"
        assert data["vehicleType"] == "Scooter"
        assert data["details"]["showroomPrice"] == 82000
        assert data["details"]["condition"] == "Average"
        assert data["details"]["conditionAdjustmentPercent"] == 0.0
        # Verify exact ±8% tolerance range
        baseline = data["details"]["adjustedFairPrice"]
        assert data["fairPriceLow"] == round(baseline * 0.92)
        assert data["fairPriceHigh"] == round(baseline * 1.08)

    def test_internal_price_valid_alias(self, client):
        """Test pricing using backward-compatible alias ID."""
        res = client.get("/internal/vehicles/RELAY_TW_001/price")
        assert res.status_code == 200
        data = res.get_json()
        assert data["vehicleId"] == "RELAY_TW_001"
        assert "fairPriceLow" in data
        assert "fairPriceHigh" in data

    def test_internal_price_nonexistent_vehicle(self, client):
        res = client.get("/internal/vehicles/NON_EXISTENT_999/price")
        assert res.status_code == 404
        data = res.get_json()
        assert "error" in data

    def test_internal_anomaly_official_chassis(self, client):
        """Test anomaly detection using official chassis ID."""
        res = client.get("/internal/vehicles/ME4A185SFWPA0255V/anomaly-score")
        assert res.status_code == 200
        data = res.get_json()
        assert data["vehicleId"] == "ME4A185SFWPA0255V"
        assert "anomalyScore" in data
        assert data["verdict"] in ["Suspicious", "Normal"]
        assert isinstance(data["reasons"], list)
        assert len(data["reasons"]) > 0

    def test_internal_anomaly_deterministic_checks_rollback_vehicle(self, client):
        """Verify API returns deterministic rollback warning for vehicle with negative odometer increments."""
        res = client.get("/internal/vehicles/ME4A185SFWPA0255V/anomaly-score")
        assert res.status_code == 200
        data = res.get_json()
        assert "deterministicChecks" in data
        dc = data["deterministicChecks"]
        assert dc["rollbackWarning"] is True
        assert dc["hasNegativeIncrement"] is True
        assert dc["negativeIncrementCount"] == 1
        assert dc["minIncrementKm"] < 0
        assert dc["warningMessage"] is not None
        assert "Odometer rollback warning" in dc["warningMessage"]
        # Verify ML evaluation is also provided separately
        assert "mlEvaluation" in data
        assert data["mlEvaluation"]["model"] == "IsolationForest"

    def test_internal_anomaly_deterministic_checks_normal_vehicle(self, client):
        """Verify API returns no rollback warning for a normal vehicle."""
        res = client.get("/internal/vehicles/ME4A23AYVN1T07BE6/anomaly-score")
        assert res.status_code == 200
        data = res.get_json()
        assert "deterministicChecks" in data
        dc = data["deterministicChecks"]
        assert dc["rollbackWarning"] is False
        assert dc["hasNegativeIncrement"] is False
        assert dc["negativeIncrementCount"] == 0
        assert dc["warningMessage"] is None

    def test_internal_anomaly_valid_alias(self, client):
        """Test anomaly detection using alias ID."""
        res = client.get("/internal/vehicles/RELAY_TW_001/anomaly-score")
        assert res.status_code == 200
        data = res.get_json()
        assert data["vehicleId"] == "RELAY_TW_001"
        assert "anomalyScore" in data

    def test_internal_anomaly_nonexistent_vehicle(self, client):
        res = client.get("/internal/vehicles/NON_EXISTENT_999/anomaly-score")
        assert res.status_code == 404
        data = res.get_json()
        assert "error" in data

    def test_internal_vehicles_list(self, client):
        res = client.get("/internal/vehicles")
        assert res.status_code == 200
        data = res.get_json()
        assert data["totalVehicles"] == 500

    def test_undefined_endpoint_returns_404(self, client):
        res = client.get("/internal/invalid-route")
        assert res.status_code == 404
