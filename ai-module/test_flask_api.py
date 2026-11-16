"""
Automated integration tests for Flask Microservice endpoints (Two-Wheeler RELAY Module 6).
Tests:
- GET /health
- GET /internal/vehicles/:id/price (valid two-wheeler, missing vehicle, 404)
- GET /internal/vehicles/:id/anomaly-score (valid two-wheeler, missing vehicle, 404)
- GET /internal/vehicles (vehicle listing)
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

    def test_internal_price_valid_two_wheeler(self, client):
        res = client.get("/internal/vehicles/RELAY_TW_001/price")
        assert res.status_code == 200
        data = res.get_json()
        assert data["vehicleId"] == "RELAY_TW_001"
        assert "fairPriceLow" in data
        assert "fairPriceHigh" in data
        assert data["fairPriceLow"] < data["fairPriceHigh"]
        assert data["currency"] == "INR"
        assert data["vehicleType"] in ["Motorcycle", "Scooter", "Two-Wheeler"]
        assert "details" in data
        assert data["details"]["showroomPrice"] > 0

    def test_internal_price_nonexistent_vehicle(self, client):
        res = client.get("/internal/vehicles/NON_EXISTENT_999/price")
        assert res.status_code == 404
        data = res.get_json()
        assert "error" in data

    def test_internal_anomaly_valid_two_wheeler(self, client):
        res = client.get("/internal/vehicles/RELAY_TW_001/anomaly-score")
        assert res.status_code == 200
        data = res.get_json()
        assert data["vehicleId"] == "RELAY_TW_001"
        assert "anomalyScore" in data
        assert data["verdict"] in ["Suspicious", "Normal"]
        assert isinstance(data["reasons"], list)

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
