"""
Automated unit tests for Two-Wheeler Fair Price Engine (RELAY Module 6).
Tests:
- Two-wheeler catalog: >= 15 models, >= 7 OEMs, Motorcycles and Scooters
- All depreciation bands (0-6m, 6-12m, 1-2y, 2-3y, 3-4y, 4-5y)
- Extended age fallback tiers (>5y: 5-8y 40%, 8-12y 30%, >12y 20%)
- Condition adjustments (Excellent, Good, Average, Poor)
- Odometer adjustments (Normal, Flagged)
- Motorcycle and Scooter valuation calculations
- Fair price range (+/- 8%)
- Input validation and edge cases
"""

import pytest
from price_engine import (
    calculate_fair_price,
    get_depreciation_ratio,
    lookup_showroom_price,
    lookup_two_wheeler,
    TWO_WHEELER_CATALOG,
    CONDITION_ADJUSTMENTS,
    ODOMETER_ADJUSTMENTS,
)


class TestTwoWheelerCatalog:
    def test_catalog_size_and_manufacturers(self):
        """Verify >= 15 models across >= 7 manufacturers."""
        assert len(TWO_WHEELER_CATALOG) >= 15, f"Catalog has {len(TWO_WHEELER_CATALOG)} models, expected >= 15"
        oems = set(make for make, _ in TWO_WHEELER_CATALOG.keys())
        assert len(oems) >= 7, f"Catalog has {len(oems)} OEMs, expected >= 7"
        expected_oems = {"Hero", "Bajaj", "TVS", "Honda", "Royal Enfield", "Suzuki", "Yamaha"}
        assert expected_oems.issubset(oems), f"Missing required OEMs: {expected_oems - oems}"

    def test_motorcycle_and_scooter_diversity(self):
        """Verify both motorcycles and scooters are present in the catalog."""
        categories = set(meta["category"] for meta in TWO_WHEELER_CATALOG.values())
        assert "Motorcycle" in categories
        assert "Scooter" in categories
        motorcycles = [m for m, meta in TWO_WHEELER_CATALOG.items() if meta["category"] == "Motorcycle"]
        scooters = [m for m, meta in TWO_WHEELER_CATALOG.items() if meta["category"] == "Scooter"]
        assert len(motorcycles) >= 10
        assert len(scooters) >= 5

    def test_lookup_showroom_price_case_insensitive(self):
        # Motorcycle lookup
        assert lookup_showroom_price("Hero", "Splendor Plus") == 75000
        assert lookup_showroom_price("hero", "splendor plus") == 75000
        assert lookup_showroom_price("Bajaj", "Pulsar 150") == 110000
        assert lookup_showroom_price("Royal Enfield", "Classic 350") == 195000
        # Scooter lookup
        assert lookup_showroom_price("Honda", "Activa") == 78000
        assert lookup_showroom_price("TVS", "Jupiter") == 76000
        assert lookup_showroom_price("Suzuki", "Access 125") == 82000
        # Unknown lookup
        assert lookup_showroom_price("Unknown", "Model") is None


class TestDepreciationBands:
    @pytest.mark.parametrize(
        "age_years,expected_ratio,expected_band,expected_extended",
        [
            (0.0, 0.95, "0–6 months", False),
            (0.25, 0.95, "0–6 months", False),
            (0.5, 0.95, "0–6 months", False),
            (0.51, 0.85, "6–12 months", False),
            (1.0, 0.85, "6–12 months", False),
            (1.5, 0.80, "1–2 years", False),
            (2.0, 0.80, "1–2 years", False),
            (2.5, 0.70, "2–3 years", False),
            (3.0, 0.70, "2–3 years", False),
            (3.5, 0.60, "3–4 years", False),
            (4.0, 0.60, "3–4 years", False),
            (4.5, 0.50, "4–5 years", False),
            (5.0, 0.50, "4–5 years", False),
            # Extended age fallback tiers (>5 years)
            (6.0, 0.40, "5–8 years (documented fallback: 40%)", True),
            (8.0, 0.40, "5–8 years (documented fallback: 40%)", True),
            (10.0, 0.30, "8–12 years (documented fallback: 30%)", True),
            (12.0, 0.30, "8–12 years (documented fallback: 30%)", True),
            (15.0, 0.20, ">12 years (documented fallback: 20%)", True),
        ],
    )
    def test_depreciation_ratios(self, age_years, expected_ratio, expected_band, expected_extended):
        ratio, band, is_ext = get_depreciation_ratio(age_years)
        assert ratio == pytest.approx(expected_ratio, abs=1e-4)
        assert band == expected_band
        assert is_ext == expected_extended

    def test_negative_age_raises_error(self):
        with pytest.raises(ValueError, match="Vehicle age cannot be negative"):
            get_depreciation_ratio(-1.0)


class TestFairPriceCalculations:
    def test_motorcycle_calculation_young_excellent(self):
        """Hero Splendor Plus: showroom 75,000, age 0.4y (95%), Excellent (+5%), Normal odo."""
        res = calculate_fair_price({
            "id": "TW_TEST_01",
            "make": "Hero",
            "model": "Splendor Plus",
            "ageYears": 0.4,
            "condition": "Excellent",
            "odometerStatus": "Normal",
        })
        # Base: 75,000 * 0.95 = 71,250
        # Condition (+5%): 71,250 * 1.05 = 74,812.5 -> 74,813
        # Range: +/- 8%
        assert res["vehicleId"] == "TW_TEST_01"
        assert res["vehicleType"] == "Motorcycle"
        assert res["currency"] == "INR"
        assert res["details"]["basePriceAfterDepreciation"] == 71250
        assert res["details"]["adjustedFairPrice"] == round(74812.5)
        assert res["fairPriceLow"] == round(74812.5 * 0.92)
        assert res["fairPriceHigh"] == round(74812.5 * 1.08)

    def test_scooter_calculation_older_average(self):
        """Honda Activa: showroom 78,000, age 4.5y (50%), Average (0%), Flagged odo (-10%)."""
        res = calculate_fair_price({
            "id": "TW_TEST_02",
            "make": "Honda",
            "model": "Activa",
            "ageYears": 4.5,
            "condition": "Average",
            "odometerStatus": "Flagged",
        })
        # Base: 78,000 * 0.50 = 39,000
        # Condition: 0% -> 39,000
        # Odometer (-10%): 39,000 * 0.90 = 35,100
        assert res["vehicleType"] == "Scooter"
        assert res["details"]["adjustedFairPrice"] == 35100
        assert res["fairPriceLow"] == round(35100 * 0.92)
        assert res["fairPriceHigh"] == round(35100 * 1.08)

    def test_older_vehicle_fallback_documented(self):
        """Royal Enfield Classic 350: showroom 195,000, age 7.0y (40% fallback)."""
        res = calculate_fair_price({
            "id": "TW_TEST_03",
            "make": "Royal Enfield",
            "model": "Classic 350",
            "ageYears": 7.0,
            "condition": "Good",  # +2%
            "odometerStatus": "Normal",
        })
        # Base: 195,000 * 0.40 = 78,000
        # Condition (+2%): 78,000 * 1.02 = 79,560
        assert res["details"]["isExtendedAgeFallback"] is True
        assert res["details"]["remainingValuePercent"] == 40.0
        assert res["details"]["adjustedFairPrice"] == 79560
        assert "documented tiered fallback" in res["details"]["assumptions"]

    def test_condition_adjustments(self):
        base_bike = {
            "id": "COND_TW",
            "showroomPrice": 100000,
            "ageYears": 2.0,  # 80% -> 80,000
            "odometerStatus": "Normal",
        }
        # Excellent (+5%): 80,000 * 1.05 = 84,000
        assert calculate_fair_price({**base_bike, "condition": "Excellent"})["details"]["adjustedFairPrice"] == 84000
        # Good (+2%): 80,000 * 1.02 = 81,600
        assert calculate_fair_price({**base_bike, "condition": "Good"})["details"]["adjustedFairPrice"] == 81600
        # Average (0%): 80,000
        assert calculate_fair_price({**base_bike, "condition": "Average"})["details"]["adjustedFairPrice"] == 80000
        # Poor (-8%): 80,000 * 0.92 = 73,600
        assert calculate_fair_price({**base_bike, "condition": "Poor"})["details"]["adjustedFairPrice"] == 73600


class TestInputValidation:
    def test_missing_vehicle_id(self):
        with pytest.raises(ValueError, match="Missing required vehicle ID"):
            calculate_fair_price({"showroomPrice": 75000, "ageYears": 1.0})

    def test_missing_and_invalid_showroom_price(self):
        with pytest.raises(ValueError, match="Invalid or missing showroomPrice"):
            calculate_fair_price({"id": "TW1", "ageYears": 1.0})
        with pytest.raises(ValueError, match="Invalid or missing showroomPrice"):
            calculate_fair_price({"id": "TW1", "showroomPrice": -5000, "ageYears": 1.0})

    def test_invalid_age(self):
        with pytest.raises(ValueError, match="Invalid or missing ageYears"):
            calculate_fair_price({"id": "TW1", "showroomPrice": 75000, "ageYears": -0.5})

    def test_invalid_condition(self):
        with pytest.raises(ValueError, match="Invalid condition"):
            calculate_fair_price({"id": "TW1", "showroomPrice": 75000, "ageYears": 1.0, "condition": "Mint"})

    def test_invalid_odometer_status(self):
        with pytest.raises(ValueError, match="Invalid odometerStatus"):
            calculate_fair_price({"id": "TW1", "showroomPrice": 75000, "ageYears": 1.0, "odometerStatus": "Tampered"})
