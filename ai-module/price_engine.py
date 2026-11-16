"""
Fair Price Calculation Engine for RELAY Module 6 (Two-Wheeler Focus).

Calculates estimated fair market value range for Indian two-wheelers (motorcycles & scooters) based on:
1. Showroom price lookup (catalog of 28 two-wheeler models across 7 Indian manufacturers)
2. Age depreciation bands (0-6 mo: 95%, 6-12 mo: 85%, 1-2 yr: 80%, 2-3 yr: 70%, 3-4 yr: 60%, 4-5 yr: 50%)
   - For vehicles >5 years: clearly documented configurable fallback tiers (5-8y: 40%, 8-12y: 30%, >12y: 20%)
3. Condition adjustment (Excellent: +5%, Good: +2%, Average: 0%, Poor: -8%)
4. Odometer status adjustment (Flagged: -10%, Normal: 0%)
5. Fair price range tolerance: +/- 8%
"""

from typing import Any, Dict, Optional, Tuple

# Two-Wheeler Benchmark Catalog: 28 models across 7 Indian manufacturers
# Notes:
# - All prices are illustrative reference benchmarks for development/testing, NOT official OEM prices.
# - Covers both Motorcycles and Scooters.
# - Key format: (Manufacturer, Model) -> (Showroom Benchmark Price in INR, Category, Introduction Year, Tier)
TWO_WHEELER_CATALOG: Dict[Tuple[str, str], Dict[str, Any]] = {
    # --- Hero MotoCorp (Older & Popular) ---
    ("Hero", "Splendor Plus"): {"showroomPrice": 75000, "category": "Motorcycle", "introYear": 2004, "tier": "older"},
    ("Hero", "HF Deluxe"): {"showroomPrice": 62000, "category": "Motorcycle", "introYear": 2012, "tier": "older"},
    ("Hero", "Passion Pro"): {"showroomPrice": 72000, "category": "Motorcycle", "introYear": 2008, "tier": "older"},
    ("Hero", "Maestro"): {"showroomPrice": 68000, "category": "Scooter", "introYear": 2012, "tier": "older"},
    # --- Hero MotoCorp (Newer) ---
    ("Hero", "Xtreme 125R"): {"showroomPrice": 95000, "category": "Motorcycle", "introYear": 2024, "tier": "newer"},

    # --- Bajaj Auto (Older & Popular) ---
    ("Bajaj", "Pulsar 150"): {"showroomPrice": 110000, "category": "Motorcycle", "introYear": 2001, "tier": "older"},
    ("Bajaj", "Platina 100"): {"showroomPrice": 65000, "category": "Motorcycle", "introYear": 2006, "tier": "older"},
    ("Bajaj", "Discover 125"): {"showroomPrice": 60000, "category": "Motorcycle", "introYear": 2004, "tier": "older"},
    # --- Bajaj Auto (Newer) ---
    ("Bajaj", "Pulsar N160"): {"showroomPrice": 130000, "category": "Motorcycle", "introYear": 2022, "tier": "newer"},
    ("Bajaj", "Pulsar NS200"): {"showroomPrice": 145000, "category": "Motorcycle", "introYear": 2012, "tier": "newer"},

    # --- TVS Motor (Older & Popular) ---
    ("TVS", "Apache RTR 160"): {"showroomPrice": 120000, "category": "Motorcycle", "introYear": 2007, "tier": "older"},
    ("TVS", "Star City Plus"): {"showroomPrice": 75000, "category": "Motorcycle", "introYear": 2014, "tier": "older"},
    ("TVS", "Jupiter"): {"showroomPrice": 76000, "category": "Scooter", "introYear": 2013, "tier": "older"},
    ("TVS", "Scooty Pep+"): {"showroomPrice": 65000, "category": "Scooter", "introYear": 2005, "tier": "older"},
    # --- TVS Motor (Newer) ---
    ("TVS", "Apache RTR 200 4V"): {"showroomPrice": 145000, "category": "Motorcycle", "introYear": 2016, "tier": "newer"},
    ("TVS", "Ntorq 125"): {"showroomPrice": 88000, "category": "Scooter", "introYear": 2018, "tier": "newer"},

    # --- Honda (Older & Popular) ---
    ("Honda", "Activa"): {"showroomPrice": 78000, "category": "Scooter", "introYear": 2001, "tier": "older"},
    ("Honda", "CB Shine 125"): {"showroomPrice": 80000, "category": "Motorcycle", "introYear": 2006, "tier": "older"},
    ("Honda", "Unicorn"): {"showroomPrice": 108000, "category": "Motorcycle", "introYear": 2004, "tier": "older"},
    # --- Honda (Newer) ---
    ("Honda", "SP 125"): {"showroomPrice": 86000, "category": "Motorcycle", "introYear": 2019, "tier": "newer"},

    # --- Royal Enfield (Older & Popular) ---
    ("Royal Enfield", "Classic 350"): {"showroomPrice": 195000, "category": "Motorcycle", "introYear": 2009, "tier": "older"},
    ("Royal Enfield", "Bullet 350"): {"showroomPrice": 175000, "category": "Motorcycle", "introYear": 2000, "tier": "older"},
    # --- Royal Enfield (Newer) ---
    ("Royal Enfield", "Hunter 350"): {"showroomPrice": 150000, "category": "Motorcycle", "introYear": 2022, "tier": "newer"},
    ("Royal Enfield", "Meteor 350"): {"showroomPrice": 205000, "category": "Motorcycle", "introYear": 2020, "tier": "newer"},

    # --- Suzuki (Older & Popular) ---
    ("Suzuki", "Access 125"): {"showroomPrice": 82000, "category": "Scooter", "introYear": 2007, "tier": "older"},
    # --- Suzuki (Newer) ---
    ("Suzuki", "Avenis 125"): {"showroomPrice": 92000, "category": "Scooter", "introYear": 2021, "tier": "newer"},

    # --- Yamaha (Newer Performance) ---
    ("Yamaha", "R15 V4"): {"showroomPrice": 182000, "category": "Motorcycle", "introYear": 2021, "tier": "newer"},
    ("Yamaha", "MT-15 V2"): {"showroomPrice": 168000, "category": "Motorcycle", "introYear": 2019, "tier": "newer"},
}

# Backward compatibility alias
BENCHMARK_CATALOG = {k: v["showroomPrice"] for k, v in TWO_WHEELER_CATALOG.items()}

# Condition multiplier adjustments (Assignment specified)
CONDITION_ADJUSTMENTS = {
    "Excellent": 0.05,
    "Good": 0.02,
    "Average": 0.00,
    "Poor": -0.08,
}

# Odometer multiplier adjustments (Assignment specified)
ODOMETER_ADJUSTMENTS = {
    "Normal": 0.00,
    "Flagged": -0.10,
}

# Configurable fallback remaining values for vehicles older than 5 years.
# The official assignment specifies depreciation bands only up to 5 years (50%).
# For older used two-wheelers, we provide clearly documented configurable assumptions:
EXTENDED_AGE_FALLBACK_TIERS = {
    "5_to_8_years": 0.40,
    "8_to_12_years": 0.30,
    "over_12_years": 0.20,
}


def get_depreciation_ratio(age_years: float) -> Tuple[float, str, bool]:
    """
    Return (remaining_value_ratio, band_description, is_extended_fallback).

    Assignment Bands (0 - 5 years):
    - 0 to 0.5 years (0–6 months): 95%
    - 0.5 to 1.0 years (6–12 months): 85%
    - 1.0 to 2.0 years (1–2 years): 80%
    - 2.0 to 3.0 years (2–3 years): 70%
    - 3.0 to 4.0 years (3–4 years): 60%
    - 4.0 to 5.0 years (4–5 years): 50%

    Configurable Fallback Tiers (>5 years):
    - 5.0 to 8.0 years: 40% (documented fallback)
    - 8.0 to 12.0 years: 30% (documented fallback)
    - > 12.0 years: 20% (documented fallback)
    """
    if age_years < 0:
        raise ValueError("Vehicle age cannot be negative")

    if age_years <= 0.5:
        return 0.95, "0–6 months", False
    elif age_years <= 1.0:
        return 0.85, "6–12 months", False
    elif age_years <= 2.0:
        return 0.80, "1–2 years", False
    elif age_years <= 3.0:
        return 0.70, "2–3 years", False
    elif age_years <= 4.0:
        return 0.60, "3–4 years", False
    elif age_years <= 5.0:
        return 0.50, "4–5 years", False
    elif age_years <= 8.0:
        return EXTENDED_AGE_FALLBACK_TIERS["5_to_8_years"], "5–8 years (documented fallback: 40%)", True
    elif age_years <= 12.0:
        return EXTENDED_AGE_FALLBACK_TIERS["8_to_12_years"], "8–12 years (documented fallback: 30%)", True
    else:
        return EXTENDED_AGE_FALLBACK_TIERS["over_12_years"], ">12 years (documented fallback: 20%)", True


def lookup_two_wheeler(make: str, model: str) -> Optional[Dict[str, Any]]:
    """Look up full model details from the two-wheeler catalog."""
    for (m, mod), details in TWO_WHEELER_CATALOG.items():
        if m.lower() == make.lower() and mod.lower() == model.lower():
            return {"make": m, "model": mod, **details}
    return None


def lookup_showroom_price(make: str, model: str) -> Optional[int]:
    """Look up showroom price from the two-wheeler catalog."""
    info = lookup_two_wheeler(make, model)
    return info["showroomPrice"] if info else None


def calculate_fair_price(vehicle: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate the fair price range for a given two-wheeler record.

    Expected vehicle fields:
    - id / vehicleId: str
    - make: Optional[str]
    - model: Optional[str]
    - category: Optional[str] ('Motorcycle' | 'Scooter')
    - showroomPrice: numeric (> 0) (optional if make & model can be looked up)
    - ageYears: numeric (>= 0)
    - condition: 'Excellent' | 'Good' | 'Average' | 'Poor'
    - odometerStatus: 'Normal' | 'Flagged'
    """
    vehicle_id = str(vehicle.get("id") or vehicle.get("vehicleId") or "").strip()
    if not vehicle_id:
        raise ValueError("Missing required vehicle ID")

    # Catalog lookup if make/model provided
    make = vehicle.get("make")
    model = vehicle.get("model")
    catalog_info = None
    if make and model:
        catalog_info = lookup_two_wheeler(str(make), str(model))

    # Showroom price resolution
    showroom_price = vehicle.get("showroomPrice")
    if showroom_price is None and catalog_info:
        showroom_price = catalog_info["showroomPrice"]

    if showroom_price is None or not isinstance(showroom_price, (int, float)) or showroom_price <= 0:
        raise ValueError(f"Invalid or missing showroomPrice: {showroom_price}")

    # Age validation
    age_years = vehicle.get("ageYears")
    if age_years is None or not isinstance(age_years, (int, float)) or age_years < 0:
        raise ValueError(f"Invalid or missing ageYears: {age_years}")

    # Condition validation
    condition_raw = str(vehicle.get("condition", "Average")).strip().capitalize()
    if condition_raw not in CONDITION_ADJUSTMENTS:
        raise ValueError(f"Invalid condition '{condition_raw}'. Allowed: {list(CONDITION_ADJUSTMENTS.keys())}")
    condition_adj = CONDITION_ADJUSTMENTS[condition_raw]

    # Odometer status validation
    odometer_raw = str(vehicle.get("odometerStatus", "Normal")).strip().capitalize()
    if odometer_raw not in ODOMETER_ADJUSTMENTS:
        raise ValueError(f"Invalid odometerStatus '{odometer_raw}'. Allowed: {list(ODOMETER_ADJUSTMENTS.keys())}")
    odometer_adj = ODOMETER_ADJUSTMENTS[odometer_raw]

    # Category resolution (Motorcycle vs Scooter)
    category = vehicle.get("category")
    if not category and catalog_info:
        category = catalog_info["category"]
    category = category or "Two-Wheeler"

    # 1. Depreciation calculation
    deprec_ratio, deprec_band, is_extended = get_depreciation_ratio(float(age_years))
    price_after_deprec = float(showroom_price) * deprec_ratio

    # 2. Condition adjustment
    price_after_condition = price_after_deprec * (1.0 + condition_adj)

    # 3. Odometer adjustment
    final_adjusted = price_after_condition * (1.0 + odometer_adj)

    # 4. Fair price range +/- 8%
    fair_price_low = int(round(final_adjusted * 0.92))
    fair_price_high = int(round(final_adjusted * 1.08))

    return {
        "vehicleId": vehicle_id,
        "vehicleType": category,
        "make": make or (catalog_info["make"] if catalog_info else None),
        "model": model or (catalog_info["model"] if catalog_info else None),
        "fairPriceLow": fair_price_low,
        "fairPriceHigh": fair_price_high,
        "currency": "INR",
        "basis": "Two-wheeler showroom benchmark + Age depreciation + Condition adjustment + Odometer status adjustment (±8% range)",
        "details": {
            "showroomPrice": int(showroom_price),
            "ageYears": round(float(age_years), 2),
            "depreciationBand": deprec_band,
            "remainingValuePercent": round(deprec_ratio * 100, 1),
            "isExtendedAgeFallback": is_extended,
            "condition": condition_raw,
            "conditionAdjustmentPercent": round(condition_adj * 100, 1),
            "odometerStatus": odometer_raw,
            "odometerAdjustmentPercent": round(odometer_adj * 100, 1),
            "basePriceAfterDepreciation": int(round(price_after_deprec)),
            "adjustedFairPrice": int(round(final_adjusted)),
            "assumptions": (
                "Showroom prices are illustrative reference benchmarks. "
                "For vehicles >5 years, a documented tiered fallback is applied (5-8y: 40%, 8-12y: 30%, >12y: 20%), "
                "as the assignment defines depreciation bands only up to 5 years."
            )
        }
    }
