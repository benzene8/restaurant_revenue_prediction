"""Feature engineering for Restaurant Revenue Prediction System."""

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    # Demographics & Location
    "location_type",
    "population_density_3km",
    "median_household_income",
    "foot_traffic_index",
    "transit_proximity_score",
    "commercial_density_score",
    "competitor_count_2km",
    "parking_spaces_available",
    # Operational
    "restaurant_type",
    "seating_capacity",
    "floor_area_sqft",
    "operating_hours_per_week",
    "avg_ticket_size",
    "menu_item_count",
    "online_order_enabled",
    "delivery_share_pct",
    "staff_headcount",
    "tech_adoption_level",
    "marketing_monthly_budget",
    "customer_rating",
    "rating_count",
    "years_in_business",
    # Engineered Ratios & Interactions
    "seats_per_sqft",
    "staff_per_seat",
    "foot_traffic_per_competitor",
    "commercial_transit_index",
    "rating_credibility_score",
    "delivery_potential_index",
    "marketing_per_seat",
]

NUMERICAL_FEATURES = [
    "population_density_3km",
    "median_household_income",
    "foot_traffic_index",
    "transit_proximity_score",
    "commercial_density_score",
    "competitor_count_2km",
    "parking_spaces_available",
    "seating_capacity",
    "floor_area_sqft",
    "operating_hours_per_week",
    "avg_ticket_size",
    "menu_item_count",
    "online_order_enabled",
    "delivery_share_pct",
    "staff_headcount",
    "marketing_monthly_budget",
    "customer_rating",
    "rating_count",
    "years_in_business",
    "seats_per_sqft",
    "staff_per_seat",
    "foot_traffic_per_competitor",
    "commercial_transit_index",
    "rating_credibility_score",
    "delivery_potential_index",
    "marketing_per_seat",
]

CATEGORICAL_FEATURES = [
    "location_type",
    "restaurant_type",
    "tech_adoption_level",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes operational and geospatial domain interaction features."""
    df_feat = df.copy()

    # Avoid division by zero
    sqft = df_feat["floor_area_sqft"].clip(lower=100)
    seats = df_feat["seating_capacity"].clip(lower=1)
    competitors = df_feat["competitor_count_2km"].clip(lower=0)

    # 1. Physical dining density
    df_feat["seats_per_sqft"] = seats / sqft

    # 2. Staffing intensity
    df_feat["staff_per_seat"] = df_feat["staff_headcount"] / seats

    # 3. Foot traffic per competitor (demand pressure)
    df_feat["foot_traffic_per_competitor"] = df_feat["foot_traffic_index"] / (competitors + 1.0)

    # 4. Urban transit-commercial synergy
    df_feat["commercial_transit_index"] = (
        df_feat["commercial_density_score"] * df_feat["transit_proximity_score"]
    )

    # 5. Review credibility (stars weighted by log review volume)
    review_weight = np.log1p(df_feat["rating_count"].clip(lower=0))
    df_feat["rating_credibility_score"] = df_feat["customer_rating"] * review_weight

    # 6. Delivery potential index
    df_feat["delivery_potential_index"] = (
        df_feat["delivery_share_pct"] * df_feat["online_order_enabled"]
    )

    # 7. Marketing investment per seat
    df_feat["marketing_per_seat"] = df_feat["marketing_monthly_budget"] / seats

    return df_feat

