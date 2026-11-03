"""Automated test suite for Restaurant Revenue Prediction System."""

import os
import sys
import pytest
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.schema import (
    RestaurantFeatures,
    LocationType,
    RestaurantType,
    TechAdoptionLevel,
    RevenuePredictionOutput
)
from src.feature_engineering import add_engineered_features, FEATURE_COLUMNS
from src.predict import RestaurantRevenuePredictor
from src.simulator import ScenarioSimulator
from data.generate_data import generate_restaurant_dataset


def test_dataset_generation():
    """Verify data generator produces valid rows and non-null distributions."""
    df = generate_restaurant_dataset(num_samples=100, random_state=123)
    assert len(df) == 100
    assert not df.isnull().values.any()
    assert (df["monthly_revenue"] > 5000).all()
    assert (df["seating_capacity"] >= 10).all()
    assert (df["avg_ticket_size"] >= 5.0).all()
    assert (df["foot_traffic_index"] >= 1.0).all()


def test_feature_engineering():
    """Verify derived domain interaction features are accurately computed."""
    df_raw = pd.DataFrame([{
        "floor_area_sqft": 2000,
        "seating_capacity": 50,
        "staff_headcount": 10,
        "competitor_count_2km": 4,
        "foot_traffic_index": 80.0,
        "commercial_density_score": 7.0,
        "transit_proximity_score": 8.0,
        "customer_rating": 4.5,
        "rating_count": 250,
        "delivery_share_pct": 30.0,
        "online_order_enabled": 1,
        "marketing_monthly_budget": 1500.0,
    }])
    df_feat = add_engineered_features(df_raw)

    assert "seats_per_sqft" in df_feat.columns
    assert np.isclose(df_feat["seats_per_sqft"].iloc[0], 50 / 2000)
    assert np.isclose(df_feat["staff_per_seat"].iloc[0], 10 / 50)
    assert np.isclose(df_feat["foot_traffic_per_competitor"].iloc[0], 80.0 / 5.0)
    assert np.isclose(df_feat["commercial_transit_index"].iloc[0], 56.0)
    assert np.isclose(df_feat["delivery_potential_index"].iloc[0], 30.0)


def test_prediction_pipeline():
    """Verify inference engine loads and outputs valid bounded revenue estimates."""
    predictor = RestaurantRevenuePredictor()
    feat = RestaurantFeatures(
        restaurant_type=RestaurantType.CASUAL_DINING,
        location_type=LocationType.DOWNTOWN,
        seating_capacity=100,
        avg_ticket_size=35.0,
        foot_traffic_index=75.0,
        delivery_share_pct=20.0
    )
    result = predictor.predict_single(feat)

    assert isinstance(result, RevenuePredictionOutput)
    assert result.predicted_monthly_revenue > 20000.0
    assert result.predicted_annual_revenue == pytest.approx(result.predicted_monthly_revenue * 12.0, rel=1e-2)
    assert result.confidence_interval_90_lower <= result.predicted_monthly_revenue
    assert result.confidence_interval_90_upper >= result.predicted_monthly_revenue
    assert result.revenue_per_seat_month > 0
    assert len(result.actionable_recommendations) > 0


def test_scenario_simulator():
    """Verify that expanding seating capacity increases projected revenue."""
    predictor = RestaurantRevenuePredictor()
    simulator = ScenarioSimulator(predictor)

    base = RestaurantFeatures(seating_capacity=60, avg_ticket_size=25.0)
    modifications = {"seating_capacity": 120}

    sim_res = simulator.simulate_what_if(base, modifications)

    assert sim_res["monthly_delta"] > 0
    assert sim_res["scenario_monthly"] > sim_res["baseline_monthly"]
    assert sim_res["percentage_lift"] > 0


def test_batch_prediction():
    """Verify batch inference produces expected columns on sample records."""
    predictor = RestaurantRevenuePredictor()
    df_sample = generate_restaurant_dataset(num_samples=15, random_state=99)
    pred_df = predictor.predict_dataframe(df_sample)

    assert len(pred_df) == 15
    assert "predicted_monthly_revenue" in pred_df.columns
    assert "conf_lower_90" in pred_df.columns
    assert "conf_upper_90" in pred_df.columns
    assert (pred_df["predicted_monthly_revenue"] > 0).all()

