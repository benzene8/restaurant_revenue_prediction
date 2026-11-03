"""Prediction engine for Restaurant Revenue Prediction System."""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from src.schema import RestaurantFeatures, RevenuePredictionOutput
from src.feature_engineering import add_engineered_features, FEATURE_COLUMNS

DEFAULT_MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "restaurant_revenue_pipeline.joblib")
)


class RestaurantRevenuePredictor:
    """Inference engine with confidence interval estimation and actionable business heuristics."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model bundle not found at {self.model_path}. Please run models/train.py first."
            )
        self.bundle = joblib.load(self.model_path)
        self.pipeline = self.bundle["pipeline"]
        self.residual_std = self.bundle.get("residual_std", 100000.0)
        self.feature_importances = self.bundle.get("feature_importances", [])
        self.champion_model_name = self.bundle.get("champion_model_name", "Gradient Boosting")
        self.metrics = self.bundle.get("metrics", {})

    def predict_dataframe(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """Batch prediction over a DataFrame of restaurant records."""
        df_feats = add_engineered_features(df_input)
        X = df_feats[FEATURE_COLUMNS]
        preds = self.pipeline.predict(X)
        
        result_df = df_input.copy()
        result_df["predicted_monthly_revenue"] = np.round(np.clip(preds, 10000.0, None), 2)
        result_df["predicted_annual_revenue"] = np.round(result_df["predicted_monthly_revenue"] * 12.0, 2)
        
        # 90% confidence bounds (z = 1.645)
        half_width = 1.645 * self.residual_std
        result_df["conf_lower_90"] = np.round(np.clip(preds - half_width, 5000.0, None), 2)
        result_df["conf_upper_90"] = np.round(preds + half_width, 2)
        
        # Unit economics
        seats = result_df["seating_capacity"].clip(lower=1)
        sqft = result_df["floor_area_sqft"].clip(lower=100)
        result_df["rev_per_seat_month"] = np.round(result_df["predicted_monthly_revenue"] / seats, 2)
        result_df["rev_per_sqft_year"] = np.round(result_df["predicted_annual_revenue"] / sqft, 2)
        
        return result_df

    def predict_single(self, features: RestaurantFeatures) -> RevenuePredictionOutput:
        """Single restaurant prediction with unit economics and tailored recommendations."""
        data_dict = features.model_dump()
        df_single = pd.DataFrame([data_dict])
        pred_df = self.predict_dataframe(df_single)
        row = pred_df.iloc[0]

        pred_monthly = float(row["predicted_monthly_revenue"])
        pred_annual = float(row["predicted_annual_revenue"])
        conf_lower = float(row["conf_lower_90"])
        conf_upper = float(row["conf_upper_90"])
        rev_per_seat = float(row["rev_per_seat_month"])
        rev_per_sqft = float(row["rev_per_sqft_year"])
        daily_estimate = round(pred_monthly / 30.4, 2)

        # Synthesize actionable insights and growth levers
        recommendations = self._generate_recommendations(features, pred_monthly)

        # Top 5 most influential global drivers
        top_drivers = self.feature_importances[:5] if self.feature_importances else []

        return RevenuePredictionOutput(
            predicted_monthly_revenue=pred_monthly,
            predicted_annual_revenue=pred_annual,
            confidence_interval_90_lower=conf_lower,
            confidence_interval_90_upper=conf_upper,
            daily_revenue_estimate=daily_estimate,
            revenue_per_seat_month=rev_per_seat,
            revenue_per_sqft_year=rev_per_sqft,
            key_drivers=top_drivers,
            actionable_recommendations=recommendations,
            input_summary={
                "concept": features.restaurant_type.value,
                "location": features.location_type.value,
                "seating": features.seating_capacity,
                "ticket_size": features.avg_ticket_size,
                "foot_traffic": features.foot_traffic_index,
                "rating": features.customer_rating,
                "delivery_pct": features.delivery_share_pct
            }
        )

    def _generate_recommendations(self, f: RestaurantFeatures, current_monthly_rev: float) -> List[str]:
        """Prescriptive recommendations based on customer location and operational attributes."""
        recs = []

        # 1. Delivery & Aggregator leverage
        if f.online_order_enabled == 0:
            recs.append(
                "Digital Aggregators Disabled: Activating online ordering and delivery aggregators "
                "could capture non-dine-in demand and expand top-line revenue by an estimated 18% - 28%."
            )
        elif f.delivery_share_pct < 20.0 and f.restaurant_type.value in ["Fast Casual", "Quick Service (QSR)", "Casual Dining"]:
            recs.append(
                f"Delivery Expansion Opportunity: Current delivery share is only {f.delivery_share_pct}%. "
                "Expanding kitchen dispatch and running targeted aggregator promos can push delivery share toward 35%, "
                f"yielding an estimated +${current_monthly_rev * 0.12:,.0f}/month."
            )

        # 2. Foot traffic vs Rating mismatch
        if f.foot_traffic_index >= 70.0 and f.customer_rating < 4.2:
            recs.append(
                f"High Foot Traffic Under-Monetization: Excellent pedestrian index ({f.foot_traffic_index:.0f}/100) "
                f"is bottle-necked by a {f.customer_rating} star rating. Elevating customer experience and service consistency "
                "to reach a 4.5+ rating typically boosts footfall conversion and table turns by 12% - 20%."
            )
        elif f.foot_traffic_index < 45.0 and f.marketing_monthly_budget < 1500.0:
            recs.append(
                f"Low Foot Traffic Catchment: Pedestrian index is modest ({f.foot_traffic_index:.0f}/100). "
                "Increasing localized hyper-targeted social ads and community catering could drive destination visits."
            )

        # 3. Space utilization
        seats_sqft_ratio = f.seating_capacity / max(f.floor_area_sqft, 100)
        if seats_sqft_ratio < 0.026:
            potential_added_seats = int((0.035 - seats_sqft_ratio) * f.floor_area_sqft)
            if potential_added_seats > 8:
                recs.append(
                    f"Floor Space Optimization: Dining room density is low ({seats_sqft_ratio:.3f} seats/sqft). "
                    f"Reconfiguring the floor plan could accommodate ~{potential_added_seats} additional seats, "
                    f"unlocking up to +${potential_added_seats * (current_monthly_rev / max(f.seating_capacity, 1) * 0.5):,.0f}/month."
                )

        # 4. Tech stack efficiency
        if f.tech_adoption_level.value.startswith("Low"):
            recs.append(
                "Technology Modernization: Upgrading from manual workflows to handheld server tablets or self-service "
                "kiosks reduces order-to-kitchen latency by 4-6 minutes per table, accelerating turnover during rush hours."
            )

        # 5. Hours & Dayparting
        if f.operating_hours_per_week < 55:
            recs.append(
                f"Operating Hours Expansion: Current hours ({f.operating_hours_per_week} hrs/week) leave valuable facility downtime. "
                "Piloting a weekend brunch, late-night bites, or afternoon cafe window can generate incremental gross margin."
            )

        # 6. Ticket size calibration vs Household income
        if f.median_household_income > 95000 and f.avg_ticket_size < 25.0 and f.restaurant_type.value in ["Casual Dining", "Bar & Grill"]:
            recs.append(
                f"Affluent Catchment Upselling: Catchment median income is high (${f.median_household_income:,.0f}) relative "
                f"to average check size (${f.avg_ticket_size:.2f}). Introducing craft beverage pairings, premium appetizers, "
                "or dessert menus could increase average spend per cover by $4 - $8."
            )

        if not recs:
            recs.append("Balanced Operation: Operations and location demographics are well-aligned with revenue potential.")

        return recs

