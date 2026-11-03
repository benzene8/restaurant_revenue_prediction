"""Scenario sensitivity and What-If analysis engine for Restaurant Revenue."""

from typing import Dict, Any, List
import pandas as pd
from src.schema import RestaurantFeatures
from src.predict import RestaurantRevenuePredictor


class ScenarioSimulator:
    """Simulates revenue lifts across operational levers and location shifts."""

    def __init__(self, predictor: RestaurantRevenuePredictor):
        self.predictor = predictor

    def simulate_what_if(
        self,
        base_features: RestaurantFeatures,
        modifications: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compares baseline vs modified scenario."""
        base_output = self.predictor.predict_single(base_features)

        # Apply modifications to base features
        modified_data = base_features.model_dump()
        modified_data.update(modifications)
        modified_features = RestaurantFeatures(**modified_data)

        scenario_output = self.predictor.predict_single(modified_features)

        monthly_diff = scenario_output.predicted_monthly_revenue - base_output.predicted_monthly_revenue
        pct_change = (monthly_diff / max(base_output.predicted_monthly_revenue, 1.0)) * 100.0
        annual_diff = monthly_diff * 12.0

        return {
            "baseline_monthly": base_output.predicted_monthly_revenue,
            "scenario_monthly": scenario_output.predicted_monthly_revenue,
            "monthly_delta": round(monthly_diff, 2),
            "annual_delta": round(annual_diff, 2),
            "percentage_lift": round(pct_change, 2),
            "baseline_output": base_output,
            "scenario_output": scenario_output,
            "applied_changes": modifications
        }

    def sweep_variable(
        self,
        base_features: RestaurantFeatures,
        variable_name: str,
        values: List[Any]
    ) -> pd.DataFrame:
        """Sweeps a single operational or location variable across values to generate sensitivity curve."""
        rows = []
        base_data = base_features.model_dump()

        for val in values:
            temp_data = base_data.copy()
            temp_data[variable_name] = val
            temp_feat = RestaurantFeatures(**temp_data)
            out = self.predictor.predict_single(temp_feat)

            rows.append({
                variable_name: val,
                "predicted_monthly_revenue": out.predicted_monthly_revenue,
                "predicted_annual_revenue": out.predicted_annual_revenue,
                "rev_per_seat_month": out.revenue_per_seat_month,
                "conf_lower": out.confidence_interval_90_lower,
                "conf_upper": out.confidence_interval_90_upper,
            })

        return pd.DataFrame(rows)

