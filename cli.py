"""Command-Line Interface (CLI) for Restaurant Revenue Prediction System."""

import argparse
import sys
import os
import json
import pandas as pd

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.schema import RestaurantFeatures, LocationType, RestaurantType, TechAdoptionLevel
from src.predict import RestaurantRevenuePredictor
from src.simulator import ScenarioSimulator


def get_cli_parser():
    parser = argparse.ArgumentParser(
        description="Restaurant Revenue Prediction System CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. Predict single restaurant
    p_single = subparsers.add_parser("predict", help="Predict revenue for a single restaurant")
    p_single.add_argument("--concept", type=str, default="Casual Dining",
                          choices=[t.value for t in RestaurantType], help="Restaurant concept")
    p_single.add_argument("--location", type=str, default="Downtown / Urban Core",
                          choices=[l.value for l in LocationType], help="Location classification")
    p_single.add_argument("--seating", type=int, default=85, help="Seating capacity")
    p_single.add_argument("--ticket", type=float, default=32.0, help="Average ticket spend ($)")
    p_single.add_argument("--foot-traffic", type=float, default=70.0, help="Foot traffic index (1-100)")
    p_single.add_argument("--delivery-share", type=float, default=25.0, help="Delivery share % (0-85)")
    p_single.add_argument("--hours", type=int, default=70, help="Operating hours per week")
    p_single.add_argument("--rating", type=float, default=4.3, help="Customer rating (1.0-5.0)")
    p_single.add_argument("--sqft", type=int, default=2600, help="Floor area in square feet")
    p_single.add_argument("--income", type=float, default=75000.0, help="Catchment median household income ($)")
    p_single.add_argument("--json", action="store_true", help="Output pure JSON format")

    # 2. Batch predict from CSV
    p_batch = subparsers.add_parser("batch", help="Batch predict on CSV file")
    p_batch.add_argument("--input", type=str, required=True, help="Input CSV path")
    p_batch.add_argument("--output", type=str, default="predictions_output.csv", help="Output CSV path")

    # 3. Model benchmarks and feature drivers
    subparsers.add_parser("benchmark", help="Display model benchmark performance & feature importances")

    # 4. What-if simulation
    p_sim = subparsers.add_parser("simulate", help="Simulate what-if changes against baseline")
    p_sim.add_argument("--add-seats", type=int, default=20, help="Additional seats to test")
    p_sim.add_argument("--add-ticket", type=float, default=5.0, help="Additional ticket price to test ($)")
    p_sim.add_argument("--add-delivery", type=float, default=15.0, help="Additional delivery share % to test")

    return parser


def main():
    # Ensure stdout handles UTF-8 on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = get_cli_parser()
    args = parser.parse_args()

    predictor = RestaurantRevenuePredictor()

    if args.command == "predict":
        feat = RestaurantFeatures(
            restaurant_type=RestaurantType(args.concept),
            location_type=LocationType(args.location),
            seating_capacity=args.seating,
            avg_ticket_size=args.ticket,
            foot_traffic_index=args.foot_traffic,
            delivery_share_pct=args.delivery_share,
            operating_hours_per_week=args.hours,
            customer_rating=args.rating,
            floor_area_sqft=args.sqft,
            median_household_income=args.income
        )
        out = predictor.predict_single(feat)

        if args.json:
            print(json.dumps(out.model_dump(), indent=2))
        else:
            print("=" * 65)
            print("RESTAURANT REVENUE PREDICTION RESULT")
            print("=" * 65)
            print(f"Concept:                     {args.concept} | Location: {args.location}")
            print(f"Seating:                     {args.seating} seats | Avg Ticket: ${args.ticket:.2f}")
            print(f"Foot Traffic:                {args.foot_traffic:.1f}/100 | Rating: {args.rating} ★")
            print("-" * 65)
            print(f"Projected Monthly Revenue:   ${out.predicted_monthly_revenue:,.2f}")
            print(f"Projected Annual Revenue:    ${out.predicted_annual_revenue:,.2f}")
            print(f"Daily Revenue Run-Rate:      ${out.daily_revenue_estimate:,.2f}")
            print(f"90% Prediction Band:         ${out.confidence_interval_90_lower:,.2f} - ${out.confidence_interval_90_upper:,.2f}")
            print(f"Revenue / Seat / Month:      ${out.revenue_per_seat_month:,.2f}")
            print(f"Annual Revenue / SqFt:       ${out.revenue_per_sqft_year:,.2f}")
            print("-" * 65)
            print("Actionable Recommendations:")
            for rec in out.actionable_recommendations:
                print(f"  * {rec}")
            print("=" * 65)

    elif args.command == "batch":
        print(f"Loading input file: {args.input}...")
        df = pd.read_csv(args.input)
        enriched = predictor.predict_dataframe(df)
        enriched.to_csv(args.output, index=False)
        print(f"Batch prediction complete! Saved {len(enriched)} records to {args.output}")

    elif args.command == "benchmark":
        print("=" * 65)
        print("MODEL BENCHMARKS & DIAGNOSTICS")
        print("=" * 65)
        print(f"Champion Architecture: {predictor.champion_model_name}")
        benchmarks = predictor.bundle.get("benchmark_results", [])
        for b in benchmarks:
            print(f"Model: {b['model_name']:<32} | R2: {b['r2_score']:.4f} | MAPE: {b['mape_pct']:.2f}% | MAE: ${b['mae']:,.2f}")
        print("\nTop 10 Global Revenue Drivers:")
        for idx, imp in enumerate(predictor.feature_importances[:10]):
            print(f"  {idx+1:2d}. {imp['feature']:<35} : {imp['importance']*100:.2f}%")
        print("=" * 65)

    elif args.command == "simulate":
        feat = RestaurantFeatures()
        sim = ScenarioSimulator(predictor)
        mods = {
            "seating_capacity": feat.seating_capacity + args.add_seats,
            "avg_ticket_size": feat.avg_ticket_size + args.add_ticket,
            "delivery_share_pct": min(85.0, feat.delivery_share_pct + args.add_delivery)
        }
        res = sim.simulate_what_if(feat, mods)
        print("=" * 65)
        print("WHAT-IF SCENARIO SIMULATION")
        print("=" * 65)
        print(f"Baseline Monthly Revenue:     ${res['baseline_monthly']:,.2f}")
        print(f"Simulated Monthly Revenue:    ${res['scenario_monthly']:,.2f}")
        print(f"Monthly Delta (Net Lift):     ${res['monthly_delta']:+,.2f} ({res['percentage_lift']:+.2f}%)")
        print(f"Annualized Net Lift:          ${res['annual_delta']:+,.2f}/year")
        print("=" * 65)


if __name__ == "__main__":
    main()
