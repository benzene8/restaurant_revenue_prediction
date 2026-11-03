"""Training and benchmarking pipeline for Restaurant Revenue Prediction System."""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ensure root directory is on sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.feature_engineering import (
    add_engineered_features,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
)


def evaluate_model(name: str, model: Pipeline, X_train, y_train, X_test, y_test):
    """Computes comprehensive regression metrics on train and test splits."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100.0

    residuals = y_test - y_pred
    residual_std = float(np.std(residuals))

    metrics = {
        "model_name": name,
        "r2_score": round(float(r2), 4),
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "mape_pct": round(float(mape), 2),
        "residual_std": round(residual_std, 2)
    }
    return metrics, model, y_pred


def train_and_benchmark(data_path: str, save_path: str):
    print("=" * 70)
    print("RESTAURANT REVENUE PREDICTION SYSTEM - MODEL TRAINING PIPELINE")
    print("=" * 70)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run generate_data.py first.")

    df_raw = pd.read_csv(data_path)
    print(f"Loaded {len(df_raw)} records from {data_path}")

    df = add_engineered_features(df_raw)
    X = df[FEATURE_COLUMNS]
    y = df["monthly_revenue"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Train set: {len(X_train)} samples | Test set: {len(X_test)} samples")

    # Column transformer for numeric scaling & categorical one-hot encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )

    candidate_models = {
        "Ridge Regression (Baseline)": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=10.0))
        ]),
        "Random Forest Regressor": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", RandomForestRegressor(
                n_estimators=160,
                max_depth=18,
                min_samples_split=4,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42
            ))
        ]),
        "HistGradientBoosting Regressor": Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", HistGradientBoostingRegressor(
                max_iter=250,
                learning_rate=0.06,
                max_depth=9,
                min_samples_leaf=15,
                l2_regularization=0.1,
                random_state=42
            ))
        ])
    }

    benchmark_results = []
    trained_pipelines = {}

    for name, pipeline in candidate_models.items():
        print(f"\nTraining and evaluating: {name}...")
        metrics, fitted_pipeline, _ = evaluate_model(
            name, pipeline, X_train, y_train, X_test, y_test
        )
        benchmark_results.append(metrics)
        trained_pipelines[name] = fitted_pipeline
        print(f"  --> R2: {metrics['r2_score']:.4f} | MAE: ${metrics['mae']:,.2f} | MAPE: {metrics['mape_pct']:.2f}% | RMSE: ${metrics['rmse']:,.2f}")

    # Select best model based on R2 score
    best_result = max(benchmark_results, key=lambda x: x["r2_score"])
    best_name = best_result["model_name"]
    best_pipeline = trained_pipelines[best_name]

    print("\n" + "=" * 70)
    print(f"CHAMPION MODEL SELECTED: {best_name}")
    print(f"R^2 Score: {best_result['r2_score']:.4f} | Mean Absolute Percentage Error: {best_result['mape_pct']:.2f}%")
    print("=" * 70)

    # Extract Feature Importances from Random Forest for explainability
    rf_pipeline = trained_pipelines["Random Forest Regressor"]
    rf_regressor = rf_pipeline.named_steps["regressor"]
    fitted_preprocessor = rf_pipeline.named_steps["preprocessor"]

    # Retrieve one-hot encoded category names
    cat_encoder = fitted_preprocessor.named_transformers_["cat"]
    encoded_cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
    all_feature_names = NUMERICAL_FEATURES + encoded_cat_names

    importances = rf_regressor.feature_importances_
    feat_importance_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    print("\nTop 10 Most Influential Revenue Drivers:")
    for idx, row in feat_importance_df.head(10).iterrows():
        print(f"  {idx+1}. {row['feature']:<35} : {row['importance'] * 100:.2f}%")

    # Package metadata & serialized artifacts
    bundle = {
        "pipeline": best_pipeline,
        "rf_explainer_pipeline": rf_pipeline,
        "champion_model_name": best_name,
        "benchmark_results": benchmark_results,
        "feature_columns": FEATURE_COLUMNS,
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "all_transformed_feature_names": all_feature_names,
        "feature_importances": feat_importance_df.to_dict(orient="records"),
        "residual_std": best_result["residual_std"],
        "metrics": best_result
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(bundle, save_path)
    print(f"\nModel pipeline and artifact bundle saved to: {save_path}")

    # Also save benchmark summary JSON for web dashboard
    summary_path = os.path.join(os.path.dirname(save_path), "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(bundle["benchmark_results"], f, indent=2)

    return bundle


if __name__ == "__main__":
    data_file = os.path.join(BASE_DIR, "data", "restaurants_data.csv")
    save_file = os.path.join(BASE_DIR, "models", "restaurant_revenue_pipeline.joblib")
    train_and_benchmark(data_file, save_file)

