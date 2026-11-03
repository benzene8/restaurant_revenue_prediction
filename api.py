"""FastAPI REST API for Restaurant Revenue Prediction System."""

import os
import sys
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.schema import RestaurantFeatures, RevenuePredictionOutput
from src.predict import RestaurantRevenuePredictor
from src.simulator import ScenarioSimulator

app = FastAPI(
    title="Restaurant Revenue Prediction API",
    description="Production REST API for forecasting restaurant revenue from location demographics and operational metrics.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor: RestaurantRevenuePredictor = None
simulator: ScenarioSimulator = None


@app.on_event("startup")
def startup_event():
    global predictor, simulator
    predictor = RestaurantRevenuePredictor()
    simulator = ScenarioSimulator(predictor)


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "Restaurant Revenue Prediction API",
        "champion_model": predictor.champion_model_name if predictor else "Loading...",
        "metrics": predictor.metrics if predictor else {},
        "version": "1.0.0"
    }


@app.post("/predict", response_model=RevenuePredictionOutput)
def predict_revenue(features: RestaurantFeatures):
    """Predict revenue for an individual restaurant configuration."""
    try:
        return predictor.predict_single(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/simulate-what-if")
def simulate_scenario(payload: Dict[str, Any]):
    """Simulate what-if changes against a baseline restaurant profile."""
    try:
        base_dict = payload.get("base_features", {})
        modifications = payload.get("modifications", {})
        base_features = RestaurantFeatures(**base_dict)
        return simulator.simulate_what_if(base_features, modifications)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/batch-predict")
def batch_predict(restaurants: List[RestaurantFeatures]):
    """Batch predict revenue for multiple restaurants."""
    try:
        records = [r.model_dump() for r in restaurants]
        df = pd.DataFrame(records)
        enriched = predictor.predict_dataframe(df)
        return enriched.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-csv-predict")
async def upload_csv_predict(file: UploadFile = File(...)):
    """Upload a CSV file of restaurant records and receive predictions."""
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        enriched = predictor.predict_dataframe(df)
        return {
            "total_records": len(enriched),
            "total_monthly_revenue": float(enriched["predicted_monthly_revenue"].sum()),
            "mean_monthly_revenue": float(enriched["predicted_monthly_revenue"].mean()),
            "predictions": enriched.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process CSV: {str(e)}")


@app.get("/model-insights")
def get_model_insights():
    """Retrieve model benchmark performance and global feature importances."""
    return {
        "champion_model": predictor.champion_model_name,
        "metrics": predictor.metrics,
        "benchmarks": predictor.bundle.get("benchmark_results", []),
        "top_feature_importances": predictor.feature_importances[:15]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)

