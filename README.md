# 🍽️ Restaurant Revenue Prediction System

A production-grade machine learning and decision-support system that predicts restaurant gross monthly and annual revenue using **customer location demographics** and **operational characteristics**.

The system features multi-model training pipelines (Ridge, Random Forest, HistGradientBoosting), an interactive **Streamlit Web Application**, a **FastAPI REST API**, a **Command-Line Interface (CLI)**, and a **What-If Scenario Simulator**.

---

## 🚀 Key Features

- **Dual-Domain Feature Matrix**:
  - **Customer Location & Demographics**: Foot traffic volume (1-100), population density (3km radius), median household income, transit proximity, commercial/office hub density, competitor clustering index, parking availability, urban setting classification.
  - **Operational Characteristics**: Seating capacity, dining format/concept (Fine Dining, Casual, Fast Casual, QSR, Cafe, Bar), average guest ticket size, weekly operating hours, online delivery aggregator share %, staff headcount, POS & kitchen tech stack, monthly marketing spend, customer ratings and review volume.
- **High-Performance Machine Learning Engine**:
  - Benchmarked models: **Ridge Regression**, **Random Forest Regressor**, and **HistGradientBoosting Regressor**.
  - Champion Model: **HistGradientBoosting** achieving **$R^2 = 0.8945$** and **$\text{MAPE} = 15.29\%$** on unseen test data.
  - Prediction intervals: 90% confidence bands ($\hat{y} \pm 1.645 \times \sigma_{\text{residual}}$).
- **Interactive Web Dashboard (`app.py`)**:
  - **Single Prediction**: Live KPI cards for monthly, annual, daily revenue run-rates, and revenue per seat.
  - **What-If Sensitivity Lab**: Sliders to test the revenue lift of adding seats, raising ticket prices, activating delivery, or improving reviews.
  - **Geospatial & Foot Traffic Map**: Interactive map with multi-metro clustering and scatter plots.
  - **Batch Portfolio Analyzer**: Upload custom CSVs or benchmark 200+ restaurants simultaneously with downloadable predictions.
  - **Model Diagnostics**: Benchmark scorecard and global feature importance charts.
- **Production REST API (`api.py`)**:
  - FastAPI service with OpenAPI / Swagger docs at `http://127.0.0.1:8000/docs`.
- **Command-Line Interface (`cli.py`)**:
  - Predict revenue, run what-if simulations, or batch process CSV files directly in terminal scripts.

---

## 🏆 Model Performance Benchmark

| Model Architecture | $R^2$ Score | Mean Absolute Error (MAE) | Mean Absolute % Error (MAPE) | Root Mean Squared Error (RMSE) |
| :--- | :---: | :---: | :---: | :---: |
| **HistGradientBoosting (Champion)** | **0.8945** | **$69,727** | **15.29%** | **$102,332** |
| Random Forest Regressor | 0.8187 | $93,836 | 22.70% | $134,184 |
| Ridge Regression (Baseline) | 0.8067 | $100,145 | 31.96% | $138,546 |

### Top 10 Revenue Drivers
1. **Seating Capacity** (53.78%) - Primary physical throughput ceiling
2. **Average Ticket Spend** (9.29%) - Revenue multiplier per cover
3. **Foot Traffic Index** (4.90%) - Pedestrian walk-in capture potential
4. **Customer Rating** (4.49%) - Conversion rate and repeat visitation driver
5. **Catchment Population Density** (4.27%) - Addressable local customer pool
6. **Delivery Potential Index** (2.64%) - Synergetic online ordering & aggregator reach
7. **Rating Credibility Score** (2.55%) - Rating weighted by review volume
8. **Concept Format (Cafe / QSR / Dining)** (2.47%) - Table turnover dynamics
9. **Delivery Share %** (2.39%) - Non-seat revenue contribution
10. **Weekly Operating Hours** (1.67%) - Time aperture for order fulfillment

---

## 📂 Repository Structure

```
Restaurant Revenue Prediction System/
├── data/
│   ├── generate_data.py                  # Economic-grounded synthetic dataset generator
│   └── restaurants_data.csv              # 5,000 restaurant records benchmark
├── models/
│   ├── train.py                          # Multi-model training & benchmarking pipeline
│   ├── restaurant_revenue_pipeline.joblib# Serialized champion model bundle
│   └── benchmark_summary.json            # Model evaluation metrics
├── src/
│   ├── __init__.py
│   ├── schema.py                         # Pydantic data schemas & enums
│   ├── feature_engineering.py            # Derived domain interaction features
│   ├── predict.py                        # Inference engine & recommendation logic
│   └── simulator.py                      # What-if scenario sensitivity analysis
├── tests/
│   └── test_pipeline.py                  # Automated unit test suite (5 passed)
├── app.py                                # Streamlit interactive web dashboard
├── api.py                                # FastAPI REST API backend
├── cli.py                                # Command-line interface
├── run_app.bat                           # 1-click Windows runner for Web App
├── run_api.bat                           # 1-click Windows runner for REST API
├── requirements.txt                      # Project dependencies
└── README.md                             # Documentation & user guide
```

---

## ⚡ Quickstart Guide

### 1. Launch the Interactive Web Dashboard
Double-click `run_app.bat` or run:
```bash
.venv\Scripts\streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 2. Launch the REST API
Double-click `run_api.bat` or run:
```bash
.venv\Scripts\python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```
Open interactive Swagger documentation at `http://127.0.0.1:8000/docs`.

### 3. Use the CLI
**Single Prediction:**
```bash
.venv\Scripts\python cli.py predict --concept "Casual Dining" --seating 90 --ticket 36.0 --foot-traffic 75.0
```

**What-If Simulation:**
```bash
.venv\Scripts\python cli.py simulate --add-seats 25 --add-ticket 4.5 --add-delivery 15.0
```

**View Model Benchmarks:**
```bash
.venv\Scripts\python cli.py benchmark
```

### 4. Run Automated Tests
```bash
.venv\Scripts\python -m pytest tests -v
```
All 5 unit and integration tests validate the data pipeline, feature engineering, and inference engine.

