"""Streamlit Web Application: Restaurant Revenue Prediction System.
An interactive AI-powered revenue forecasting and what-if simulation dashboard.
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is on sys.path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.schema import (
    LocationType,
    RestaurantType,
    TechAdoptionLevel,
    RestaurantFeatures,
)
from src.predict import RestaurantRevenuePredictor
from src.simulator import ScenarioSimulator

st.set_page_config(
    page_title="Restaurant Revenue Prediction System",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #0f172a 100%);
        color: white;
        padding: 24px 28px;
        border-radius: 14px;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .metric-container {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }
    .rec-box {
        background-color: #f0fdf4;
        border-left: 4px solid #16a34a;
        padding: 12px 16px;
        border-radius: 6px;
        margin-bottom: 10px;
    }
    .card-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 8px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor():
    return RestaurantRevenuePredictor()


@st.cache_data
def load_sample_dataset():
    data_path = os.path.join(BASE_DIR, "data", "restaurants_data.csv")
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return pd.DataFrame()


try:
    predictor = get_predictor()
    simulator = ScenarioSimulator(predictor)
    model_ready = True
except Exception as e:
    model_ready = False
    init_error = str(e)

# Header Section
st.markdown("""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div>
            <h1 style="margin: 0; font-size: 2.1rem; font-weight: 700;">🍽️ Restaurant Revenue Prediction System</h1>
            <p style="margin: 6px 0 0 0; font-size: 1.05rem; opacity: 0.9;">
                Predict gross revenue using granular Customer Location Demographics & Operational Characteristics.
            </p>
        </div>
        <div style="background: rgba(255,255,255,0.15); padding: 8px 16px; border-radius: 8px; margin-top: 8px;">
            <span style="font-weight: 600; font-size: 0.9rem;">Champion Model:</span>
            <span style="background: #22c55e; color: white; padding: 2px 8px; border-radius: 4px; font-size: 0.85rem; margin-left: 6px;">HistGradientBoosting</span>
            <span style="margin-left: 8px; font-size: 0.85rem;">R² = 0.8945 | MAPE = 15.3%</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if not model_ready:
    st.error(f"❌ Model failed to load: {init_error}. Please run `models/train.py` first.")
    st.stop()

# -------------------------------------------------------------
# SIDEBAR: Quick Presets & System Navigation
# -------------------------------------------------------------
st.sidebar.title("⚙️ Restaurant Setup & Presets")
st.sidebar.markdown("Quickly populate typical restaurant profiles:")

presets = {
    "Custom Configuration": None,
    "🏙️ Downtown Fine Dining Bistro": {
        "location_type": LocationType.DOWNTOWN,
        "restaurant_type": RestaurantType.FINE_DINING,
        "seating_capacity": 85,
        "floor_area_sqft": 3200,
        "avg_ticket_size": 115.0,
        "operating_hours_per_week": 50,
        "online_order_enabled": 0,
        "delivery_share_pct": 0.0,
        "foot_traffic_index": 85.0,
        "population_density_3km": 16000.0,
        "median_household_income": 110000.0,
        "transit_proximity_score": 9.0,
        "commercial_density_score": 9.0,
        "competitor_count_2km": 35,
        "parking_spaces_available": 10,
        "staff_headcount": 28,
        "tech_adoption_level": TechAdoptionLevel.MEDIUM,
        "marketing_monthly_budget": 4500.0,
        "customer_rating": 4.6,
        "rating_count": 850,
        "years_in_business": 4.5,
        "latitude": 40.7128,
        "longitude": -74.0060
    },
    "⚡ Suburban Fast Casual Bowl Hub": {
        "location_type": LocationType.SUBURBAN_STRIP,
        "restaurant_type": RestaurantType.FAST_CASUAL,
        "seating_capacity": 65,
        "floor_area_sqft": 1800,
        "avg_ticket_size": 18.5,
        "operating_hours_per_week": 80,
        "online_order_enabled": 1,
        "delivery_share_pct": 42.0,
        "foot_traffic_index": 55.0,
        "population_density_3km": 4800.0,
        "median_household_income": 82000.0,
        "transit_proximity_score": 4.5,
        "commercial_density_score": 5.5,
        "competitor_count_2km": 14,
        "parking_spaces_available": 80,
        "staff_headcount": 14,
        "tech_adoption_level": TechAdoptionLevel.HIGH,
        "marketing_monthly_budget": 2000.0,
        "customer_rating": 4.4,
        "rating_count": 520,
        "years_in_business": 2.8,
        "latitude": 30.2672,
        "longitude": -97.7431
    },
    "☕ College Town Artisan Cafe": {
        "location_type": LocationType.COMMERCIAL_DISTRICT,
        "restaurant_type": RestaurantType.CAFE_BAKERY,
        "seating_capacity": 38,
        "floor_area_sqft": 1100,
        "avg_ticket_size": 11.5,
        "operating_hours_per_week": 65,
        "online_order_enabled": 1,
        "delivery_share_pct": 20.0,
        "foot_traffic_index": 78.0,
        "population_density_3km": 8500.0,
        "median_household_income": 62000.0,
        "transit_proximity_score": 8.0,
        "commercial_density_score": 8.2,
        "competitor_count_2km": 18,
        "parking_spaces_available": 15,
        "staff_headcount": 7,
        "tech_adoption_level": TechAdoptionLevel.HIGH,
        "marketing_monthly_budget": 900.0,
        "customer_rating": 4.7,
        "rating_count": 1100,
        "years_in_business": 5.0,
        "latitude": 47.6062,
        "longitude": -122.3321
    },
    "🍔 High-Volume Highway QSR / Drive-Thru": {
        "location_type": LocationType.HIGHWAY_CORRIDOR,
        "restaurant_type": RestaurantType.QSR,
        "seating_capacity": 55,
        "floor_area_sqft": 2200,
        "avg_ticket_size": 13.0,
        "operating_hours_per_week": 110,
        "online_order_enabled": 1,
        "delivery_share_pct": 35.0,
        "foot_traffic_index": 60.0,
        "population_density_3km": 2800.0,
        "median_household_income": 58000.0,
        "transit_proximity_score": 2.5,
        "commercial_density_score": 4.0,
        "competitor_count_2km": 6,
        "parking_spaces_available": 120,
        "staff_headcount": 22,
        "tech_adoption_level": TechAdoptionLevel.HIGH,
        "marketing_monthly_budget": 3500.0,
        "customer_rating": 4.0,
        "rating_count": 950,
        "years_in_business": 6.5,
        "latitude": 34.0522,
        "longitude": -118.2437
    }
}

selected_preset = st.sidebar.selectbox("Load Profile Preset", list(presets.keys()))
preset_vals = presets[selected_preset] if presets[selected_preset] else {}

# Helper to get preset or default
def pval(key, default):
    return preset_vals.get(key, default)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📌 Key Input Factors")
st.sidebar.caption("• **Location**: Foot traffic, population density, median income, transit access, competitors.")
st.sidebar.caption("• **Operations**: Concept, seats, ticket size, hours, delivery share, staff, tech level, rating.")

# -------------------------------------------------------------
# MAIN APP TABS
# -------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predict Revenue & What-If",
    "🗺️ Geospatial & Location Explorer",
    "📊 Batch Analysis & Benchmarks",
    "🧠 Model Diagnostics & Explainability"
])

# -------------------------------------------------------------
# TAB 1: PREDICT REVENUE & WHAT-IF SIMULATOR
# -------------------------------------------------------------
with tab1:
    col_loc, col_ops = st.columns(2, gap="large")

    with col_loc:
        st.markdown("### 📍 Customer Location & Demographics")
        loc_type_options = list(LocationType)
        loc_type = st.selectbox(
            "Location Urban Classification",
            loc_type_options,
            index=loc_type_options.index(pval("location_type", LocationType.DOWNTOWN))
        )
        
        c1, c2 = st.columns(2)
        with c1:
            foot_traffic = st.slider(
                "Foot Traffic Index (1-100)",
                min_value=1.0, max_value=100.0,
                value=float(pval("foot_traffic_index", 72.0)),
                step=1.0,
                help="Pedestrian density and sidewalk volume score."
            )
            pop_density = st.number_input(
                "Pop. Density within 3km (people/km²)",
                min_value=200, max_value=40000,
                value=int(pval("population_density_3km", 7500)),
                step=250
            )
            transit_score = st.slider(
                "Transit Proximity Score (1-10)",
                min_value=1.0, max_value=10.0,
                value=float(pval("transit_proximity_score", 7.5)),
                step=0.5
            )
        with c2:
            income = st.number_input(
                "Median Household Income ($/yr)",
                min_value=25000, max_value=250000,
                value=int(pval("median_household_income", 78000)),
                step=1000
            )
            comp_count = st.number_input(
                "Competing Restaurants within 2km",
                min_value=0, max_value=80,
                value=int(pval("competitor_count_2km", 16)),
                step=1
            )
            parking = st.number_input(
                "Dedicated / Validated Parking Spaces",
                min_value=0, max_value=300,
                value=int(pval("parking_spaces_available", 25)),
                step=5
            )
        
        comm_density = st.slider(
            "Commercial & Office Hub Density (1-10)",
            min_value=1.0, max_value=10.0,
            value=float(pval("commercial_density_score", 7.0)),
            step=0.5
        )

    with col_ops:
        st.markdown("### ⚙️ Restaurant Operational Characteristics")
        rest_type_options = list(RestaurantType)
        rest_type = st.selectbox(
            "Dining Concept / Restaurant Type",
            rest_type_options,
            index=rest_type_options.index(pval("restaurant_type", RestaurantType.CASUAL_DINING))
        )

        o1, o2 = st.columns(2)
        with o1:
            seating = st.number_input(
                "Seating Capacity (Dining + Bar)",
                min_value=10, max_value=450,
                value=int(pval("seating_capacity", 85)),
                step=5
            )
            ticket_size = st.number_input(
                "Average Check Spend / Ticket ($)",
                min_value=6.0, max_value=250.0,
                value=float(pval("avg_ticket_size", 34.0)),
                step=1.0
            )
            hours_week = st.number_input(
                "Operating Hours per Week",
                min_value=30, max_value=126,
                value=int(pval("operating_hours_per_week", 68)),
                step=2
            )
            delivery_share = st.slider(
                "Delivery & Takeout Sales Share (%)",
                min_value=0.0, max_value=85.0,
                value=float(pval("delivery_share_pct", 24.0)),
                step=1.0
            )
        with o2:
            sqft = st.number_input(
                "Floor Area (Square Feet)",
                min_value=300, max_value=12000,
                value=int(pval("floor_area_sqft", 2600)),
                step=100
            )
            staff = st.number_input(
                "Total Staff Headcount",
                min_value=2, max_value=75,
                value=int(pval("staff_headcount", 16)),
                step=1
            )
            cust_rating = st.slider(
                "Customer Review Rating (★)",
                min_value=1.5, max_value=5.0,
                value=float(pval("customer_rating", 4.3)),
                step=0.1
            )
            online_order = st.selectbox(
                "Online / App Ordering Enabled?",
                [1, 0],
                index=0 if pval("online_order_enabled", 1) == 1 else 1,
                format_func=lambda x: "Yes (Integrated Delivery)" if x == 1 else "No (Dine-in/Walk-in Only)"
            )

        t_opts = list(TechAdoptionLevel)
        tech_level = st.selectbox(
            "Technology & POS Stack Level",
            t_opts,
            index=t_opts.index(pval("tech_adoption_level", TechAdoptionLevel.MEDIUM))
        )
        
        m1, m2 = st.columns(2)
        with m1:
            marketing = st.number_input(
                "Monthly Marketing Budget ($)",
                min_value=0, max_value=40000,
                value=int(pval("marketing_monthly_budget", 2200)),
                step=250
            )
        with m2:
            rating_count = st.number_input(
                "Total Online Reviews Count",
                min_value=5, max_value=8000,
                value=int(pval("rating_count", 450)),
                step=50
            )

    years_biz = float(pval("years_in_business", 3.5))

    current_features = RestaurantFeatures(
        location_type=loc_type,
        population_density_3km=float(pop_density),
        median_household_income=float(income),
        foot_traffic_index=float(foot_traffic),
        transit_proximity_score=float(transit_score),
        commercial_density_score=float(comm_density),
        competitor_count_2km=int(comp_count),
        parking_spaces_available=int(parking),
        restaurant_type=rest_type,
        seating_capacity=int(seating),
        floor_area_sqft=int(sqft),
        operating_hours_per_week=int(hours_week),
        avg_ticket_size=float(ticket_size),
        menu_item_count=int(pval("menu_item_count", 45)),
        online_order_enabled=int(online_order),
        delivery_share_pct=float(delivery_share),
        staff_headcount=int(staff),
        tech_adoption_level=tech_level,
        marketing_monthly_budget=float(marketing),
        customer_rating=float(cust_rating),
        rating_count=int(rating_count),
        years_in_business=years_biz,
        latitude=float(pval("latitude", 40.7128)),
        longitude=float(pval("longitude", -74.0060))
    )

    st.markdown("---")
    res = predictor.predict_single(current_features)

    # REVENUE FORECAST DISPLAY
    st.markdown("## 📈 Revenue Forecast & Key Performance Indicators")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="metric-container">
            <span style="color: #64748b; font-size: 0.9rem; font-weight: 600;">PROJECTED MONTHLY REVENUE</span>
            <h2 style="color: #1e3a8a; margin: 8px 0; font-size: 2rem;">${res.predicted_monthly_revenue:,.0f}</h2>
            <span style="color: #059669; font-size: 0.85rem;">Est. ${res.daily_revenue_estimate:,.0f} / day</span>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
        <div class="metric-container">
            <span style="color: #64748b; font-size: 0.9rem; font-weight: 600;">PROJECTED ANNUAL REVENUE</span>
            <h2 style="color: #047857; margin: 8px 0; font-size: 2rem;">${res.predicted_annual_revenue:,.0f}</h2>
            <span style="color: #64748b; font-size: 0.85rem;">12-month gross run-rate</span>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="metric-container">
            <span style="color: #64748b; font-size: 0.9rem; font-weight: 600;">MONTHLY REVENUE / SEAT</span>
            <h2 style="color: #d97706; margin: 8px 0; font-size: 2rem;">${res.revenue_per_seat_month:,.0f}</h2>
            <span style="color: #64748b; font-size: 0.85rem;">Capacity productivity</span>
        </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
        <div class="metric-container">
            <span style="color: #64748b; font-size: 0.9rem; font-weight: 600;">ANNUAL SALES / SQFT</span>
            <h2 style="color: #4f46e5; margin: 8px 0; font-size: 2rem;">${res.revenue_per_sqft_year:,.0f}</h2>
            <span style="color: #64748b; font-size: 0.85rem;">Floor space yield</span>
        </div>
        """, unsafe_allow_html=True)

    # Confidence Interval Visualizer
    st.markdown("#### 🎯 90% Confidence Interval Range")
    ci_fig = go.Figure()
    ci_fig.add_trace(go.Bar(
        y=["Monthly Revenue ($)"],
        x=[res.confidence_interval_90_lower],
        orientation='h',
        marker=dict(color='rgba(0,0,0,0)'),
        showlegend=False,
        hoverinfo='none'
    ))
    ci_fig.add_trace(go.Bar(
        y=["Monthly Revenue ($)"],
        x=[res.confidence_interval_90_upper - res.confidence_interval_90_lower],
        base=res.confidence_interval_90_lower,
        orientation='h',
        name="90% Prediction Band",
        marker=dict(color='#93c5fd', line=dict(color='#2563eb', width=2)),
        hovertemplate=f"Lower 90%: ${res.confidence_interval_90_lower:,.0f}<br>Upper 90%: ${res.confidence_interval_90_upper:,.0f}<extra></extra>"
    ))
    ci_fig.add_trace(go.Scatter(
        y=["Monthly Revenue ($)"],
        x=[res.predicted_monthly_revenue],
        mode='markers',
        marker=dict(color='#1e3a8a', size=16, symbol='diamond'),
        name=f"Point Estimate (${res.predicted_monthly_revenue:,.0f})",
        hovertemplate=f"Expected: ${res.predicted_monthly_revenue:,.0f}<extra></extra>"
    ))
    ci_fig.update_layout(
        height=140,
        margin=dict(l=10, r=10, t=10, b=25),
        barmode='stack',
        xaxis=dict(tickprefix="$", tickformat=","),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(ci_fig, use_container_width=True)

    # -------------------------------------------------------------
    # WHAT-IF SENSITIVITY SIMULATOR
    # -------------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🧪 What-If Scenario Simulator & Growth Levers")
    st.caption("Adjust operational parameters below to instantly calculate the projected revenue impact against your baseline.")

    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    with s_col1:
        delta_seats = st.slider("Expand/Reduce Seating", -30, 60, 0, step=5)
    with s_col2:
        delta_ticket = st.slider("Ticket Price Shift ($)", -10.0, 25.0, 0.0, step=1.0)
    with s_col3:
        delta_delivery = st.slider("Delivery Share Shift (%)", -20.0, 35.0, 0.0, step=5.0)
    with s_col4:
        delta_rating = st.slider("Customer Rating Shift (★)", -0.8, 0.8, 0.0, step=0.1)

    sim_modifications = {
        "seating_capacity": max(10, seating + delta_seats),
        "avg_ticket_size": max(6.0, ticket_size + delta_ticket),
        "delivery_share_pct": float(np.clip(delivery_share + delta_delivery, 0.0, 85.0)),
        "customer_rating": float(np.clip(cust_rating + delta_rating, 1.5, 5.0))
    }

    if delta_seats != 0 or delta_ticket != 0 or delta_delivery != 0 or delta_rating != 0:
        sim_res = simulator.simulate_what_if(current_features, sim_modifications)
        lift_color = "#16a34a" if sim_res["monthly_delta"] >= 0 else "#dc2626"
        lift_symbol = "▲" if sim_res["monthly_delta"] >= 0 else "▼"

        st.markdown(f"""
        <div style="background: #f1f5f9; border-radius: 12px; padding: 18px; margin: 16px 0; border: 1px solid #cbd5e1;">
            <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap;">
                <div style="text-align: center;">
                    <div style="font-size: 0.85rem; color: #64748b; font-weight: 600;">BASELINE REVENUE</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #334155;">${sim_res['baseline_monthly']:,.0f}/mo</div>
                </div>
                <div style="font-size: 1.8rem; color: #94a3b8;">➔</div>
                <div style="text-align: center;">
                    <div style="font-size: 0.85rem; color: #64748b; font-weight: 600;">SIMULATED REVENUE</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: #1e3a8a;">${sim_res['scenario_monthly']:,.0f}/mo</div>
                </div>
                <div style="font-size: 1.8rem; color: #94a3b8;">➔</div>
                <div style="text-align: center;">
                    <div style="font-size: 0.85rem; color: #64748b; font-weight: 600;">PROJECTED NET IMPACT</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: {lift_color};">
                        {lift_symbol} ${abs(sim_res['monthly_delta']):,.0f}/mo ({sim_res['percentage_lift']:+.1f}%)
                    </div>
                    <div style="font-size: 0.8rem; color: #64748b;">(${sim_res['annual_delta']:+,.0f} / year)</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("💡 Adjust any of the 4 sliders above to see instant revenue lift simulations.")

    # Strategic Actionable Recommendations
    st.markdown("### 💡 Tailored Strategic Recommendations")
    for rec in res.actionable_recommendations:
        st.markdown(f'<div class="rec-box">📌 {rec}</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 2: GEOSPATIAL & LOCATION EXPLORER
# -------------------------------------------------------------
with tab2:
    st.markdown("### 🗺️ Geospatial Demographics & Foot Traffic Landscape")
    st.caption("Visualizing restaurant placement within metro catchment zones, foot traffic index, and competitor density.")

    df_sample = load_sample_dataset()
    if not df_sample.empty:
        # Filter by selected restaurant type or location type
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            loc_filter = st.multiselect(
                "Filter by Location Type",
                options=df_sample["location_type"].unique(),
                default=df_sample["location_type"].unique()[:3]
            )
        with f_col2:
            concept_filter = st.multiselect(
                "Filter by Concept Type",
                options=df_sample["restaurant_type"].unique(),
                default=df_sample["restaurant_type"].unique()
            )

        filtered_geo = df_sample[
            df_sample["location_type"].isin(loc_filter) &
            df_sample["restaurant_type"].isin(concept_filter)
        ]

        # Interactive Map using Plotly Scatter Mapbox
        map_fig = px.scatter_mapbox(
            filtered_geo.head(500),
            lat="latitude",
            lon="longitude",
            color="monthly_revenue",
            size="seating_capacity",
            hover_name="restaurant_id",
            hover_data={
                "restaurant_type": True,
                "location_type": True,
                "monthly_revenue": ":$,.0f",
                "foot_traffic_index": True,
                "avg_ticket_size": ":$,.2f",
                "latitude": False,
                "longitude": False
            },
            color_continuous_scale="Viridis",
            size_max=16,
            zoom=3.8,
            title="Restaurant Geographic Footprint & Revenue Heatmap"
        )
        map_fig.update_layout(
            mapbox_style="carto-positron",
            margin=dict(l=0, r=0, t=35, b=0),
            height=520
        )
        st.plotly_chart(map_fig, use_container_width=True)

        # Foot traffic vs Monthly Revenue scatter
        st.markdown("#### 🚶 Pedestrian Foot Traffic vs Revenue Relationship")
        scatter_fig = px.scatter(
            filtered_geo.head(800),
            x="foot_traffic_index",
            y="monthly_revenue",
            color="location_type",
            size="seating_capacity",
            trendline="ols",
            labels={
                "foot_traffic_index": "Foot Traffic Index (0-100)",
                "monthly_revenue": "Monthly Revenue ($)",
                "location_type": "Urban Setting"
            },
            title="Correlation: Pedestrian Footfall vs Gross Revenue by Location Type"
        )
        scatter_fig.update_layout(height=450)
        st.plotly_chart(scatter_fig, use_container_width=True)
    else:
        st.warning("Benchmark dataset not found in data/restaurants_data.csv.")

# -------------------------------------------------------------
# TAB 3: BATCH ANALYSIS & BENCHMARKS
# -------------------------------------------------------------
with tab3:
    st.markdown("### 📊 Batch Prediction & Portfolio Analyzer")
    st.caption("Upload a CSV file of restaurant candidates or run batch predictions across the benchmark portfolio.")

    uploaded_file = st.file_uploader("Upload Restaurant CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(batch_df)} records from uploaded file.")
    else:
        st.info("ℹ️ No file uploaded. Displaying sample benchmark portfolio (First 200 restaurants).")
        sample_all = load_sample_dataset()
        batch_df = sample_all.head(200) if not sample_all.empty else pd.DataFrame()

    if not batch_df.empty:
        if st.button("🚀 Run Batch Revenue Prediction", type="primary"):
            with st.spinner("Processing batch predictions with feature pipeline..."):
                enriched_batch = predictor.predict_dataframe(batch_df)

            st.markdown("#### Batch Summary Statistics")
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Total Restaurants", f"{len(enriched_batch):,}")
            b2.metric("Mean Predicted Monthly", f"${enriched_batch['predicted_monthly_revenue'].mean():,.0f}")
            b3.metric("Total Monthly Portfolio Rev", f"${enriched_batch['predicted_monthly_revenue'].sum():,.0f}")
            b4.metric("Avg Rev / Seat", f"${enriched_batch['rev_per_seat_month'].mean():,.0f}")

            # Revenue Distribution Histogram
            hist_fig = px.histogram(
                enriched_batch,
                x="predicted_monthly_revenue",
                color="restaurant_type",
                nbins=30,
                labels={"predicted_monthly_revenue": "Predicted Monthly Revenue ($)"},
                title="Portfolio Revenue Distribution by Restaurant Type"
            )
            hist_fig.update_layout(height=380)
            st.plotly_chart(hist_fig, use_container_width=True)

            # Table preview & Download
            st.markdown("#### Detailed Prediction Records")
            display_cols = [
                "location_type", "restaurant_type", "seating_capacity",
                "avg_ticket_size", "foot_traffic_index",
                "predicted_monthly_revenue", "predicted_annual_revenue",
                "conf_lower_90", "conf_upper_90", "rev_per_seat_month"
            ]
            valid_cols = [c for c in display_cols if c in enriched_batch.columns]
            st.dataframe(
                enriched_batch[valid_cols].head(50).style.format({
                    "predicted_monthly_revenue": "${:,.0f}",
                    "predicted_annual_revenue": "${:,.0f}",
                    "conf_lower_90": "${:,.0f}",
                    "conf_upper_90": "${:,.0f}",
                    "rev_per_seat_month": "${:,.0f}",
                    "avg_ticket_size": "${:.2f}"
                }),
                use_container_width=True
            )

            # Download CSV
            csv_data = enriched_batch.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Predictions CSV",
                data=csv_data,
                file_name="restaurant_revenue_predictions.csv",
                mime="text/csv"
            )

# -------------------------------------------------------------
# TAB 4: MODEL DIAGNOSTICS & EXPLAINABILITY
# -------------------------------------------------------------
with tab4:
    st.markdown("### 🧠 Model Architecture, Benchmarking & Global Feature Importance")
    st.caption("Comprehensive audit of machine learning models, validation scores, and revenue drivers.")

    # Model Benchmark Scorecard
    benchmarks = predictor.bundle.get("benchmark_results", [])
    if benchmarks:
        st.markdown("#### 🏆 Multi-Model Benchmark Scorecard")
        bench_df = pd.DataFrame(benchmarks)
        st.dataframe(
            bench_df.rename(columns={
                "model_name": "Model Architecture",
                "r2_score": "R² Score",
                "mae": "Mean Absolute Error ($)",
                "rmse": "Root Mean Squared Error ($)",
                "mape_pct": "MAPE (%)",
                "residual_std": "Residual Std ($)"
            }).style.format({
                "R² Score": "{:.4f}",
                "Mean Absolute Error ($)": "${:,.2f}",
                "Root Mean Squared Error ($)": "${:,.2f}",
                "MAPE (%)": "{:.2f}%",
                "Residual Std ($)": "${:,.2f}"
            }),
            use_container_width=True
        )

    # Global Feature Importance Chart
    importances = predictor.feature_importances
    if importances:
        st.markdown("#### 🌟 Top 15 Most Influential Revenue Drivers")
        imp_df = pd.DataFrame(importances).head(15)
        imp_df["importance_pct"] = imp_df["importance"] * 100.0

        bar_fig = px.bar(
            imp_df.sort_values(by="importance_pct", ascending=True),
            x="importance_pct",
            y="feature",
            orientation="h",
            color="importance_pct",
            color_continuous_scale="Blues",
            labels={"importance_pct": "Relative Importance (%)", "feature": "Model Feature"},
            title="Global Feature Importance (Derived from Ensemble Trees)"
        )
        bar_fig.update_layout(height=480, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(bar_fig, use_container_width=True)

    st.markdown("""
    #### 📐 Methodological & Econometric Notes
    - **Ensemble Architecture**: HistGradientBoosting Regressor and Random Forest capture non-linear thresholds (e.g. minimum seating capacity required for viable table turns) and interaction terms between physical density and delivery demand.
    - **Customer Location Signals**: Captures pedestrian footfall, population catchment density, local purchasing power ($), transit proximity, and competitive agglomeration vs cannibalization.
    - **Operational Signals**: Evaluates seating capacity, dining format, average ticket size, delivery aggregators, staffing efficiency, and online reputation.
    """)

