"""Synthetic dataset generator for Restaurant Revenue Prediction System.
Produces realistic restaurant operational and location data grounded in dining economics.
"""

import os
import numpy as np
import pandas as pd


def generate_restaurant_dataset(num_samples: int = 5000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)

    # City hubs with base coordinates
    city_hubs = [
        {"name": "New York Metro", "lat": 40.7128, "lon": -74.0060, "weight": 0.22, "income_base": 88000, "density_base": 14000},
        {"name": "Chicago Metro", "lat": 41.8781, "lon": -87.6298, "weight": 0.16, "income_base": 76000, "density_base": 8500},
        {"name": "Los Angeles Metro", "lat": 34.0522, "lon": -118.2437, "weight": 0.18, "income_base": 82000, "density_base": 7800},
        {"name": "Austin Metro", "lat": 30.2672, "lon": -97.7431, "weight": 0.14, "income_base": 85000, "density_base": 5200},
        {"name": "Seattle Metro", "lat": 47.6062, "lon": -122.3321, "weight": 0.12, "income_base": 96000, "density_base": 6500},
        {"name": "Atlanta Metro", "lat": 33.7490, "lon": -84.3880, "weight": 0.10, "income_base": 72000, "density_base": 4800},
        {"name": "Miami Metro", "lat": 25.7617, "lon": -80.1918, "weight": 0.08, "income_base": 70000, "density_base": 6200},
    ]

    city_choices = np.random.choice(
        len(city_hubs),
        size=num_samples,
        p=[c["weight"] for c in city_hubs]
    )

    records = []
    location_types = [
        "Downtown / Urban Core",
        "Business / Commercial District",
        "Shopping Mall / Food Court",
        "Suburban Strip Mall",
        "Residential Neighborhood",
        "Highway / Travel Corridor"
    ]
    loc_weights = [0.28, 0.22, 0.16, 0.18, 0.11, 0.05]

    restaurant_types = [
        "Fast Casual",
        "Casual Dining",
        "Fine Dining",
        "Quick Service (QSR)",
        "Cafe & Bakery",
        "Bar & Grill"
    ]
    rest_weights = [0.25, 0.28, 0.09, 0.20, 0.10, 0.08]

    tech_levels = [
        "Low (Manual POS / Paper)",
        "Medium (Standard POS & Cards)",
        "High (Kiosks, QR & Integrated Kitchen)"
    ]
    tech_weights = [0.20, 0.55, 0.25]

    for i in range(num_samples):
        hub = city_hubs[city_choices[i]]
        
        # Spatial jitter around metro center (approx 0 to 25 km)
        lat_jitter = np.random.normal(0, 0.08)
        lon_jitter = np.random.normal(0, 0.08)
        lat = round(hub["lat"] + lat_jitter, 5)
        lon = round(hub["lon"] + lon_jitter, 5)

        loc_type = np.random.choice(location_types, p=loc_weights)
        rest_type = np.random.choice(restaurant_types, p=rest_weights)
        tech_level = np.random.choice(tech_levels, p=tech_weights)

        # Context-dependent location demographics
        if loc_type == "Downtown / Urban Core":
            pop_density = np.random.normal(hub["density_base"] * 1.6, 2500)
            foot_traffic = np.random.normal(82, 10)
            transit_score = np.random.normal(8.8, 1.0)
            comm_density = np.random.normal(8.5, 1.0)
            parking_spaces = np.random.randint(0, 35)
            comp_count = np.random.randint(15, 60)
        elif loc_type == "Business / Commercial District":
            pop_density = np.random.normal(hub["density_base"] * 1.1, 2000)
            foot_traffic = np.random.normal(74, 12)
            transit_score = np.random.normal(7.5, 1.2)
            comm_density = np.random.normal(8.8, 0.9)
            parking_spaces = np.random.randint(20, 90)
            comp_count = np.random.randint(10, 45)
        elif loc_type == "Shopping Mall / Food Court":
            pop_density = np.random.normal(hub["density_base"] * 0.9, 1800)
            foot_traffic = np.random.normal(88, 8)
            transit_score = np.random.normal(6.5, 1.5)
            comm_density = np.random.normal(7.2, 1.4)
            parking_spaces = np.random.randint(100, 250)
            comp_count = np.random.randint(8, 30)
        elif loc_type == "Suburban Strip Mall":
            pop_density = np.random.normal(hub["density_base"] * 0.65, 1200)
            foot_traffic = np.random.normal(48, 14)
            transit_score = np.random.normal(4.2, 1.5)
            comm_density = np.random.normal(5.0, 1.5)
            parking_spaces = np.random.randint(40, 160)
            comp_count = np.random.randint(4, 22)
        elif loc_type == "Residential Neighborhood":
            pop_density = np.random.normal(hub["density_base"] * 0.85, 1400)
            foot_traffic = np.random.normal(42, 12)
            transit_score = np.random.normal(5.8, 1.8)
            comm_density = np.random.normal(3.8, 1.3)
            parking_spaces = np.random.randint(5, 45)
            comp_count = np.random.randint(2, 18)
        else: # Highway / Travel Corridor
            pop_density = np.random.normal(hub["density_base"] * 0.35, 800)
            foot_traffic = np.random.normal(52, 15)
            transit_score = np.random.normal(2.5, 1.0)
            comm_density = np.random.normal(4.0, 1.5)
            parking_spaces = np.random.randint(60, 220)
            comp_count = np.random.randint(2, 14)

        pop_density = float(np.clip(pop_density, 300, 38000))
        foot_traffic = float(np.clip(foot_traffic, 5, 100))
        transit_score = float(np.clip(transit_score, 1.0, 10.0))
        comm_density = float(np.clip(comm_density, 1.0, 10.0))
        parking_spaces = int(np.clip(parking_spaces, 0, 250))
        comp_count = int(np.clip(comp_count, 0, 75))

        # Household income
        income = np.random.normal(hub["income_base"], 18000)
        income = float(np.clip(income, 28000, 230000))

        # Operational characteristics linked to restaurant concept
        if rest_type == "Fine Dining":
            seating = np.random.randint(40, 140)
            sqft = int(seating * np.random.uniform(32, 45))
            ticket_size = np.random.normal(110.0, 30.0)
            hours_week = np.random.randint(42, 65)
            menu_items = np.random.randint(24, 55)
            online_order = 1 if np.random.rand() > 0.65 else 0
            delivery_pct = np.random.uniform(0, 12) if online_order else 0.0
            staff = int(np.clip(seating * 0.32 + np.random.normal(0, 3), 10, 65))
            base_turnover = np.random.uniform(1.1, 1.7)
        elif rest_type == "Casual Dining":
            seating = np.random.randint(70, 220)
            sqft = int(seating * np.random.uniform(24, 34))
            ticket_size = np.random.normal(36.0, 8.0)
            hours_week = np.random.randint(56, 85)
            menu_items = np.random.randint(45, 95)
            online_order = 1 if np.random.rand() > 0.15 else 0
            delivery_pct = np.random.uniform(10, 35) if online_order else 0.0
            staff = int(np.clip(seating * 0.18 + np.random.normal(0, 3), 8, 48))
            base_turnover = np.random.uniform(2.1, 3.4)
        elif rest_type == "Fast Casual":
            seating = np.random.randint(35, 110)
            sqft = int(seating * np.random.uniform(18, 26))
            ticket_size = np.random.normal(18.5, 3.5)
            hours_week = np.random.randint(65, 95)
            menu_items = np.random.randint(20, 50)
            online_order = 1 if np.random.rand() > 0.05 else 0
            delivery_pct = np.random.uniform(25, 55) if online_order else 0.0
            staff = int(np.clip(seating * 0.14 + np.random.normal(0, 2), 6, 28))
            base_turnover = np.random.uniform(3.8, 5.6)
        elif rest_type == "Quick Service (QSR)":
            seating = np.random.randint(25, 95)
            sqft = int(seating * np.random.uniform(16, 24))
            ticket_size = np.random.normal(12.5, 2.8)
            hours_week = np.random.randint(75, 115)
            menu_items = np.random.randint(25, 65)
            online_order = 1 if np.random.rand() > 0.08 else 0
            delivery_pct = np.random.uniform(30, 65) if online_order else 0.0
            staff = int(np.clip(seating * 0.15 + np.random.normal(0, 2), 7, 30))
            base_turnover = np.random.uniform(5.2, 8.0)
        elif rest_type == "Cafe & Bakery":
            seating = np.random.randint(20, 75)
            sqft = int(seating * np.random.uniform(18, 28))
            ticket_size = np.random.normal(11.0, 2.5)
            hours_week = np.random.randint(50, 78)
            menu_items = np.random.randint(30, 60)
            online_order = 1 if np.random.rand() > 0.20 else 0
            delivery_pct = np.random.uniform(12, 38) if online_order else 0.0
            staff = int(np.clip(seating * 0.12 + np.random.normal(0, 2), 4, 18))
            base_turnover = np.random.uniform(4.0, 6.2)
        else: # Bar & Grill
            seating = np.random.randint(65, 190)
            sqft = int(seating * np.random.uniform(26, 36))
            ticket_size = np.random.normal(42.0, 10.0)
            hours_week = np.random.randint(60, 95)
            menu_items = np.random.randint(35, 75)
            online_order = 1 if np.random.rand() > 0.30 else 0
            delivery_pct = np.random.uniform(8, 25) if online_order else 0.0
            staff = int(np.clip(seating * 0.20 + np.random.normal(0, 3), 9, 45))
            base_turnover = np.random.uniform(1.8, 3.0)

        ticket_size = float(np.clip(ticket_size, 7.0, 240.0))
        delivery_pct = float(round(np.clip(delivery_pct, 0.0, 75.0), 1))
        
        # Marketing, Rating, Age
        years_in_biz = float(round(np.random.exponential(4.5) + 0.5, 1))
        years_in_biz = min(years_in_biz, 40.0)
        
        # Customer rating and review volume
        rating_mean = 4.1 + (0.15 if tech_level == "High (Kiosks, QR & Integrated Kitchen)" else 0)
        cust_rating = float(round(np.clip(np.random.normal(rating_mean, 0.38), 1.8, 5.0), 1))
        
        rating_count = int(np.clip(
            years_in_biz * np.random.uniform(40, 180) + foot_traffic * 3.5, 
            15, 6500
        ))

        marketing_budget = float(round(np.clip(
            np.random.gamma(2.5, 800) + (1000 if "Downtown" in loc_type else 300),
            150, 28000
        ), 0))

        # -------------------------------------------------------------
        # GROUND TRUTH REVENUE ECONOMIC MODEL
        # -------------------------------------------------------------
        # 1. Base operational capacity
        daily_capacity = seating * base_turnover
        
        # 2. Foot traffic & Location demand factor
        foot_traffic_factor = 0.55 + (foot_traffic / 100.0) * 0.65
        pop_density_factor = 0.85 + (pop_density / 20000.0) * 0.30
        income_factor = 0.80 + (income / 100000.0) * 0.25
        
        # 3. Competitive pressure / dining synergy
        # Moderate competitors enhance cluster appeal; excessive saturated competitors dilute revenue
        if comp_count < 15:
            comp_factor = 1.0 - (comp_count * 0.004)
        elif comp_count < 35:
            comp_factor = 0.94 + ((comp_count - 15) * 0.003) # Dining cluster synergy
        else:
            comp_factor = 1.0 - ((comp_count - 35) * 0.006) # Cannibalization

        # 4. Reputation & Rating impact
        rating_factor = ((cust_rating / 4.0) ** 1.6) * (1.0 + np.log10(rating_count) * 0.035)

        # 5. Technology throughput boost
        tech_mult = {
            "Low (Manual POS / Paper)": 0.92,
            "Medium (Standard POS & Cards)": 1.00,
            "High (Kiosks, QR & Integrated Kitchen)": 1.12
        }[tech_level]

        # 6. Marketing lift
        marketing_lift = 1.0 + np.log1p(marketing_budget / 800.0) * 0.05

        # 7. Operational hours factor (standard week is ~65 hours)
        hours_factor = (hours_week / 65.0) ** 0.65

        # Effective daily in-house covers
        effective_in_house_covers = (
            daily_capacity * 
            foot_traffic_factor * 
            pop_density_factor * 
            comp_factor * 
            rating_factor * 
            tech_mult * 
            marketing_lift *
            hours_factor
        )

        # Base dine-in daily revenue
        dine_in_daily_rev = effective_in_house_covers * ticket_size * income_factor

        # 8. Delivery revenue addition
        if online_order and delivery_pct > 0:
            delivery_rev_ratio = (delivery_pct / (100.0 - delivery_pct * 0.65))
            delivery_ticket_ratio = 1.08 # Delivery tickets often 8% higher due to bundled items
            delivery_daily_rev = dine_in_daily_rev * delivery_rev_ratio * delivery_ticket_ratio
        else:
            delivery_daily_rev = 0.0

        total_daily_rev = (dine_in_daily_rev + delivery_daily_rev)

        # Monthly revenue (average 30.4 days) with realistic business variance
        monthly_base = total_daily_rev * 30.4

        # Add Gaussian residual noise (epsilon ~ N(0, 0.05 * monthly_base))
        noise_pct = np.random.normal(0, 0.05)
        monthly_revenue = monthly_base * (1.0 + noise_pct)

        # Bounds check
        monthly_revenue = float(round(max(monthly_revenue, 12000.0), 2))
        annual_revenue = float(round(monthly_revenue * 12.0, 2))

        records.append({
            "restaurant_id": f"REST-{i+1:05d}",
            "location_type": loc_type,
            "population_density_3km": round(pop_density, 1),
            "median_household_income": round(income, 0),
            "foot_traffic_index": round(foot_traffic, 1),
            "transit_proximity_score": round(transit_score, 1),
            "commercial_density_score": round(comm_density, 1),
            "competitor_count_2km": comp_count,
            "parking_spaces_available": parking_spaces,
            "latitude": lat,
            "longitude": lon,
            "restaurant_type": rest_type,
            "seating_capacity": seating,
            "floor_area_sqft": sqft,
            "operating_hours_per_week": hours_week,
            "avg_ticket_size": round(ticket_size, 2),
            "menu_item_count": menu_items,
            "online_order_enabled": online_order,
            "delivery_share_pct": delivery_pct,
            "staff_headcount": staff,
            "tech_adoption_level": tech_level,
            "marketing_monthly_budget": round(marketing_budget, 0),
            "customer_rating": cust_rating,
            "rating_count": rating_count,
            "years_in_business": years_in_biz,
            "monthly_revenue": monthly_revenue,
            "annual_revenue": annual_revenue,
        })

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    output_path = os.path.join(os.path.dirname(__file__), "restaurants_data.csv")
    print(f"Generating 5,000 restaurant benchmark samples...")
    df = generate_restaurant_dataset(num_samples=5000, random_state=42)
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {len(df)} records saved to {output_path}")
    print(df[["location_type", "restaurant_type", "seating_capacity", "foot_traffic_index", "monthly_revenue"]].head())

