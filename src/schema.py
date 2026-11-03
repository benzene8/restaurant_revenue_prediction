"""Data schemas and feature definitions for Restaurant Revenue Prediction System."""

from typing import Optional, List, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class LocationType(str, Enum):
    DOWNTOWN = "Downtown / Urban Core"
    COMMERCIAL_DISTRICT = "Business / Commercial District"
    SHOPPING_MALL = "Shopping Mall / Food Court"
    SUBURBAN_STRIP = "Suburban Strip Mall"
    RESIDENTIAL = "Residential Neighborhood"
    HIGHWAY_CORRIDOR = "Highway / Travel Corridor"


class RestaurantType(str, Enum):
    FAST_CASUAL = "Fast Casual"
    CASUAL_DINING = "Casual Dining"
    FINE_DINING = "Fine Dining"
    QSR = "Quick Service (QSR)"
    CAFE_BAKERY = "Cafe & Bakery"
    BAR_GRILL = "Bar & Grill"


class TechAdoptionLevel(str, Enum):
    LOW = "Low (Manual POS / Paper)"
    MEDIUM = "Medium (Standard POS & Cards)"
    HIGH = "High (Kiosks, QR & Integrated Kitchen)"


class RestaurantFeatures(BaseModel):
    """Input features capturing Location and Operational characteristics."""
    # Location & Customer Demographics
    location_type: LocationType = Field(
        default=LocationType.DOWNTOWN,
        description="General classification of the restaurant's geographic setting."
    )
    population_density_3km: float = Field(
        default=5500.0, ge=200.0, le=40000.0,
        description="Estimated population within 3km radius (people/km^2)."
    )
    median_household_income: float = Field(
        default=75000.0, ge=25000.0, le=250000.0,
        description="Median annual household income in the local catchment ($)."
    )
    foot_traffic_index: float = Field(
        default=65.0, ge=1.0, le=100.0,
        description="Pedestrian volume and foot traffic score (1-100 scale)."
    )
    transit_proximity_score: float = Field(
        default=7.0, ge=1.0, le=10.0,
        description="Proximity score to subway, train, and bus transit hubs (1-10)."
    )
    commercial_density_score: float = Field(
        default=6.5, ge=1.0, le=10.0,
        description="Density of offices, colleges, and retail establishments nearby (1-10)."
    )
    competitor_count_2km: int = Field(
        default=12, ge=0, le=80,
        description="Number of directly competing restaurants within 2km."
    )
    parking_spaces_available: int = Field(
        default=25, ge=0, le=250,
        description="Dedicated or validated customer parking spaces."
    )
    latitude: Optional[float] = Field(
        default=40.7128, description="Latitude coordinate for spatial mapping."
    )
    longitude: Optional[float] = Field(
        default=-74.0060, description="Longitude coordinate for spatial mapping."
    )

    # Operational Characteristics
    restaurant_type: RestaurantType = Field(
        default=RestaurantType.CASUAL_DINING,
        description="Restaurant concept and dining format."
    )
    seating_capacity: int = Field(
        default=85, ge=10, le=450,
        description="Total indoor and outdoor dining seats."
    )
    floor_area_sqft: int = Field(
        default=2600, ge=300, le=12000,
        description="Total restaurant floor area in square feet."
    )
    operating_hours_per_week: int = Field(
        default=70, ge=30, le=126,
        description="Total hours the restaurant operates per week."
    )
    avg_ticket_size: float = Field(
        default=32.0, ge=6.0, le=250.0,
        description="Average check spend per guest in USD ($)."
    )
    menu_item_count: int = Field(
        default=45, ge=8, le=160,
        description="Total distinct food and beverage items on menu."
    )
    online_order_enabled: int = Field(
        default=1, ge=0, le=1,
        description="Whether online ordering and delivery aggregation is enabled (0 or 1)."
    )
    delivery_share_pct: float = Field(
        default=25.0, ge=0.0, le=85.0,
        description="Percentage of orders originating from delivery and takeout (0-85%)."
    )
    staff_headcount: int = Field(
        default=16, ge=2, le=75,
        description="Total operational staff headcount (kitchen, service, management)."
    )
    tech_adoption_level: TechAdoptionLevel = Field(
        default=TechAdoptionLevel.MEDIUM,
        description="Degree of technology integration in ordering and kitchen workflows."
    )
    marketing_monthly_budget: float = Field(
        default=2200.0, ge=0.0, le=40000.0,
        description="Monthly marketing and digital advertising expenditure ($)."
    )
    customer_rating: float = Field(
        default=4.3, ge=1.5, le=5.0,
        description="Average public rating (e.g. Google Maps, Yelp) on 1.0-5.0 scale."
    )
    rating_count: int = Field(
        default=420, ge=5, le=8000,
        description="Total count of customer reviews online."
    )
    years_in_business: float = Field(
        default=3.5, ge=0.1, le=50.0,
        description="Number of years the establishment has been operating."
    )


class RevenuePredictionOutput(BaseModel):
    """Model output payload for restaurant revenue forecasting."""
    predicted_monthly_revenue: float
    predicted_annual_revenue: float
    confidence_interval_90_lower: float
    confidence_interval_90_upper: float
    daily_revenue_estimate: float
    revenue_per_seat_month: float
    revenue_per_sqft_year: float
    key_drivers: List[Dict[str, Any]]
    actionable_recommendations: List[str]
    input_summary: Dict[str, Any]

