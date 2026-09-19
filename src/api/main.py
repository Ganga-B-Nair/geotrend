"""FastAPI service -- the delivery proof for the mobile client.

Run:  uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
Docs: http://localhost:8000/docs
"""
from __future__ import annotations

from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="GeoTrend API",
    description="Micro-neighbourhood price estimation with explanations.",
    version="0.1.0",
)


class PropertyQuery(BaseModel):
    latitude: float = Field(..., ge=8.0, le=9.0)
    longitude: float = Field(..., ge=76.0, le=78.0)
    area_sqft: float = Field(..., gt=200, lt=10000)
    bedrooms: int = Field(..., ge=1, le=8)
    bathrooms: int = Field(2, ge=1, le=6)
    property_age_years: float = Field(5, ge=0, le=60)
    property_type: Literal["apartment", "villa", "independent_house"] = "apartment"
    furnishing: Literal["unfurnished", "semi", "full"] = "unfurnished"
    parking: int = 1


class ShapItem(BaseModel):
    feature: str
    value: float
    contribution_inr: float


class PredictionResponse(BaseModel):
    estimated_price_inr: float
    interval_low_inr: float
    interval_high_inr: float
    cluster_id: int
    cluster_label: str
    top_drivers: list[ShapItem]


class TrendPoint(BaseModel):
    month: str
    median_price_per_sqft: float
    is_forecast: bool


@app.on_event("startup")
def load_artifacts() -> None:
    """Load model, DBSCAN cluster assigner, SHAP explainer, trend tables."""
    ...


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(q: PropertyQuery) -> PredictionResponse:
    """Estimate price + return the SHAP drivers behind this specific number."""
    raise HTTPException(501, "not implemented")


@app.get("/cluster/{cluster_id}/trend", response_model=list[TrendPoint])
def cluster_trend(cluster_id: int, horizon: int = 6) -> list[TrendPoint]:
    """Historical monthly series + ARIMA/Prophet forecast for the cluster."""
    raise HTTPException(501, "not implemented")


@app.get("/clusters")
def clusters() -> dict:
    """GeoJSON of micro-neighbourhood hulls for the mobile map layer."""
    raise HTTPException(501, "not implemented")


@app.get("/infrastructure-impact")
def infrastructure_impact() -> dict:
    """DiD estimates per event: effect size, CI, p-value, control set used."""
    raise HTTPException(501, "not implemented")
