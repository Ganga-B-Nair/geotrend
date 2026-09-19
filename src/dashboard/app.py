"""Streamlit dashboard -- doubles as the backup deliverable.

Run: streamlit run src/dashboard/app.py

Five tabs, mapped 1:1 to the report chapters so a reviewer can follow the
argument by clicking rather than reading:
  1. Data Quality   -- missingness, defect detection, DGP summary
  2. Explore        -- distributions, price-per-sqft by locality, correlations
  3. Micro-neighbourhoods -- Moran's I result, k-distance plot, Folium clusters
  4. Trends         -- per-cluster decomposition, ADF verdict, forecast
  5. Infrastructure -- distance-decay curve, event-study plot, DiD table
  6. Estimate       -- pick a point on the map -> price + SHAP waterfall
"""
import streamlit as st

st.set_page_config(page_title="GeoTrend", layout="wide")
st.title("GeoTrend — micro-neighbourhood price intelligence")

TABS = [
    "Data Quality",
    "Explore",
    "Micro-neighbourhoods",
    "Trends",
    "Infrastructure Impact",
    "Estimate a Property",
]
tabs = st.tabs(TABS)

with tabs[0]:
    st.info("Missingness matrix, defect counts, synthetic DGP disclosure.")
with tabs[1]:
    st.info("Univariate + bivariate EDA, price-per-sqft normalisation.")
with tabs[2]:
    st.info("Moran's I (I, p), k-distance elbow, Folium cluster map.")
with tabs[3]:
    st.info("STL decomposition, ADF test, rolling-origin backtest, forecast.")
with tabs[4]:
    st.info("Distance-decay OLS, parallel-trends check, event study, DiD table.")
with tabs[5]:
    st.info("Map picker -> /predict -> price, interval, SHAP waterfall.")
