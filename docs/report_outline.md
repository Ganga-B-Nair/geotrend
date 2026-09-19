# Report outline (maps 1:1 to notebooks)

1. Introduction — problem, why spatial + temporal framing, contributions
2. Background — hedonic pricing, spatial autocorrelation, DiD in urban economics
3. Data — real source, hybrid design rationale, what is real vs synthetic, limitations
4. Data quality & preprocessing — profile, decisions, before/after scorecard
5. Exploratory analysis — distributions, suburb comparison, temporal structure
6. Spatial analysis — Moran's I, LISA, DBSCAN, Weka cross-check
7. Temporal analysis — decomposition, stationarity, forecasting, backtest
8. Infrastructure impact
   8a. Distance decay on real prices (correlational, real finding)
   8b. Simulation study: overlay, parallel trends, event study, DiD
   8c. Inference: the few-clusters problem and how it was handled
   8d. Recovery against ground truth + robustness
9. Predictive modelling — baseline, XGBoost, metrics, residual diagnostics
10. Explainability — SHAP global/local, case studies
11. System — architecture, API contract, dashboard, mobile client
12. Discussion — DS contribution vs engineering wrapper
13. Ethics & limitations — geocoding bias, price != value, overclaim risk
14. Conclusion & future work
