# Terra — GeoTrend

**A Data Science Framework for Micro-Neighbourhood Real Estate Valuation Using Geospatial Clustering, Temporal Trend Analysis, and Infrastructure Impact Quantification**

*Terra* is the project; **GeoTrend** is its working component — the analytical
pipeline and the application built on top of it. This repository is GeoTrend.

> Solo academic project — subject: Data Science.
> Machine learning is used as one component; the graded contribution is the data pipeline, statistical reasoning, and causal analysis.

---

## Problem

Property valuation is usually posed as regression over property attributes. That framing ignores two facts about housing markets:

1. **Prices are spatially dependent** — a property's value is partly the value of its surroundings, and administrative suburb boundaries are a poor proxy for the real economic neighbourhood.
2. **Prices respond to infrastructure over time** — a new mall, park, or transit station changes those surroundings, and a static cross-sectional model cannot see it.

GeoTrend addresses both: it derives micro-neighbourhoods from the data itself, tracks their price trajectories, and quantifies how much of a price movement is attributable to a specific infrastructure event rather than to market-wide drift.

## Central research questions

| # | Question | Method | Data |
|---|---|---|---|
| RQ1 | Is residential price spatially autocorrelated, and at what scale? | Global + Local Moran's I | **Real** |
| RQ2 | What are the data-driven micro-neighbourhoods, and do they differ from suburb boundaries? | DBSCAN over (lat, lon, log price-per-sqft) | **Real** |
| RQ3 | How do micro-neighbourhood price trajectories differ, and are they forecastable? | STL decomposition, ADF, ARIMA/Prophet | **Real** |
| RQ4 | What is the marginal price of proximity to an amenity? | Distance-decay regression | **Real** |
| RQ5 | Can we correctly estimate the causal effect of an infrastructure opening? | Event study + Difference-in-Differences | **Real + synthetic overlay** |
| RQ6 | Which factors drive an individual valuation, and can we explain it in plain language? | XGBoost + SHAP | **Real** |

Five of six research questions run entirely on real data. One synthetic component exists, and it is confined to RQ5.

## Data: a hybrid design

### Base layer — real

**Melbourne Housing Market** ([Kaggle](https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot)) — real coordinates, real sale dates, real prices, plus rooms, land size, building area and distance to CBD. It is one of the few public housing datasets carrying genuine *space and time* together; most give you one or the other, and this project needs both.

```bash
# download Melbourne_housing_FULL.csv into data/raw/, then:
python -m src.data.load_real --raw data/raw/Melbourne_housing_FULL.csv
```

Measured on the downloaded file (19 September 2026): **34,857 rows over 27
months** (2016-01-28 to 2018-03-17). Price is missing for 7,610 rows,
coordinates for 7,976, and building area for 21,115.

The project uses a **two-track sample design**: clustering runs on the 20,993
rows with valid price *and* coordinates, because DBSCAN is density-based and a
better-populated sample yields a better estimate of neighbourhood structure;
everything downstream runs on the 10,647-row complete-case subset, which
inherits cluster labels geographically. That subset is tested for
representativeness against the dropped rows before it is used.

Notably, the price and coordinate gaps overlap in only 1,722 rows — they are
two distinct missingness mechanisms rather than one pattern of incomplete
records, and Notebook 02 treats them separately.

The loader normalises the source to an internal schema so every notebook reads the same column names. It handles the source file's misspelled `Lattitude` / `Longtitude` columns; **do not fix the CSV by hand** — leaving the download untouched is what makes the pipeline reproducible for anyone who re-downloads it.

An offline stand-in with the same schema is available for running tests before you have downloaded anything:

```bash
python -m src.data.load_real --fixture     # writes data/raw/_fixture.csv
```

The fixture is a test artefact, never a result. It writes to a deliberately different filename so the two can never be confused.

### Event layer — synthetic, and only for RQ5

Estimating the causal effect of new infrastructure requires knowing exactly **when and where** infrastructure opened. Public housing datasets do not carry that, and reconstructing it retrospectively is unreliable.

Rather than abandon the causal question or answer it with an unverifiable claim, this project treats it as a **simulation study**: an effect of known magnitude is injected into designated clusters at designated months, and the Difference-in-Differences estimator is judged by how accurately it recovers that known value.

```
price_augmented = price_real × (1 + effect × ramp(t − event_month))
```

Everything else — spatial structure, market noise, seasonality, property attributes — is real. One term is added, and its exact value is recorded in `data/processed/overlay_manifest.json`.

**The result is not a claim about Melbourne. It is a demonstration that the estimator works, with a measurable error.** That distinction is stated in the report, not buried in it.

```bash
python -m src.data.event_overlay --clustered data/processed/clustered.parquet
```

### Why the order matters

```
real data → clean → Moran's I → DBSCAN clusters → ★ OVERLAY APPLIED HERE ★ → DiD
```

The overlay runs **after** clustering, never before. Clusters must come from genuine spatial structure; injecting price changes first would let the synthetic layer contaminate RQ1 and RQ2.

Notebooks 02–05 and 07 read the `price` column. Only Notebook 06 reads `price_augmented`. The columns are separately named so they cannot be silently swapped.

## Pipeline

```
load real → quality profile → clean → EDA
         → Moran's I → DBSCAN micro-neighbourhoods → Weka cross-check
         → per-cluster series → decomposition → ADF → ARIMA/Prophet → backtest
         → distance decay (real)
         → event overlay → parallel-trends test → event study → DiD → recovery vs truth
         → temporal split → OLS baseline → XGBoost → residual Moran's I → SHAP
         → FastAPI service → Streamlit dashboard + mobile client
```

## Repository layout

```
geotrend/
├── CLAUDE.md                   # conventions + hard constraints for Claude Code
├── config.yaml                 # single source of truth for all parameters
├── requirements.txt
├── data/
│   ├── raw/                    # downloaded CSV (+ offline fixture)
│   ├── processed/              # cleaned, clustered, augmented, manifest
│   └── external/               # cached amenity data from OSM
├── notebooks/
│   ├── 01_data_acquisition.ipynb
│   ├── 02_data_quality_and_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_spatial_clustering.ipynb
│   ├── 05_timeseries_trends.ipynb
│   ├── 06_infrastructure_impact.ipynb      ← the only synthetic chapter
│   └── 07_modelling_and_shap.ipynb
├── src/
│   ├── data/       load_real.py · event_overlay.py · clean.py · enrich_osm.py
│   ├── features/   build_features.py
│   ├── analysis/   spatial.py · timeseries.py · causal.py
│   ├── models/     train.py · explain.py
│   ├── api/        main.py           (FastAPI)
│   └── dashboard/  app.py            (Streamlit)
├── reports/figures/
├── tests/
└── docs/
```

Notebooks tell the story; `src/` holds the reusable logic. Nothing important should live only inside a notebook cell.

## Quickstart

```bash
git clone https://github.com/Ganga-B-Nair/geotrend.git && cd geotrend
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

python -m src.data.load_real --fixture               # offline stand-in
pytest -q                                            # 12 tests, no download needed

# then download the real CSV and:
python -m src.data.load_real --raw data/raw/Melbourne_housing_FULL.csv
jupyter lab                                          # notebooks 01 → 07

uvicorn src.api.main:app --reload                    # API at :8000/docs
streamlit run src/dashboard/app.py                   # dashboard at :8501
```

## Deliverables and how effort is split

| Component | Role | Share |
|---|---|---|
| Python DS pipeline (notebooks 01–07 + `src/`) | The data science contribution — the graded core | ~70% |
| FastAPI + mobile client | Proof the pipeline works on unseen input; deployment stage of the DS lifecycle | ~20% |
| Streamlit dashboard | Interactive visualisation layer; also the demo fallback if the mobile build runs long | included above |
| Weka cluster cross-check | Methodological validation appendix — one tool, one section | ~5% |

Power BI is deliberately excluded: it duplicates the Streamlit layer and fragments the narrative.

## Key methodological commitments

- **Real data does the real work.** Five of six research questions never touch the synthetic layer. The one that does says so, every time it reports a number.
- **Temporal split, never random.** A random split lets later months train a model evaluated on earlier ones. Reported metrics would be inflated and meaningless.
- **Leakage discipline in features.** Any price-derived feature (cluster medians, suburb averages) is computed on the training window only.
- **Parameters are justified, not defaulted.** DBSCAN `eps` comes from a k-distance elbow; ARIMA order from ACF/PACF; both are shown. `config.yaml` ships with `eps_km: null` so it cannot be left at a default by accident.
- **Assumptions are tested before they are used.** Moran's I before geospatial clustering; ADF before forecasting; parallel trends before DiD.
- **Cluster-robust standard errors are treated with suspicion at this cluster count.** With only a few treated clusters, cluster-robust SEs are severely downward-biased — during development this pipeline produced an SE of 0.0004 on an estimate 0.019 from the known truth. The project uses wild cluster bootstrap or randomisation inference instead, and reports why.
- **Residuals are re-tested spatially.** If Moran's I on residuals is still significant, the spatial specification is incomplete — and that is reported, not hidden.

## Limitations

- The infrastructure effect sizes recovered in Notebook 06 are recoveries of *designed* effects. They demonstrate that the estimator is sound; they say nothing about the true value of a mall in Melbourne, and are never presented as if they did.
- Melbourne's sale records span a limited window, which constrains how late an event can be placed and still be identified. Event months are chosen to leave sufficient post-ramp observation, and this constraint is documented rather than worked around.
- DiD identifies the effect only under parallel trends; robustness checks (placebo dates, alternative control sets) bound but do not eliminate this.
- DBSCAN is sensitive to density variation; sparse peripheral areas are more likely to be labelled noise, which biases coverage toward dense suburbs.
- Predicted price is a market-price estimate, not intrinsic value, and should not be read as investment guidance.

## Author

GG (Ganga B Nair) — B.Tech Computer Science & AI, MBCET Thiruvananthapuram
