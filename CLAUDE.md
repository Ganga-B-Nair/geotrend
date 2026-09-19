# Project conventions for Claude Code

**Project:** Terra (umbrella) — **GeoTrend** is this component: the data pipeline and app.
**Subject:** Data Science. Solo, graded, with a viva.

---

## The most important rule

**This project is graded on the author's own reasoning, and she has to defend
it verbally.** Do not write analysis, interpretation, or methodological
justification for her. That includes: choosing DBSCAN parameters, interpreting
Moran's I, deciding how to handle missingness, writing the report narrative,
or explaining why a result came out as it did.

What to help with instead:
- environment setup, dependency and install problems
- boilerplate: plotting scaffolds, file I/O, refactoring notebook code into `src/`
- debugging errors she brings you, explained so she understands the fix
- reviewing her code for bugs and leakage, pointing at problems without fixing silently
- explaining a statistical concept when asked, so she can then apply it herself

If asked to "do notebook 04", decline and offer to explain the method or review
what she writes. She will be asked in the viva why she chose `eps`. She needs
to have actually chosen it.

---

## Architecture

```
real data → clean → Moran's I → DBSCAN clusters → ★ OVERLAY ★ → DiD
```

Notebooks tell the story; `src/` holds reusable logic. Nothing important
lives only inside a notebook cell.

| Path | Purpose |
|---|---|
| `src/data/load_real.py` | Loads/normalises the Melbourne CSV to internal schema |
| `src/data/event_overlay.py` | The ONLY synthetic component. Injects known effects |
| `src/data/clean.py` | Quality profiling + cleaning |
| `src/analysis/spatial.py` | Moran's I, k-distance, DBSCAN |
| `src/analysis/timeseries.py` | Decomposition, ADF, ARIMA/Prophet |
| `src/analysis/causal.py` | Distance decay (real) + DiD (augmented) |
| `src/models/train.py` | OLS baseline + XGBoost |
| `src/models/explain.py` | SHAP |
| `config.yaml` | Single source of truth for parameters |

---

## Hard constraints — do not violate these

**1. Real vs synthetic separation.**
`price` is real. `price_augmented` is real + injected effect.
Notebooks 02–05 and 07 use `price`. ONLY notebook 06 Part B uses
`price_augmented`. Never swap them, never merge them into one column.

**2. The overlay runs AFTER clustering.**
Clusters must come from genuine spatial structure. Applying the overlay
before clustering contaminates RQ1 and RQ2 and invalidates the project.

**3. Never read the manifest into an estimator.**
`overlay_manifest.json` holds ground-truth effect sizes. It is for SCORING
results, never for producing them. Reading it into the DiD is circular.

**4. Temporal split, never random.**
`split: temporal` in config. A random split lets later months train a model
evaluated on earlier ones. If you see `train_test_split(..., shuffle=True)`
on this data, flag it.

**5. No price-derived feature computed on the full dataset.**
Cluster medians, suburb averages — train window only, then applied forward.
This is the most likely source of silent leakage here.

**6. Cluster-robust SEs are not trustworthy at this cluster count.**
With ~3 treated clusters, cluster-robust SEs are severely downward-biased
(development runs produced SE=0.0004 on an estimate 0.019 from truth).
Use wild cluster bootstrap or randomisation inference. If you see
`cov_type="cluster"` in the DiD, flag it.

**7. `config.yaml` ships with `dbscan.eps_km: null` deliberately.**
It must be set from a k-distance elbow, by the author, with the plot shown.
Do not fill in a plausible default.

---

## Two-track sample design

Melbourne FULL has ~34.8k rows: Price 22% missing, BuildingArea 61% missing.

- **cluster_sample** (~27.2k): valid Price + coordinates → Moran's I, DBSCAN
- **analysis_sample** (~11–13k): also valid BuildingArea → everything downstream

Rationale: DBSCAN is a density algorithm, so dropping 60% of points degrades
the structure it finds. Cluster membership is geographic, so the smaller
sample inherits labels from the larger one.

Before using `analysis_sample`, its representativeness must be tested against
the dropped rows (chi-square on suburb mix, KS on price and date). A
significant result is a limitation to report, not a failure.

---

## Known data defects (confirmed present)

| Column | Defect | Note |
|---|---|---|
| `YearBuilt` | min 1196 | Melbourne founded 1835 — impossible |
| `YearBuilt` | max 2106 | 90 years after data ends |
| `BuildingArea` | min 0 | No such building |
| `BuildingArea` | max 44500 | 327× the median of 136 sqm |
| `Lattitude`/`Longtitude` | misspelled in source | Loader handles both; do NOT edit the CSV |

`BuildingArea` is in square metres (median 136). The loader converts to sqft.

---

## Conventions

- Target: `log(price / area_sqft)`. Log because prices are right-skewed;
  per-sqft because otherwise area dominates every SHAP plot.
- Run tests before committing: `pytest -q` (expect 12 passing).
- Don't commit the source CSV — `.gitignore` excludes it; README says where to get it.
- DO commit `overlay_manifest.json` — it is ground truth for the recovery table.
- Commit messages: plain and descriptive. No emoji.

## Environment

Windows. Python 3.11 in `venv/`. `prophet` and `geopandas` are the two
dependencies most likely to fail on Windows — if they do, comment them out of
`requirements.txt` and proceed; `statsmodels` ARIMA covers notebook 05 and
Folium covers mapping. Revisit via conda later if wanted.
