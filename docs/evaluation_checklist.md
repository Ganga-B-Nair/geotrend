# Pre-submission checklist

## Hybrid-design integrity (new — check these first)
- [ ] Report states clearly which RQs use real data and which uses the overlay
- [ ] Every number in NB06 Part B labelled as coming from `price_augmented`
- [ ] Every number elsewhere confirmed to come from `price`
- [ ] Overlay applied AFTER clustering — verify NB04 ran on unmodified prices
- [ ] Manifest values never read by any estimator (only by the scoring table)
- [ ] Fixture (`_fixture.csv`) does not appear in any reported result
- [ ] Written justification for the simulation-study approach, in your own words

## Data science depth
- [ ] Data quality section shows detected defects with counts
- [ ] Every preprocessing decision has a stated reason
- [ ] Temporal coverage checked before committing to event months
- [ ] Moran's I reported with permutation p-value BEFORE any clustering
- [ ] DBSCAN eps justified with the k-distance plot (config ships as null on purpose)
- [ ] ADF verdict stated per cluster with the differencing decision
- [ ] Forecasts backtested with rolling origin, not a single split
- [ ] Parallel trends shown visually AND tested
- [ ] SE method chosen deliberately; few-clusters problem addressed explicitly
- [ ] Placebo test included and genuinely null
- [ ] Recovery table: designed vs estimated vs error, with CI coverage
- [ ] Robustness re-run with noise_sd=0.05
- [ ] Temporal train/test split, explicitly justified
- [ ] Residual Moran's I reported
- [ ] SHAP local cases chosen for narrative reasons, stated

## Presentation
- [ ] One figure per claim; no unlabelled axes
- [ ] Event-study plot is the hero figure
- [ ] README renders correctly on GitHub
- [ ] Repo has description + topics
- [ ] Notebooks run top-to-bottom on a clean checkout
- [ ] pytest passes
- [ ] Report separates DS contribution from engineering wrapper

## Demo
- [ ] Streamlit runs offline as the fallback
- [ ] Mobile app has 3 screens: map picker, result + SHAP, cluster trend
- [ ] One-paragraph answer ready: "why a synthetic event layer?"
- [ ] One-paragraph answer ready: "why DBSCAN and not K-Means?"
- [ ] One-paragraph answer ready: "why not cluster-robust standard errors?"
