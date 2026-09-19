"""Spatial dependence testing + micro-neighbourhood discovery.

Order matters for the report:
  1. Moran's I  -> establishes that space carries signal (statistical claim)
  2. k-distance -> justifies the DBSCAN eps you chose (parameter defence)
  3. DBSCAN     -> derives micro-neighbourhoods
  4. validation -> silhouette + cross-check against Weka's implementation
"""
def morans_i(gdf, value_col, k=8, permutations=999):
    """Return I, expected I, p-value (permutation-based)."""
def k_distance_curve(coords, k): ...
def fit_dbscan(coords, eps_km, min_samples): ...
def cluster_diagnostics(coords, labels): """silhouette, n_clusters, noise share"""
def export_for_weka(df, path): """ARFF/CSV export for the validation appendix"""
