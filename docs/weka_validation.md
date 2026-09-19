# Weka cross-validation appendix (keep this small)

Purpose: show the micro-neighbourhood structure is a property of the data,
not of one library's implementation.

**Use REAL prices only.** Clustering happens before the overlay exists, so
the Weka export must come from `data/processed/features.parquet`, never from
`augmented.parquet`. Cross-checking the synthetic layer would validate nothing.

1. Export via `src.analysis.spatial.export_for_weka`
   (columns: latitude, longitude, log_price_per_sqft — scaled identically
   to the Python run)
2. Weka Explorer -> Cluster -> `DBSCAN` (or `SimpleKMeans` at the same k
   for a coarse check)
3. Match eps / minPoints to your tuned config.yaml values
4. Compare: number of clusters, noise share, Adjusted Rand Index against
   the scikit-learn labels
5. Report one table + two sentences. Do not let this grow into a chapter.

If ARI is high (> 0.8), state that the structure is implementation-independent.
If it is low, investigate scaling differences first — that is usually the
cause, and saying so is itself a good finding.
