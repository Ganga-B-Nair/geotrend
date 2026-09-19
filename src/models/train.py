"""Price model. Temporal split, not random -- a random split leaks future
market drift into the training set and inflates your metrics.
"""
def temporal_split(df, test_size, date_col="transaction_date"): ...
def baseline_ols(X, y): """interpretable benchmark -- report it, don't skip it"""
def train_xgb(X, y, params=None): ...
def evaluate(model, X, y): """RMSE, MAE, R2, MAPE + residual-vs-space check"""
def residual_spatial_autocorrelation(model, gdf):
    """Re-run Moran's I on residuals: if still significant, your spatial
    features are under-specified. Excellent discussion point."""
def save(model, path): ...
