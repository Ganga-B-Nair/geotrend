"""Feature engineering. Keep leakage discipline: any feature derived from
price must be computed on the TRAIN window only, then applied forward.
"""
def add_geo_features(df): """dist to CBD, dist to each amenity type, bearing"""
def add_temporal_features(df): """month, quarter, months_since_start, season"""
def add_property_features(df): """price-free ratios: area_per_bed, age bucket"""
def add_cluster_features(df, labels): """cluster id, cluster median (train-only)"""
def build(df, cfg): ...
