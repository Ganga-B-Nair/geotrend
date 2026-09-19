"""Optional amenity enrichment via OpenStreetMap Overpass API.

Since the base dataset is synthetic, amenity coordinates are already known.
Use this module only if you extend to a real city -- then cache results to
data/external/ so the pipeline stays reproducible offline.
"""
def fetch_amenities(bbox, kinds=("mall", "park", "station", "school")): ...
def distance_to_nearest(df, amenities, kind): ...
