"""SHAP attribution -- global, local, and interaction."""
def global_summary(model, X, out): ...
def dependence(model, X, feature, out, interaction=None): ...
def local_waterfall(model, X, index, out):
    """Pick: (a) a pre-event property, (b) the same locality post-event,
    (c) a far-from-amenity property. Three stories, not three random rows."""
def amenity_contribution_table(model, X):
    """Mean |SHAP| for each proximity feature -- ties XAI back to the
    infrastructure research question."""
