"""Infrastructure impact: distance-decay (real data) + DiD (augmented data).

TWO ANALYSES, TWO DATA COLUMNS -- keep them straight:

  distance_decay()  runs on `price`           -> REAL correlational finding
  did_estimate()    runs on `price_augmented` -> SIMULATION STUDY

Say which column produced every number you report. Conflating them is the
single easiest way to turn a careful project into an overclaim.

IDENTIFICATION NOTES (defend these in your viva):
  - Treated = clusters carrying a designed event; controls = clusters matched
    on pre-period LEVEL and SLOPE (see event_overlay.cluster_pre_period_profile).
  - Parallel trends must be shown visually AND tested: pre-period lead
    coefficients should be jointly insignificant.
  - Drop the ramp-in window so the post coefficient estimates the terminal
    effect rather than a blend of partial and full treatment.
  - Cluster standard errors by cluster -- BUT SEE BELOW.

THE FEW-CLUSTERS PROBLEM (verified on this project's own data):
  Cluster-robust standard errors are asymptotic in the NUMBER OF CLUSTERS,
  not the number of rows. With 3 treated clusters and a handful of controls,
  cluster-robust SEs are badly downward-biased -- during development this
  pipeline produced an SE of 0.0004 on an estimate that was 0.019 away from
  the known truth, i.e. an interval excluding the true value by ~45 sigma.
  That is a broken inference, not a precise one.

  Do one of these and say which:
    (a) wild cluster bootstrap (Cameron-Gelbach-Miller) -- the standard fix
    (b) randomisation inference: permute treatment across clusters, compare
        the observed estimate to the permutation distribution
    (c) report heteroskedasticity-robust SEs and state plainly that
        cluster-robust inference is unreliable at this cluster count
  Catching this yourself and handling it is a genuinely strong result to
  report. Quietly printing a tiny SE is not.
"""

from __future__ import annotations


# --------------------------------------------------------------------------
# Correlational -- REAL prices
# --------------------------------------------------------------------------

def distance_decay(df, dist_col, price_col="price", area_col="area_sqft"):
    """OLS of log price-per-sqft on distance to nearest amenity.

    Report the % change per km with a confidence interval, and control for
    property attributes so you are not just recovering "big houses are far
    from the centre". This is a REAL finding from REAL data -- it is the
    part of the infrastructure story you can state without qualification.
    """
    raise NotImplementedError


def decay_by_amenity_type(df, dist_cols, price_col="price"):
    """Compare decay slopes across amenity types (transit vs park vs mall)."""
    raise NotImplementedError


# --------------------------------------------------------------------------
# Causal -- AUGMENTED prices (simulation study)
# --------------------------------------------------------------------------

def check_parallel_trends(df, treated, controls, event_month, pre_window=12):
    """Test pre-period leads jointly. Returns the F-test and a tidy table.

    If this fails, your control set is wrong. Fix the controls -- do not
    proceed to the DiD and hope no one asks.
    """
    raise NotImplementedError


def event_study(df, treated, controls, event_month, leads=6, lags=12,
                price_col="price_augmented"):
    """Lead/lag coefficients relative to the month before the event.

    The single strongest figure in the report: flat and near zero before the
    event, rising through the ramp, plateauing after. Plot with CIs.
    """
    raise NotImplementedError


def did_estimate(df, treated, controls, event_month, ramp_months,
                 price_col="price_augmented", se="wild_bootstrap"):
    """Two-way DiD: log_pps ~ treated*post + property controls.

    Drop the ramp window [event_month, event_month + ramp_months).
    `se` should be 'wild_bootstrap' or 'randomisation' -- read the
    few-clusters note at the top of this module before choosing 'cluster'.
    """
    raise NotImplementedError


def placebo_test(df, treated, controls, fake_event_month, **kw):
    """Run the identical specification at a date with no designed effect.

    Expect a null. A significant placebo effect means the specification is
    absorbing something other than treatment.
    """
    raise NotImplementedError


def recover_vs_truth(estimates, manifest_path):
    """Compare each DiD estimate against the designed effect.

    Output the headline table of the project:
        event | designed | estimated | abs error | CI covers truth?

    Note the manifest stores PROPORTIONAL effects while the regression
    returns LOG points -- compare log(1 + effect) against the coefficient,
    not the raw percentage.
    """
    raise NotImplementedError
