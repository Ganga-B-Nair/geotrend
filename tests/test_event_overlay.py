"""Tests for the synthetic event overlay.

These assert the OVERLAY is exact and non-contaminating. They deliberately
do NOT assert that a DiD estimate recovers the true effect within a fixed
tolerance -- that depends on sample size, noise and specification, and is
the thing Notebook 06 is supposed to investigate rather than assume.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import DBSCAN

from src.data.load_real import make_fixture, SCHEMA
from src.data.event_overlay import (
    apply_overlay,
    cluster_pre_period_profile,
    default_events,
    placebo_events,
    ramp_weight,
    suggest_treated_and_controls,
)


@pytest.fixture(scope="module")
def clustered():
    df = make_fixture(8000, seed=42)
    df["cluster"] = DBSCAN(eps=0.006, min_samples=40).fit_predict(
        np.c_[df.latitude, df.longitude]
    )
    return df


def test_fixture_matches_schema():
    df = make_fixture(500)
    assert list(df.columns) == SCHEMA


def test_ramp_weight_bounds():
    w = ramp_weight(np.array([-5, 0, 3, 6, 20]), ramp_months=6)
    assert w[0] == 0.0          # before the event, nothing
    assert w[1] == 0.0          # on the event month, nothing yet
    assert 0 < w[2] < 1         # mid-ramp
    assert w[3] == 1.0          # fully phased in
    assert w[4] == 1.0          # and stays there


def test_injected_uplift_is_exact(clustered):
    """The whole point: we know the truth precisely."""
    prof = cluster_pre_period_profile(clustered)
    assign = suggest_treated_and_controls(prof, n_events=3)
    events = default_events(list(assign.keys()), 36)
    aug = apply_overlay(clustered, events)

    for e in events:
        m = aug.cluster == e.cluster_id
        post = aug.month_index >= e.event_month + e.ramp_months
        got = np.log(aug.loc[m & post, "price_augmented"]
                     / aug.loc[m & post, "price"]).mean()
        assert abs(got - np.log(1 + e.effect)) < 1e-3


def test_control_and_pre_period_rows_are_untouched(clustered):
    """No contamination outside the designed treatment cells."""
    prof = cluster_pre_period_profile(clustered)
    assign = suggest_treated_and_controls(prof, n_events=3)
    aug = apply_overlay(clustered, default_events(list(assign.keys()), 36))

    untreated = aug[aug.is_treated_cluster == 0]
    assert (untreated.price_augmented == untreated.price).all()

    pre = aug[(aug.is_treated_cluster == 1) & (aug.months_since_event < 0)]
    assert (pre.price_augmented == pre.price).all()


def test_real_price_column_is_never_modified(clustered):
    aug = apply_overlay(clustered, default_events([0], 36))
    assert aug["price"].equals(clustered["price"])


def test_placebo_events_inject_nothing(clustered):
    """Placebo must be a genuine null, or the robustness check is theatre."""
    real = default_events([0, 1], 36)
    aug = apply_overlay(clustered, placebo_events(real))
    assert (aug.price_augmented == aug.price).all()


def test_controls_are_distinct_from_treated(clustered):
    prof = cluster_pre_period_profile(clustered)
    assign = suggest_treated_and_controls(prof, n_events=3)
    treated = set(assign.keys())
    controls = [c for v in assign.values() for c in v]
    assert treated.isdisjoint(controls)
    assert len(controls) == len(set(controls))   # no control reused


def test_insufficient_clusters_raises_clearly():
    df = make_fixture(500)
    df["cluster"] = 0
    prof = cluster_pre_period_profile(df)
    with pytest.raises(ValueError, match="usable clusters"):
        suggest_treated_and_controls(prof, n_events=3)


def test_noise_option_perturbs_but_does_not_bias(clustered):
    events = default_events([0], 36)
    clean = apply_overlay(clustered, events, noise_sd=0.0)
    noisy = apply_overlay(clustered, events, noise_sd=0.05)
    assert not clean.price_augmented.equals(noisy.price_augmented)
    m = (clean.cluster == 0) & (clean.month_index >= 16)
    if m.sum() > 100:
        rel = (noisy.loc[m, "price_augmented"]
               / clean.loc[m, "price_augmented"]).mean()
        assert abs(rel - 1.0) < 0.02


# --- identifiability guard (added after Melbourne window turned out short) ---

def test_short_window_is_rejected():
    """A 12-month window cannot support pre + ramp + post. Must raise."""
    from src.data.event_overlay import default_events as de
    with pytest.raises(ValueError, match="observation window"):
        de([0, 1, 2], n_months=12)


def test_every_default_event_is_identifiable():
    from src.data.event_overlay import check_identifiable
    for n in (20, 26, 36, 48):
        for e in default_events([0, 1, 2], n_months=n):
            assert check_identifiable(e.event_month, e.ramp_months, n)["ok"]


def test_shorter_ramps_can_rescue_a_tight_window():
    """Escape hatch: caller may pass custom kinds with shorter ramps."""
    from src.data.event_overlay import default_events as de
    tight = [("mall", 0.115, 2), ("park", 0.045, 2)]
    evs = de([0, 1], n_months=15, kinds=tight)
    assert len(evs) == 2
