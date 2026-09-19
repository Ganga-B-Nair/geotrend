"""
Synthetic infrastructure-event overlay on top of REAL housing data.

THE ONE SYNTHETIC COMPONENT IN THIS PROJECT. Everything else -- coordinates,
prices, dates, attributes, spatial structure, noise -- is real.

WHY THIS EXISTS (this paragraph goes in your report almost verbatim):
    Estimating the causal effect of new public infrastructure on nearby
    property prices requires knowing exactly when and where infrastructure
    opened. Public housing datasets do not carry that information, and
    reconstructing it retrospectively is unreliable. Rather than abandon the
    causal question or answer it with an unverifiable claim, this project
    treats it as a SIMULATION STUDY: an effect of known magnitude is injected
    into designated clusters at designated months, and the Difference-in-
    Differences estimator is then judged by how accurately it recovers that
    known value. The result is not a claim about Melbourne; it is a
    demonstration that the estimator works, with a measurable error.

ORDER OF OPERATIONS -- THIS MATTERS:
    real data -> clean -> Moran's I -> DBSCAN clusters -> *** OVERLAY HERE ***
    -> DiD

    The overlay is applied AFTER clustering, never before. Clusters must be
    derived from genuine spatial structure; injecting price changes first
    would let the synthetic layer contaminate RQ1 and RQ2. Notebooks 02-05
    and 07 all run on unmodified real prices. Only Notebook 06 touches the
    augmented column, and it is named `price_augmented` -- distinct from
    `price` -- so the two can never be silently swapped.

THE MODEL:
    price_augmented = price_real * (1 + effect * ramp(t - event_month))
    ramp(k) = 0                for k < 0
            = k / ramp_months  for 0 <= k < ramp_months
            = 1                for k >= ramp_months

Usage:
    python -m src.data.event_overlay \
        --clustered data/processed/clustered.parquet \
        --out data/processed/augmented.parquet
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42


@dataclass
class Event:
    """One designed infrastructure opening."""
    cluster_id: int
    kind: str             # mall | park | transit
    event_month: int      # months since first sale in the dataset
    effect: float         # terminal proportional uplift, e.g. 0.115 = 11.5%
    ramp_months: int      # months to reach full effect
    label: str = ""


def ramp_weight(elapsed: np.ndarray, ramp_months: int) -> np.ndarray:
    """Linear phase-in. Prices do not jump the day a mall opens."""
    w = np.clip(elapsed / max(ramp_months, 1), 0.0, 1.0)
    return np.where(elapsed < 0, 0.0, w)


# --------------------------------------------------------------------------
# Choosing where to put the events
# --------------------------------------------------------------------------

def cluster_pre_period_profile(
    df: pd.DataFrame,
    cluster_col: str = "cluster",
    price_col: str = "price",
    area_col: str = "area_sqft",
    upto_month: int | None = None,
) -> pd.DataFrame:
    """Pre-period level, slope and volume per cluster.

    Control selection is judged on these three. A control cluster must have
    a similar pre-period TREND, not merely a similar price level -- that is
    the parallel-trends assumption, and matching on level alone does not
    deliver it.
    """
    d = df.copy()
    if upto_month is not None:
        d = d[d["month_index"] < upto_month]
    d["lpps"] = np.log(d[price_col] / d[area_col])

    rows = []
    for cid, g in d.groupby(cluster_col):
        if cid == -1 or len(g) < 30:          # -1 is DBSCAN noise
            continue
        monthly = g.groupby("month_index")["lpps"].median()
        if len(monthly) < 6:
            continue
        slope = np.polyfit(monthly.index.values, monthly.values, 1)[0]
        rows.append({
            "cluster": cid,
            "n": len(g),
            "level": float(monthly.mean()),
            "slope": float(slope),
            "n_months": len(monthly),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def suggest_treated_and_controls(
    profile: pd.DataFrame,
    n_events: int = 3,
    n_controls_each: int = 2,
) -> dict:
    """Propose treated clusters and their matched controls.

    A suggestion, not a decision. Inspect the match quality in Notebook 06,
    override if a pairing looks weak, and say in the report which pairings
    you chose and why -- that reasoning is graded.
    """
    p = profile[profile["n_months"] >= 12].copy()
    if len(p) < n_events * (1 + n_controls_each):
        raise ValueError(
            f"only {len(p)} usable clusters; need "
            f"{n_events * (1 + n_controls_each)}. Loosen DBSCAN eps or "
            f"reduce n_events."
        )

    treated = p.nlargest(n_events, "n")["cluster"].tolist()
    pool = p[~p["cluster"].isin(treated)]

    assignments = {}
    used: set = set()
    for t in treated:
        trow = p[p["cluster"] == t].iloc[0]
        cand = pool[~pool["cluster"].isin(used)].copy()
        # distance in standardised (level, slope) space
        cand["d"] = np.sqrt(
            ((cand["level"] - trow["level"]) / (p["level"].std() + 1e-9)) ** 2
            + ((cand["slope"] - trow["slope"]) / (p["slope"].std() + 1e-9)) ** 2
        )
        picks = cand.nsmallest(n_controls_each, "d")["cluster"].tolist()
        used.update(picks)
        assignments[int(t)] = [int(c) for c in picks]

    return assignments


MIN_PRE_MONTHS = 6    # enough to establish a trend and test parallel trends
MIN_POST_MONTHS = 6   # enough post-ramp observation at full effect

# (kind, terminal effect, ramp months). Ramps are deliberately short: the
# Melbourne window is ~26 months, and every ramp month is excluded from
# estimation, so a long ramp buys realism at the cost of identification.
DEFAULT_KINDS = [
    ("mall", 0.115, 4),
    ("park", 0.045, 3),
    ("transit", 0.150, 4),
]


def check_identifiable(event_month: int, ramp_months: int, n_months: int) -> dict:
    """Can this event actually be estimated in the observation window?

    Budget:  pre-period | ramp (excluded) | post-period at full effect
    """
    pre = event_month
    post = n_months - (event_month + ramp_months)
    return {
        "pre_months": pre,
        "post_months": post,
        "ok": pre >= MIN_PRE_MONTHS and post >= MIN_POST_MONTHS,
    }


def default_events(
    treated: list[int],
    n_months: int,
    kinds: list[tuple] | None = None,
    strict: bool = True,
) -> list[Event]:
    """Space events across the identifiable middle of the observation window.

    Raises if the window is too short to place an identifiable event. That is
    deliberate: a silently unidentifiable event produces a DiD estimate that
    looks fine and means nothing.

    Melbourne note: the SNAPSHOT release spans ~20 months, which leaves almost
    no slack. Use Melbourne_housing_FULL.csv (~26 months) if at all possible,
    and check load_real.temporal_coverage() before relying on these defaults.
    """
    kinds = kinds or DEFAULT_KINDS
    max_ramp = max(k[2] for k in kinds[: max(len(treated), 1)])
    budget = MIN_PRE_MONTHS + max_ramp + MIN_POST_MONTHS

    if strict and n_months < budget:
        raise ValueError(
            f"observation window is {n_months} months; need at least {budget} "
            f"({MIN_PRE_MONTHS} pre + {max_ramp} ramp + {MIN_POST_MONTHS} post). "
            f"Use a longer dataset (Melbourne_housing_FULL.csv), shorten the "
            f"ramps via `kinds`, or reduce the number of events."
        )

    lo = MIN_PRE_MONTHS
    hi = n_months - max_ramp - MIN_POST_MONTHS
    months = np.linspace(lo, max(hi, lo), num=max(len(treated), 1)).round().astype(int)

    events = []
    for i, cid in enumerate(treated):
        kind, eff, ramp = kinds[i % len(kinds)]
        m = int(months[i])
        diag = check_identifiable(m, ramp, n_months)
        if strict and not diag["ok"]:
            raise ValueError(
                f"event for cluster {cid} at month {m} (ramp {ramp}) is not "
                f"identifiable: {diag['pre_months']} pre / "
                f"{diag['post_months']} post months available."
            )
        events.append(Event(
            cluster_id=int(cid),
            kind=kind,
            event_month=m,
            effect=eff,
            ramp_months=ramp,
            label=f"{kind}_c{cid}_m{m}",
        ))
    return events


# --------------------------------------------------------------------------
# Applying the overlay
# --------------------------------------------------------------------------

def apply_overlay(
    df: pd.DataFrame,
    events: list[Event],
    cluster_col: str = "cluster",
    price_col: str = "price",
    noise_sd: float = 0.0,
    seed: int = SEED,
) -> pd.DataFrame:
    """Add `price_augmented` and treatment bookkeeping columns.

    The real `price` column is left untouched. Downstream code chooses which
    to use, explicitly.

    noise_sd adds multiplicative lognormal noise to the injected effect only.
    Leave it at 0 for your headline recovery figure (cleanest demonstration),
    then re-run with noise_sd=0.05 as a robustness check and report both --
    an estimator that only works on a noiseless effect is not much of an
    estimator.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()

    uplift = np.zeros(len(out))
    treated_flag = np.zeros(len(out), dtype=int)
    event_label = np.array([""] * len(out), dtype=object)
    event_month = np.full(len(out), np.nan)

    for ev in events:
        mask = (out[cluster_col] == ev.cluster_id).to_numpy()
        elapsed = out["month_index"].to_numpy() - ev.event_month
        w = ramp_weight(elapsed, ev.ramp_months)
        uplift += mask * w * ev.effect
        treated_flag |= mask.astype(int)
        event_label[mask] = ev.label
        event_month[mask] = ev.event_month

    if noise_sd > 0:
        uplift = uplift * rng.lognormal(0, noise_sd, len(out))

    out["price_augmented"] = (out[price_col] * (1 + uplift)).round(-2)
    out["is_treated_cluster"] = treated_flag
    out["event_label"] = event_label
    out["event_month"] = event_month
    out["months_since_event"] = out["month_index"] - out["event_month"]
    out["post"] = (out["months_since_event"] >= 0).astype("Int64")

    return out


def write_manifest(
    events: list[Event],
    assignments: dict,
    path: str | Path,
    noise_sd: float = 0.0,
    source_note: str = "",
) -> None:
    """Ground truth. Notebook 06 compares its estimates against this file.

    Do NOT let any modelling code read effect sizes from here -- it is for
    scoring your results, not for producing them.
    """
    manifest = {
        "overlay_version": 1,
        "seed": SEED,
        "noise_sd": noise_sd,
        "base_data": source_note or "real housing dataset (see README)",
        "model": "price_augmented = price_real * (1 + effect * ramp(t - event_month))",
        "events": [asdict(e) for e in events],
        "control_assignments": {str(k): v for k, v in assignments.items()},
        "warning": (
            "Ground truth for validation only. Recovering these values is the "
            "result; feeding them into the estimator is circular."
        ),
    }
    Path(path).write_text(json.dumps(manifest, indent=2))


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def placebo_events(events: list[Event], shift: int = -12) -> list[Event]:
    """Same clusters, fake earlier dates, zero real effect at those dates.

    Run the full DiD on these. A well-specified estimator should return
    effects statistically indistinguishable from zero. If it does not, your
    specification is picking up something other than the treatment, and
    that is a finding worth reporting honestly.
    """
    return [
        Event(e.cluster_id, e.kind, e.event_month + shift, 0.0,
              e.ramp_months, f"placebo_{e.label}")
        for e in events
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clustered", default="data/processed/clustered.parquet")
    ap.add_argument("--out", default="data/processed/augmented.parquet")
    ap.add_argument("--manifest", default="data/processed/overlay_manifest.json")
    ap.add_argument("--n-events", type=int, default=3)
    ap.add_argument("--noise-sd", type=float, default=0.0)
    args = ap.parse_args()

    df = pd.read_parquet(args.clustered)
    n_months = int(df["month_index"].max()) + 1

    profile = cluster_pre_period_profile(df)
    assignments = suggest_treated_and_controls(profile, n_events=args.n_events)
    events = default_events(list(assignments.keys()), n_months)

    aug = apply_overlay(df, events, noise_sd=args.noise_sd)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    aug.to_parquet(args.out, index=False)
    write_manifest(events, assignments, args.manifest, noise_sd=args.noise_sd)

    print(f"wrote {len(aug):,} rows -> {args.out}")
    for e in events:
        print(f"  {e.label}: cluster {e.cluster_id}, month {e.event_month}, "
              f"effect {e.effect:.1%}, controls {assignments[e.cluster_id]}")


if __name__ == "__main__":
    main()
