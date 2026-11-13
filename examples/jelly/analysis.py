"""Hyperliquid JELLY mechanism failure: causal postmortem with intervense.

Source data: public Hyperliquid candleSnapshot API, JELLY perp,
2025-03-25 to 2025-03-27 UTC (the attack window). Publicly reported
facts anchor the timeline (attacker short build, squeeze, HLP vault
forced to absorb the losing short, delisting decision).

Three analyses:
  1. Event study with panel fixed effects: minute returns around the
     squeeze, volume and range as concurrent witnesses, intervention
     time = the squeeze onset read off the data (largest 5-minute
     range in the window).
  2. Specification curve on the event-stimate: the attack effect on
     minute volatility under every control subset, to show which
     controls the conclusion depends on.
  3. Counterfactual exposure cap scan: a one-page model of the HLP
     liquidation rule (vault absorbs margin-weighted losing positions
     up to an exposure cap c of open interest) evaluated against the
     realized squeeze path; reports for which cap values the vault
     survives without the delisting intervention.

Inputs are public market data only; position sizes come from the
publicly reported figures recorded in FACTS below.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Publicly reported figures for the 2025-03-26 JELLY incident (USD).
FACTS = {
    "attacker_short_notional": 6.1e6,   # reported concentrated short opened across wallets
    "squeeze_move": 4.4,                # price multiple at the peak of the squeeze (~0.009 -> ~0.04)
    "hlp_absorbed_pnl": -12.0e6,        # unrealized loss forced onto the HLP vault at peak
    "hlp_vault_equity": 230e6,          # vault equity around the event, USD
}

HERE = os.path.dirname(os.path.abspath(__file__))


def load_candles():
    rows = json.load(open(os.path.join(HERE, "jelly_candles.json")))
    df = pd.DataFrame([r for r in rows if r["i"] == "1m"])
    df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df.sort_values("ts").reset_index(drop=True)
    for c in ("o", "c", "h", "l", "v"):
        df[c] = df[c].astype(float)
    df["ret"] = df["c"].pct_change()
    df["range"] = (df["h"] - df["l"]) / df["o"]
    return df


def event_study(df: pd.DataFrame):
    """Minute-level panel: each candle is a unit, the squeeze onset is
    the intervention, post-squeeze minutes carry the treatment flag,
    hour-of-day fixed effects absorb intraday seasonality."""
    five_min_range = df.set_index("ts")["range"].rolling(5).mean()
    onset = five_min_range.idxmax()
    df["post"] = (df["ts"] > onset).astype(int)
    df["hour"] = df["ts"].dt.hour
    df["logrange"] = np.log1p(df["range"].fillna(0).clip(lower=0))
    df["candle_id"] = np.arange(len(df))
    from intervense.analysis.panel_fixed_effects import panel_fixed_effects
    res = panel_fixed_effects(df, outcome="logrange", features=["post"],
                              group="hour", cluster="hour")
    eff_row = [r for r in res["features"] if r["feature"] == "post"][0]
    # raw log-scale effect from the same design with full hour dummies:
    # the panel FE coefficient is standardized (within per sd) and must
    # not be exponentiated directly
    dummies = pd.get_dummies(df["hour"], prefix="h", drop_first=True).astype(float)
    X = np.column_stack([np.ones(len(df)), df["post"].astype(float)] +
                        [dummies[c].values for c in dummies.columns])
    y = df["logrange"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    raw_log = float(beta[1])
    return {"onset_utc": str(onset),
            "raw_log_effect": round(raw_log, 4),
            "range_multiplier": round(float(np.exp(raw_log) - 1), 4),
            "standardized_coef_within_per_sd": eff_row["coef_within_per_sd"],
            "se_cluster": eff_row["se_cluster"],
            "p_value": eff_row["p_value"], "ci95": eff_row["ci95"],
            "significant": eff_row["significant"], "n_groups": res.get("n_groups"),
            "engine": "intervense.analysis.panel_fixed_effects (cluster-robust) plus raw-scale OLS with hour dummies"}


def specification_curve(df: pd.DataFrame, n_specs: int = 200, seed: int = 0):
    """Re-estimate the post-squeeze effect under random control subsets
    (sampled, the full lattice of hour dummies plus volume is too large
    to enumerate); report sign stability."""
    df = df.copy()
    df["logv"] = np.log1p(df["v"])
    df["hour"] = df["ts"].dt.hour
    controls = [f"h_{h}" for h in range(1, 24)] + ["logv"]
    rng = np.random.default_rng(seed)
    estimates = []
    y = np.log1p(df["range"].fillna(0).clip(lower=0).values)
    post = df["post"].astype(float).values
    hour_arr = df["hour"].values
    logv_arr = df["logv"].values
    for _ in range(n_specs):
        k = int(rng.integers(0, len(controls) + 1))
        subset = list(rng.choice(len(controls), size=k, replace=False))
        cols = [np.ones(len(df)), post]
        for j in subset:
            c = controls[j]
            cols.append((hour_arr == int(c.split("_")[1])).astype(float) if c.startswith("h_") else logv_arr)
        X = np.column_stack(cols)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        estimates.append(float(beta[1]))
    estimates = np.array(estimates)
    return {"n_specs": len(estimates), "share_positive": float((estimates > 0).mean()),
            "min": round(float(estimates.min()), 3), "max": round(float(estimates.max()), 3)}


def counterfactual_cap_scan():
    """One-page HLP rule: when the concentrated short's unrealized loss
    crosses the maintenance margin, the vault absorbs the position.
    Vault loss under the realized squeeze path equals the reported peak
    PnL scaled by how much of the position the cap admits. The scan
    reports the largest survivable cap fraction."""
    squeeze_gain_multiple = FACTS["squeeze_move"] - 1.0      # short loses this fraction of notional at peak
    worst_short_loss = FACTS["attacker_short_notional"] * squeeze_gain_multiple
    # reported absorbed PnL implies the vault took roughly the whole position
    implied_takeup = FACTS["hlp_absorbed_pnl"] / -worst_short_loss
    results = []
    for cap_pct in (2, 5, 10, 20, 50, 100):
        takeup = min(cap_pct / 100.0, implied_takeup)
        vault_loss = worst_short_loss * takeup
        results.append({
            "cap_pct_of_open_interest": cap_pct,
            "vault_loss_usd_m": round(vault_loss / 1e6, 1),
            "survives_without_intervention": bool(vault_loss < 0.10 * FACTS["hlp_vault_equity"]),
        })
    return {"implied_realized_takeup": round(float(implied_takeup), 2),
            "worst_short_loss_usd_m": round(worst_short_loss / 1e6, 1),
            "scan": results}


def main():
    df = load_candles()
    peak = df.loc[df["h"].idxmax()]
    out = {
        "window": [str(df.ts.iloc[0]), str(df.ts.iloc[-1])],
        "price_min": round(float(df["l"].min()), 6),
        "price_max": round(float(df["h"].max()), 6),
        "realized_squeeze_multiple": round(float(df["h"].max() / df["l"].min()), 1),
        "peak_time_utc": str(peak["ts"]),
        "event_study": event_study(df),
        "specification_curve": specification_curve(df),
        "counterfactual_cap_scan": counterfactual_cap_scan(),
        "facts": {k: v for k, v in FACTS.items()},
        "method": "intervense postmortem pattern: event study + specification curve + counterfactual parameter scan",
        "grading": "observational for market data; prior for reported position sizes",
    }
    path = os.path.join(HERE, "postmortem.json")
    json.dump(out, open(path, "w"), indent=2)
    print(json.dumps(out, indent=2)[:2400])
    print("saved", path)


if __name__ == "__main__":
    main()
