"""Temporal structure discovery with lagged causal graphs.

The static backends in discovery.py assume i.i.d. rows. Real panels
(protocol telemetry, market data, operational metrics) are time
series: causes lead effects, and feedback loops appear as lagged
edges in both directions. This module runs PCMCI (condition
selection then MCI tests) over a time-indexed panel and returns
lagged edges with measured effects.

Every edge carries:
  - the lag tau, meaning source at t-tau -> target at t,
  - a standardized lag-regression coefficient estimated from the
    same data (the edge's partial association given the target's
    own lags), not a placeholder,
  - the MCI p-value,
  - a grading note: temporal priority gives the orientation under
    the assumptions of lagged cause-effect; contemporaneous or
    instantaneous confounding is not resolved by this method.

Robustness contract mirrors discovery.py: non-finite rows are
dropped and counted, constant columns are dropped and named, and a
dimension guard rejects panels wider than max_columns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    import tigramite.data_processing as tp

    HAS_TIGRAMITE = True
except Exception:  # pragma: no cover
    HAS_TIGRAMITE = False


def _lag_matrix(values: np.ndarray, tau_max: int):
    """Stack [x_{t-tau_max} .. x_t]; rows are aligned samples."""
    n, p = values.shape
    rows = []
    for tau in range(tau_max, -1, -1):
        rows.append(values[tau_max - tau: n - tau])
    return np.hstack(rows)  # columns: lag tau_max..0 for var 0, then var 1, ...


def temporal_discovery(
    df: pd.DataFrame,
    columns: list[str],
    tau_max: int = 5,
    pc_alpha: float = 0.05,
    max_edges: int = 25,
    max_columns: int = 40,
    focal: str | None = None,
) -> dict:
    """Run PCMCI over the given time-ordered columns of df.

    focal: if named, edges into this column always survive the
    max_edges cap (the audit's target of interest).
    """
    if not HAS_TIGRAMITE:
        raise ImportError(
            "tigramite is not installed. Install it with: pip install tigramite")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"columns not in frame: {missing}")
    if focal is not None and focal not in columns:
        raise ValueError(f"focal {focal!r} not among columns")

    X = df[columns].to_numpy(dtype=float)
    finite = np.isfinite(X).all(axis=1)
    n_dropped = int((~finite).sum())
    X = X[finite]

    stds = X.std(axis=0)
    keep = np.where(stds > 1e-12)[0]
    dropped_const = [columns[j] for j in np.where(stds <= 1e-12)[0]]
    if len(keep) < 2:
        raise ValueError("fewer than 2 usable columns after NaN and constant filtering")
    if len(keep) > max_columns:
        raise ValueError(
            f"{len(keep)} columns exceeds the {max_columns}-column guard; "
            "reduce the panel or pre-select columns")
    if dropped_const:
        X = X[:, keep]
        names = [columns[j] for j in keep]
        if focal is not None:
            focal = focal if focal in names else None
    else:
        names = list(columns)

    dataframe = tp.DataFrame(X, var_names=names)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr())
    res = pcmci.run_pcmci(tau_min=1, tau_max=tau_max, pc_alpha=pc_alpha)

    graph = res["graph"]
    pvals = res["p_matrix"]
    valmat = res["val_matrix"]

    # collect lagged links the MCI stage kept
    found = []
    for i in range(len(names)):
        for j in range(len(names)):
            for tau in range(1, tau_max + 1):
                link = graph[i, j, tau]
                if link in ("-->", "o->", "x->"):
                    found.append((i, j, tau, float(pvals[i, j, tau]),
                                  float(valmat[i, j, tau]), link))

    # measured effect: lag regression controlling the target's own lags
    k_max = min(3, tau_max)

    def lag_effect(i: int, j: int, tau: int) -> float:
        n = len(X)
        start = max(tau, k_max)
        y = X[start:, j]
        src = X[start - tau: n - tau, i]
        ctrl = [X[start - k: n - k, j] for k in range(1, k_max + 1)]
        M = np.column_stack([src] + ctrl)
        try:
            beta, *_ = np.linalg.lstsq(M, y, rcond=None)
            sx = M[:, 0].std()
            sy = y.std()
            return float(beta[0] * sx / sy) if sx > 1e-12 and sy > 1e-12 else 0.0
        except np.linalg.LinAlgError:
            return 0.0

    edges = []
    focal_idx = names.index(focal) if focal else None
    for i, j, tau, p, _, link in found:
        eff = lag_effect(i, j, tau)
        edges.append({
            "source": names[i],
            "target": names[j],
            "lag": tau,
            "effect_standardized": round(eff, 4),
            "mci_p_value": p,
            "link_type": link,
            "detail": (
                f"PCMCI lag-{tau}: {names[i]}(t-{tau}) -> {names[j]}(t), "
                f"standardized lag effect {eff:+.3f}, MCI p={p:.2g}, link {link}"
            ),
        })
    edges.sort(key=lambda e: e["mci_p_value"])

    focal_edges = [e for e in edges if focal_idx is not None and names.index(e["target"]) == focal_idx] \
        if focal_idx is not None else []
    other = [e for e in edges if e not in focal_edges]
    kept = focal_edges + other[: max(0, max_edges - len(focal_edges))]

    reduction = []
    if n_dropped:
        reduction.append(f"{n_dropped} non-finite rows dropped")
    if dropped_const:
        reduction.append(f"constant columns dropped: {', '.join(dropped_const)}")

    return {
        "n_samples": int(len(X)),
        "columns": names,
        "tau_max": tau_max,
        "edges": kept,
        "n_edges_found": len(edges),
        "reduction_notes": reduction,
        "verdict": (
            f"{len(kept)} lagged edges kept of {len(edges)} found; "
            f"orientations rely on temporal priority and the ParCorr linearity assumption"
        ),
        "method": "pcmci_parcorr",
        "reference": "Runge et al., Detecting and quantifying causal associations in "
                     "large nonlinear time series datasets, Science Advances 2019",
        "grading_note": (
            "lagged association with causal orientation under temporal priority; "
            "instantaneous confounding is not resolved; linear conditional independence"
        ),
    }
