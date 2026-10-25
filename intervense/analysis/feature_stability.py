"""Construction stability: does a feature depend on how the dataset
was built rather than on the world?

A feature matrix is the output of a pipeline with construction
parameters: history depth, sample window, entity filtering,
normalization scope. A feature whose distribution or target
relationship moves when those parameters change is a construction
artifact: it looks informative under the construction it was tuned
on and collapses under the one production uses.

The check takes two matrices over the SAME rows (same entities, same
order) built under two constructions and measures, per feature:
  - distribution shift between constructions (PSI and KS)
  - when a target is provided, rank IC against the target in both
    constructions, plus the IC gap
A feature is flagged unstable when its PSI clears the threshold or
its IC flips or collapses across constructions.

Evidence tier: observational. The check compares constructions, it
does not intervene; flagged features are candidates for
depth-invariant reformulation, not proven defects.
"""
from __future__ import annotations

import numpy as np

from .drift import _ks_statistic, _psi


def _rank_ic(a: np.ndarray, b: np.ndarray) -> float:
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 20:
        return np.nan
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def construction_stability(X_a, X_b, feature_names, y=None,
                           psi_threshold: float = 0.25,
                           ic_collapse: float = 0.5) -> dict:
    """Compare two constructions of the same feature matrix.

    X_a and X_b must be row-aligned: the same entities in the same
    order, built under construction A and construction B. y, when
    given, is the shared target. Returns per-feature PSI, KS, rank IC
    per construction, IC gap, a stability verdict per feature, and an
    overall summary with the flagged features sorted by PSI.
    """
    A = np.asarray(X_a, dtype=float)
    B = np.asarray(X_b, dtype=float)
    if A.shape != B.shape:
        raise ValueError("X_a and X_b must have the same shape")
    if A.shape[1] != len(feature_names):
        raise ValueError("feature_names length does not match the matrices")
    y = None if y is None else np.asarray(y, dtype=float)

    rows = []
    for j, name in enumerate(feature_names):
        a, b = A[:, j], B[:, j]
        psi = _psi(a, b)
        ks = _ks_statistic(a[~np.isnan(a)], b[~np.isnan(b)]) if (~np.isnan(a)).any() and (~np.isnan(b)).any() else np.nan
        ic_a = _rank_ic(a, y) if y is not None else np.nan
        ic_b = _rank_ic(b, y) if y is not None else np.nan
        ic_gap = (ic_b - ic_a) if (np.isfinite(ic_a) and np.isfinite(ic_b)) else np.nan
        shifted = np.isfinite(psi) and psi > psi_threshold
        ic_unstable = (np.isfinite(ic_gap)
                       and (np.sign(ic_a) != np.sign(ic_b)
                            or abs(ic_gap) > ic_collapse * max(abs(ic_a), 1e-9)))
        rows.append({
            "feature": name,
            "psi": round(float(psi), 4) if np.isfinite(psi) else None,
            "ks": round(float(ks), 4) if np.isfinite(ks) else None,
            "ic_a": round(ic_a, 4) if np.isfinite(ic_a) else None,
            "ic_b": round(ic_b, 4) if np.isfinite(ic_b) else None,
            "ic_gap": round(ic_gap, 4) if np.isfinite(ic_gap) else None,
            "stable": bool(not shifted and not ic_unstable),
        })

    flagged = [r for r in rows if not r["stable"]]
    flagged.sort(key=lambda r: -(r["psi"] or 0.0))
    share = len(flagged) / max(len(rows), 1)
    verdict = ("construction_unstable" if share >= 0.1
               else "borderline" if flagged else "stable")
    return {
        "verdict": verdict,
        "n_features_total": len(rows),
        "n_features_unstable": len(flagged),
        "psi_threshold": psi_threshold,
        "features": rows,
        "flagged": flagged,
        "advice": (
            "flagged features change with the construction parameters; "
            "reformulate them to be invariant (fixed lookback, rank or "
            "log transforms) or validate them under the production "
            "construction before trusting any score they helped produce"
        ) if flagged else "all features invariant across the given constructions",
    }
