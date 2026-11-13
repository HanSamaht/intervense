"""Temporal discovery tests: PCMCI lagged edges on synthetic SCMs.

Ground truth SCM: a is autoregressive, b responds to a with a
two-step lag plus its own autoregression, c is independent of both.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intervense.analysis.temporal_discovery import (
    HAS_TIGRAMITE,
    temporal_discovery,
)

pytestmark = pytest.mark.skipif(
    not HAS_TIGRAMITE, reason="tigramite is not installed"
)


@pytest.fixture(scope="module")
def lag_panel():
    rng = np.random.default_rng(1)
    n = 600
    x = np.zeros((n, 3))
    for t in range(5, n):
        x[t, 0] = 0.8 * x[t - 1, 0] + rng.normal(scale=0.3)
        x[t, 1] = 0.6 * x[t - 2, 0] + 0.5 * x[t - 1, 1] + rng.normal(scale=0.3)
        x[t, 2] = 0.7 * x[t - 1, 2] + rng.normal(scale=0.3)
    return pd.DataFrame(x, columns=["a", "b", "c"])


def _edges(res):
    return {(e["source"], e["target"], e["lag"]) for e in res["edges"]}


def test_finds_planted_lag_two_edge(lag_panel):
    res = temporal_discovery(lag_panel, ["a", "b", "c"], tau_max=4)
    assert ("a", "b", 2) in _edges(res)
    edge = [e for e in res["edges"] if (e["source"], e["target"], e["lag"]) == ("a", "b", 2)][0]
    assert abs(edge["effect_standardized"]) > 0.2
    assert edge["mci_p_value"] < 0.01


def test_independent_column_stays_isolated(lag_panel):
    res = temporal_discovery(lag_panel, ["a", "b", "c"], tau_max=4)
    edges = _edges(res)
    assert not any(s in ("a", "b") and t == "c" for s, t, _ in edges)
    assert not any(s == "c" and t in ("a", "b") for s, t, _ in edges)


def test_nan_rows_dropped_and_reported(lag_panel):
    df = lag_panel.copy()
    df.iloc[:20, 0] = np.nan
    res = temporal_discovery(df, ["a", "b", "c"], tau_max=4)
    assert res["n_samples"] == len(lag_panel) - 20
    assert any("20 non-finite rows" in n for n in res["reduction_notes"])
    assert ("a", "b", 2) in _edges(res)


def test_constant_column_dropped_and_named(lag_panel):
    df = lag_panel.copy()
    df["flat"] = 1.5
    res = temporal_discovery(df, ["a", "b", "c", "flat"], tau_max=4)
    assert "flat" not in res["columns"]
    assert any("flat" in n for n in res["reduction_notes"])


def test_dimension_guard_rejects_wide_panel(lag_panel):
    rng = np.random.default_rng(2)
    wide = pd.DataFrame(rng.normal(size=(200, 41)),
                        columns=[f"v{i}" for i in range(41)])
    with pytest.raises(ValueError, match="column guard"):
        temporal_discovery(wide, list(wide.columns), tau_max=2)


def test_focal_edges_survive_cap(lag_panel):
    rng = np.random.default_rng(3)
    df = lag_panel.copy()
    for k in range(6):
        df[f"z{k}"] = rng.normal(size=len(df))
    res = temporal_discovery(df, ["a", "b", "c"] + [f"z{k}" for k in range(6)],
                             tau_max=3, max_edges=2, focal="b")
    into_b = [e for e in res["edges"] if e["target"] == "b"]
    assert ("a", "b", 2) in {(e["source"], e["target"], e["lag"]) for e in into_b}


def test_missing_column_raises(lag_panel):
    with pytest.raises(ValueError, match="not in frame"):
        temporal_discovery(lag_panel, ["a", "nope"], tau_max=2)
