"""Structure discovery tests: causal-learn backends (pc, ges, lingam).

Uses a synthetic chain with known ground truth: a drives b drives c,
plus an independent noise column. Discovery should place edges among
the chain nodes and leave the noise column isolated.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intervense.analysis import build_causal_graph, discovery_edges
from intervense.analysis.discovery import HAS_CAUSAL_LEARN
from intervense.demos import train_finance_demo
from intervense.drivers import detect_driver

pytestmark = pytest.mark.skipif(
    not HAS_CAUSAL_LEARN, reason="causal-learn is not installed"
)

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def chain_driver():
    rng = np.random.default_rng(1)
    n = 400
    a = rng.normal(size=n)
    b = 0.9 * a + 0.4 * rng.normal(size=n)
    c = 0.9 * b + 0.4 * rng.normal(size=n)
    noise = rng.normal(size=n)
    X = np.column_stack([a, b, c, noise])
    y = (b > np.median(b)).astype(int)

    model = LogisticRegression().fit(X, y)
    return detect_driver(model, ["a", "b", "c", "noise"], X, name="chain")


def test_pc_discovers_chain_edges(chain_driver):
    edges = discovery_edges(chain_driver, method="pc", alpha=0.05)
    assert len(edges) > 0
    assert all(e.kind == "discovered" for e in edges)
    pairs = {(e.source, e.target) for e in edges}
    chain_pairs = {p for p in pairs if "noise" not in p and "prediction" not in p}
    assert chain_pairs, f"expected chain edges, got {sorted(pairs)}"
    # the isolated noise column must not gain structural edges
    assert not any("noise" in p for p in pairs)


def test_ges_discovers_edges(chain_driver):
    edges = discovery_edges(chain_driver, method="ges")
    assert len(edges) > 0
    assert all(e.kind == "discovered" for e in edges)


def test_lingam_returns_weighted_dag(chain_driver):
    edges = discovery_edges(chain_driver, method="lingam")
    assert len(edges) > 0
    weighted = [e for e in edges if e.effect != 1.0]
    assert weighted, "lingam should return non-trivial weights"


def test_unknown_method_rejected(chain_driver):
    with pytest.raises(ValueError, match="unknown discovery method"):
        discovery_edges(chain_driver, method="notamethod")


def test_build_causal_graph_with_discovery(chain_driver):
    anchor = np.zeros(4)
    g = build_causal_graph(chain_driver, anchor, discovery="pc")
    kinds = {e.kind for e in g.edges}
    assert "direct" in kinds
    assert "discovered" in kinds
    assert "pc_discovery" in g.method


def test_build_causal_graph_without_discovery_unchanged(chain_driver):
    anchor = np.zeros(4)
    g = build_causal_graph(chain_driver, anchor)
    assert all(e.kind != "discovered" for e in g.edges)


def test_session_causal_graph_discovery():
    model, feats, bg, _ = train_finance_demo()
    driver = detect_driver(model, feats, bg, name="fin_discovery")
    from intervense.core import MicSession

    session = MicSession(driver)
    out = session.get_causal_graph(discovery="pc")
    assert "discovered_edges" in out
    assert out["method"].endswith("pc_discovery")


def test_discovery_drops_constant_columns():
    """Zero variance one-hot columns must be dropped or PC never terminates."""
    rng = np.random.default_rng(5)
    n = 300
    a = rng.normal(size=n)
    b = 0.8 * a + rng.normal(0, 0.4, size=n)
    y = (0.5 * a + b + rng.normal(0, 0.4, size=n) > 0).astype(int)
    X = np.column_stack([a, b, np.zeros(n)])  # third column constant
    model = LogisticRegression().fit(X, y)
    driver = detect_driver(model, ["a", "b", "dead_col"], X, name="const_col")

    edges = discovery_edges(driver, method="pc")
    assert len(edges) > 0
    assert not any("dead_col" in (e.source, e.target) for e in edges)


def test_discovery_all_constant_rejected():
    rng = np.random.default_rng(5)
    X = np.column_stack([np.zeros(100), np.ones(100)])
    model = LogisticRegression().fit(np.column_stack([np.zeros(100), np.ones(100),
                                                      rng.normal(size=100)]),
                                     rng.integers(0, 2, size=100))
    driver = detect_driver(model, ["z1", "z2", "r"], np.column_stack(
        [np.zeros(100), np.ones(100), rng.normal(size=100)]), name="dead")
    with pytest.raises(ValueError, match="usable columns"):
        discovery_edges(driver, method="pc")


class _NaNChainDriver:
    """Reuse the chain driver but poison a few background rows with NaN,
    the shape real panel data arrives in."""

    def __init__(self, base):
        self._base = base
        bg = base.background.copy()
        bg[:20, 0] = np.nan
        bg[5:10, 2] = np.inf
        self._poisoned = bg

    def __getattr__(self, name):
        return getattr(self._base, name)

    @property
    def background(self):
        return self._poisoned


def test_discovery_survives_nan_rows(chain_driver):
    wrapped = _NaNChainDriver(chain_driver)
    edges = discovery_edges(wrapped, method="pc")
    assert edges, "discovery returned nothing on NaN-bearing data"
    assert all(np.isfinite(e.effect) for e in edges)
    assert any("20 rows dropped" in e.detail for e in edges), (
        "the NaN reduction note must be reported in edge details")


def test_pc_effects_are_measured_not_placeholder(chain_driver):
    edges = discovery_edges(chain_driver, method="pc")
    assert edges
    values = [e.effect for e in edges]
    assert any(v != 1.0 for v in values), (
        "PC edges must carry regression-estimated effects, not the 1.0 placeholder")


def test_ges_effects_are_measured_not_placeholder(chain_driver):
    edges = discovery_edges(chain_driver, method="ges")
    assert edges
    assert any(e.effect != 1.0 for e in edges), (
        "GES edges must carry regression-estimated effects, not the 1.0 placeholder")


def test_dimension_guard_filters_wide_background():
    rng = np.random.default_rng(3)
    n = 300
    signal = rng.normal(size=n)
    filler = rng.normal(size=(n, 80))
    X = np.column_stack([signal, filler])
    y = (signal > 0).astype(int)
    model = LogisticRegression().fit(X, y)
    names = ["signal"] + [f"f{i}" for i in range(80)]
    driver = detect_driver(model, names, X, name="wide")
    edges = discovery_edges(driver, method="pc", max_features=20)
    assert edges
    assert any("dimension guard" in e.detail for e in edges), (
        "the guard note must name the reduction")
    used = {e.source for e in edges} | {e.target for e in edges}
    assert "signal" in used, "the target-correlated feature must survive the guard"
