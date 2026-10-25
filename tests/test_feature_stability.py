"""Construction stability tests: synthetic features with known
dependence on the dataset construction parameters."""
import sys, os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from intervense.analysis.feature_stability import construction_stability


def test_flags_distribution_shifting_feature():
    rng = np.random.default_rng(0)
    n = 600
    y = rng.normal(size=n)
    base = rng.normal(size=n)
    stable = base + 0.1 * rng.normal(size=n)
    # counter feature grows with the construction depth parameter
    depth_a, depth_b = 10.0, 90.0
    counter_a = base + rng.normal(0, 0.05, n) + depth_a
    counter_b = base + rng.normal(0, 0.05, n) + depth_b
    X_a = np.column_stack([stable, counter_a])
    X_b = np.column_stack([stable, counter_b])
    r = construction_stability(X_a, X_b, ["stable_feat", "depth_counter"], y=y)
    names = {f["feature"]: f for f in r["features"]}
    assert names["stable_feat"]["stable"] is True
    assert names["depth_counter"]["stable"] is False
    assert names["depth_counter"]["psi"] > 0.25
    assert r["verdict"] in ("construction_unstable", "borderline")
    assert r["flagged"][0]["feature"] == "depth_counter"


def test_flags_ic_flip_with_stable_distribution():
    rng = np.random.default_rng(1)
    n = 800
    y = rng.normal(size=n)
    # same marginal distribution in both constructions, opposite signal
    sig = rng.normal(size=n)
    flip_a = sig
    flip_b = -sig
    noise = np.column_stack([rng.normal(size=n) for _ in range(3)])
    X_a = np.column_stack([flip_a, noise])
    X_b = np.column_stack([flip_b, noise])
    X_b[:, 1:] = X_b[:, 1:]  # noise identical across constructions
    r = construction_stability(X_a, X_b, ["flip", "n0", "n1", "n2"], y=y)
    names = {f["feature"]: f for f in r["features"]}
    assert names["flip"]["ic_a"] * names["flip"]["ic_b"] < 0
    assert names["flip"]["stable"] is False
    assert all(names[f"n{i}"]["psi"] == 0.0 or names[f"n{i}"]["psi"] is None
               for i in range(3))


def test_shape_mismatch_raises():
    rng = np.random.default_rng(2)
    with __import__("pytest").raises(ValueError):
        construction_stability(rng.normal(size=(50, 2)), rng.normal(size=(60, 2)), ["a", "b"])
