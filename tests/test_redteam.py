"""Red-team search tests.

Tiny mechanism: a vault absorbs a directional loss above a margin,
uncapped. The optimizer must find that the damage-maximizing action
under the threat objective is a large adverse-side position plus a
push against it.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intervense.analysis.redteam import attack_search


def spec():
    def transition(state, t, ex, params):
        price = ex["price"][t]
        move = price / 1.0 - 1.0
        loss = params["notional"] * max(0.0, -params["direction"] * move)
        margin = params["notional"] * 0.2
        absorbed = loss > margin
        state["price"] = price
        state["vault_loss"] = loss if absorbed else 0.0

    return {
        "params": {"notional": 1e6, "direction": 1.0},
        "variables": {"price": 1.0, "vault_loss": 0.0},
        "transition": transition,
        "invariants": [],
    }


N = 20


def base_path():
    rng = np.random.default_rng(3)
    return list(1.0 + 0.002 * np.cumsum(rng.normal(size=N)))


def build_run(action):
    base = base_path()
    entry = int(action["entry_frac"] * N)
    impact = action["push_m"] * 1e6 / 3e6
    path = [p * (1.0 + impact * min(1.0, max(0, t - entry) / (N - entry)))
            for t, p in enumerate(base)]
    return {"do_param": {"direction": float(np.sign(action["direction"])),
                         "notional": action["notional_m"] * 1e6},
            "exogenous": {"price": path}}


def threat(res, action):
    vl = res["summary"].get("vault_loss", {}).get("max", 0.0) or 0.0
    return vl - action["push_m"] * 1e6 - action["notional_m"] * 1e6 * 0.2


@pytest.mark.slow
def test_search_finds_adverse_side_and_push():
    res = attack_search(spec(), N, {"price": base_path()},
                        [("direction", -1.0, 1.0), ("notional_m", 0.5, 10.0),
                         ("push_m", 0.0, 5.0), ("entry_frac", 0.0, 0.5)],
                        build_run, threat, seed=0, maxiter=30, popsize=12)
    a = res["best_action"]
    assert np.sign(a["direction"]) < 0, "optimizer should land on the adverse side"
    assert a["push_m"] > 0.5, "spending to move price against the position"
    assert res["best_simulation"]["summary"]["vault_loss"]["max"] > 1e6


def test_infeasible_evaluations_do_not_crash_search():
    def bad_build(action):
        return {"exogenous": {"price": [1.0]}}  # too short: simulate raises

    res = attack_search(spec(), 5, {"price": [1.0] * 5},
                        [("x", 0.0, 1.0)],
                        bad_build, lambda r, a: 0.0,
                        seed=0, maxiter=3, popsize=4)
    assert res["n_evaluations"] > 0
    assert all(h["profit"] == -1e12 for h in res["history"])
