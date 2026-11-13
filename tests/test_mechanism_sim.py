"""Mechanism simulation tests: do-semantics, invariants, scans.

Synthetic mechanism: a leveraged pool holds an asset whose price
path is exogenous. A rule absorbs a losing position into the pool
up to an exposure cap fraction. Solvency invariant: pool loss below
ten percent of equity.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from intervense.analysis.mechanism_sim import scan_params, simulate_mechanism


def make_spec(entry=1.0, notional=100.0, margin=5.0, cap_frac=1.0, equity=1000.0):
    def transition(state, t, ex, params):
        price = ex["price"][t]
        loss = params["notional"] * max(0.0, price / params["entry"] - 1.0)
        absorbed = loss > params["margin"]
        takeup = min(params["cap_frac"], 1.0) if absorbed else 0.0
        state["price"] = price
        state["short_loss"] = loss
        state["vault_loss"] = loss * takeup
        state["insolvent_position"] = absorbed

    def solvency(state):
        return state["vault_loss"] < 0.10 * equity

    solvency.label = "pool solvency under 10 percent equity"
    return {
        "params": {"entry": entry, "notional": notional,
                   "margin": margin, "cap_frac": cap_frac},
        "variables": {"price": entry, "short_loss": 0.0,
                      "vault_loss": 0.0, "insolvent_position": False},
        "transition": transition,
        "invariants": [solvency],
    }


def ramp_path(peak=2.0, n=10):
    return list(np.linspace(1.0, peak, n))


def test_loss_absorption_without_cap():
    spec = make_spec()
    res = simulate_mechanism(spec, n_steps=10, exogenous={"price": ramp_path(1.9)})
    # peak loss 90 stays under the 10 percent equity bound (100)
    assert res["summary"]["short_loss"]["max"] == pytest.approx(90.0)
    assert res["summary"]["vault_loss"]["max"] == pytest.approx(90.0)
    assert res["n_violations"] == 0


def test_do_param_cap_cuts_vault_loss():
    spec = make_spec()
    full = simulate_mechanism(spec, 10, {"price": ramp_path(2.0)})
    capped = simulate_mechanism(spec, 10, {"price": ramp_path(2.0)},
                                do_param={"cap_frac": 0.05})
    assert capped["summary"]["vault_loss"]["max"] == pytest.approx(
        0.05 * full["summary"]["short_loss"]["max"], rel=0.01)
    assert capped["effective_params"]["cap_frac"] == 0.05


def test_do_variable_clamps_state():
    spec = make_spec()
    res = simulate_mechanism(spec, 10, {"price": ramp_path(2.0)},
                             do_variable={"vault_loss": 0.0})
    assert set(res["trajectory"]["vault_loss"]) == {0.0}
    assert res["n_violations"] == 0


def test_do_path_probes_the_tail():
    spec = make_spec()
    base = simulate_mechanism(spec, 10, {"price": ramp_path(1.5)})
    tail = simulate_mechanism(spec, 10, {"price": ramp_path(4.0)})
    assert tail["summary"]["short_loss"]["max"] > base["summary"]["short_loss"]["max"]
    # a 4x path with no cap breaches the 10 percent solvency bound
    assert tail["n_violations"] > 0
    assert tail["invariant_violations"][0]["invariant"] == "pool solvency under 10 percent equity"


def test_invariant_violation_records_step():
    spec = make_spec()
    res = simulate_mechanism(spec, 10, {"price": ramp_path(20.0)})
    assert res["n_violations"] >= 1
    assert res["invariant_violations"][0]["step"] >= 0


def test_margin_gate_blocks_small_losses():
    spec = make_spec(margin=50.0)
    res = simulate_mechanism(spec, 10, {"price": ramp_path(1.3)})
    # max short loss 30 < margin 50: nothing is absorbed
    assert res["summary"]["vault_loss"]["max"] == 0.0


def test_scan_params_grid():
    spec = make_spec()
    rows = scan_params(spec, "cap_frac", [0.02, 0.05, 0.2, 1.0], 10,
                       {"price": ramp_path(20.0)}, outcome="max:vault_loss")
    assert [r["cap_frac"] for r in rows] == [0.02, 0.05, 0.2, 1.0]
    losses = [r["outcome"] for r in rows]
    assert losses == sorted(losses)
    assert rows[-1]["n_violations"] > 0


def test_unknown_do_param_rejected():
    spec = make_spec()
    with pytest.raises(ValueError, match="unknown params"):
        simulate_mechanism(spec, 2, {"price": [1.0, 1.0]},
                           do_param={"nope": 1.0})


def test_short_exogenous_rejected():
    spec = make_spec()
    with pytest.raises(ValueError, match="needs"):
        simulate_mechanism(spec, 5, {"price": [1.0, 1.0]})
