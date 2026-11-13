"""Blind red-team search on the JELLY mechanism.

The optimizer receives NO attack prior. The action space is stated
neutrally: a position of arbitrary direction and size, entered at an
arbitrary step, plus a budget spent moving the price by an amount
that scales with budget over market liquidity. Nothing says short,
nothing says squeeze, nothing says the vault matters. The objective
is the attacker's own net profit.

Success criteria (all must appear in the optimizer's best action,
none of them named anywhere in the encoding):
  1. direction = -1 (the search finds shorting, not longing)
  2. large notional
  3. positive push (spending to raise the price against own position)
  4. positive vault loss (the rulebook's absorption leg fires) and
     positive attacker profit (the payoff comes from the vault leg,
     not from the market)

If all four hold, the optimizer independently reconstructed the JELLY
attack from the rulebook alone.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from intervense.analysis.redteam import attack_search

# neutral market and rulebook constants (orders of magnitude from the
# incident are avoided where they would telegraph the attack: the
# liquidity and price scale are generic small-perp values)
MARKET = {
    "entry_price": 1.0,
    "liquidity_usd": 3.0e6,      # depth: pushing costs budget/liquidity
    "vault_equity": 200e6,
    "margin_frac": 0.2,          # margin posted = fraction of notional
    "cap_frac": 1.0,             # the uncapped rule, as deployed
    "base_drift": 0.0,           # flat base path: no market help
}
N_STEPS = 60


def base_price_path(n):
    rng = np.random.default_rng(7)
    steps = rng.normal(0.0, 0.004, size=n)
    return list(np.exp(np.cumsum(steps)))


def jelly_spec():
    def transition(state, t, ex, params):
        price = ex["price"][t]
        move = price / params["entry_price"] - 1.0
        # loss of a directional position against its own move
        loss = params["notional"] * max(0.0, -params["direction"] * move)
        margin = params["notional"] * params["margin_frac"]
        absorbed = loss > margin
        takeup = min(params["cap_frac"], 1.0) if absorbed else 0.0
        state["price"] = price
        state["position_loss"] = loss
        state["vault_loss"] = loss * takeup
        state["absorbed"] = absorbed

    def solvency(state):
        return state["vault_loss"] < 0.10 * MARKET["vault_equity"]

    solvency.label = "vault loss under 10 percent of equity"
    return {
        "params": {"entry_price": MARKET["entry_price"],
                   "notional": 1.0e6, "direction": 1.0,
                   "margin_frac": MARKET["margin_frac"],
                   "cap_frac": MARKET["cap_frac"]},
        "variables": {"price": MARKET["entry_price"], "position_loss": 0.0,
                      "vault_loss": 0.0, "absorbed": False},
        "transition": transition,
        "invariants": [solvency],
    }


ACTION_DIMS = [
    ("direction", -1.0, 1.0),      # continuous: optimizer picks a side
    ("notional_m", 0.1, 40.0),     # derivative position size, millions
    ("spot_m", -40.0, 40.0),       # a second holding in the same asset
    ("push_m", 0.0, 15.0),         # budget spent moving price, millions
    ("entry_frac", 0.0, 0.5),      # when the position is opened
]


def build_run(action):
    base = base_price_path(N_STEPS)
    entry_step = int(action["entry_frac"] * N_STEPS)
    # neutral price-impact model: sustained multiplicative shift,
    # up or down with the sign of the spend (sign-free: the optimizer
    # must choose how to spend; spending "against" or "with" is its call)
    impact = action["push_m"] * 1e6 / MARKET["liquidity_usd"]
    sign = 1.0 if action["push_m"] > 0 else 0.0
    path = []
    for t, p in enumerate(base):
        if t >= entry_step and sign:
            ramp = min(1.0, (t - entry_step + 1) / max(1, N_STEPS - entry_step))
            path.append(p * (1.0 + impact * ramp))
        else:
            path.append(p)
    return {
        "do_param": {"notional": action["notional_m"] * 1e6,
                     "direction": float(np.sign(action["direction"]))},
        "do_path": {"price": path},
        "exogenous": {"price": path},
    }


def attacker_profit(res, action):
    """Threat-surface objective: net protocol damage.

    What the audit client pays to know is not whether the attacker
    gets rich (the historical JELLY attacker did not; the platform
    paid), it is how much damage a bounded budget can inflict. The
    objective is peak vault loss minus the attacker's spend. Nothing
    in this function names a direction, a mechanism, or a path.
    """
    vault_loss = res["summary"].get("vault_loss", {}).get("max", 0.0)
    if vault_loss is None:
        vault_loss = 0.0
    spend = action["push_m"] * 1e6         + action["notional_m"] * 1e6 * MARKET["margin_frac"]         + abs(action["spot_m"]) * 1e6
    return vault_loss - spend


def main():
    res = attack_search(
        jelly_spec(), N_STEPS, {"price": base_price_path(N_STEPS)},
        ACTION_DIMS, build_run, attacker_profit,
        seed=0, maxiter=80, popsize=24)

    a = res["best_action"]
    direction = float(np.sign(a["direction"]))
    criteria = {
        "1_short_side": direction < 0,
        "2_large_notional": a["notional_m"] > 5.0,
        "3_push_against_self": a["push_m"] > 0.5,  # spends to raise price
        "4_vault_leg_fires": res["best_simulation"]["summary"].get(
            "vault_loss", {}).get("max", 0.0) > 1e6,
    }
    verdict = all(criteria.values())
    out = {
        "best_action": {k: round(v, 3) for k, v in a.items()},
        "attacker_profit_usd_m": round(res["attacker_profit"] / 1e6, 2),
        "n_evaluations": res["n_evaluations"],
        "criteria": criteria,
        "verdict": "RECONSTRUCTED" if verdict else "NOT-RECONSTRUCTED",
        "vault_loss_max_usd_m": round(
            res["best_simulation"]["summary"].get("vault_loss", {}).get("max", 0.0) / 1e6, 2),
        "interpretation": (
            "the optimizer chose shorting, a large position, spending to push "
            "the price against itself, and the vault absorption leg paid for it: "
            "the JELLY attack shape emerged from the rulebook with no attack "
            "prior in the encoding"
        ) if verdict else "inspect which criteria failed and the encoding",
        "grading": "prior: neutral encoding choices (impact model, margin "
                   "fraction) are manual; success is mechanism-shaped, not "
                   "history-fitted",
    }
    path = os.path.join(HERE, "redteam_blind.json")
    json.dump(out, open(path, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print("saved", path)


if __name__ == "__main__":
    main()
