"""Adversarial search over an encoded mechanism: automated red team.

The premise this module tests: a designer's blind spots live in the
parameter corners humans never inspect. An optimizer that only sees
a black-box map from action parameters to attacker profit has no
intuition about which corners are absurd, so it searches them all.

The caller provides:
  base_spec        a mechanism_sim spec (the protocol rulebook)
  n_steps          horizon
  base_exogenous   the exogenous paths before attacker influence
  action_dims      list of (name, low, high) boxes; the optimizer
                   searches inside them
  build_run(action_params) -> dict
                   adapter that turns action values into
                   simulate_mechanism arguments (params overrides,
                   modified exogenous paths, clamped variables)
  attacker_profit(result, action_params) -> float
                   the objective to MAXIMIZE; define profit, not
                   protocol damage: an attacker with no budget
                   constraint defining the search, damage-maximizing
                   solutions are unrealistic

The module runs differential evolution and returns the best action
found, its profit, the full simulation result, and the search
trace. Grading: outputs are prior statements about the encoded
rulebook; a found attack is a candidate vulnerability that still
needs a human to confirm the encoding is faithful.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import differential_evolution

from .mechanism_sim import simulate_mechanism


def attack_search(
    base_spec: dict,
    n_steps: int,
    base_exogenous: dict,
    action_dims: list,
    build_run,
    attacker_profit,
    seed: int = 0,
    maxiter: int = 60,
    popsize: int = 20,
) -> dict:
    """Maximize attacker_profit over the action box via differential
    evolution. Every evaluation is one full mechanism simulation."""
    names = [d[0] for d in action_dims]
    bounds = [(d[1], d[2]) for d in action_dims]
    history = []

    def objective(x):
        action = {n: float(v) for n, v in zip(names, x)}
        run_args = build_run(action)
        try:
            res = simulate_mechanism(base_spec, n_steps,
                                     exogenous=run_args.get("exogenous", base_exogenous),
                                     do_param=run_args.get("do_param"),
                                     do_variable=run_args.get("do_variable"),
                                     do_path=run_args.get("do_path"))
            profit = float(attacker_profit(res, action))
        except Exception:
            profit = -1e12  # infeasible encodings rank last
        history.append({"action": action, "profit": profit})
        return -profit  # scipy minimizes

    result = differential_evolution(
        objective, bounds, seed=seed, maxiter=maxiter, popsize=popsize,
        polish=True, init="sobol")

    best_action = {n: float(v) for n, v in zip(names, result.x)}
    run_args = build_run(best_action)
    try:
        best_sim = simulate_mechanism(
            base_spec, n_steps,
            exogenous=run_args.get("exogenous", base_exogenous),
            do_param=run_args.get("do_param"),
            do_variable=run_args.get("do_variable"),
            do_path=run_args.get("do_path"))
        best_summary = {"summary": best_sim["summary"],
                        "effective_params": best_sim["effective_params"],
                        "n_violations": best_sim["n_violations"]}
    except Exception as exc:
        best_summary = {"error": f"best action replay failed: {type(exc).__name__}"}
    return {
        "best_action": best_action,
        "attacker_profit": -float(result.fun),
        "n_evaluations": len(history),
        "converged": bool(result.success),
        "best_simulation": best_summary,
        "history": history,
        "grading": "prior: candidate attack on the encoded rulebook; "
                   "faithfulness of the encoding must be confirmed by hand",
    }
