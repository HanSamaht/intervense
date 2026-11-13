"""Mechanism simulation with a do-operator for encoded protocols.

Why this module exists. Structure discovery on market data cannot see
variables that are deterministic functions of other variables: the
JELLY vault exposure leg was invisible to every purely data-driven
method we ran (static PC, LiNGAM, PCMCI), because it carries no
independent noise. Such legs live in the protocol's rulebook, not in
the residuals. This module takes the rulebook as code.

The caller encodes a mechanism as:
  params        numbers the rules depend on (caps, thresholds, sizes)
  variables     state entries initialized once
  transition    fn(state, t, exogenous, params) -> None (mutates state)
  invariants    list of fn(state) -> bool that must hold on every step

Three do-operations are supported:
  do_param      override a params entry for the whole run
  do_variable   clamp a state variable to a fixed value every step
  do_path       replace an exogenous path (for example scale the
                realized price path to probe the tail)

Outputs: the full state trajectory, the first violation of each
invariant with its step, and summary statistics. scan_params grids
do_param over a range and returns one row per setting, the engine
behind counterfactual cap scans.

Grading note for reports: mechanism encodings are manual evidence;
simulation outputs are prior statements about the encoded rule, not
observations of the world.
"""
from __future__ import annotations

import numpy as np


def _check_spec(spec):
    for key in ("params", "transition"):
        if key not in spec:
            raise ValueError(f"spec missing required key {key!r}")
    if not callable(spec["transition"]):
        raise ValueError("spec['transition'] must be callable")
    for inv in spec.get("invariants", []):
        if not callable(inv):
            raise ValueError("invariants must be callables")
    for name in spec.get("do_variable", {}):
        if name not in spec.get("variables", {}):
            raise ValueError(f"do_variable target {name!r} not in variables")


def simulate_mechanism(
    spec: dict,
    n_steps: int,
    exogenous: dict | None = None,
    do_param: dict | None = None,
    do_variable: dict | None = None,
    do_path: dict | None = None,
) -> dict:
    """Run the encoded mechanism under interventions.

    exogenous: dict of 1-D sequences, all at least n_steps long.
    do_param / do_variable / do_path: interventions applied from
    step 0 onward. Returns trajectory, invariant violations, and a
    copy of the effective params used.
    """
    _check_spec(spec)
    exogenous = exogenous or {}
    for name, path in exogenous.items():
        if len(path) < n_steps:
            raise ValueError(
                f"exogenous path {name!r} has {len(path)} steps, needs {n_steps}")

    params = dict(spec["params"])
    if do_param:
        unknown = [k for k in do_param if k not in params]
        if unknown:
            raise ValueError(f"do_param on unknown params: {unknown}")
        params.update(do_param)

    ex = {k: list(v) for k, v in exogenous.items()}
    if do_path:
        for k, v in do_path.items():
            if k not in ex:
                raise ValueError(f"do_path on unknown exogenous path {k!r}")
            if len(v) < n_steps:
                raise ValueError(f"do_path {k!r} too short")
            ex[k] = list(v)

    clamps = dict(spec.get("do_variable", {}))
    if do_variable:
        for k, v in do_variable.items():
            if k not in spec.get("variables", {}):
                raise ValueError(f"do_variable target {k!r} not in variables")
            clamps[k] = v

    state = dict(spec.get("variables", {}))
    transition = spec["transition"]
    invariants = spec.get("invariants", [])

    names = sorted(state.keys())
    trajectory = {n: [] for n in names}
    violations = []
    for t in range(n_steps):
        transition(state, t, ex, params)
        for k, v in clamps.items():
            state[k] = v
        for n in names:
            trajectory[n].append(state[n])
        for inv in invariants:
            label = getattr(inv, "label", inv.__name__)
            if not inv(state):
                violations.append({"invariant": label, "step": t})

    def summarize(name):
        arr = np.asarray(trajectory[name], dtype=float)
        if arr.size == 0 or not np.isfinite(arr).all():
            return {}
        return {"min": float(arr.min()), "max": float(arr.max()),
                "final": float(arr[-1])}

    return {
        "n_steps": n_steps,
        "effective_params": params,
        "clamped_variables": list(clamps.keys()),
        "trajectory": trajectory,
        "summary": {n: summarize(n) for n in names},
        "invariant_violations": violations,
        "n_violations": len(violations),
        "grading": "manual mechanism encoding; outputs are prior statements",
    }


def scan_params(
    spec: dict,
    param_name: str,
    values,
    n_steps: int,
    exogenous: dict | None = None,
    outcome: str = "n_violations",
) -> list:
    """Grid one parameter under do_param and collect outcomes."""
    rows = []
    for v in values:
        res = simulate_mechanism(spec, n_steps, exogenous=exogenous,
                                 do_param={param_name: v})
        if outcome == "n_violations":
            out = res["n_violations"]
        elif outcome.startswith("max:"):
            var = outcome.split(":", 1)[1]
            out = res["summary"].get(var, {}).get("max")
        elif outcome.startswith("final:"):
            var = outcome.split(":", 1)[1]
            out = res["summary"].get(var, {}).get("final")
        else:
            raise ValueError(f"unknown outcome {outcome!r}")
        rows.append({param_name: v, "outcome": out,
                     "n_violations": res["n_violations"],
                     "first_violation": res["invariant_violations"][0] if res["invariant_violations"] else None})
    return rows
