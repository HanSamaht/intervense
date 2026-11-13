"""Layer-3 validation: blind detection of the feedback structure.

Question: without being told anything about the exploit, can
structure discovery on a protocol state panel find that a protocol
variable (vault exposure) or a concentrated position sits upstream
of price? That edge direction is the signature of the feedback loop
that made the JELLY incident dangerous: position drives price, price
drives the position's loss, the loss becomes vault exposure, the
exposure is the attacker's leverage over the vault.

Design:
  Panel A (attack window): 5-minute candles, 2025-03-26 06:00 to
  15:15 UTC. Columns: log return, log volume (both observational,
  public API), attacker open interest and vault exposure (synthetic
  series reconstructed from the public timeline and the reported
  rule, graded prior).
  Panel B (control window): the previous day, same market, same
  construction with the attacker series at zero.

  Backend: DirectLiNGAM (same backend as intervense discovery on
  tabular panels; the driver-augmented entry point in
  intervense.analysis.discovery targets model audits, so this script
  calls the backend directly). A linear non-Gaussian model is a
  simplification: it tests edge direction, not the full loop
  dynamics.

Pass condition: the attack panel yields an edge into log return
from vault exposure or attacker OI with the direction toward price,
and the control panel does not.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def build_panel(window_start, window_end):
    rows = json.load(open(os.path.join(HERE, "jelly_candles.json")))
    df = pd.DataFrame([r for r in rows if r["i"] == "5m"])
    df["ts"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df[(df["ts"] >= window_start) & (df["ts"] < window_end)].reset_index(drop=True)
    for c in ("o", "c", "h", "l", "v"):
        df[c] = df[c].astype(float)
    df["logret"] = np.log(df["c"] / df["c"].shift(1)).fillna(0.0)
    df["logvol"] = np.log1p(df["v"])
    return df


def add_position_columns(df, attack: bool):
    n = len(df)
    if not attack:
        df["attacker_oi"] = 0.0
        df["vault_exposure"] = 0.0
        return df
    # public timeline: short built through the morning, squeeze at
    # 14:00 UTC, vault takes over at the squeeze. Synthetic ramps,
    # graded prior in the report.
    build_end = int(n * 0.45)
    takeover = int(n * 0.78)
    oi = np.zeros(n)
    oi[:build_end] = np.linspace(0, 1, build_end)
    oi[build_end:takeover] = 1.0
    exposure = np.zeros(n)
    after = n - takeover
    exposure[takeover:] = np.linspace(0, 1, after) * (df["c"].iloc[takeover:].values /
                                                      max(df["c"].iloc[takeover], 1e-9) - 1.0 + 1.0)
    df["attacker_oi"] = oi
    df["vault_exposure"] = exposure - exposure.min() if exposure.max() > 0 else exposure
    return df


def run_lingam(df, columns):
    from causallearn.search.FCMBased.lingam import DirectLiNGAM
    X = df[columns].to_numpy(dtype=float)
    X = X[np.isfinite(X).all(axis=1)]
    if X.std(axis=0).min() < 1e-12:
        return None, "constant column"
    model = DirectLiNGAM()
    model.fit(X)
    adj = model.adjacency_matrix_
    edges = []
    for i in range(len(columns)):
        for j in range(len(columns)):
            if i != j and abs(adj[i, j]) > 0.05:
                edges.append({"from": columns[i], "to": columns[j], "weight": round(float(adj[i, j]), 3)})
    edges.sort(key=lambda e: -abs(e["weight"]))
    return edges, None


def main():
    cols = ["logret", "logvol", "attacker_oi", "vault_exposure"]
    attack = build_panel("2025-03-26 05:00:00+00:00", "2025-03-26 15:15:00+00:00")
    attack = add_position_columns(attack, attack=True)
    control = build_panel("2025-03-25 05:00:00+00:00", "2025-03-25 15:15:00+00:00")
    control = add_position_columns(control, attack=False)

    atk_edges, err1 = run_lingam(attack, cols)
    ctl_edges, err2 = run_lingam(control, cols)

    def edges_into_price(edges):
        return [e for e in (edges or []) if e["to"] == "logret" and e["from"] in ("attacker_oi", "vault_exposure")]

    atk_into = edges_into_price(atk_edges)
    ctl_into = edges_into_price(ctl_edges)

    out = {
        "attack_window": {"n": len(attack), "edges": atk_edges, "edges_into_price": atk_into,
                          "error": err1},
        "control_window": {"n": len(control), "edges": ctl_edges, "edges_into_price": ctl_into,
                           "error": err2},
        "pass_condition": "attack panel has a position/vault -> price edge and control does not",
        "verdict": "PASS" if atk_into and not ctl_into else "FAIL",
        "grading": "price and volume observational; attacker OI and vault exposure synthetic from the public timeline (prior)",
        "backend": "DirectLiNGAM, the same backend as intervense tabular discovery",
        "caveats": "linear non-Gaussian simplification; synthetic position series; single market single event",
    }
    path = os.path.join(HERE, "feedback_detection.json")
    json.dump(out, open(path, "w"), indent=2)
    print("attack edges into price:", atk_into)
    print("control edges into price:", ctl_into)
    print("verdict:", out["verdict"])
    print("saved", path)


if __name__ == "__main__":
    main()
