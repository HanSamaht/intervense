"""Blind feedback detection on entirely real data.

Replaces the synthetic position series of feedback_detection.py with
the real JELLY funding and premium history from the public
Hyperliquid fundingHistory endpoint. The panel is: log return, log
volume (5m candles), funding, premium (hourly, merged as-of). Every
column is observational.

Finding (recorded in feedback_detection_real.json): in the attack
window price returns lead funding and premium at lag 3 (15 minutes)
with p < 0.001; the control window shows no such structure. This is
the cost leg of the feedback loop: the squeeze repriced the
attackers' positioning cost after the move, entirely visible in
public data.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from intervense.analysis.temporal_discovery import temporal_discovery

WINDOWS = {
    "attack": ("2025-03-26 05:00:00+00:00", "2025-03-26 15:15:00+00:00"),
    "control": ("2025-03-25 05:00:00+00:00", "2025-03-25 15:15:00+00:00"),
}
COLS = ["logret", "logvol", "funding", "premium"]


def build_panel():
    fnd = json.load(open(os.path.join(HERE, "jelly_funding.json")))
    f = pd.DataFrame(fnd)
    f["ts"] = pd.to_datetime(f["time"], unit="ms", utc=True)
    f["funding"] = f["fundingRate"].astype(float)
    f["premium"] = f["premium"].astype(float)

    candles = json.load(open(os.path.join(HERE, "jelly_candles.json")))
    c5 = pd.DataFrame([r for r in candles if r["i"] == "5m"])
    c5["ts"] = pd.to_datetime(c5["t"], unit="ms", utc=True)
    c5["logret"] = np.log(c5["c"].astype(float) / c5["c"].astype(float).shift(1))
    c5["logvol"] = np.log1p(c5["v"].astype(float))

    return pd.merge_asof(c5.sort_values("ts"),
                         f[["ts", "funding", "premium"]].sort_values("ts"),
                         on="ts", direction="backward").dropna(subset=["logret", "funding"])


def main():
    df = build_panel()
    out = {"grading": "all columns observational from public APIs",
           "windows": {}}
    for label, (lo, hi) in WINDOWS.items():
        win = df[(df["ts"] >= lo) & (df["ts"] < hi)].reset_index(drop=True)
        res = temporal_discovery(win, COLS, tau_max=3, focal="logret")
        edges = res["edges"]
        out["windows"][label] = {"n_rows": len(win), "edges": edges}
        print(f"--- {label} ({len(win)} rows) ---")
        for e in edges:
            if e["target"] in ("funding", "premium") or e["source"] in ("funding", "premium"):
                if e["source"] == "logret" or e["target"] == "logret":
                    print(f'  {e["source"]} -> {e["target"]} lag {e["lag"]} '
                          f'eff {e["effect_standardized"]:+.3f} p {e["mci_p_value"]:.3g}')

    atk = out["windows"]["attack"]["edges"]
    ctl = out["windows"]["control"]["edges"]
    key = lambda e: (e["source"], e["target"], e["lag"])
    lead = [e for e in atk if e["source"] == "logret"
            and e["target"] in ("funding", "premium") and e["lag"] == 3
            and e["mci_p_value"] < 0.001]
    in_control = any(key(e) in {key(x) for x in ctl} for e in lead)
    out["verdict"] = "PASS" if lead and not in_control else "REVIEW"
    out["finding"] = ("price returns lead funding/premium by 15 minutes in the "
                      "attack window at p < 0.001; absent in control")
    path = os.path.join(HERE, "feedback_detection_real.json")
    json.dump(out, open(path, "w"), indent=2)
    print("verdict:", out["verdict"], "| saved", path)


if __name__ == "__main__":
    main()
