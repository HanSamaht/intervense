"""Positioning and liquidation data pipeline for causal monitoring.

What it collects, from the public Hyperliquid API only:
  - fundingHistory per market: funding rate and premium time series,
    the real positioning-imbalance witness (validated on the JELLY
    incident: price -> funding/premium lag-3 structure appears in
    the attack window and not in the control window, entirely from
    real data).
  - metaAndAssetCtxs snapshots: open interest, mark, oracle, daily
    volume for every market, appended to a growing local store for
    future panels.

Layout:
  examples/defi_risk/data/funding.parquet     long panel, one row per
      (market, ts): funding, premium
  examples/defi_risk/data/snapshots.parquet   long panel appended on
      every run: (market, ts, oi, mark, oracle, day_ntl)

Usage:
  python examples/defi_risk/pipeline.py backfill_funding [start_ms]
  python examples/defi_risk/pipeline.py snapshot

The snapshot subcommand is idempotent per day-hour: reruns within the
same hour replace that hour's rows instead of duplicating them.
Grading: all rows observational from a public API.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
API = "https://api.hyperliquid.xyz/info"


def _post(body):
    req = urllib.request.Request(API, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(4):
        try:
            return json.load(urllib.request.urlopen(req, timeout=30))
        except Exception:
            if attempt == 3:
                raise
            time.sleep(8)


def _universe():
    meta = _post({"type": "metaAndAssetCtxs"})
    names = [u["name"] for u in meta[0]["universe"]]
    return names, meta[1]


def backfill_funding(start_ms: int, sleep_s: float = 0.4):
    """Pull funding history for every listed market since start_ms."""
    os.makedirs(DATA, exist_ok=True)
    names, _ = _universe()
    rows = []
    failures = []
    for i, coin in enumerate(names):
        try:
            hist = _post({"type": "fundingHistory", "coin": coin,
                          "startTime": start_ms})
        except Exception:
            failures.append(coin)
            continue
        for h in hist:
            rows.append({"market": coin,
                         "ts": pd.to_datetime(h["time"], unit="ms", utc=True),
                         "funding": float(h["fundingRate"]),
                         "premium": float(h["premium"])})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(names)} markets, {len(rows):,} rows", flush=True)
        time.sleep(sleep_s)
    df = pd.DataFrame(rows)
    path = os.path.join(DATA, "funding.parquet")
    if os.path.exists(path):
        old = pd.read_parquet(path)
        df = pd.concat([old, df]).drop_duplicates(["market", "ts"]).sort_values(["market", "ts"])
    df.to_parquet(path, index=False)
    print(f"funding panel: {len(df):,} rows across {df.market.nunique()} markets -> {path}")
    if failures:
        print("failed markets:", failures)


def snapshot():
    """Append one metaAndAssetCtxs snapshot, idempotent per hour."""
    os.makedirs(DATA, exist_ok=True)
    names, ctxs = _universe()
    now = pd.Timestamp.utcnow().floor("h")
    rows = []
    for name, c in zip(names, ctxs):
        try:
            rows.append({
                "market": name, "ts": now,
                "oi": float(c["openInterest"]),
                "mark": float(c["markPx"]),
                "oracle": float(c["oraclePx"]),
                "day_ntl": float(c["dayNtlVlm"]),
                "funding_now": float(c["funding"]),
            })
        except (KeyError, ValueError, TypeError):
            continue
    df = pd.DataFrame(rows)
    path = os.path.join(DATA, "snapshots.parquet")
    if os.path.exists(path):
        old = pd.read_parquet(path)
        old = old[old["ts"] != now]
        df = pd.concat([old, df]).sort_values(["ts", "market"])
    df.to_parquet(path, index=False)
    print(f"snapshot {now}: {len(df)} markets stored ({len(pd.read_parquet(path))} total rows)")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or sys.argv[1] == "snapshot":
        snapshot()
    elif sys.argv[1] == "backfill_funding":
        start = int(sys.argv[2]) if len(sys.argv) > 2 else int(
            (time.time() - 60 * 86400) * 1000)
        backfill_funding(start)
    else:
        raise SystemExit("usage: pipeline.py [snapshot|backfill_funding start_ms]")
