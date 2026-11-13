"""Layer-2 validation: pattern scan for the JELLY signature.

The postmortem reduces the incident to a causal signature with three
measurable conditions on any perpetual market:

  S1  Position stock dwarfs flow: open interest notional over daily
      notional volume is high, so unwinds cannot clear at current
      liquidity.
  S2  Price pressure is live: mark deviates from oracle, or funding
      is extreme, meaning one-sided positioning is already paying to
      hold.
  S3  The market is small: absolute daily volume is thin, so a
      modest attacker budget moves price.

This script scores every listed Hyperliquid perp against the
signature from one public read-only snapshot and reports the markets
matching the most conditions. It is defensive pattern scanning: it
predicts nothing, exploits nothing, and names markets only so a risk
review can check whether exposure invariants exist there.

Inputs: market_snapshot.json (public metaAndAssetCtxs snapshot).
Grading: observational snapshot, single point in time.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    d = json.load(open(os.path.join(HERE, "market_snapshot.json")))
    universe = d["ctxs"][0]["universe"]
    ctxs = d["ctxs"][1]
    names = [u["name"] for u in universe]
    assert len(names) == len(ctxs)
    return d["pulled_at"], list(zip(names, ctxs))


def score(pulled_at, markets):
    rows = []
    for name, c in markets:
        try:
            mark = float(c["markPx"])
            oracle = float(c["oraclePx"])
            oi_base = float(c["openInterest"])
            day_ntl = float(c["dayNtlVlm"])
            funding = float(c["funding"])
        except (ValueError, KeyError, TypeError):
            continue
        if mark <= 0 or day_ntl <= 0:
            continue
        oi_usd = oi_base * mark
        oi_over_volume = oi_usd / day_ntl
        mark_oracle_gap = abs(mark / oracle - 1.0) if oracle > 0 else 0.0
        # S3 thresholds: thin market under 2M USD daily notional
        thin = day_ntl < 2e6
        # S1: stock over flow above 0.5 means unwinding the book takes
        # more than half a day of all traded liquidity
        top_heavy = oi_over_volume > 0.5
        # S2: live pressure, mark gap above 0.5 percent or funding
        # above the 8-hour drift implied by 40 percent APR
        pressured = mark_oracle_gap > 0.005 or abs(funding) > 0.00045
        hits = sum((top_heavy, pressured, thin))
        rows.append({
            "market": name,
            "oi_usd_m": round(oi_usd / 1e6, 2),
            "day_ntl_m": round(day_ntl / 1e6, 2),
            "oi_over_volume": round(oi_over_volume, 2),
            "mark_oracle_gap_pct": round(mark_oracle_gap * 100, 3),
            "funding": funding,
            "S1_top_heavy": top_heavy,
            "S2_pressured": pressured,
            "S3_thin": thin,
            "signature_hits": hits,
        })
    rows.sort(key=lambda r: (-r["signature_hits"], -r["oi_over_volume"]))
    return rows


def main():
    pulled_at, markets = load()
    rows = score(pulled_at, markets)
    flagged = [r for r in rows if r["signature_hits"] >= 2]
    jelly_now = next((r for r in rows if r["market"] == "JELLY"), None)
    out = {
        "snapshot_utc": pulled_at,
        "n_markets": len(rows),
        "n_flagged_two_plus": len(flagged),
        "n_flagged_all_three": sum(1 for r in rows if r["signature_hits"] == 3),
        "top10": rows[:10],
        "jelly_today": jelly_now,
        "signature_definition": {
            "S1_top_heavy": "open interest notional / daily notional volume > 0.5",
            "S2_pressured": "|mark/oracle - 1| > 0.5% or |funding| > 0.00045 per 8h",
            "S3_thin": "daily notional volume < 2M USD",
        },
        "grading": "observational snapshot, single point in time; no prediction or trading",
    }
    path = os.path.join(HERE, "pattern_scan.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"markets scored: {len(rows)} | flagged (>=2 of 3): {len(flagged)} | all three: {out['n_flagged_all_three']}")
    print("top of the list:")
    for r in rows[:10]:
        print(f"  {r['market'][:22]:24s} hits={r['signature_hits']} "
              f"OI/vol={r['oi_over_volume']:7.2f} dayM={r['day_ntl_m']:8.2f} "
              f"gap%={r['mark_oracle_gap_pct']:6.3f}")
    if jelly_now:
        print("JELLY today:", {k: jelly_now[k] for k in ('oi_over_volume', 'day_ntl_m', 'signature_hits')})
    print("saved", path)


if __name__ == "__main__":
    main()
