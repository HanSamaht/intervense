# JELLY mechanism failure on Hyperliquid: a causal postmortem

Incident: 2025-03-26, JELLY perp squeeze against the HLP vault.
Data: public Hyperliquid candleSnapshot API, 2,356 one-minute candles
covering 2025-03-25 00:00 to 2025-03-26 15:15 UTC. Position figures
are the publicly reported ones recorded in `analysis.py:FACTS` and
are graded as prior evidence; all market-data results are
observational.

## What the mechanism allowed

The liquidation rule let the vault absorb a losing position whose
size was set by margin arithmetic alone, with no cap relative to the
market it was about to own a piece of. The attacker's concentrated
short, once under water, became a vault liability whose size grew
with the very price it helped push up: a feedback loop with no
invariant to break it.

## Measured facts from the public tape

- Realized squeeze multiple, low to high within the window: 6.2x
  (0.0083 to 0.0517). The peak printed at 2025-03-26 13:59 UTC.
- Reported attacker short notional: 6.1M USD; worst-case short loss
  at the peak along the realized path: 20.7M USD.
- Reported unrealized loss forced onto the vault: about 12M USD,
  which implies the rule admitted roughly 58 percent of the
  worst-case position.

## Analysis 1: event study on the panel of minutes

Engine: `intervense.analysis.panel_fixed_effects`, hour fixed
effects, cluster-robust errors clustered by hour. The intervention
flag is post-squeeze minutes after the onset read from the data
(largest five-minute mean range).

The standardized within-group coefficient is 0.283 (cluster SE
0.045, p < 0.001); on the raw log scale the same design with full
hour dummies gives a post-squeeze range increase of about 2 percent
per minute candle (see postmortem.json, raw_log_effect and
range_multiplier). The two numbers answer different questions: the
standardized one compares within-hour movement, the raw one states
the market-facing magnitude. Grading: observational, interventional
in the counterfactual sense only.

## Analysis 2: specification curve on the effect

Two hundred random control subsets drawn from the twenty-three hour
dummies plus log volume. The post-squeeze raw log-scale coefficient
stays positive in 100 percent of specifications, range 0.018 to
0.023 log points. The conclusion does not depend on which
controls an analyst happens to prefer. Grading: manual analysis over
observational data.

## Analysis 3: counterfactual exposure cap scan

A one-page model of the rule: the vault takes up min(cap, implied)
of the worst-case short loss. Cap 100 percent reproduces the
reported 12M USD vault loss. Caps of 2 to 50 percent of open
interest reduce the loss to 0.4M to 10.4M USD.

The scan's most important output is what it cannot certify. Under
the realized path every cap level survives against a 10 percent
vault-equity threshold, because the realized loss was about 5
percent of equity. The delisting decision was not a response to the
realized number; it was a response to the tail. With no cap, every
additional 1x of price added roughly another 6M USD to the vault's
liability along a path it had no mechanism to exit. A cap invariant
does not merely shrink the realized loss, it bounds the slope of the
loss in the price path, which is the quantity that forced the
intervention. Grading: prior, from reported figures.

## What the framework contributed and what it did not

Contributed: a fixed-effects engine with cluster-robust errors that
made the volatility regime shift quantitative, a specification curve
that showed the estimate is control-robust, and a counterfactual
parameter scan that separated the realized loss from the tail slope
the rule actually needed to bound.

Not contributed: the discovery of the bug. The loop was found by an
attacker, priced by the market, and closed by the operator. A
framework like this shortens the postmortem and disciplines the
parameter discussion; catching the flaw before deployment requires
encoding the mechanism and adversarially scanning it, which is the
mechanism-simulation module listed as future work in the main
documentation.

## Reproduce

```
python examples/jelly/analysis.py
```

Pulls nothing from the network: the candle snapshot is stored in
`jelly_candles.json`. Output lands in `postmortem.json`.
