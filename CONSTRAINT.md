# VIC dispatch constraints — canon

Two envelopes, for two different jobs. Both derive from the same table
(`data/preprocessed/hist/5min/.../table.parquet`, 500,832 rows, 2021-10-01 → 2026-07-05).

- **Caps** are the empirical max in both. A percentile cap would forbid output
  levels the fleet demonstrably reaches, which is a different and unjustified
  claim. Lower bound 0 for all six.
- **Ramps** differ: the max envelope for validating history, the p99.9 envelope
  for anything a model is trained or scored through.

---

## A. Validation envelope — empirical max (`ml/check_caps.py`)

Rounded outward to 0.1 MW, so 0 historical violations by construction. Use this
when the question is "is this real dispatch feasible" — never declare recorded
data infeasible.

| target | cap (MW) | ramp down (MW/5min) | ramp up (MW/5min) |
| --- | ---: | ---: | ---: |
| hydro | 2168.0 | −735.4 | +956.5 |
| coal_brown | 4895.8 | −1553.6 | +333.4 |
| gas_steam | 516.4 | −499.9 | +82.5 |
| gas_ocgt | 1748.6 | −363.9 | +400.3 |
| battery_charging | 1611.6 | −863.1 | +795.0 |
| battery_discharging | 1687.5 | −670.9 | +715.1 |

`coal_brown` down = a single 2024-02-13 unit trip. It is 8.6× the p99.9 and makes
the coal down-ramp effectively non-binding — which is why envelope B exists.

## B. Modelling envelope — p99.9 ramps (`planB/limits.py`)

Per-channel, per-direction 99.9th percentile of the historical 5-minute step
change. Caps unchanged. **This is what planB trains and evaluates through.**

| target | cap (MW) | ramp down (MW/5min) | ramp up (MW/5min) | down tightened |
| --- | ---: | ---: | ---: | ---: |
| hydro | 2168.0 | −336.5 | +377.7 | 2.2× |
| coal_brown | 4895.8 | −180.2 | +183.5 | **8.6×** |
| gas_steam | 516.4 | −54.8 | +45.8 | **9.1×** |
| gas_ocgt | 1748.6 | −112.8 | +119.3 | 3.2× |
| battery_charging | 1611.6 | −254.6 | +264.0 | 3.4× |
| battery_discharging | 1687.5 | −250.2 | +251.4 | 2.7× |

Battery SOC reservoir: 4735.75 MWh, η 0.834 (unchanged, and **not enforced** by
any current head — measured slack on 3h gaps, but not guaranteed).

### Why p99.9 and not p99

Both were measured by projecting the **truth** onto each envelope. What survives
is error no model can remove:

| envelope | real steps over | floor MAE | worst balance shortfall |
| --- | ---: | ---: | ---: |
| max | 0.00% | 0.00 MW | 0.0 MW |
| **p99.9** | **0.23%** | **0.21 MW** | **12.8 MW** |
| p99 | 2.42% | 2.63 MW | 189.3 MW |

The accuracy cost of p99 is small — 2.63 MW against a ~50 MW scale, so it is not
why models trail interpolation. The feasibility cost is the problem: at p99 the
balance target becomes unreachable inside the tightened box on ~0.1% of steps, so
*"balance exact by construction"* — the entire point of the constraint layer —
degrades to *"exact except where the envelope forbids it"*. p99.9 keeps that claim
for a floor of 0.21 MW.

### One envelope for imputation and counterfactual alike

An earlier plan was max for imputation and a percentile for counterfactuals, on
the grounds that max validates history while a percentile simulates behaviour.
The counterfactual evidence argued against splitting: at p99 the recursive head
put **16%** of a demand rise on coal, where `dispatch_study/` measures the real
grid at **43%**. p99 was over-correcting coal in the direction that was already
wrong. p99.9 is used everywhere.

### Verified under envelope B

- bridge feasibility `|pR − pL| ≤ 37·ρ`: **0/400** test 3h-gap windows fail; tightest slack 1,326 MW (gas_steam)
- real dispatch exceeds the envelope on **0.20%** of channel-steps, as a two-sided p99.9 implies
- both planB heads over 86,400 gap cells and 88,800 consecutive pairs: ramp overshoot 0.00 MW, below zero 0.00 MW, above cap 0.00 MW
- `hardnet` is idempotent to 1.3e-9 MW — the identity on an already-feasible input

Consequence to state wherever it matters: **real data violates envelope B on
0.20% of channel-steps by construction.** A historical actual showing a "ramp
violation" against it is expected, not a bug.
