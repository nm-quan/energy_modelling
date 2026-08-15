     # No-model counterfactual fill (NEM-style)

10 days 2026-06-01 .. 2026-06-10, window 11:00–14:00, rebound 2.4% / reduce 2.4%. No neural network. Every rule is measured in `dispatch_study/`.

## Curtailment — filled, then partially released

Curtailment inside the gap is UNKNOWN, so it is filled by interpolating between its own pinned boundaries, exactly like dispatch. Nothing inside the window is read from the actual. The base-case fill is then scored against the truth, and the counterfactual releases only the part priced below zero.

| | wind | solar | total |
| --- | ---: | ---: | ---: |
| ACTUAL in window (MWh) | 9,088 | 1,605 | 10,693 |
| base-case FILL (MWh) | 10,334 | 2,213 | 12,546 |
| fill MAE vs actual (MW) | 84.5 | 31.9 | 58.2 |
| counterfactual (MWh) | 6,309 | 1,597 | 7,906 |
| released by the rebound | 4,024 | 616 | 4,640 |
| **retained** | **6,309** | **1,597** | **63% of the fill** |

The retained part is curtailment at non-negative prices — network congestion and system-strength limits, which extra Melbourne load does not relieve. Measured over 2022–2026 that is 15.3% of all curtailment MWh.

## Dispatch response (MWh in the window)

| channel | no-model fill | share | measured 3h grid response |
| --- | ---: | ---: | ---: |
| hydro | +10,549 | 30.5% | 29.3% |
| coal | +11,441 | 33.1% | 43.4% |
| gas steam | +1,302 | 3.8% | 3.7% |
| gas OCGT | +3,380 | 9.8% | 9.7% |
| battery in (load) | -5,196 | 15.0% | 6.6% |
| battery out | +2,722 | 7.9% | 7.4% |
| **Σ signed** | +34,590 | | |
| _demand rise_ | _+39,230_ | | |
| _met by released curtailment_ | _+4,640_ | | |

## Feasibility

balance >1 MW: 0 | ramp: 0 | negative: 0 | unserved energy: 0 MWh

**Isolated-Victoria closure.** No interconnector, so local supply must equal local demand at every step:

```
SIGN.dispatch + wind_delivered + solar_delivered  =  demand_local
worst residual over all 360 window steps: 3.638e-12 MW
```

`demand_local` is defined as the local fleet's output rather than taken from AEMO's `demand_mw`, which the fleet exceeds by 196.7 MW on average (4.2% of demand, up to 2,508 MW). That surplus is export; with export forbidden it has nowhere to go, so it would otherwise sit in every figure as an unexplained gap. Curtailment is still drawn above the demand line — it was available and was not delivered.

