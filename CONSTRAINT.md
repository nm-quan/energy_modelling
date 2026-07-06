Version 1: Fixed Constraint

1. Battery must remain balanced at all times
2. No changes of energy can surpass ramp-rate

# Aggregate VIC constraints by target

Source: OpenElectricity /v4/facilities (capacity, operating units only), 365-day generation data (observed), AEMO registry (nameplate ramp). Pulled 2026-05-26. Full values in data/capacity_constraints.json.

# Power caps (MW)

| target     | data max 5min | facility cap |
| ---------- | ------------- | ------------ |
| hydro      | 1833.6        | 2402.5       |
| coal_brown | 4895.7        | 5095.0       |
| gas_steam  | 513.8         | 510.0        |
| gas_ocgt   | 1748.6        | 2157.0       |

gas_steam note: single unit (Newport), data max 513.8 slightly exceeds its 510 facility cap (runs above rating about 0.5 percent of the time).

# Demand cap (MW)

Source: historical 5-min demand_mw, full 12 months (2025-05-19 to 2026-05-18), 105,120 rows.

| quantity        | value (MW) |
| --------------- | ---------- |
| max (all data)  | 10783.7    |
| 99.9 pctile     | 8908.6     |
| 99 pctile       | 7814.7     |
| mean            | 4919.8     |
| max (train)     | 10783.7    |
| max (val)       | 7288.1     |
| max (test)      | 7771.4     |

demand note: max occurs 2026-01-27 18:00 (summer evening peak). The elastic load shift must never push reshaped demand above the historical max (10783.7 MW). At phi=0.2, sat_cap=0.5 the free-window fill peaks ~6400 MW, well under the cap, so the per-interval sat_cap (+50% growth) is the active limit, not the demand ceiling.

# Ramp rate (MW per 5min)

| target     | data ramp up | data ramp down | facility ramp |
| ---------- | ------------ | -------------- | ------------- |
| hydro      | 790.7        | -575.3         | 3250          |
| coal_brown | 268.4        | -566.8         | 5000          |
| gas_steam  | 58.9         | -499.9         | 100           |
| gas_ocgt   | 262.9        | -193.0         | 2185          |

ramp note: facility ramp is one symmetric rate that applies to both up and down (AEMO has no separate up/down). data ramp is always smaller than facility ramp, so the nameplate ramp rate is never the binding limit.

# Battery (15 operating bidirectional units)

| quantity        | data max 5min | facility |
| --------------- | ------------- | -------- |
| discharge       | 1687.4 MW     | 2260 MW  |
| charge          | 1611.5 MW     | 2210 MW  |
| state of charge | 285.5 MWh     | 4736 MWh |

state of charge note: only batteries that publish telemetry report SOC, so the data max is about 6 percent of installed storage.

# Verification results

Same 10,656 test rows (data/gru_predictions_5min.csv)

Real-world data baseline (test-set actuals):

| target              | n_over_cap | n_negative | min   | n_ramp_up | n_ramp_down |
| ------------------- | ---------- | ---------- | ----- | --------- | ----------- |
| hydro               | 0          | 4          | -6.39 | 0         | 0           |
| coal_brown          | 0          | 0          | 0.00  | 0         | 0           |
| gas_steam           | 0          | 0          | 0.00  | 0         | 0           |
| gas_ocgt            | 0          | 79         | -4.00 | 0         | 0           |
| battery_charging    | 0          | 0          | 0.14  | 0         | 0           |
| battery_discharging | 0          | 0          | 0.00  | 0         | 0           |
| total               | 0          | 83         |       | 0         | 0           |

GRU prediction violations:

| Check                                             | Result                                                                                                                                                                                 |
| ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Upper cap (data max)                              | 0 violations across all 6 targets                                                                                                                                                      |
| Negative predictions (lower bound < 0)            | 13,057 cells: hydro 970 (min -28.9), gas_steam 4,154 (min -10.9), gas_ocgt 4,251 (min -49.9), battery_charging 1,763 (min -27.9), battery_discharging 1,919 (min -21.7). coal_brown: 0 |
| Ramp (asymmetric 5-min data ramps)                | 2 violations, both gas_steam ramp-up, max excess +18.1 MW                                                                                                                              |
| Battery power caps                                | 0                                                                                                                                                                                      |
| Battery mutual exclusion                          | not a constraint at aggregate level                                                                                                                                                    |
| Battery SOC                                       | skipped                                                                                                                                                                                |
| Demand constraint (sum of dispatch == net_demand) | 9,553 violations (residual > 10 MW), max 617.84 MW                                                                                                                                     |

Negative predictions on targets that are non-negative in reality:

| target              | actual min | n_neg_pred | most negative pred |
| ------------------- | ---------- | ---------- | ------------------ |
| gas_steam           | 0.00       | 4,154      | -10.90             |
| battery_charging    | 0.14       | 1,763      | -27.87             |
| battery_discharging | 0.00       | 1,919      | -21.68             |

# Battery SOC feasibility

1. eta_rt is the round-trip efficiency, total discharged / total charged over the full year = 0.834.
2. Capacity = 4735.75 MWh.

Result (itransformer_totdem_nonneg, 10,656 test rows, capacity 4735.75 MWh, eta_rt 0.834):

| series | swing MWh | % of capacity | feasible | feasible SOC0 window MWh |
| ------ | --------- | ------------- | -------- | ------------------------ |
| actual | 4124.7    | 87.1          | yes      | [2087.4, 2698.4]         |
| pred   | 4047.0    | 85.5          | yes      | [2064.2, 2753.0]         |

Run: python script/check_caps.py --csv no_constraints/benchmarks/5min/itransformer_totdem_nonneg/predictions.csv --res 5min
