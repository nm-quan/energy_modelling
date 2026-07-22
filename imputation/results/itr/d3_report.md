# D3 — counterfactual change report, FixedPercentageShift(q=3.699%)

185 test days, whole-day blackout imputation, Option A (no pinned dispatch anywhere), guarantee map = free-endpoint Π on C2 posthoc Π(C0).

| fuel | ΔE free-window (MWh) | ΔE off-window (MWh) | peak base→shift (MW) |
| --- | --- | --- | --- |
| hydro | +147215 | -305779 | 1834 → 1834 |
| coal_brown | +536082 | -333580 | 4896 → 4896 |
| gas_steam | +48361 | -26183 | 514 → 514 |
| gas_ocgt | +66903 | -71652 | 1283 → 1211 |
| battery_charging | -60462 | +76908 | 521 → 838 |
| battery_discharging | +53991 | -4755 | 365 → 674 |

**Violations (must be 0):** bal>1MW steps 2, ramp cells 0, neg cells 0, SOC days 0.

**Consistency (Σ SIGN·ΔE vs shifted net-demand energy):**
free-window: dispatch +913014 MWh vs demand +913014 MWh (gap 0);  off-window: -818858 vs -818877 MWh (gap 20).
