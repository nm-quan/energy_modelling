# VIC dispatch constraints — canon

Empirical max over the full history (`data/preprocessed/hist/5min/net_dispatch_totdem/table.parquet`, 500,832 rows, 2021-10-01 to 2026-07-05), rounded outward to 0.1 MW → 0 historical violations. Mirrored in `ml/check_caps.py`.

| target | cap (MW) | ramp down (MW/5min) | ramp up (MW/5min) |
| --- | --- | --- | --- |
| hydro | 2168.0 | −735.4 | +956.5 |
| coal_brown | 4895.8 | −1553.6 | +333.4 |
| gas_steam | 516.4 | −499.9 | +82.5 |
| gas_ocgt | 1748.6 | −363.9 | +400.3 |
| battery_charging | 1611.6 | −863.1 | +795.0 |
| battery_discharging | 1687.5 | −670.9 | +715.1 |

Lower bound 0 for all six. Battery SOC reservoir: 4735.75 MWh, η 0.834 (unchanged). coal_brown down = a single 2024-02-13 unit-trip, effectively non-binding.
