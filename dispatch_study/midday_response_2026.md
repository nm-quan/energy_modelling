# Who serves extra demand at 11:00–14:00? — VIC, 2026

6,696 five-minute intervals, 2026-01-01 to 2026-07-05. Descriptive, no model.

Every cell is **MW of that channel per MW of demand increase** — the OLS slope of Δchannel on Δdemand at that horizon. Battery charging is signed negative, so a positive entry under *battery in* means charging **falls**, which is how a load supplies power.

## The state of the window

- mean price **$21/MWh**, negative on **43%** of intervals and under $10 on 56%
- mean demand 4,590 MW — the daily minimum
- renewables curtailed on **83%** of intervals, mean 624 MW, max 4,847 MW
- battery charging 427 MW mean against 17 MW discharging
- VIC exports 250 MW net on average

The 3-hour horizon is absent at midday because the window is itself 3 hours wide, so no pair of endpoints 3 hours apart fits inside it.

## A. Closed framing — the VIC fleet only

`demand_mw` already has net imports subtracted (lib/pipeline.py:27), so this is the world the models see and the signed slopes close to ~1.

| horizon | coal | gas OCGT | gas steam | hydro | battery out | battery in (load) | wind | solar | Σ signed | n pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 min | +0.106 | +0.002 | +0.000 | +0.020 | +0.047 | +0.453 | +0.244 | +0.145 | 1.016 | 6,510 |
| 30 min | +0.184 | +0.005 | +0.001 | +0.032 | +0.043 | +0.372 | +0.243 | +0.112 | 0.991 | 5,580 |
| 1 hour | +0.233 | +0.009 | +0.002 | +0.036 | +0.029 | +0.381 | +0.227 | +0.076 | 0.994 | 4,464 |

### A1. When renewables are being spilled (curtailment > 100 MW, 62% of the window)

| horizon | coal | gas OCGT | gas steam | hydro | battery out | battery in (load) | wind | solar | Σ signed | n pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 min | +0.084 | -0.000 | +0.000 | +0.005 | +0.032 | +0.402 | +0.327 | +0.175 | 1.025 | 4,041 |
| 30 min | +0.183 | +0.001 | +0.001 | +0.011 | +0.030 | +0.332 | +0.305 | +0.129 | 0.992 | 3,465 |
| 1 hour | +0.229 | +0.001 | +0.002 | +0.012 | +0.029 | +0.353 | +0.283 | +0.086 | 0.995 | 2,776 |

### A2. When nothing is being spilled (38% of the window)

| horizon | coal | gas OCGT | gas steam | hydro | battery out | battery in (load) | wind | solar | Σ signed | n pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 min | +0.154 | +0.007 | +0.000 | +0.053 | +0.080 | +0.565 | +0.060 | +0.078 | 0.998 | 2,469 |
| 30 min | +0.190 | +0.014 | +0.002 | +0.086 | +0.072 | +0.462 | +0.092 | +0.069 | 0.987 | 2,115 |
| 1 hour | +0.247 | +0.030 | +0.002 | +0.103 | +0.026 | +0.435 | +0.096 | +0.053 | 0.993 | 1,688 |

## B. Open framing — VIC consumer demand, interconnector free to move

Regressor is `demand_mw + net_import_mw`, the market's VIC demand.

| horizon | coal | gas OCGT | gas steam | hydro | battery out | battery in (load) | wind | solar | net import | Σ signed | n pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 min | +0.147 | +0.003 | +0.001 | +0.006 | +0.024 | +0.083 | -0.036 | -0.003 | +0.662 | 0.886 | 6,508 |
| 30 min | +0.200 | +0.005 | +0.004 | +0.022 | +0.027 | +0.286 | +0.075 | +0.019 | +0.304 | 0.942 | 5,579 |
| 1 hour | +0.237 | +0.009 | +0.005 | +0.029 | +0.029 | +0.310 | +0.067 | +0.009 | +0.275 | 0.969 | 4,463 |

## C. Contrast — same closed regression at the 17:00–21:00 evening peak

| horizon | coal | gas OCGT | gas steam | hydro | battery out | battery in (load) | wind | solar | Σ signed | n pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 min | +0.100 | +0.019 | +0.002 | +0.181 | +0.504 | +0.020 | +0.187 | +0.007 | 1.021 | 8,742 |
| 30 min | +0.227 | +0.040 | +0.006 | +0.269 | +0.322 | +0.016 | +0.277 | -0.138 | 1.019 | 7,812 |
| 1 hour | +0.291 | +0.038 | +0.006 | +0.277 | +0.301 | +0.012 | +0.320 | -0.217 | 1.029 | 6,696 |
| 3 hours | +0.383 | +0.037 | +0.014 | +0.251 | +0.211 | +0.010 | +0.365 | -0.239 | 1.033 | 2,232 |

## D. Mechanism — what happens to curtailment itself

Slope of Δcurtailment on Δdemand, closed framing. Negative means the extra demand is served by spilling less.

| condition | horizon | wind curtailed | solar curtailed | total |
| --- | --- | ---: | ---: | ---: |
| all midday | 5 min | -0.205 | -0.111 | -0.317 |
| all midday | 30 min | -0.175 | -0.087 | -0.262 |
| all midday | 1 hour | -0.141 | -0.077 | -0.219 |
| spilling | 5 min | -0.290 | -0.155 | -0.445 |
| spilling | 30 min | -0.244 | -0.117 | -0.360 |
| spilling | 1 hour | -0.203 | -0.104 | -0.307 |

