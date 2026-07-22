# Feasible counterfactual (q=3.699%) — iTransformer T4_rayen + RAYEN(soc=True)

Span: 2026-05-05..2026-05-08 (highest-median-demand 4 consecutive test days with real wind/solar/curtailment coverage; the last365 renewables extract ends 2026-05-18, so the usual Jun 2-5 D2/D3 span has no curtailment data).

Protocol: whole-day blackout imputation, drivers overridden with FixedPercentageShift(q=3.699%) (off-window demand cut, 11:00-14:00 rebound, price->0 in the window, renewables fixed so dnd = ddemand). Option A: no dispatch outside the day is pinned. Guarantee map = rayen_traj_project(soc=True) with the anchor overridden to the Option-A projected reference (P_ref = actual dispatch free-endpoint-projected onto the shifted set; actual dispatch for the base run) and pins = the anchor's own endpoints -- the default interp anchor stalls at the ~2000 MW one-step nd discontinuity at the free-window edges (total 1-step ramp capacity is 2891 MW up / 3369 down). Base and counterfactual differ only in the drivers and their anchors.

Violations, base run (4 days x 288 steps): {'bal>1MW': 0, 'ramp': 0, 'neg': 0, 'soc_over': 0}
Violations, counterfactual run:            {'bal>1MW': 0, 'ramp': 0, 'neg': 0, 'soc_over': 0}

| fuel | dE free-window (MWh) | dE off-window (MWh) |
| --- | --- | --- |
| hydro | +3937 | -2920 |
| coal_brown | +3825 | -6930 |
| gas_steam | +3546 | +0 |
| gas_ocgt | +3844 | -513 |
| battery_charging | -3491 | +6932 |
| battery_discharging | +3937 | -2734 |

Consistency: SIGN.dE free +22579 MWh vs demand shift +22579 MWh; off -20030 vs -20030 MWh.

Figures: stack_4day.png (all-energy stack incl. wind, solar, curtailment hatched, battery charging negative, demand dashed); line_<fuel>.png x6 (model-output lines, base vs counterfactual).
