# plan1 — counterfactual energy change, size_aware model

Rebound 2.4% / reduce 2.4% (max feasible over the WHOLE day, balance+SOC across 288 steps; q_max 2.41%), free window 11:00–14:00. nd = nd + Δ(total demand), renewables + curtailment fixed. Totals over 186 full test days; % change vs the actual (no-shift) dispatch. Energy in MWh (= Σ MW × 5/60 h).

## Drivers

| quantity | actual (MWh) | counterfactual (MWh) | Δ energy | % change |
| --- | --- | --- | --- | --- |
| total demand | 24,834,380 | 24,895,846 | +61,467 | +0.25% |
| net demand (demand − renewables − curtailment) | 15,707,631 | 15,769,098 | +61,467 | +0.39% |

## Per-source total energy change (full-day counterfactual)

| source | actual (MWh) | counterfactual (MWh) | Δ energy (MWh) | Δ % |
| --- | --- | --- | --- | --- |
| hydro | 1,317,140 | 1,661,246 | +344,105 | +26.13% |
| coal_brown | 16,312,465 | 15,351,698 | -960,767 | -5.89% |
| gas_steam | 34,975 | 139,960 | +104,986 | +300.18% |
| gas_ocgt | 370,477 | 633,302 | +262,824 | +70.94% |
| battery_charging | 569,128 | 353,576 | -215,551 | -37.87% |
| battery_discharging | 474,699 | 569,466 | +94,767 | +19.96% |

_Counterfactual = full-day shifted scenario (off-window dispatch × nd_after/nd_before + model-filled free window); the single total delta of the whole scenario vs actual._
