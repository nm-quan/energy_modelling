# How the Victorian grid actually dispatches

Descriptive study of the real fleet, 2022-01-01 to 2026-07-05 (474,336 five-minute
intervals, local AEST). **No model anywhere in here.** These numbers are the
empirical target that any counterfactual or imputation output should be judged
against.

| file | what it is |
| --- | --- |
| `profiles.py` → `monthly_2025.png`, `monthly_2026.png` | 12 panels, the average day of each calendar month, stacked by fuel |
| `profiles.py` → `weekly_2025.png`, `weekly_2026.png` | 4 panels, the average day of each week of that year's highest-demand full month (Jul 2025, Jun 2026) |
| `peaks.py` → `peaks.md`, `peaks.png` | who serves the evening peak, how often, and how much each is turned up |
| `peak_shares.py` → `peak_shares.md`, `peak_shares.png` | fuel proportions in the **morning** peak vs the **evening** peak |
| `response.py` → `response.md`, `response.png` | Study A (marginal response) and Study B (activation thresholds) |
| `intermittency.py` → `intermittency.md` | how long each fuel goes missing — gas is absent for whole months |
| `common.py` | shared definitions — peak window, trough window, palette |

Definitions used everywhere: **evening peak 17:00–21:00** (chosen from the data —
mean demand exceeds 5,900 MW across exactly those hours, topping out 18:00–19:00),
**midday trough 11:00–14:00**, **active = above 10 MW**, **net demand =
demand − wind − solar − curtailment** (what the six dispatchable fuels must cover).

> Note on the stacks: fuels + wind + solar sits slightly *above* the demand line,
> because Victoria is a net exporter on average and curtailed energy is not drawn.
> The gap is the interconnector.

---

## The short answer

The Victorian evening peak is served by **coal running harder, hydro switching on,
and the battery flipping from charging to discharging** — in that order of
magnitude. Gas is a top-decile event, not a daily one.

But *who responds* depends entirely on **how fast the change is**:

| timescale | biggest responder | second | coal's share |
| --- | --- | --- | --- |
| 5 minutes | **battery, 38%** | hydro, 29% | 28% |
| 3 hours | **coal, 43%** | hydro, 29% | 43% |

Fast changes are absorbed by storage and hydro. Sustained changes are absorbed by
coal. Any study that reports a single "merit order" without naming a timescale is
answering the wrong question.

---

## Per-fuel signs and interpretation

### coal_brown — the floor
- **On 100% of the time.** Never switches off, ever, in 474k intervals.
- Runs 3,046 MW at the midday trough, 3,853 MW at the evening peak. Turned up
  **+807 MW** for the peak.
- **76.8% of all peak energy** but only **40.4% of the peak turn-up.** It dominates
  the *level* and merely participates in the *change*. Confusing these two is the
  single easiest mistake in this dataset.
- Median price when running: **$65/MWh** — the cheapest dispatchable.
- Slow: supplies 0.150 MW per MW of 5-minute change, but 0.336 per MW of 3-hour
  change. It more than doubles its share as the horizon lengthens.
- **Sign it is being used:** always. **Sign it is being ramped:** a sustained
  (>30 min) rise in net demand.

### hydro — the peak-follower
- Active 70% of all steps, and on **99.5% of days during the peak**.
- 62 MW at the trough → **665 MW at the peak, a 10.7× swing.** The most dramatic
  proportional ramp of any fuel.
- **30.2% of the peak turn-up**, second only to coal.
- Switches on at **net demand ≈ 2,739 MW** — roughly the 3rd decile, so it is in
  play most of the day. Median price when active $91/MWh.
- Uniquely, its response is **flat across timescales**: 0.154 at 5 minutes,
  0.227 at 3 hours. It is the all-purpose responder.
- **Sign:** net demand above ~2,700 MW. Almost any evening.

### gas_ocgt — the peaker
- Active 27% of steps overall, **62% of days during the peak.**
- 32 MW at the trough → 253 MW at the peak. **11.1% of the turn-up.**
- Switches on at **net demand ≈ 5,058 MW** (9th decile). Median price when active
  **$148/MWh**.
- **Sign:** net demand above ~5,000 MW, or price above ~$150/MWh. If OCGT is
  running, the day is in its top 20%.

### gas_steam — the last resort
- Active only 12.8% of steps, **26% of days during the peak.**
- 12 MW → 111 MW. **4.9% of the turn-up.**
- Switches on at **net demand ≈ 6,708 MW** — the top decile only. Median price
  when active **$171/MWh**, the most expensive fuel in the stack.
- **Sign:** if gas_steam is on, net demand is in the top 10% of all intervals.
  It is a scarcity indicator more than a supply source.

### battery — the fast one, and the one that changed
- **Two states, and the clock decides which.** Charging peaks at **83% of steps at
  12:00**; discharging peaks at **82% of steps at 18:00**. It is a solar-shifting
  machine.
- Price confirms it: charging median price **$11/MWh**, discharging **$97/MWh**.
  In the cheapest price quintile it charges 74% of the time; in the dearest it
  discharges 56% of the time.
- Small in energy (**2.5% of peak energy**) but large in response: at the 5-minute
  horizon it supplies **0.198 MW per MW** — *more than coal or hydro* — because
  both directions count (charging less is supplying).
- **Sign:** hour of day first, price second. Net demand is the weakest of the three
  signals for the battery.

---

## Morning peak vs evening peak

Victoria has two demand peaks and they are **not the same event**. Full tables in
`peak_shares.md`.

| window | hours | mean demand | mean net demand |
| --- | --- | ---: | ---: |
| overnight trough | 02:00–05:00 | 4,999 MW | 3,561 MW |
| **morning peak** | 06:00–09:00 | 5,484 MW | 3,809 MW |
| midday trough | 11:00–14:00 | 4,373 MW | 2,319 MW |
| **evening peak** | 17:00–21:00 | 6,317 MW | 4,719 MW |

**Proportion of generation:**

| fuel | morning 06–09 | evening 17–21 | both pooled |
| --- | ---: | ---: | ---: |
| coal | **87.8%** | **76.9%** | 81.1% |
| hydro | 7.9% | 13.3% | 11.2% |
| gas OCGT | 2.2% | 5.1% | 4.0% |
| gas steam | 0.7% | 2.2% | 1.7% |
| battery out | 1.4% | 2.5% | 2.1% |

**Turn-up — and this is where they diverge completely:**

| fuel | morning turn-up | share | evening turn-up | share |
| --- | ---: | ---: | ---: | ---: |
| coal | **−51 MW** | **−16.4%** | +807 MW | 40.4% |
| hydro | +159 MW | **50.8%** | +603 MW | 30.2% |
| gas OCGT | +57 MW | 18.1% | +221 MW | 11.1% |
| gas steam | +18 MW | 5.7% | +99 MW | 4.9% |
| battery out | +48 MW | 15.2% | +119 MW | 6.0% |
| battery in (reduction) | +33 MW | 10.3% | +148 MW | 7.4% |
| **total** | **314 MW** | | **1,997 MW** | |

Three things follow.

**Coal goes DOWN into the morning peak.** It runs 3,685 MW overnight and 3,634 MW
at the morning peak, despite net demand being 248 MW *higher*. Coal holds a flat
overnight block and simply does not follow the morning rise — the entire morning
increment comes from hydro, gas and the battery.

**The morning peak is a demand peak but barely a net-demand peak.** Demand rises
485 MW into it, but net demand only 248 MW, and the total turn-up is 314 MW —
about **one sixth** of the evening's 1,997 MW. Overnight wind is high and solar is
just starting, so most of the morning rise is already covered.

**Hydro is the morning peaker; coal is the evening one.** Hydro carries 50.8% of the
morning turn-up against 30.2% of the evening; coal is −16.4% versus +40.4%. If you
study "the peak" without saying which one, these cancel into nonsense.

## Gas is absent for months at a time

The two gas channels are not intermittent in the ordinary sense — they are simply
**off** for long stretches while hydro and the battery carry the flexible work.
Full detail in `intermittency.md`.

| fuel | longest continuous absence | days it never runs |
| --- | ---: | ---: |
| coal | 0.0 days | 0.0% |
| hydro | 1.7 days | 0.1% |
| battery out | 1.0 days | 0.0% |
| gas OCGT | 16.8 days | 30.0% |
| **gas steam** | **101.7 days** | **71.6%** |
| **no gas at all** | **16.8 days** | **29.1%** |

`gas_steam` went **101.7 consecutive days** below 10 MW (2026-01-30 → 2026-05-12),
and there are **8 full calendar months out of 54** in which it never ran once:
`2023-10`, `2023-11`, `2024-01`, `2024-10`, `2025-11`, `2026-02`, `2026-03`,
`2026-04`. Both gas types were off together for 16.8 straight days over
Christmas 2025.

Hydro and battery discharge, by contrast, have never been absent for more than
about two days. **They are always-on channels; gas is an event.**

What separates the two states is a threshold, not a season: days with no gas
average a peak net demand of **4,631 MW**, days with gas **5,752 MW**.

### Consequence for the model

The correct prediction for `gas_steam` is **exactly zero on ~72% of days**. An MSE
loss on z-scored targets cannot express that — it settles on a small positive
number every step, which is wrong whenever the channel is off and barely penalised
when it is on. That is the direct cause of the **248% gas_steam MRE** in the
interpolation benchmark: the denominator is near zero because the truth is near
zero, while the prediction is not.

`gas_steam`, and to a lesser degree `gas_ocgt`, wants zero-inflated treatment —
predict P(on) and the level separately — or at minimum a metric that does not
reward a small constant. Hydro and battery need no such thing.

## The evening peak, year by year

| year | mean peak demand | peak hour | trough→peak net-demand rise |
| --- | ---: | ---: | ---: |
| 2022 | 6,277 MW | 18:00 | 1,934 MW |
| 2023 | 6,121 MW | 19:00 | 2,208 MW |
| 2024 | 6,249 MW | 18:00 | 2,446 MW |
| 2025 | 6,419 MW | 19:00 | 2,851 MW |
| 2026 | 6,713 MW | 18:00 | 2,714 MW |

**The peak is not growing much; the ramp into it is.** Demand rose 7% across the
period, but the trough-to-peak net-demand rise grew **40%** (1,934 → 2,714 MW).
That is solar hollowing out the middle of the day, not load growth.

### Who serves that ramp has changed completely

| year | coal | hydro | gas OCGT | gas steam | **battery (both)** |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2022 | 28.9% | 38.2% | 17.1% | 11.0% | **4.8%** |
| 2023 | 46.5% | 34.3% | 9.7% | 4.4% | **5.1%** |
| 2024 | 46.0% | 30.3% | 12.4% | 3.9% | **7.5%** |
| 2025 | 41.5% | 23.5% | 9.5% | 3.7% | **21.8%** |
| 2026 | 36.7% | 24.6% | 4.8% | 0.7% | **33.1%** |

Battery went from **4.8% to 33.1%** of the evening turn-up in four years, and it
came almost entirely out of gas: OCGT 17.1% → 4.8%, gas steam 11.0% → 0.7%. Hydro
also gave up ground (38.2% → 24.6%). **Storage has displaced the gas peakers.**

If you model the peak on 2022–2023 behaviour you will badly over-predict gas and
under-predict battery. Train and evaluate on recent data, or the fuel mix is wrong
before the model starts.

---

## Study A — marginal response

*Deliverable: "how much coal gets ramped up?"*

OLS slope of Δfuel on Δnet-demand — MW of that fuel per MW of net-demand change.

| horizon | coal | hydro | gas OCGT | gas steam | battery out | battery in | Σ signed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 min | +0.150 | +0.154 | +0.019 | +0.005 | +0.089 | −0.109 | 0.527 |
| 30 min | +0.252 | +0.214 | +0.057 | +0.018 | +0.068 | −0.058 | 0.666 |
| 1 hour | +0.291 | +0.224 | +0.067 | +0.023 | +0.066 | −0.049 | 0.721 |
| **3 hours** | **+0.336** | **+0.227** | **+0.075** | **+0.029** | **+0.057** | **−0.051** | **0.775** |

Read the last row as: over three hours, **coal supplies 34% of any net-demand
change, hydro 23%, gas 10%, battery 11%, and the interconnector takes the
remaining 22%.**

The Σ column is the local fleet's share. At 5 minutes it is only 0.527 — **half of
a fast change leaves the state** rather than being met locally. The interconnector
is the fastest unit on the system, and any model without it will attribute its work
to some generator.

## Study B — activation thresholds

*Deliverable: "when does battery get used? when does hydro? signs that gas gets used?"*

| fuel | switches on above | median price when active | active overall |
| --- | ---: | ---: | ---: |
| coal | always | $65/MWh | 100.0% |
| battery in (charging) | always (low nd) | $11/MWh | 41.7% |
| hydro | 2,739 MW | $91/MWh | 70.3% |
| battery out | 5,058 MW | $97/MWh | 37.6% |
| gas OCGT | 5,058 MW | $148/MWh | 26.9% |
| gas steam | 6,708 MW | $171/MWh | 12.8% |

Ordering the fuels by their median active price recovers the merit order **from
data alone**, without assuming it:

> battery charging ($11) → coal ($65) → hydro ($91) → battery discharge ($97)
> → gas OCGT ($148) → gas steam ($171)

That ordering matches `data/fuel_costs.json` almost exactly, which is a useful
independent check on the price table.

---

## Why this matters for the model

The 3-hour row of Study A is the **direct benchmark** for a 3-hour counterfactual.
Normalised to the local fleet's share:

| | coal | hydro | gas OCGT | gas steam | battery |
| --- | ---: | ---: | ---: | ---: | ---: |
| **actual grid, 3 h** | **43%** | **29%** | **10%** | **4%** | **14%** |
| incumbent `size_aware` head | 82% | 2% | 1% | 0.5% | 15% |
| recursive head, canon ramps | 24% | 30% | 12% | 3% | 30% |
| recursive head, p99 coal ramp | 16% | 36% | 14% | 3% | 31% |

The incumbent head puts twice as much on coal as reality and essentially nothing on
hydro. The recursive head fixes hydro and gas almost exactly, but **over-corrects on
coal and battery** — and tightening coal's ramp to p99 makes that over-correction
worse, not better.

So on this evidence the **canon-ramp recursive head is the better of the two**, and
the remaining error is that coal is under-used. That is a calibration target with a
number attached, which is a much better position than arguing about weight vectors.

Two caveats: the model rows are 4 winter days and 40 January days respectively,
while the 43/29/10/4/14 row is all hours of 4.5 years, so the populations differ.
And the actual grid's response includes the interconnector, which the model has no
channel for — some of the model's coal share is standing in for imports.
