# planB evaluation plan

Five tests. Each one names the problem it targets, the measurement that
established the problem, the visualisation, and an acceptance criterion taken
from `dispatch_study/patterns_2025_2026.md` rather than from taste.

**All benchmarks are 2025–2026, not pooled 2022–2026.** Battery went from 4.8% to
33.1% of the evening turn-up over that span and gas OCGT fell 17.1% → 4.8%; the
pooled figure describes a fleet that no longer exists. The test split is 2026, so
it must be judged against the fleet that produced it.

---

## The problems, and what established each

| # | problem | the measurement |
|---|---|---|
| P1 | every arm is worse than linear interpolation | 50.6 MW vs 80–103 MW dispatch MAE, 400 random gaps |
| P2 | the constraint head decides the allocation, not the model | `rayen`'s anchor alone scores 63.7 MW against 82.8 with the network |
| P3 | zero-inflated channels are mishandled | `gas_steam` ≤1 MW in 96% of cells; MRE 142–370% at an MAE of 13–34 MW |
| P4 | curtailment level is not learned | predicted 4,684–11,206 MWh against 10,693 actual, one arm the wrong sign |
| P5 | dispatch is feasible but not necessarily plausible | coal share of the counterfactual response 33–74% against a measured 44.1% |
| P6 | shape-following is untested | the demand shift is a rectangular step, so a model can match the integral without tracking anything |

---

## Test 1 — reconstruction on random windows

**Targets P1, P3.** Built: `planB/eval.py`.

400 random 3-hour gaps from the test split, per-channel MAE and MRE, plus the
feasibility audit (balance / ramp / box, 86,400 cells and 88,800 pairs).

**Addition needed for P3.** MRE is meaningless where the truth is mostly zero —
the denominator collapses. Split the zero-heavy channels into two numbers:

- **on/off F1** — did the model know the channel was off?
- **MAE | on** — given it was on, was the level right?

Applies to `gas_steam` (96% zero), `gas_ocgt` (75%), `wind_curt` (56%),
`solar_curt` (66%).

**Acceptance:** dispatch aggregate below **50.6 MW**, curtailment below **43.0 MW**
— linear interpolation. Nothing has cleared either yet, and one epoch was the
standing excuse.

**Visualisation:** per-channel MAE bars, models against the interpolation
reference line.

---

## Test 2 — peak filling, 18:00–21:00

**Targets P1, P5.** New.

The current counterfactual window is 11:00–14:00, which is the **midday trough**
— the one part of the day where neither peak's dispatch logic applies. From the
2025–26 data:

- evening turn-up **2,270 MW** against the morning's 271 MW, an 8× difference
- midday has **no battery-discharge signal at all**: corr(nd, discharge) = +0.035,
  against +0.287 in the evening
- midday curtailment is 1,810 MW in the bottom net-demand decile; evening is 11 MW

So a peak-filling claim cannot be tested on a midday window.

**Two numbers, not one.**

1. **Per-channel MAE/MRE inside the window, over the ENTIRE test set** (186 days
   × 36 steps). This is the statistic.
2. **10 consecutive days** for the figure. This is the illustration, and it should
   never be quoted as the result.

**The criterion that matters is not MAE.** It is the **turn-up composition** —
who supplies the rise from the midday trough to the evening peak — against the
measured 2025–26 shares:

| coal | hydro | gas OCGT | gas steam | battery out | battery in |
|---:|---:|---:|---:|---:|---:|
| 39.9% | 23.9% | 7.9% | 2.7% | 11.3% | 14.3% |

**Visualisation:** 10-day stacked panels, actual on top and one row per arm, the
window shaded. Plus a per-fuel line overlay inside the window only — actual
against predicted — because a stacked area hides shape error in the small
channels, which is exactly where the interesting behaviour is.

---

## Test 3 — curtailment

**Targets P4.** Partly built.

Three separate questions, currently collapsed into one MAE:

- **occurrence** — on/off F1, since 56%/66% of cells are exactly zero
- **level** — MAE conditional on being curtailed
- **response** — under a demand rise, does spill fall, and by how much

The response criterion comes from the measured release rate, 2025–26, 3-hour
horizon:

```
released  =  min( C,  k · Δ )
k = 0.42   if C > 200 MW and price < 0
k = 0.33   if C >  10 MW
k = 0.24   otherwise
retained >= 13.6% of C          the network-bound floor
```

**Acceptance:** release direction negative in every arm; magnitude within a factor
of 2 of `k·Δ`; and the retained fraction not below 13.6%, because that part is
transmission-bound and more demand cannot free it.

**Visualisation:** predicted against actual curtailment as a time series inside the
window, and the release Δ per arm against the `k·Δ` band.

---

## Test 4 — is the dispatch plausible, not merely feasible

**Targets P5.** New. This is a **distributional** test — pointwise error can look
fine while the behaviour is nonsense.

Four checks, each against a measured historical distribution:

| check | historical reference (2025–26) |
|---|---|
| activation vs net-demand decile | `gas_steam` switches on above ~6,700 MW; `gas_ocgt` ~5,058; `hydro` ~2,739; coal always |
| implied merit order | median price when active: batt-charge $11 → coal $65 → hydro $91 → batt-discharge $97 → OCGT $148 → steam $171 |
| intermittency | `gas_steam` produces nothing on 72% of days and has gone 101 consecutive days below 10 MW |
| ramp distribution | not just "under the cap" — the *shape* of the step-change histogram |

A model that emits a small positive `gas_steam` at every step is feasible, scores
acceptably on MSE, and is wrong about the plant 72% of the time. Only this test
catches that.

**Visualisation:** activation curve (model vs historical) per channel across
net-demand deciles; ramp histogram overlay.

---

## Test 5 — shaped demand, to test gradual ramping

**Targets P2, P6.** New, and the strongest test in the list.

`FixedPercentageShift` adds a **flat +q%** across the free window, so the imposed
net demand is a boxcar. A model can match its integral without tracking anything.
Worse, it cannot distinguish a model that reshapes the fleet from one that scales
every channel by a constant.

**Replace the boxcar with a raised cosine** over the window, same total energy:

```
Δ(t) = A · (1 − cos(2π t / T)) / 2 ,      t ∈ [0, T)
```

Zero at both seams, so no discontinuity for the ramp constraint to absorb; peak in
the middle; and — the point — **a rate of change that varies along the window**,
steep at the quarter-points and flat at the crest.

### Why this is the right test, from the historical data

Study A measures the response composition as **timescale-dependent**:

| horizon | coal | hydro | battery (both) |
|---:|---:|---:|---:|
| 5 min | 15.0% | 15.4% | **19.8%** |
| 3 hours | **33.6%** | 22.7% | 10.8% |

Fast changes are absorbed by storage and hydro; sustained changes by coal.
Curtailment release shows the same signature — k = 0.67 at 5 minutes falling to
0.46 at 3 hours, because releasing spilled energy costs no ramping.

So along a raised cosine, a model that has learned the physics should **shift its
mix**: battery and hydro dominant where the curve is steep, coal taking over on
the plateau. A model that has only learned to scale will hold the same mix
throughout. **The boxcar cannot tell these apart. The cosine can.**

**Metrics:**
- shape correlation between the imposed Δnd and the model's Δ(SIGN·P) within the window
- per-step tracking error, reported against position on the curve
- **mix-vs-local-rate**: bin the window steps by |dΔnd/dt| and report the fuel
  composition in each bin. Compare against Study A's timescale gradient.

**Visualisation:** the imposed curve against the model's supply, per day; and a
stacked area of the response composition as a function of local rate of change —
which should slope from battery/hydro toward coal if the model is right.

**Controls:** run the boxcar alongside as the reference, and a linear
triangle as an intermediate (constant rate, so a flat mix is the correct answer
there — a useful negative control).

---

## Build order

| test | status | effort |
|---|---|---|
| 1 reconstruction | built; add on/off F1 + MAE-given-on | small |
| 2 peak filling 18:00–21:00 | new script, reuses the counterfactual plumbing | medium |
| 3 curtailment | metrics exist; add the split and the `k·Δ` band | small |
| 4 plausibility | new; all reference numbers already computed in `dispatch_study/` | medium |
| 5 shaped demand | new shift class + the mix-vs-rate decomposition | medium |

Tests 1 and 3 should run before the Colab job so the outputs are comparable.
Tests 2, 4 and 5 read checkpoints, so they can run on whatever comes back.

## One thing to settle

`planB/counterfactual.py` still hardcodes the **pooled** benchmark
(coal 43.4 / hydro 29.3 / battery 14.0). The 2025–26 targets are
**44.1 / 22.4 / 24.5**. Hydro is 7 points out and battery 10 — enough to change
which arm looks closest. Every table above assumes the recent numbers.
