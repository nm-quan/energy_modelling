# Four hard constraints at once, on the whole-day gap

Status 2026-08-23. Reproduce with the four commands at the bottom.
Data: `data/updata` only, plus the interconnector series from `data/`.

The task is imputation, not forecasting: a whole day (288 five-minute steps) of the VIC
dispatchable fuel mix is missing. Demand, price, wind, solar, curtailment and
interconnector flow are observed *through* the gap; the reservoir level is read at the two
edges. Recover the dispatch so the result is feasible on all four constraints
simultaneously:

| | constraint | form | timesteps coupled |
| --- | --- | --- | --- |
| 1 | balance | `SIGN . P(t) = nd(t)` | 1 |
| 2 | battery SOC | `E_min <= E(t) <= E_max`, `E` linear in `P` | all of them |
| 3 | capacity | `P_min(t) <= P(t) <= P_max(t)`, monthly `C(t)` | 1 |
| 4 | ramp table | `|P(t+k) - P(t)| <= R(k)` for **every** k | k+1 |

---

## 0. What was already here, and what was missing

Three of the four were solved separately in `planB/heads.py`, on the older
`data/preprocessed` tree. What was missing:

- **SOC was not enforced.** `CONSTRAINT.md:46` says so outright. The swing form that
  exists (`max E - min E <= cap`) never binds at any horizon — `hard_constraints_with_soc.md`
  calls it "decoration".
- **The ramp table was not hard.** `exp4` built `R(k)`, but only its `k=1` row reached
  `hardnet`; the full table appeared as a soft penalty inside the QP.
- **Capacity was static**, while the battery fleet grew 735 → 2,260 MW across the window.
- **`nd` came from the truth.** `planB/PUBLICATION_REVIEW.md` measured the honest version
  at a 3,272 MW residual, i.e. worse than linear interpolation.
- And the projection **cost accuracy**: in `exp6` it moved microWAPE 0.2441 → 0.2638 and
  gas_steam MAE 48.7 → 155.5 MW.

---

## 1. The one design decision everything else follows from

**Carry the battery as a single signed channel `b = discharge − charge`.**

The reservoir moves by `dE = contrib(b_t) + contrib(b_{t-1}) − draw`, where

```
contrib(b) = (eta*DT/2) * max(-b, 0)  -  (DT/(2*eta)) * max(b, 0)
```

is continuous, piecewise linear and **strictly decreasing** in `b` — slope `-eta*DT/2`
below zero, `-DT/(2*eta)` above, and `eta < 1` makes the second steeper. A strictly
decreasing function maps an interval to an interval, so

```
dE in [L, U]      <=>      b in [ contrib^-1(U) , contrib^-1(L) ]
```

The SOC constraint becomes an **exact box on one coordinate**, in closed form, with no
conservatism. That is what keeps the per-step feasible set a *box intersected with a
hyperplane* — the shape the closed-form projection needs.

Carried as two non-negative channels it is not a box at all. The separable surrogate the
repo uses today (`chg <= U/(eta*DT)` assuming `dis = 0`, and conversely) is simultaneously
too loose to guarantee the reservoir and too tight to stay compatible with balance:
measured on this task it overflows by **310–5,700 MWh**.

## 1b. What one signed channel cannot say, and how it is recovered

`b` is everything the reservoir and the balance plane can see. It is not everything the
fleet does. On **54% of intervals** some VIC batteries charge while others discharge, and
the reported channels differ from the net view by the **overlap** `m = min(chg, dis)`:

```
battery_charging = m + max(-b, 0)        battery_discharging = m + max(b, 0)
```

Measured over the span: mean 6.3 MW, p99 101 MW, **max 454 MW**, holding **6.4%** of all
discharge energy. A signed-only head pays that on both battery channels.

**This was invisible to the first version of the audit, and the reason is worth stating.**
`battery_reconstruct` aggregated the net and then clipped it, rather than aggregating each
unit and summing — different numbers, and only the second is what the meters saw. The
feasibility floor then compared the head's output against that same collapsed reference, so
the overlap was missing from *both* sides and the battery columns read 0.000 whatever the
head did. A check whose reference has been collapsed cannot detect a collapse.

**The fix.** `m` is predicted alongside the dispatch and reattached by
`head_traj.attach_overlap`. None of the four guarantees can be damaged by it, and each
reason is structural:

| | why m is safe |
| --- | --- |
| balance | `m` adds to **both** channels, so `b = dis - chg` is unchanged |
| ramp | likewise — the table is defined on `b`, which `m` does not touch |
| capacity | clipped so `m + max(-b,0) <= C` and `m + max(b,0) <= C` |
| SOC | `eta < 1` makes overlap pure round-trip loss, so it only ever **lowers** the level, and the bound that binds on this fleet is the ceiling |

Two details had to be right, and both were found by measurement rather than argument.

**It runs as a separate pass on a settled trajectory.** Computed inside the sweep, `m`
changes the SOC state, which moves `b`, which moves the balance correction: feeding the
**truth** through that version walked the output **877 MW** away from itself. Run
afterwards, `P` is bit-identical with the overlap on or off.

**The SOC clip is a cumulative budget, not a per-step one.** Overlap is pure loss, so every
MW lowers the level permanently; clipping only against the floor at the current step lets
the level walk down until a later step goes under with `m = 0` there and nothing left to
give back — measured, a **170 MWh** breach. Because the trajectory is settled the whole
future is known, so the budget is exact. With `S_t` the cumulative overlap and
`G(t) = (E_b(t) - floor)/delta`,

```
E(t) = E_b(t) - delta*( S_t + S_{t-1} + m_0 )   =>   S_t + S_{t-1} + m_0 <= G(t)  for all t
```

giving a cap for now and a suffix-minimum cap for every step after. It holds to
**2e-11 MWh** against an adversarial `m = 1e5` request.

---

## 2. The head

One greedy forward sweep. Once `P[t-1], P[t-2], ...` are fixed, everything constraining
step `t` collapses to an interval per channel:

```
lo[t] = max( P_min(t),  max_k (P[t-k] - R_dn(k)),  pR - R_up(N+1-t),  soc_lo[t] )
hi[t] = min( P_max(t),  min_k (P[t-k] + R_up(k)),  pR + R_dn(N+1-t),  soc_hi[t] )
P[t]  = argmin sum_i w_i (P_i - F_i)^2   s.t.  lo <= P <= hi,  SIGN . P = nd[t]
      = clip(F + lam * SIGN/w, lo, hi)
```

**Why `[lo, hi]` is never empty.** Induction on `t`. The forward cone from `P[t-1]` and the
backward cone to the pinned right endpoint intersect exactly when
`R(m+1) <= R(m) + R(1)` — **subadditivity**, which `constraint_set.ramp_table` enforces by
DP closure. Longer-`k` terms cannot empty it either: each is implied by the sum of the
single steps it spans, again by subadditivity.

**Why every `k`, not a subset.** With `k` in `{1,3,6,12,24,36}` and a linear seam guard
`k*R(1)`, the induction breaks and the sweep violated ramps by up to **900 MW** under
stress. With the full grid and the exact remaining distance `R(N+1-t)`, the worst violation
over the same inputs is **1.1e-13 MW**.

**Why the projection is weighted.** One scalar `lam` moves every free channel by the same
number of MW, so a 510 MW peaker takes the same correction as a 5,095 MW coal fleet — the
exp6 regression. `w = 1/softmax(logits)` makes the share channel `i` absorbs exactly `p_i`,
and softmax is the right family because the balance identity forces those shares to sum to
one.

**Where SOC meets balance.** `SIGN` is all `+1`, so `b = nd − G` with `G` the thermal sum,
and `G` is confined to its box. Balance alone therefore already confines
`b in [nd − sum hi_th, nd − sum lo_th]`. Intersecting the SOC interval with *that*, before
the projection runs, is what stops the two constraints fighting.

---

## 3. Two claims, and they are different

**Balance, capacity and ramp are architectural.** The sweep only ever looks backward, or
forward through a reachability guard that is exact, so they hold for *any* input —
including a constant ±1e5. Worst violation over truth, truth+N(0,150), truth+N(0,1000),
pure noise, an untrained network and both adversaries: **≤ 1.5e-6 MW**.

**SOC is empirical.** The reservoir couples every step to every other, and a single
forward sweep commits to `P[t]` before it has seen `t+1`; once the thermal channels are
pinned against their ramp box the battery is *fully determined* by balance, with no freedom
left. So SOC is restored by an iterative repair — alternate the sweep with a reservoir
projection, then shrink the bounds by any residual excursion and re-solve. It is exact on
everything a network can plausibly emit and degrades gracefully beyond.

The map is **idempotent to 1.8e-11 MW** — the identity on an already-feasible trajectory,
which is what makes the ablation below mean anything.

---

## 4. Why the empirical max, not p99.9

The repo tightened its per-step ramp to a percentile because a single-step max is set by
unit trips and is therefore inert. A *duration table* removes that reason:

| | 15m | 60m | 180m | 1440m |
| --- | ---: | ---: | ---: | ---: |
| `R(k) / [k * R(1)]`, hydro | 0.529 | 0.150 | 0.065 | **0.008** |
| coal_brown | 0.360 | 0.252 | 0.119 | **0.016** |
| battery | 0.489 | 0.122 | 0.058 | **0.009** |

Over a day the per-step box is **33–140× looser** than the real envelope, and the table
recovers all of it while still admitting every recorded step. The percentile costs what the
max does not — measured on real windows:

| envelope | worst balance shortfall | recorded steps rejected |
| --- | ---: | ---: |
| p99.9 table | 443 MW | 0.19 % |
| **empirical max table** | **0.00 MW** | **0.00 %** |

At p99.9 "balance exact by construction" degrades to "exact except where the envelope
forbids it". At the max it does not.

## The inverse view — the ramp table as asked for

Shortest time in which the fleet has ever moved a given fraction of installed capacity:

| channel | 5% | 10% | 20% | 30% | 50% | 75% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hydro | 5 min | 5 min | 5 min | 5 min | 20 min | 195 min |
| coal_brown | 5 min | 5 min | 30 min | 55 min | 350 min | never |
| gas_steam | 5 min | 5 min | 5 min | 10 min | 20 min | 35 min |
| gas_ocgt | 5 min | 10 min | 20 min | 45 min | 150 min | never |
| battery | 5 min | 5 min | 5 min | 5 min | 10 min | 125 min |

---

## 5. The reservoir, measured rather than assumed

Rebuilt from `vic_battery_power.csv` / `vic_battery_soc.csv` on a **fixed 12-unit set**
(a fleet total is only well posed if membership is held constant, or a commissioning event
reads as a reservoir jump). The fueltech rollup differs from this sum by std 56.7 MW and up
to 470 MW, so the channels the model predicts had to be rebuilt to drive the reservoir
whose level is measured.

- `E_max` = **3,391 MWh** observed, against 4,735.75 MWh nameplate.
- `eta` = **0.924** one-way plus a **222 MWh/day** standing draw. The two terms must be
  fitted together — alone, each absorbs the other's mean, which is how the repo canon got
  `eta_rt = 0.834` with one parameter doing two jobs.
- **The level form binds, the swing form does not.** Over 268 clean days the median day
  consumes **78.1 %** of its charging headroom (p90 94.3 %, >95 % on 8.6 % of days), while
  the largest daily swing is 3,103 MWh against a 4,536 MWh swing budget.
- **The floor carries a calibrated margin, the ceiling does not.** The recursion is open
  loop, so it inherits a drift the telemetry does not have — up to 44 % of `E_max` over a
  day. Enforcing a floor of exactly 0 would declare recorded dispatch infeasible, so the
  floor is lowered to admit every recorded day and the amount is reported. The ceiling
  needs no margin, and the ceiling is the side that binds.
- **No terminal pin.** Landing on the measured right-edge level would need a ~950 MWh
  tolerance on a 3,391 MWh reservoir — looser than the bounds themselves. The terminal
  error is reported as a diagnostic instead.

## 6. Balance on observables

Every constrained model in this repo has been fed `nd = SIGN . truth`, which makes "balance
exact by construction" a restatement of the answer. Fitted on `demand`, `wind`,
`solar_utility` and `net_import`:

| | out-of-sample MAE | std | max |
| --- | ---: | ---: | ---: |
| naive identity | 196.8 MW | 142.4 | 783.9 |
| **fitted map** | **54.8 MW** | **71.9** | 478.8 |

against a 4,579 MW level on the test gaps. The balance constraint is now a real one, and
both scorings (against the observable `nd` and against `SIGN . truth`) are reported.

---

## 7. Results

300 three-day windows with a complete SOC-bracketed middle day; chronological tail split,
test = 60 days, **2026-04-30 .. 2026-07-11**. One backbone per seed, 3 seeds, the head
switched arm by arm so a row-to-row difference is the constraint and not a different init.
Violation columns are MAGNITUDES (MW, MWh for SOC), worst over seeds and over every window
-- not counts at a tolerance.

| arm | microWAPE | MAE (MW) | macroR2 | balance | box | ramp | SOC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| linear interpolation | 0.1671 | 134.63 | 0.211 | 2.9e+03 | 0 | 0 | 5.6e+03 |
| persistence (prev day) | 0.2031 | 163.63 | 0.131 | 3.9e+03 | 0 | 9.2e+02 | 2.3e+03 |
| interp + head (0 params) | 0.1359 | 109.51 | 0.080 | 4.9e-11 | 0 | 2.3e-13 | 3.0e-02 |
| `none` (unconstrained net) | 0.1306 | 105.22 | 0.480 | 1.7e+03 | 3.4e+02 | 4.0e+01 | 4.0e+03 |
| `+balance` | 0.1311 | 105.58 | 0.330 | **2.7e-12** | 3.9e+02 | 1.5e+02 | 5.2e+03 |
| `+box` | 0.1113 | 89.68 | 0.513 | 2.7e-12 | **0** | 8.3e+01 | 6.6e+03 |
| `+box C(t)` | 0.1113 | 89.68 | 0.513 | 2.7e-12 | **0** | 8.3e+01 | 6.6e+03 |
| `+ramp k=1` | 0.1113 | 89.68 | 0.513 | 2.7e-12 | **0** | 3.9e+01 | 6.6e+03 |
| `+ramp R(k)` | 0.1113 | 89.68 | 0.513 | 2.7e-12 | **0** | **1.1e-13** | 6.6e+03 |
| **`+SOC` (the model)** | **0.1133** | **91.26** | 0.474 | **7.3e-11** | **0** | **3.4e-13** | **5.7e-14** |

**The model is feasible on all four at machine precision and still beats every baseline**
-- 32% below linear interpolation on MAE and 13% below the unconstrained network that has
the same weights.

What the ladder says, row by row:

- **Balance is free.** 0.1306 -> 0.1311, inside the seed spread, and it removes a 1,700 MW
  residual. Note this is against the *observable* `nd`, not the truth.
- **The box is the big accuracy win, not a cost.** 0.1311 -> 0.1113, a 15% improvement,
  because the unconstrained network spends error on outputs the fleet cannot produce.
- **The ramp table is free and it is the only thing that closes the ramp gap.** Accuracy is
  identical to four decimal places across `+box`, `+ramp k=1` and `+ramp R(k)`, while the
  violation goes 83 MW -> 39 MW -> 1.1e-13 MW. A per-step limit leaves 39 MW of multi-step
  violation standing; the table costs nothing to remove it.
- **SOC is the only constraint with a measurable price**: 0.1113 -> 0.1133, +1.8%, to turn
  a 6,600 MWh reservoir excursion into 5.7e-14. That is the honest trade, and it is small.
- **The head is doing a lot of the work**, as the repo has found before: `interp + head`
  with zero trained parameters scores 0.1359, better than the unconstrained network. The
  trained backbone buys 18 MW of MAE on top of it.

Two things this table does **not** show, stated rather than buried:

- `+box C(t)` is identical to `+box`. The test slice is 2026-04-30..07-11, by which point
  the battery fleet has reached its final size, so the monthly capacity and the static max
  coincide *there*. Time-varying capacity matters for the training-era windows (the fleet
  grew 2.2x across the span) and cannot be shown to matter by this split.
- `macroWAPE` rises on the constrained arms while `microWAPE` falls. That is `gas_steam`,
  which is zero 96% of the time, so its WAPE denominator is tiny and a few MW of projected
  correction dominate the unweighted mean of ratios. `microWAPE` is the stable headline --
  the same convention the rest of the repo uses.

The feasibility floor (`exp7` check 4) is **0.000 MW on every channel**: the map applied to
the recorded dispatch returns it unchanged. So none of the error above is the constraint
set; all of it is the network.

## 8. Files

| file | what |
| --- | --- |
| `constraint_set.py` | the four constraints as data. `ramp_table` (max, monotone, subadditive closure), `capacity_bounds`, `Reservoir`, `fit_nd_map`. Imported by both the head and the scorer, so they cannot drift apart |
| `battery_reconstruct.py` | channels + reservoir from per-unit telemetry; loss-model fit |
| `head_traj.py` | `hardnet_traj` (single pass, differentiable — the training map) and `hardnet_traj_polished` (the deployed map) |
| `exp7_head_audit.py` | the feasibility proof: non-emptiness, adversarial feasibility, idempotence, feasibility floor, an independent re-check, historical admissibility |
| `exp8_constrained_imputer.py` | the model and the constraint ablation ladder |

## 9. Reproduce

```
python colab/capacity_experiment/battery_reconstruct.py     # reservoir + loss model
python colab/capacity_experiment/constraint_set.py          # the four constraints + checks
python colab/capacity_experiment/exp7_head_audit.py         # the proof
python colab/capacity_experiment/exp8_constrained_imputer.py   # model + ladder
```
