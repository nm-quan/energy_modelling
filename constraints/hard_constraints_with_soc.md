# Hard constraints for the full set, SOC included

Status 2026-08-21. Reproduce section 1-3 with `python3 constraints/soc_trackability.py`.
Extends `constraint_research.md`; supersedes section 3 of `planB/SOC_DESIGN.md`.

---

## 0. The short answer

**Yes, we have the total battery SOC, and yes it can be tracked.**

`data/updata/vic_battery_soc.csv` (pulled 2026-08-17) carries measured reservoir
level in genuine MWh for each VIC battery. A fleet total is well posed as long as
the unit set is held fixed: on the 12 units with a stable fleet over
2026-02-01..2026-08-13 the aggregate level `E(t)` is present on **79.7%** of the
5-minute grid, with `E_max = 3,422 MWh`, and it uses its reservoir — it ranges
3.3% to 97.1% of `E_max` and sits within 5% of an edge on 1.3% of intervals.

That matters because every SOC constraint in the repo today is built around the
assumption that the level is *unobservable*. `planB/SOC_DESIGN.md` section 3 states
it directly: "The data gives you power in and power out. It never gives you `E`."
That was true when it was written. It is no longer true, and the constraint that
was designed around the limitation is roughly **2.6x looser than the real one**.

---

## 1. Tracking the reservoir from dispatch

The reason the level matters for a *constraint* is that a model does not get to
observe `E` inside its own forecast — it has to propagate it from the dispatch it
is predicting. So the question is how fast an open-loop recursion drifts away from
the truth when anchored on a measured starting level.

Three details are load-bearing and each was verified before the numbers below:

- **Power leads SOC by one interval.** Power is timestamped at the end of its
  dispatch interval (NEM convention), SOC is an instantaneous snapshot at its
  label. Use `power.shift(-1)`. On KESSB1 this is the difference between R2 0.944
  and R2 0.986, and it is not documented on either metric.
- **Energy is the trapezoid, not `P·dt`.** OpenElectricity power is instantaneous,
  so `E_i = (P_i + P_{i-1})/2 · dt`.
- **The loss has two components and both are real.** An efficiency alone and a
  constant draw alone fit almost equally well at step level, because each absorbs
  the other's mean. What separates them is the day-to-day *variation* of the gap:
  regress the daily (fuel − ΔSOC) gap on daily charge energy and the slope is
  **0.161** with R2 0.31, i.e. losses are 16% of throughput, where a constant
  explains none of the variation by construction. Fitting both gives a one-way
  `eta = 0.942` (`eta_rt = 0.887`) plus a residual constant draw of 139 MWh/day.
  Fitting the constant alone inflates it to 482 MWh/day, because it is then
  standing in for the efficiency term as well.

Worst error inside the horizon, as % of `E_max`, median / p90:

| driver of the recursion | step R2 | 3 h | 6 h | 24 h |
| --- | ---: | ---: | ---: | ---: |
| summed units, frictionless | 0.804 | 1.5 / 4.2 | 3.2 / 7.9 | 14.6 / 25.0 |
| summed units, constant draw only | 0.807 | 0.8 / 2.7 | 1.6 / 5.2 | 4.3 / 18.6 |
| summed units, `eta = 0.942` only | 0.807 | 1.0 / 2.3 | 1.9 / 4.6 | 5.3 / 20.1 |
| **summed units, `eta = 0.942` + constant** | **0.808** | **0.7 / 2.5** | **1.4 / 4.6** | **3.3 / 17.7** |
| fueltech rollup, `eta = 0.942` + constant | 0.799 | 0.9 / 4.2 | 1.6 / 7.5 | 4.5 / 25.8 |
| fueltech, `eta = 0.913`, no lag/trapezoid (as used today) | 0.784 | 1.0 / 4.2 | 1.9 / 7.4 | 5.3 / 22.8 |

Read the rows in pairs. Modelling no loss at all costs a factor of four over a day
(14.6% vs 3.3%), because a bias integrates. Either single-term correction gets most
of that back; both together are best at every horizon. The canon `eta = 0.913`
applied as it is today — without the lag shift and without the trapezoid — is the
worst of the corrected rows, and the fix is the alignment, not the value.

Note the fitted `eta_rt = 0.887` is above the repo canon 0.834. That is not a
disagreement: canon was fitted with `eta` as the only free parameter, so it had to
absorb the standing draw as well. Separating the two moves `eta` up and leaves
139 MWh/day behind. This is the same trap in the opposite direction from the
per-unit note that first reported a pure constant.

**Verdict on trackability:** anchored on a measured level, the reservoir is
predictable to about **±0.7% of `E_max` over 3 h and ±3% over a day**. That is well
inside the tolerance a hard constraint needs. The 24 h p90 of 17.7% is the honest
caveat — the tail days are real and a constraint set at the nominal edge will
occasionally clip valid dispatch, so the operating bound should carry a margin of
roughly 5% of `E_max`, not the current arbitrary 200 MWh.

One caveat that is specific to *our* pipeline: the model's target channels are the
`vic_generation` fueltech rollup, and that rollup double-counts a battery for a
period after it commissions. It shows up in the table above as the p90 columns
degrading (4.2% vs 2.5% at 3 h) while the medians nearly match. Rebuilding
`battery_charging` / `battery_discharging` from `vic_battery_power.csv` by
splitting the signed per-unit power on sign fixes it, and should happen before any
SOC constraint is attached to those channels.

---

## 2. Why SOC is a different kind of constraint

| constraint | timesteps coupled | form | per-step cost |
| --- | --- | --- | --- |
| box `0 <= P <= cap` | 1 | inequality on the value | trivial |
| ramp `\|P(t) - P(t-1)\| <= r` | 2 | inequality on the difference | trivial |
| balance `sum(sign · P) = nd` | 1 | one equality | trivial |
| **SOC `0 <= E(t) <= E_max`** | **all of them** | **inequality on the accumulation** | **needs a running state** |

The first three constrain a *value*. SOC constrains a *pattern*: discharging at
the full 1,687 MW for one step is fine, doing it for three hours straight is not,
because that is 5,061 MWh out of a 3,422 MWh bucket. There is no box on the
outputs that expresses this.

The saving grace is that `E` is **linear** in the dispatch:

```
E(t) = E_0 + sum_{s<=t} ( chg(s)·eta - dis(s)/eta )·DT - c·t
```

so `0 <= E(t) <= E_max` is just `2(N+1)` more linear inequalities on the same
trajectory the other constraints already live on. The feasible set stays a
polytope. Every projection method in the repo therefore extends to it in
principle; what differs is cost and conservatism.

---

## 3. The gain from knowing the level

Today's heads implement the **swing** form — `max(E) - min(E) <= cap`, with
`cap = 4,735.75 - 200 = 4,535.75 MWh` from `default_soc_cap()`. Swing was the
right choice when `E_0` was unknown, because the unknown constant cancels in a
difference. It costs two things:

1. **It is two-directional.** From any starting point it permits a full `cap`
   excursion up *and* a full `cap` down — 9,071 MWh of admissible excursion. The
   level form permits `E_0` down and `E_max - E_0` up, which sums to `E_max`
   exactly, so at the measured `E_max = 3,422 MWh` the level form is **2.65x
   tighter**. Half of that is the two-directional slack, the rest is nameplate.
2. **It is set from nameplate, not from the reservoir.** 4,735.75 MWh is registered
   storage; the fleet's observed maxima sum to 3,422 MWh (12 units) or ~4,100 MWh
   including MLB01. The cap therefore permits excursions the fleet cannot perform.

Measured on real windows:

| horizon | swing form binds | level form: charge headroom used (median, >95%) | discharge (median, >95%) |
| --- | ---: | ---: | ---: |
| 3 h | 0.00% | 2.8%, 0.00% | 5.4%, 0.00% |
| 6 h | 0.00% | 12.7%, 0.44% | 20.1%, 0.00% |
| 24 h | 0.00% | 78.2%, 2.48% | 62.3%, 0.00% |

The swing form never binds at any horizon — it is decoration. The level form is
also non-binding at 3 h, which independently confirms the earlier finding that SOC
cannot improve a 3 h gap fit. But at 24 h it is a real constraint: the median day
consumes 78% of its charging headroom and 2.5% of days consume more than 95% of
it. **The constraint activates on the whole-day imputation and on counterfactual
rollouts, and nowhere else.** That is where to spend the effort, and it is also
exactly the setting where a model can drift somewhere physically impossible.

Note the asymmetry: charging headroom binds, discharge headroom does not. The
fleet's exposure is running out of room to absorb, not running out of energy.

---

## 4. How to implement it — three families, all already partly built

### A. Recursive tightened box — `planB/heads.py::_Soc`

Carry `E` as a running state through the sequential head. At each step convert the
admissible `dE` into a tightening of the `chg` / `dis` rows of the box, then let
the existing exact projection onto `{box} ∩ {balance hyperplane}` run unchanged.

This is the strongest option and it is 90% written. `_Soc.tighten` already does
the conversion, already has backward reachability toward a terminal `target`,
already handles the case where the guard forces a *lower* bound on discharge, and
already implements the correct precedence — balance outranks SOC, with the miss
reported in `short` rather than hidden.

What it needs is small and local, in `_Soc`:

```python
# now (swing, unknown level):
U = self.Emin + self.cap - self.E      # room to rise
L = self.Emax - self.cap - self.E      # room to fall

# level form (measured E_0):
U = self.E_res_max - self.E            # room to rise
L = 0.0 - self.E                       # room to fall
```

with `self.E` seeded from the measured level at the window start instead of `0`,
`E_res_max` set from observed maxima rather than nameplate, and the constant draw
subtracted in `advance()`. `soc_seed` is already the plumbing for the seed.

Cost: one extra state variable, no extra iterations. Guarantee: exact, because the
projection at each step already respects whatever box it is handed.

Its real limitation is myopia, not the level form. Tightening the box stops the
step from overflowing *now*; it does not stop the head from walking into a corner
it cannot get out of later. That is what the `target` reachability guard is for,
and it should be extended from a terminal point to a full backward interval —
see section 5.

### B. RAYEN wall — `imputation/constraint_layers.py::rayen_traj_project(soc=True)`

Because `E` is linear in the dispatch, every ordered pair `(t, t')` contributes one
more `alpha` wall, so SOC folds into the existing ray-shoot with no new machinery.
Already implemented and already validated: it fixed C3's 92 SOC-infeasible days to
0 and improved blackout MAE 146.6 -> 134.8.

Two known problems, both structural rather than fixable by switching to the level
form: a single `alpha` for the whole trajectory is conservative, and `alpha*` has
been observed to collapse to ~0 when the anchor lands on a box face, at which point
the output is the anchor and the model contributes nothing. The pair grid is also
`O(N^2)` and already needs chunking.

Switching to the level form here is a one-line change to the same walls —
`cap_eff - dEA` becomes two bounds against `E_max - E_A(t)` and `-E_A(t)` — but it
makes `alpha` collapse *more* likely, since the set is 2.6x smaller. Use this arm
only where a single differentiable shot is required.

### C. Exact projection / cyclic POCS — `imputation/constraints.py::cyclic_project`

`balance -> ramp -> box -> SOC` cycled to convergence. Already the reference
implementation and already SOC-aware. It is the right thing for evaluation and
post-hoc use, and the wrong thing for the inner training loop: it needs ~120
cycles once SOC is on, because the SOC damping runs last and partially undoes
balance each pass.

### Recommendation

Use **A** as the deployed head, **C** as the scoring projection, and keep **B**
only for the arm that needs a single differentiable shot. That is the arrangement
the code already assumes ("the consistent scoring rule" at the top of
`constraint_layers.py`); the work is to move all three from the swing form to the
level form and to give A a proper reachability tube.

---

## 5. The missing piece: backward reachability on the reservoir

A greedy tightening is feasible at every step and can still be infeasible overall.
The fix is cheap here because the reservoir dynamics are one-dimensional and
monotone, so the reachable interval can be computed backwards in closed form
before the sweep starts:

```
E_hi(N) = E_max ,  E_lo(N) = 0          (or the terminal cycle condition)
E_hi(j) = min( E_max , E_hi(j+1) + max_discharge_per_step + c )
E_lo(j) = max( 0     , E_lo(j+1) - max_charge_per_step    + c )
```

Then at step `j` the admissible `dE` is intersected with `[E_lo(j+1) - E, E_hi(j+1) - E]`.
That is `O(N)`, done once, and it converts "does not violate now" into "cannot be
forced to violate later". `_box` already applies exactly this trick for the ramps
and the pinned right endpoint; `_Soc` applies a scalar version of it toward
`target`. Generalising it is a small amount of code and it is the difference
between a guard and a guarantee.

The terminal condition should stay the **cycle** condition already argued for in
`heads.py::set_soc` — a swing bound alone is blind to a slow drain, and the real
dispatch cycles almost exactly (discharge/charge 0.8341 against `eta^2` 0.8340).
With a measured level the cycle condition becomes checkable rather than assumed.

---

## 6. Risks and open items

- **History coverage.** Measured SOC starts 2025-06-30 and is only trustworthy
  from 2026-02 (the 2025-09..12 misalignment destroys the fit — KESSB1 collapses
  to R2 0.198 on the full window). The modelling table runs from 2021-10. So we
  can *evaluate* a SOC constraint on ~6 months and *seed* it in deployment, but we
  cannot train through a measured-level constraint on the full history. For older
  windows the level has to be reconstructed by recursion from a seeded start, and
  the 24 h p90 drift of 17.7% is then the error bar.
- **Fleet composition, not fleet reservoir.** MLB01 (frozen telemetry, starts
  2026-03), MRNBESS1 and TRGBESS1 (enter the feed 2026-07-30) each change the unit
  set mid-window. `E_max` is a function of time and has to be handled the same way
  `exp1_capacity_bounds` handles time-varying capacity, or a commissioning event
  reads as a reservoir jump.
- **Aggregate hides the binding constraint.** Per unit, SOC is within 5% of an
  edge on 21-47% of intervals; aggregated, 1.3%. The fleet sum is a genuine
  relaxation of 12 separate reservoirs, not the same constraint. Satisfying all 12
  per-unit bounds implies the aggregate bound, so the aggregate is necessary but
  not sufficient — enforcing it is sound, just weak. A per-unit model would enforce
  something much stronger. That is a separate, larger project.
- **`nd` leak.** None of this touches the outstanding problem that the head is fed
  `nd = SIGN · truth`. SOC is the one hard constraint that does not involve `nd`,
  which is exactly why it is worth having.
- **Not yet done.** Nothing above has been trained through. The claims in sections
  1-3 are measurements on data; the claims in sections 4-5 are design.

---

## 7. Next steps, in order

1. Rebuild `battery_charging` / `battery_discharging` from `vic_battery_power.csv`
   so the target channels match the reservoir they are supposed to drive.
2. Replace `default_soc_cap()`'s nameplate swing budget with an observed,
   time-varying `E_max`, and refit the loss as `eta` **plus** a constant rather
   than `eta` alone — canon's 0.834 is a single parameter doing two jobs.
3. Switch `_Soc` to the level form with a measured seed, keeping the swing form
   behind a flag so the existing numbers stay reproducible.
4. Generalise the backward reachability from a terminal `target` to the full
   `[E_lo(j), E_hi(j)]` tube.
5. Re-run the whole-day imputation and the `q*` counterfactual — the two settings
   section 3 shows the constraint actually binds in — and report WAPE against the
   swing-form baseline. H1 from `constraint_research.md` predicts the cost is
   near zero for a projection-family method; this is a clean test of that.
