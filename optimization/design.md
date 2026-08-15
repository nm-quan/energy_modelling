# Inverse optimization for VIC dispatch — design

Replaces the "network predicts dispatch, head repairs it" architecture with
"network predicts *bids*, an optimizer produces the dispatch".

Two things this has to do:

1. **Match the real grid.** Reproduce observed dispatch on held-out data.
2. **Survive a counterfactual.** Re-dispatch under a shifted demand, obeying every
   constraint.

---

## 1. The idea

AEMO's dispatch engine is an optimizer. Every five minutes NEMDE minimizes total
as-bid cost subject to energy balance, capacity limits and ramp rates. So rather
than learn a map `features -> dispatch`, learn the **objective function** of the
optimizer that produced the data, then re-run that optimizer for the
counterfactual:

```
dispatch  =  argmin_p  Cost_theta(p)      s.t.  balance, box, ramp, SOC
```

`theta` (the bid parameters) is the only free part.

**Why this answers requirement 2.** The constraints sit *inside* the argmin, so
every output is feasible by construction. There is no projection head, so planB's
P2 ("the constraint head decides the allocation, not the model") cannot occur —
there is no head to decide anything. The allocation is decided by the recovered
cost curves, which are fit to data.

**Why this answers requirement 1 more strongly than a regression.** Demand enters
**only through the constraint**, never through the objective. The recovered supply
curve is a behavioural primitive that can be argued invariant to a demand shift.
A black-box regression cannot make that argument: its weights encode the demand
distribution it was trained on, so extrapolating them to a shifted demand is
exactly the thing that has no justification.

**The rule that keeps this true.** At every rung of the ladder below, the bid
parameters may depend on calendar, renewable availability, plant availability and
recent bid state. They may **never** depend on demand or price. Break that and it
is a black box with an optimizer stapled on.

---

## 2. What the data says

Four measurements taken before committing to the design. All on 2025–2026
(158,688 five-minute steps), because the fleet composition changed enough that
pooled 2022–2026 describes a system that no longer exists.

### 2.1 The premise holds

Mean output by price band:

| price band | batt_chg | coal | hydro | batt_dis | ocgt | steam |
|---|---:|---:|---:|---:|---:|---:|
| < −$1 | 196 | 2375 | 15 | 14 | 1 | 0 |
| $0–20 | 122 | 3251 | 52 | 34 | 3 | 1 |
| $60–80 | 46 | 4024 | 293 | 100 | 35 | 6 |
| $100–150 | 36 | 3991 | 577 | 135 | 184 | 41 |
| $150–200 | 47 | 3983 | 704 | 163 | 335 | 125 |
| > $1000 | 9 | 3628 | 1063 | 426 | 1192 | 460 |

Every channel is monotone in price with a crisp activation threshold. Share of
steps active (>10 MW) goes 1% -> 100% for OCGT and 0% -> 99% for gas steam across
the same bands. That is the signature of a price-driven optimizer, and it is the
evidence the whole approach rests on.

### 2.2 A static capacity cap is wrong

| channel | daily max, p5 | median | p95 | envelope cap |
|---|---:|---:|---:|---:|
| coal_brown | 3068 | 4144 | 4763 | 4895.8 |
| hydro | 114 | 961 | 1614 | 2168.0 |
| gas_ocgt | 0 | 262 | 880 | 1748.6 |
| gas_steam | 0 | **0** | 510 | 516.4 |

Coal's daily max ranges 2106–4896 MW. The envelope cap sits above the actual
daily max on **100% of days**. Gas steam's median daily max is **zero** — the
plant is simply not available most days.

A model with static caps will over-dispatch coal on outage days and switch steam
on when the plant does not exist. This is very likely the mechanism behind planB's
P3 ("zero-inflated channels are mishandled"), and time-varying availability fixes
it structurally rather than with a loss term.

### 2.3 Coal has a must-run floor

| statistic | value |
|---|---:|
| coal daily min / daily max, median | **0.60** |
| coal daily min (MW), median | 2445 |
| coal mean output when price < 0 | 2418 |
| negative price, share of steps | 23.6% |

Coal holds ~2,400 MW straight through negative prices. Without a lower bound
`p_coal >= P_min` the optimizer will switch coal off whenever price goes negative,
which never happens in reality. So the day's commitment state supplies **both**
bounds, `P_min_it` and `P_bar_it`, not just the upper one.

### 2.4 A static supply curve gets the counterfactual wrong, and the reason names the architecture

Marginal response shares, three ways of measuring:

| source | coal | hydro | ocgt | steam | batt_in | batt_out |
|---|---:|---:|---:|---:|---:|---:|
| cross-sectional price slope | **negative** | high | high | high | — | — |
| within-day price slope | 32.4 | 28.4 | 12.7 | 5.3 | 9.0 | 12.2 |
| within-day net-demand slope | 45.4 | 24.1 | 7.2 | 2.7 | 10.9 | 9.7 |
| **Study A, 3-hour (target)** | **44.1** | **22.4** | **6.5** | **2.4** | **11.5** | **13.0** |

Cross-sectionally coal output *falls* as price rises, because high-price hours are
hours when coal is on outage. That confound is what section 2.2 fixes.

The remaining gap between the within-day price slope and Study A is **ramping**.
Coal is cheap but slow, so its instantaneous response is muted and its three-hour
response is large. This means Study A's timescale gradient (coal 15% -> 34%,
battery 20% -> 11% going from 5 min to 3 h) is an **emergent** property of ramp
limits plus storage.

A multiperiod optimizer reproduces that gradient for free. This is the strongest
argument for the approach and the result no planB arm can produce.

---

## 3. Scope decisions (settled)

| decision | choice | consequence |
|---|---|---|
| system boundary | **net demand only** — the model boundary is the six dispatchable channels | exports, losses, non-scheduled wind and rooftop all fall outside by construction; the balance is an exact identity, not an approximation |
| interconnector | **excluded**, and exports removed from demand | achieved automatically by the net-demand boundary; nothing to reconcile |
| curtailment | **endogenous** — spill is a decision variable | preserved under the net-demand boundary by splitting `nd = nd_free + C` (section 4.3) |
| availability `P_bar_it` | **both, reported side by side** | trailing rolling max for reconstruction (no leak), day's observed max for counterfactuals (genuinely exogenous) |

---

## 4. The system boundary: why net demand, not demand

### 4.1 Demand does not equal total generation, and the reason is definitional

Three findings, in order of size. All 2025–2026.

**(i) The preprocessed `demand_mw` is not AEMO demand.** It already has net exports
added, exactly:

```
table.demand_mw  ==  market.demand_mw  −  net_import        mean |err| 0.0000 MW
```

So the export term is already inside the demand column. Removing it means going
back to `data/vic_market_*.parquet`.

**(ii) Even after removing it, generation exceeds demand by 198.7 MW (4.0%).**

| term | mean MW |
|---|---:|
| scheduled + semi-scheduled generation | 5717.0 |
| − battery charging (a load) | −98.2 |
| + net imports | −496.0 |
| = supply available to operational load | **5122.8** |
| AEMO operational demand | **4925.7** |
| **residual** | **198.7** (std 139.9) |

Stable across years: 199.4 in 2025, 197.2 in 2026. Not noise.

**(iii) The residual is 10.7% of wind output plus 40 MW** (`R^2 = 0.65`,
correlation with wind +0.807). Ruled out by measurement:

| candidate | verdict |
|---|---|
| coal auxiliary load | no — correlation with coal is **−0.247** |
| rooftop PV mis-netted | no — adding rooftop back makes the residual **891 MW**, worse |
| demand-proportional losses alone | no — correlation with demand is **−0.015** |

**The explanation.** AEMO's operational demand counts only generation the market
dispatches: scheduled, semi-scheduled and *significant* non-scheduled plant. Small
and distribution-connected generators are treated exactly like rooftop PV — they
reduce the demand figure rather than appearing as supply against it.
OpenElectricity's generation file meters everything. So roughly 11% of the metered
VIC wind fleet, plus ~40 MW (0.8% of demand, consistent with transmission losses),
sits on the supply side of one meter and the demand side of the other definition.

**Consequence.** Both parts are definitional or physical and neither is something a
dispatch model should explain. But an exact demand-side balance cannot be built
from this data: any demand-side formulation carries a 198 ± 140 MW error no model
can remove, sitting as a permanent floor under every accuracy number.

### 4.2 So the boundary is net demand

```
nd(t)  =  hydro + coal_brown + gas_steam + gas_ocgt − battery_charging + battery_discharging
```

Already built: `data/preprocessed/hist/5min/net_dispatch_totdem/table.parquet`.
The identity holds to **9e-15 MW**.

What this buys:

- no accounting residual, no reconciliation, no constructed target series
- **planB's 50.6 MW interpolation baseline stays directly comparable** — the model
  is scored against the raw meter, unmodified
- exports, losses, non-scheduled wind and rooftop fall outside the boundary by
  construction rather than by assumption, which is a stronger form of "isolated
  Victoria" than deleting the interconnector by hand
- the counterfactual input is a net-demand path, supplied directly

### 4.3 Endogenous curtailment survives this, via `nd = nd_free + C`

A dispatchable-only balance appears to put renewables outside the model. It does
not, if net demand is split into its exogenous part and the spill:

```
nd_free(t)  =  nd(t)  −  C(t)              C = wind_curtailment + solar_curtailment
balance:       SIGN . P  =  nd_free  +  C          identity holds to 9e-13 MW
```

`nd_free` is net demand as it would be if every available renewable MW had run.
That is the exogenous input. `C >= 0` stays a decision variable, bounded above by
available renewables less the network floor.

**The mechanism is in the data.** Spill is forced when `nd_free` falls below what
the must-run fleet can physically produce:

| | `nd_free` < coal must-run (2418 MW) | elsewhere |
|---|---:|---:|
| share of steps | **24.0%** | 76.0% |
| mean curtailment | **1048 MW** | 52 MW |
| price negative | **85%** | 4% |

That 24.0% matches the 23.6% unconditional negative-price rate. So a must-run floor
plus a negative renewable bid reproduces midday curtailment *and* negative prices
without either being fitted. Hourly profile of `nd_free` bottoms at 1907 MW at
midday against coal's 2418 MW floor, which is exactly where curtailment peaks
(707 MW at 12h).

---

## 5. The model

Per 5-minute step `t`. Six dispatchable channels `i`, sign vector
`s = [+1,+1,+1,+1,−1,+1]` in TARGETS order (charging is a load). Two spill
variables `c_w` (wind) and `c_v` (utility solar). One slack `u` (unserved energy).

Exogenous input: `nd_free_t`, net demand as it would be if every available
renewable MW had run (section 4.3).

```
min   sum_t [ sum_i ( a_i * p_it  +  0.5 * b_i * p_it^2 )
              +  g_w * c_w,t  +  g_v * c_v,t
              +  VOLL * u_t ]

s.t.  sum_i s_i * p_it  +  u_t  =  nd_free_t + c_w,t + c_v,t   balance
      0 <= c_w,t <= W_bar_t ,  0 <= c_v,t <= V_bar_t           spill <= availability
      c_w,t + c_v,t  >=  Cmin_t                                network-bound floor
      P_min_it <= p_it <= P_bar_it                             must-run and availability
      −r_dn_i <= p_it − p_i,t−1 <= r_up_i                      ramps
      E_t = E_t−1 + eta*p_chg,t*dt − p_dis,t*dt/eta            SOC
      0 <= E_t <= E_bar
```

Curtailment `C_t = c_w,t + c_v,t` is a decision, and net demand as observed is
recovered as `nd_t = nd_free_t + C_t`. Spilling costs `g_w` per MW (the forgone
certificate value), so the model spills only when the must-run floor leaves it no
choice. `Cmin_t` is the non-releasable component, set to 13.622% of base-case
curtailment and **held fixed in counterfactuals**, because a network limit does
not move when demand does.

**Where negative prices come from.** The balance dual is
`lambda_t = dCost / d(nd_free_t)`. When spill is active and interior, one more MW
of `nd_free` is absorbed by releasing one MW of spill rather than by generating,
so `lambda_t = −g_w < 0`. When spill is zero, `lambda_t` is the marginal
generator's cost and is positive. The negative price is therefore *equal to the
renewable bid floor*, which is a sharp, falsifiable prediction: measured 1st
percentile price is −$60.

### Why each piece is there

- **`a_i`** is the bid price at zero output. It does the **on/off** work: a channel
  sits at exactly zero until the balance shadow price clears `a_i`. This is what
  produces genuine zeros instead of a small positive smear on every channel.
- **`b_i`** is the supply-curve slope. It does the smoothing, makes the solution
  unique, and makes the argmin differentiable.
- **`g_w`, `g_v` are negative marginal costs** for renewables — the certificate
  value that makes a wind farm willing to pay to generate. This is what generates
  **negative prices**. Without it the price floors at zero and 23.6% of the data
  is unreproducible.
- **`u_t` priced at VOLL** means the problem is **never infeasible**. The
  q_max = 4.856% counterfactual that is infeasible on 63/186 days stops being a
  blocker and becomes an economically meaningful output: "this shift sheds N MWh."
- **`N_t`** is the transmission/system-strength limit that makes 13.622% of
  curtailment non-releasable no matter how high demand goes.
- **Multiperiod with ramps and SOC** is what converts a static merit order into
  the observed timescale-dependent response (section 2.4). Solve over the full day
  (288 x 8 variables) so the battery has something to arbitrage against.

### Parameters, and what the data already says about each

Roughly 16 global parameters at rung 1, generating predictions for six dispatch
channels, two curtailment channels and the price series. Every one has a measured
anchor before any fitting, so a nonsense fit is obvious immediately.

| parameter | role | data anchor |
|---|---|---|
| `a_i` | bid intercept, drives on/off | median price when active: chg $11, coal $65, hydro $91, dis $97, OCGT $148, steam $171 (`dispatch_study/response.md` Study B); cross-check `data/fuel_costs.json` |
| `b_i` | supply-curve slope | within-day dp/dlambda: coal 0.256, hydro 0.224, OCGT 0.100, steam 0.042 MW per $/MWh |
| `g_w`, `g_v` | renewable negative bid | 1st-percentile price −$60; market floor −$1000 |
| `P_bar_it` | availability | daily max: coal 2106–4896, steam median 0 |
| `P_min_it` | must-run | coal daily min/max median 0.60; coal 2418 MW at negative price |
| `N_t` | network limit | 13.622% non-releasable curtailment floor |
| `VOLL` | unserved energy | $17,500/MWh market cap |
| ramps, caps, SOC | envelope | `planB/limits.py` unchanged (p99.5 override on coal and steam) |

---

## 6. Fitting theta — three rungs

Each rung is a working model. Later rungs fit strictly better and give a natural
ablation table.

### Rung 1 — read the costs off the data (closed form, seconds)

The spot price **is** the dual variable of the balance constraint. For any step
where a channel is interior (not at 0, not at its cap, ramp not binding),
optimality says

```
a_i  +  b_i * p_it   =   lambda_t   ~=   observed price_t
```

So fitting `(a_i, b_i)` is a linear regression of observed price on observed
output. Boundary observations give inequalities instead: at `p_i = 0` we learn
`a_i >= lambda_t`; at the cap, `a_i + b_i * P_bar <= lambda_t`. So the honest
estimator is a constrained least squares. Convex, closed form, seconds on 500k
rows.

**Treat this as a starting point and an interpretability check, not the final
estimator.** The within-day price slope is biased by ramp-binding, which is
precisely what rung 2 fixes.

### Rung 2 — decision loss (the real model)

Fit what we actually care about:

```
min_theta   sum_t  || p*(nd_free_t ; theta)  −  p_obs,t ||^2
```

where `p*` is the argmin of section 5. Requires differentiating through the
solver.

**Implementation.** The structure is favourable. Per step the only coupling
constraint is one scalar, so for a fixed `lambda_t` each channel is closed form:

```
p_i  =  clip( (lambda_t − a_i) / b_i ,  P_min_i ,  P_bar_i )
```

and `lambda_t` is found by **1-D bisection** on the balance residual. That is
classic lambda-iteration economic dispatch: fully vectorised over the whole
series, and its derivative follows from the implicit function theorem on a scalar
equation, so backprop is cheap and exact. Ramp and SOC couple across time and get
an outer dual-ascent / ADMM loop over the day-sized problem.

`cvxpy` is installable (network verified) and should be used as a slow reference
implementation to validate the fast solver against on a sample of days.

### Rung 3 — contextual bids (closes the accuracy gap)

Sixteen global parameters will not fit tightly, because real bids move with fuel
prices, outages, season and strategic behaviour. So let a small network predict
them:

```
a_it, b_it, P_bar_it, P_min_it  =  f_phi( exogenous context_t )
```

trained end-to-end through the solver. **Context may contain calendar, renewable
availability, plant availability and recent bid state. It may not contain demand
or price.** This is the rung that can plausibly beat the interpolation floor while
keeping every structural guarantee.

---

## 7. Requirement 1 — showing it matches the real grid

| # | test | why it is evidence |
|---|---|---|
| 1 | 400 random 3h test gaps, per-channel MAE/MRE, plus on/off F1 and MAE-given-on | `planB/EVALUATION_PLAN.md` Test 1. **Directly comparable to the 50.6 MW interpolation floor** — same series, unmodified, because the net-demand boundary needs no reconciliation |
| 2 | **model dual `lambda_t` vs observed spot price** | at rungs 2–3 price is never fit against, so this is a genuine out-of-sample prediction |
| 3 | **recovered `a_i` vs measured merit order** | if the intercepts land near chg $11 / coal $65 / hydro $91 / dis $97 / OCGT $148 / steam $171 the recovered object is economically meaningful, not just curve-fitted |
| 4 | **negative-price reproduction** | model should produce price < 0 on ~23.6% of steps, and 99% of those should coincide with active curtailment |
| 5 | curtailment MAE, plus occurrence F1 and level-given-on | against the 43.0 MW interpolation reference; P4 |
| 6 | activation curve by net-demand decile | `EVALUATION_PLAN.md` Test 4; should be nailed once availability is time-varying |
| 7 | full test set, 186 days x 288 steps | not just sampled gaps |

Tests 2, 3 and 4 are the ones to lead with. MAE says "it interpolates well".
Recovering the price series, the merit order and the negative-price regime says
"the mechanism is right", which is what licenses the counterfactual.

---

## 8. Requirement 2 — the counterfactual

- **Feasibility is by construction.** Run `ml/check_caps.py` anyway and expect
  exact zeros. Any nonzero result is a solver bug, not a modelling failure.
- **Infeasible shifts return a number, not an error.** Unserved energy `u_t` is the
  output.
- **Response composition** against Study A 2025–26: coal 44.1 / hydro 22.4 /
  OCGT 6.5 / steam 2.4 / batt out 13.0 / batt in 11.5.
- **Timescale gradient.** Run the shift at 5 min / 30 min / 1 h / 3 h and check the
  model reproduces the *gradient*, not just the 3-hour point (coal 15% -> 34%,
  battery 20% -> 11%). This is the emergent-behaviour test and the strongest
  available result.
- **Shaped demand** (`EVALUATION_PLAN.md` Test 5, raised cosine). The mix should
  slide from battery/hydro on the steep parts toward coal on the plateau, with no
  extra machinery. Boxcar and linear triangle as controls.
- **Curtailment release.** Under a demand rise, spill should fall at a rate near
  the measured `k = 0.42 / 0.33 / 0.24` rule, with the retained fraction not below
  13.622%.

---

## 9. Build order

1. **Forward solver with hand-set parameters from the section 5 table.** Before
   any fitting, run the optimizer on historical demand and look at the output. If
   a 16-parameter guess already gives a sane merit order and sane curtailment, the
   approach is confirmed in a day. If it does not, we find out cheaply.
2. **Build the input series.** `nd_free = nd − curtailment`, the availability and
   must-run bounds `P_bar_it` / `P_min_it` (both variants), the renewable
   availability `W_bar` / `V_bar`, and `Cmin_t`. No target reconstruction is
   needed: the target is the observed dispatch, unmodified.
3. **Rung 1 fit.** Constrained least squares. Produces the interpretability table
   (test 3) immediately.
4. **Rung 2.** Differentiable lambda-iteration solver in torch, `cvxpy` as
   reference.
5. **Rung 3.** Contextual bid network.

Steps 1–4 are enough to answer both questions. Step 5 is what closes the accuracy
gap.

---

## 10. Known risks

| risk | mitigation |
|---|---|
| `Cmin_t` (the non-releasable curtailment floor) is derived from base-case curtailment, so it is not truly exogenous | it is the only constructed input; report counterfactual results with `Cmin = 0` as a sensitivity, which bounds the effect |
| the net-demand boundary means exports, losses and non-scheduled wind are outside the model, so the model cannot answer questions about them | intended, and stated: the claim is about the dispatchable fleet's allocation, not about Victoria's full energy balance |
| identification: gas steam is active 12.8% of the time, so its cost is only set-identified | report `a_steam` as a band, not a point. A model that knows what it cannot identify is a feature |
| a whole-day solve has perfect foresight and will arbitrage the battery too well | rolling-horizon variant (solve 1h ahead, commit the first step) as a realism check; the gap task pins both boundaries anyway |
| `lambda` from a VIC-only model need not equal the VIC regional price when the interconnector is marginal | expected, and it is a consequence of the isolation choice; report the price test with that caveat, not as a failure |
| commitment (which units are on) is genuinely discrete and we are keeping the problem convex | on/off comes from the price threshold plus `P_bar_it`; if steam's on/off F1 is still poor, supply commitment as an exogenous input rather than going MILP |
| marginal loss factors ignored | note as a refinement; unlikely to matter at this resolution |
