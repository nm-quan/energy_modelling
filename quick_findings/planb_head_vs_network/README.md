# Is SAITS constraint-following? No. Neither is any other planB arm.

400 random test 3h gaps, seed 123, trained checkpoints from `colab/planB_results`
(not `planB/results`, which still holds the 1-epoch versions).

## 1. The raw output before the head

`raw_feasibility.py`. F = interp skeleton + network residual, in MW, audited
against the same p99.9 envelope the head enforces. `head move` is
`hardnet_head(..., return_debug=True)["correction"]` — the MW the projection then
has to shift. hardnet is the identity on a feasible input, so this is a direct
read of how far off the network was.

| raw output, pre-head | bal mean | bal max | %>1MW | neg max | cap max | ramp mean | head move mean | % identity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| interp (no network) | 200.1 | 1502.2 | 99.6% | 0.0 | 0.0 | 0.00 | 200.1 | 0.4% |
| TRUTH | 0.0 | 0.0 | 0.0% | 0.0 | 0.0 | 0.16 | 0.4 | 98.9% |
| unconstrained (bilstm) | 110.1 | 1370.2 | 99.2% | 114.9 | 6.5 | 3.85 | 124.2 | 0.3% |
| hardnet (bilstm) | 315.0 | 1726.0 | 99.7% | 409.6 | 0.0 | 3.12 | 646.4 | 0.0% |
| brits | 820.3 | 3762.4 | 99.9% | 509.2 | 78.8 | 22.02 | 1452.3 | 0.0% |
| **saits** | **339.9** | **2081.6** | **99.8%** | **325.5** | **167.8** | **4.84** | **540.1** | **0.0%** |

SAITS raw is 340 MW off balance on average — worse than a straight line between
the pinned boundaries (200 MW) and worse than the head-free BiLSTM (110 MW). It
also runs 325 MW negative and 168 MW over cap. The projection moves it 540 MW per
step and is never within 1 MW of the identity. Every feasibility claim for the
`saits` arm is the hardnet head, not the backbone.

The ordering is not an accident. `unconstrained` is trained on its own raw output,
so the loss pushes it toward the truth, which is balanced. The head arms are
scored only *after* projection, and the projection is not injective, so their raw
output is free to sit anywhere on the fibre. Nothing ever penalises it.

## 2. What the head alone achieves

`head_ablation.py`. Project the interp skeleton with the network residual set to
zero. Anything an arm gains over that row is the network; anything it does not is
hard-coded.

| | hydro | coal | steam | ocgt | bat_chg | bat_dis | disp agg | curt agg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| interp (no head, no net) | 63.4 | 85.8 | 2.0 | 15.0 | 64.7 | 72.5 | 50.6 | — |
| **hardnet_head(interp) — HEAD ONLY** | 53.3 | 80.9 | 19.9 | 27.7 | 55.2 | 58.0 | **49.2** | — |
| hardnet (bilstm) = head + net | 52.2 | 72.3 | 5.9 | 22.6 | 42.7 | 51.4 | 41.2 | 59.8 |
| brits = head + net | 156.4 | 231.0 | 43.6 | 90.7 | 84.5 | 70.2 | 112.7 | 120.2 |
| saits = head + net | 57.3 | 80.0 | 3.1 | 20.2 | 50.9 | 52.6 | 44.0 | 40.2 |

The head-only floor is **49.2 MW** with no network at all. hardnet's network buys
8.0 MW on top of it, SAITS's buys 5.2 MW. So roughly 85-90% of every dispatch
number in `eval.md` is the interp skeleton plus the projection.

Curtailment is the opposite case: no head, no skeleton, so `curt agg` is the
network and nothing else. SAITS 40.2 is the only value under the 43.0
interpolation reference; the BiLSTM arms are 60-77 and BRITS is 120.

## 3. Is hardnet exact? Yes — and `eval.md`'s 62 MW is stale

`hardnet_exactness.py`. Post-head residual `|SIGN.P - nd|` over the same 400
windows, 14,400 steps:

| input to the head | exact | mean | max | below zero | above cap | ramp overshoot |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TRUTH | 100.00% | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 |
| interp, zero network | 100.00% | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 |
| hardnet (bilstm) raw | 100.00% | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 |
| saits raw | 100.00% | 0.000 | 0.00 | 0.00 | 0.00 | 0.00 |

Same in float32, which is what `eval.py` runs — so the 62.24 / 62.48 / 63.13 MW
in `colab/planB_results/eval.md` is not a live number. It was measured under the
**p99.9** envelope; commit 308162f (Aug 3) switched `limits.py` to empirical-max
ramps with p99.5 overrides on coal_brown and gas_steam only. The wider box now
always meets the balance plane. The hardcoded note in `eval.py` ("feasibility
floor at p99.9 is 0.21 MW MAE and 12.8 MW worst balance") is stale for the same
reason — the floor is now 0.00. Re-run `planB/eval.py` to refresh both.

Headroom, not a coincidence — attainable range of `SIGN.P` over the per-step box
vs where `nd` falls in it:

```
box width      mean 3,691 MW, min 2,667 MW
nd slack       min 431.2 MW, p1 738.2, median 1,115.0 MW
plane misses the box:  0 / 14,400 steps
slack < 100 MW:        0 / 14,400 steps
```

Box and ramp are exact *by construction* (the last operation is a clip into an
interval built from them). Balance is exact *whenever the plane meets the box* —
a data-dependent condition, not a theorem, and currently satisfied with ~431 MW
to spare at the tightest step. SOC is enforced by neither head.

## 4. rayen vs hardnet against the "this is just hardcoded" charge

`rayen_vs_hardnet.py`. Each head's fixed part = its output with zero network
contribution. For hardnet that is `hardnet_head(interp)`; for rayen it is the
anchor `A`, which `rayen_head` returns in its debug dict.

Run under both envelopes, because the sensitivity is itself the discriminator.

| dispatch MAE (MW) | p99.9 (as trained/evaluated) | current limits.py |
| --- | ---: | ---: |
| interp (no head, no network) | 50.6 | 50.6 |
| **hardnet fixed part** `hardnet_head(interp)` | 49.2 | 49.2 |
| hardnet full | **41.2** (net buys +8.0) | **41.2** (net buys +8.0) |
| **rayen fixed part 1** glide `c`, geometry only | 56.8 | 64.8 |
| **rayen fixed part 2** anchor `A`, zero network | 57.4 | 66.7 |
| rayen full | **57.5** (net buys **−0.1**) | **66.4** (net buys +0.3) |

Under the envelope it was actually trained and evaluated on, **rayen's entire
accuracy is the anchor** — the ray makes it 0.1 MW worse. And this is not the
plan1 `alpha* -> 0` collapse: median `sigmoid(step)` is 0.789, the head walks a
median 18.5 MW per step, and only 7.3% of steps are no-ops. The network moves;
it just doesn't help. The anchor is a hand-written rule (glide toward `pR`, then
a room-weighted balance snap), so the rayen row in `eval.md` is a report on that
rule, not on the BiLSTM.

Three structural asymmetries behind the numbers:

1. **Identity on a feasible input.** hardnet returns the truth unchanged on 98.2%
   of steps (mean residual 0.30 MW; the rest is where the truth itself breaks the
   envelope). If the network learns to be feasible the layer disappears. rayen
   cannot be tested this way at all — it consumes a *direction*, not a dispatch
   level — and can never be the identity, because it always starts at `A`.
2. **Envelope sensitivity.** hardnet scores 41.2 under both envelopes, unchanged.
   rayen goes 57.5 -> 66.4, a 16% swing from editing a hand-specified constant,
   because `A` and `alpha_max` are both *constructed from the box*. rayen's answer
   is partly a function of the envelope you wrote down; hardnet's is not.
3. **No free design choices.** "Closest point" is a definition. The anchor's glide
   rate and its room-weighted snap are choices, and moving them moves the answer.

The one real point in rayen's favour: under the tight p99.9 envelope rayen was
balance-exact (0.00 MW) where hardnet was not (62.24 MW worst step). rayen's
conservatism keeps its trajectory away from states where the plane misses the
box. But that is the same property as the weakness — it stays near the anchor
because it ignores the network.

Not a free pass for hardnet: 49.2 of its 41.2 is the fixed part, and it rides the
interp skeleton on top. The difference is that hardnet has a *defence* against
the hardcoding charge and rayen does not.

## 5. Why hardnet gives gas steam 8% — it is the projection, not the model

`steam_decompose.py`. Wrap `heads.hardnet_head` to record `(F_in, P_out)` on every
call, run the normal 10-day counterfactual for the trained hardnet arm, then split
the response exactly:

```
resp      = E_c - E_b                    (what the report table shows)
resp_net  = F_c - F_b                    (raw BiLSTM, pre-head)
resp_proj = (P_c - F_c) - (P_b - F_b)    (what the projection added)
```

Masked mode, MWh in the window:

| channel | response | share | from BiLSTM | from projection | real grid |
| --- | ---: | ---: | ---: | ---: | ---: |
| hydro | +9,822 | 25.0% | +1,879 | +7,944 | 29.3% |
| coal | +12,742 | 32.5% | +3,052 | +9,690 | 43.4% |
| **gas steam** | **+3,276** | **8.3%** | **+118** | **+3,158** | **3.7%** |
| gas OCGT | +3,258 | 8.3% | −333 | +3,591 | 9.7% |
| battery in | −6,628 | 16.9% | −93 | −6,535 | 6.6% |
| battery out | +3,504 | 8.9% | −144 | +3,648 | 7.4% |

**96% of the gas-steam response is the head.** Across all channels the BiLSTM
supplies 4,665 MWh of the +39,230 required — **12%**. The projection supplies 88%.

The mechanism is visible in the mean levels inside the counterfactual window:

| channel | raw F | after head | moved |
| --- | ---: | ---: | ---: |
| hydro | 169.1 | 364.9 | **+195.8** |
| coal | 3,992.7 | 4,160.7 | **+168.0** |
| gas steam | **−60.6** | 109.4 | **+170.1** |
| gas OCGT | **−87.7** | 108.6 | **+196.3** |
| battery in | 134.4 | 86.4 | −48.0 |
| battery out | **−76.3** | 121.0 | **+197.3** |

Every unclipped channel moves by nearly the same number of MW. That is the
signature of `P = clip(F + lam*s, lo, hi)`: one scalar `lam`, so the correction is
split **equally in MW**, never in proportion to channel size. Coal takes +168 on a
3,993 MW base; gas steam takes +170 on a base of −61. The equal-split limit is
1/6 = 16.7% per free channel; gas steam lands at 8.3% only because its ramp cap
clips it partway there. Nothing in the map knows that gas steam is a 516 MW
peaker and coal is a 4,896 MW baseload fleet.

Three channels — gas steam, OCGT, battery discharging — have **negative** raw
network output. Their entire level in the fill is manufactured by the clip at zero
plus the `lam` shift.

For contrast, the pure BiLSTM (`unconstrained`, no head) allocates: coal 71%,
hydro 16%, OCGT 3%, battery out 3%, **gas steam 1%**. Its shape is "coal absorbs
everything", and it misses the energy target by 2,020 MWh (358 balance
violations). Caveat: that checkpoint is 1 epoch (see section 0 note).

### 5b. Like-for-like: same data, same epoch, head vs no head

`unconstrained_2025` trained here to match `hardnet_2025` exactly — 1 epoch,
hidden 192, seed 0, 40k windows, `--train-range 2025-01-01 2026-03-31`:

    python3 planB/fit.py --arm unconstrained --epochs 1 \
        --train-range 2025-01-01 2026-03-31 --tag unconstrained_2025
    python3 planB/counterfactual.py --days 10 \
        --arms hardnet_2025 unconstrained_2025 --suffix _2025ab

val loss 0.7723 (no head) against 0.7009 (hardnet head). Masked mode, current
empirical-max envelope, MWh in the window:

| channel | hardnet_2025 | unconstrained_2025 | real-grid share of +39,230 |
| --- | ---: | ---: | ---: |
| hydro | +8,253 (21%) | +337 (7%) | 11,494 (29.3%) |
| coal | +11,506 (29%) | +3,534 (**72%**) | 17,026 (43.4%) |
| **gas steam** | +4,585 (**12%**) | +150 (**3%**) | 1,452 (3.7%) |
| gas OCGT | +4,683 (12%) | +198 (4%) | 3,805 (9.7%) |
| battery in | −5,348 (14%) | −452 (9%) | 2,589 (6.6%) |
| battery out | +4,855 (12%) | +234 (5%) | 2,903 (7.4%) |
| **Σ signed** | **+39,230** (exact) | **+4,905** (12.5%) | needs +39,230 |
| raw violations bal/ramp/neg | 0 / 24 / 0 | 360 / 26 / 565 | — |

The pure BiLSTM's gas-steam *share* is 3%, essentially the real 3.7%. But that is
3% of a response that is **8x too small** — in absolute MWh it delivers 150 where
the real grid's share would be 1,452. It under-delivers every channel and dumps
what little it has on coal (72%).

So the correct statement is not "the projection invents a gas-steam response the
network did not want". It is: **the network produces only 12.5% of the required
energy, so the head has to supply the other 88%, and it splits that 88% equally in
MW.** Both numbers are wrong in opposite directions — hardnet 3.2x over on steam,
the bare BiLSTM 10x under. The 12.5% here matches the 12% measured independently
in section 5 on the full-history trained hardnet's own backbone.

Artifacts: `planB/results/unconstrained_2025.pt`,
`counterfactual_10day_2025ab.png`, `counterfactual_2025ab.md`.

**Reading.** hardnet's allocation is not a learned merit order — it is
least-squares. "Smallest correction" is scale-blind, and every counterfactual
share it reports is closer to `1/n` than to the merit order. This is the strongest
form of the "just hardcoded" objection, and unlike the accuracy question it has no
defence: the number in the results table is the head's arithmetic. A size- or
cost-weighted norm (MinT-style, `lit_review.md` theme 4) would change the
allocation without touching feasibility.

## 6. Can a weighted norm fix it? Partly — and it exposes a real inconsistency

`weighted_head.py`. Minimise the weighted norm instead:

    P* = argmin sum_i w_i (P_i - F_i)^2   s.t.  s.P = nd,  lo <= P <= hi

KKT gives the same one-scalar shape, `P = clip(F + lambda*s/w, lo, hi)`, and
`g(lambda)` is still monotone (slope `sum_{free} 1/w_i > 0`), so the same
bisection works; the differentiable formula divides by `sum_{i in A} 1/w_i`
instead of `|A|`. Drop-in, same exactness, ~10 changed lines.

**The inconsistency it fixes.** `fit.py:forward` computes the loss as MSE in
Z-SPACE but calls the head in RAW MW. So the head's notion of "closest" and the
loss's notion of "closest" disagree. `w = 1/sigma^2` makes them agree.

hardnet_2025, masked, 10 days. Inference-time re-weighting on a model trained
through the equal head, so this is indicative of the allocation rule only:

| channel | equal `w=1` (current) | `w=1/sigma` | `w=1/sigma^2` (loss units) | REAL |
| --- | ---: | ---: | ---: | ---: |
| hydro | +8,253 (21%) | +11,342 (**29%**) | +11,890 (30%) | **29.3%** |
| coal | +11,506 (29%) | +21,170 (54%) | +24,981 (64%) | **43.4%** |
| **gas steam** | +4,585 (**12%**) | +1,973 (**5%**) | +600 (2%) | **3.7%** |
| gas OCGT | +4,683 (12%) | +3,538 (**9%**) | +1,862 (5%) | **9.7%** |
| battery in | −5,348 (14%) | −841 (2%) | +136 (0%) | **6.6%** |
| battery out | +4,855 (12%) | +367 (1%) | +34 (0%) | **7.4%** |
| Σ signed | +39,230 | +39,230 | +39,230 | needs +39,230 |

`w = 1/sigma` is the best of the three: hydro 29 vs 29.3 and OCGT 9 vs 9.7 are
essentially exact, gas steam drops 12% -> 5%. It overshoots coal (54 vs 43.4) and
**collapses the battery** (2%/1% against 6.6%/7.4%).

Why the battery breaks: its `y_scale` is 52-56 MW, the smallest of the six,
because the historical fleet was small. Batteries are HIGH-response and
LOW-variance in this data, and no sigma-derived weight can express that. Tuning
`w` until the shares match `EMP` would reproduce the measured response by
construction — which is precisely the "hardcoded" charge in a new place.

**So: land `w = 1/sigma` (or `1/sigma^2`) for consistency with the loss and to
kill the worst artifact, but do not present any fixed `w` as the fix.** The head
is deciding ~88% of the allocation only because the network supplies 12.5% of the
response (sections 5, 5b). Every weighting is a choice about how to distribute an
amount the model never predicted. The root cause is that training only ever sees
reconstruction of historical gaps, where `nd == SIGN.P` already holds, so there is
no gradient that teaches the network what to do when demand moves — the
perturbation / counterfactual-training gap already named in
`constraints/lit_review.md` themes 8 and 9 (stage 4).

## 7. The evening peak, and whether random sampling is to blame

From `planB/results/peak_18_21_bilstm.md` — gap 18:00-21:00, 185 test days:

| arm | battery-out MAE | battery-out share of gross gen | dispatch agg MAE |
| --- | ---: | ---: | ---: |
| **actual** | — | **6.8%** | — |
| interp | 129.8 | 5.9% | 55.1 |
| rayen | **97.5** | **6.0%** | 63.9 |
| unconstrained | 201.4 | 3.2% | 106.5 |
| **hardnet** | **226.7** | **2.7%** | **120.4** |

hardnet at the peak is 3x worse than its own random-gap score (41.2) and 2x worse
than a straight line, and it loses 60% of the battery. Confirmed.

**But the sampling hypothesis does not survive measurement** (`coverage.py`). A
timestep is masked by any of `gap`=36 possible starts, so
`P(never masked) = (1 - 36/N)^n`:

```
--train-range 2025-01-01 2026-03-31  (hardnet_2025, unconstrained_2025)
  timesteps NEVER masked   2 of 130,753   (0.002%)
  times masked per step    mean 11.0, p1 4, median 11
  18-21h  10.8x/step   vs elsewhere 11.1x   -> ratio 0.977
  share of masked cells in 18-21h: 12.2%    (uniform would be 12.5%)

DEFAULT npz train split (every headline planB arm)
  timesteps NEVER masked   10,042 of 394,272  (2.5%)
  times masked per step    mean 3.7
```

**Sliding window would not change this.** Stride-1 is also uniform over time of
day: 3.3x more windows (10x on the full split) at the *same* 12.5% peak share. It
closes a 2.5% coverage gap that does not exist on the recent range, at 3-10x the
epoch cost, and does not touch the peak's share of the gradient.

Two other candidate causes also die on inspection:

* **"It does not know it is the peak."** It does. `hour_sin`, `hour_cos` and an
  explicit `is_peak` flag are features 9, 10 and 16, and none of them are zeroed
  inside the gap — only the 6 dispatch channels and the 2 curtailment channels
  are. Demand, price, wind and solar are all visible through the gap too.
* **"The loss under-weights battery."** The opposite. Loss is MSE in z-space and
  battery `sigma` is 52 MW against coal's 681. A 226.7 MW battery error is 4.3
  z-units; coal's 175.9 MW error is 0.26. The loss already weights battery ~13x
  harder per MW. It is screaming and the model still cannot do it.

**What the numbers do say.** The model is *worse than a straight line* on battery
at the peak. That is not an under-fit rare event, it is a learned residual adding
error. The ordering — rayen (mostly its boundary-driven anchor) best, interp
(boundaries only) second, and the two arms that lean hardest on the network worst
— says the pinned boundaries already carry most of the peak battery signal and
every learned component degrades it. Same 12%-contribution story as sections 5
and 5b, from another angle.

**The steel-manned version of the hypothesis is still worth testing**, and it is
not about coverage but about the peak's share of the *gradient*: 12.2%. Two cheap
experiments that do change it, where sliding window does not:

1. **Peak-stratified sampling** — force a fixed fraction (say 30%) of training
   windows to have a gap covering 17:00-21:00. One line in `fit.sample`.
2. **Early-stop on peak val loss, not global.** Val loss is currently a global
   average, so the 14-epoch stop for hardnet optimised the bulk; nothing in that
   criterion knows the peak exists.

Beyond that, `lit_review.md` theme 6 already records that every model and
persistence floors at WAPE 0.25-0.27 on the battery channels because batteries
dispatch by price arbitrage, not autocorrelation — a generic sequence model may
be the wrong function class for that channel regardless of sampling.

## 8. Peak-stratified sampling — the experiment, and it works

Sliding window was the wrong lever (section 7: coverage is already ~11x/step and
uniform over the clock). The right one is the peak's share of the GRADIENT.
`planB/fit.py` gained `--peak-frac`, `--peak-hours`, `--early-on` and a second
peak-only validation set, all defaulting to the previous behaviour.

    python3 planB/fit.py --arm hardnet --epochs 1 \
        --train-range 2025-01-01 2026-03-31 --tag hardnet_2025_base
    python3 planB/fit.py --arm hardnet --epochs 1 \
        --train-range 2025-01-01 2026-03-31 --peak-frac 0.30 --tag hardnet_2025_pk30
    python3 planB/peak_eval.py --arms hardnet_2025_base hardnet_2025_pk30 --hour 18

One epoch each, identical seed/config, single variable.

| | train | val (global) | **val_peak** |
| --- | ---: | ---: | ---: |
| base — uniform (8.3% of windows open 17-18h) | 0.6131 | 0.5249 | **0.8389** |
| pk30 — 30% forced | 0.6906 | 0.5222 | **0.7479** |

**val_peak −10.8% at no cost to the global val** (0.5249 -> 0.5222). Train loss
rises because peak windows are genuinely harder, not because anything regressed.

Peak test, 185 test days, gap 18:00-21:00:

| arm | hydro | coal | steam | ocgt | bat_chg | **bat_dis** | **disp agg** | curt agg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| interp | 95.3 | 74.8 | 3.9 | 17.7 | 8.8 | 129.8 | **55.1** | 27.5 |
| hardnet_2025_base | 77.6 | 85.1 | 13.2 | 36.9 | 6.6 | 108.0 | **54.6** | 26.3 |
| **hardnet_2025_pk30** | 82.6 | 79.5 | 9.7 | 34.0 | 6.6 | **92.4** | **50.8** | **25.5** |

Battery-out share of gross generation: actual **6.8%**, interp 5.9, base 5.5,
**pk30 6.1** — the closest any arm has come.

Two results:

1. **pk30 is the first planB arm to beat linear interpolation at the evening
   peak** (50.8 vs 55.1). Every arm in `peak_18_21_bilstm.md` lost to it.
2. Stratification alone is worth **−7.0% dispatch agg** (54.6 -> 50.8) and
   **−14.4% battery discharge** (108.0 -> 92.4). bat_dis 92.4 also beats the
   previous best of any arm, rayen's 97.5.

**Do not compare these to the 120.4 / 226.7 in `peak_18_21_bilstm.md`.** That
`hardnet.pt` has no `residual` key, so `peak_eval.run_arm` runs it with
`residual=False` — a different forward path, on top of a different data range and
epoch count. The base-vs-pk30 pair above is the only single-variable comparison
here.

`--early-on peak` is a no-op at one epoch, but the 0.8389 vs 0.5249 spread shows
the signal is there: a global-average val loss can improve while the peak decays,
and 14-epoch early stopping was being driven entirely by the former.

Artifacts: `planB/results/hardnet_2025_{base,pk30}.pt`, `peak_18_21_pk30ab.md`,
`peak_18_21_dayavg_pk30ab.png`, `peak_18_21_channels_pk30ab.png`.

## 9. Does the peak fix carry to the counterfactual? No.

    python3 planB/counterfactual.py --days 10 \
        --arms hardnet_2025_base hardnet_2025_pk30 --suffix _pk30cf

Masked mode, MWh in the window. Both feasible (0 balance / 24 ramp / 0 negative):

| channel | base | pk30 | REAL |
| --- | ---: | ---: | ---: |
| hydro | +8,253 (21%) | +7,980 (20%) | 29.3% |
| coal | +11,506 (29%) | +10,711 (27%) | 43.4% |
| gas steam | +4,585 (12%) | +4,877 (**12%**) | 3.7% |
| gas OCGT | +4,683 (12%) | +5,618 (14%) | 9.7% |
| battery in | −5,348 (14%) | −4,951 (13%) | 6.6% |
| battery out | +4,855 (12%) | +5,092 (13%) | 7.4% |
| Σ signed | +39,230 | +39,230 | needs +39,230 |

Network-vs-projection split (`steam_decompose.py`), Σ signed of the BiLSTM part
as a share of the +39,230 required:

| | network supplies | projection supplies |
| --- | ---: | ---: |
| hardnet_2025_base | **5.8%** | 94.2% |
| hardnet_2025_pk30 | **5.0%** | 95.0% |
| hardnet (full history, 14 ep) | 12% | 88% |

**Peak stratification changed nothing about the counterfactual.** Gas steam stays
at 12%, the network share does not rise, and the allocation is the projection's
equal-MW split either way. Two reasons, and only the first is fixable by
stratification:

1. **Window mismatch.** The stratification targeted gaps opening 17:00-18:59; the
   counterfactual free window is **11:00-14:00**. Different time of day, no
   reason to expect transfer. Testable in ~7 min with `--peak-hours 10 11`.
2. **Training never teaches demand response.** Every training window is gap
   reconstruction on historical data, where `nd == SIGN.P` holds by identity.
   Changing WHICH gaps are reconstructed cannot create a gradient for "demand
   moved, now what". Reason 2 survives any choice of `--peak-hours`.

**What did improve** — the head-free channel, again. Base-case predicted
curtailment 12,223 -> **10,085** against an actual 10,693: the error drops from
1,530 to 608 MWh, −60%. Consistent with the peak eval (curt agg 26.3 -> 25.5,
solar-curtailment MRE 110.1 -> 63.3). And the raw output starts closer to
feasible: base-case `|correction|` 19,756 -> 14,685 MWh.

So stratification produced a genuinely better backbone — visible everywhere the
projection is not in the way — and left the counterfactual untouched, because the
counterfactual is decided by the head.

## 10. `hardnet_alloc` — the split as a model output

New head in `planB/heads.py`. The network emits 12 outputs: 6 dispatch levels and
6 allocation logits. `p = softmax(logits)`, `w = 1/p`, fed to
`project_box_plane_w`. Softmax is the right parameterisation because the
allocation MUST be a distribution -- the balance identity forces
`sum_i s_i a_i = 1` -- so the network cannot emit an inconsistent split, the same
way the projection cannot emit an infeasible dispatch. Balance / box / ramp stay
exact for any `p`, and the map is still the identity on a feasible `F`.

    python3 planB/fit.py --arm hardnet_alloc --epochs 1 \
        --train-range 2025-01-01 2026-03-31 --alloc-weight 1.0  # supervised
    python3 planB/fit.py --arm hardnet_alloc --epochs 1 \
        --train-range 2025-01-01 2026-03-31 --alloc-weight 0.0  # reconstruction only

| arm | val | val_peak |
| --- | ---: | ---: |
| hardnet_2025_base (plain head) | 0.5249 | 0.8389 |
| hardnet_2025_alloc_sup | 0.4830 | 0.7448 |
| **hardnet_2025_alloc_unsup** | **0.4612** | **0.7195** |

Both beat the plain head on reconstruction AND at the peak; the UNSUPERVISED one
is best (−12% val, −14% val_peak). Giving the projection a learned split is not a
tax on accuracy, it is an improvement.

### Merit-order score (`merit_score.py`, 400 held-out test gaps)

`p` against the measured `s_i*a_i`. `p` is a network output computed from the
inputs at inference, so this measures whether the allocation is LEARNED.

| | hydro | coal | steam | ocgt | batt in | batt out | MAE | CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MEASURED (target) | 0.250 | 0.344 | 0.006 | 0.050 | 0.167 | 0.184 | — | — |
| equal split 1/n (plain hardnet) | .167 | .167 | .167 | .167 | .167 | .167 | 0.1643 | 1.7918 |
| alloc_sup | 0.221 | 0.368 | 0.013 | 0.052 | 0.178 | 0.168 | **0.0727** | **1.2600** |
| alloc_unsup | 0.189 | 0.208 | 0.070 | 0.097 | 0.193 | 0.242 | 0.1360 | 1.6092 |

Supervised recovers 56% of the gap from the equal split to the measured merit
order. **Unsupervised recovers 17%** -- told nothing about merit order, it still
moved gas steam 0.167 -> 0.070 and coal 0.167 -> 0.208, in the right directions.

### Counterfactual (masked)

| channel | base | alloc_unsup | alloc_sup | pooled real |
| --- | ---: | ---: | ---: | ---: |
| hydro | 21% | 21% | 17% | 29.3% |
| coal | 29% | 47% | 52% | 43.4% |
| **gas steam** | **12%** | **3%** | **1%** | **3.7%** |
| gas OCGT | 12% | 4% | 3% | 9.7% |
| battery in | 14% | 14% | 17% | 6.6% |
| battery out | 12% | 11% | 10% | 7.4% |

mean |share error| vs pooled: base 7.55 -> **alloc_unsup 4.88** -> alloc_sup 7.22
(sup overshoots coal). Gas-steam error 8.3pp -> **0.7pp**. Feasibility unchanged
(0 balance / 24 ramp / 0 negative for all three).

Network share of the response energy: base 5.8% -> **unsup 12.3%, sup 11.3%**.
But that decomposition now UNDERSTATES the network, because it only counts the
contribution from positioning `F`; the projection's remaining ~88% is itself
being aimed by `p`, which is a network output.

**The finding that matters.** `alloc_unsup` has the BEST reconstruction and the
WORSE merit order of the two alloc arms. That is the whole thesis in one line:
gap reconstruction with pinned boundaries does not require dispatch logic, so
optimising it does not produce dispatch logic. The signal is there -- 17% of it
gets picked up for free -- but the objective barely rewards it.

## Reading

SAITS is the best *learner* in planB and the worst *constraint follower* of the
converged arms. Those are not in tension — it is scored post-projection, so there
is no gradient telling it to be feasible. If raw feasibility is wanted, the loss
has to see the raw output: a penalty on `|SIGN.F - nd|` before the head, or
supervision on F as well as P.

## 11. Constraint verification of the counterfactual

`verify_cf.py` wraps every head call so the EXACT tensors the head produced are
audited, rather than re-derived. Worst value over the 10 counterfactual days,
masked mode, free window 11:00-14:00:

| arm | case | balance | below 0 | above cap | ramp in-window | ramp seams | SOC swing |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hardnet_2025_base | base | 5.00e-04 | 0 | 0 | 3.81e-06 | 1.12e-04 | 1,732 MWh |
| hardnet_2025_base | cf | 7.55e-04 | 0 | 0 | 1.12e-04 | 3.56e-04 | 956 MWh |
| alloc_unsup | base | 4.88e-04 | 0 | 0 | 2.13e-04 | 1.12e-04 | 1,733 MWh |
| alloc_unsup | cf | 6.10e-04 | 0 | 0 | 1.12e-04 | 3.56e-04 | 1,065 MWh |
| alloc_sup | base | 5.09e-04 | 0 | 0 | 2.13e-04 | 1.12e-04 | 1,739 MWh |
| alloc_sup | cf | 6.10e-04 | 0 | 0 | 1.12e-04 | 8.45e-04 | 1,156 MWh |

All four hard constraints hold to **float32 rounding** (worst 8.45e-04 MW on a
~5 GW stack), including both seams, and `below 0` / `above cap` are exactly zero
because the final operation is a clip. The learned allocation head is exactly as
feasible as the plain one -- as it must be, since `w` only rescales the travel
direction and cannot move the feasible set.

**SOC is not enforced and is not violated here either**, but only by luck of the
window length: the 3h swing peaks at 1,739 MWh against a 4,736 MWh reservoir
(4,536 MWh effective after `cyclic_project`'s 200 MWh margin). That is 38% of the
reservoir in three hours, and it says nothing about the absolute state of charge,
which is unobserved. A daily-window check would be the real test.

### The 24 ramp flags are the data, not the model

`counterfactual.md` reports 24 ramp flags per arm, counted over the FULL DAY after
the filled window is spliced back in. The untouched actual dispatch over the same
10 days has **29**, all in `coal_brown`:

    per day: [2, 5, 4, 3, 1, 0, 0, 5, 6, 3]

So the models flag FEWER violations than the historical data does -- the 3h they
fill replaces some real coal ramps with feasible ones. Every remaining flag comes
from actual dispatch outside the window. And they are all coal, which is the one
channel `limits.py` deliberately tightens to p99.5, with the documented cost that
"these two channels' own history exceeds their limit ~1.0% of the time".
