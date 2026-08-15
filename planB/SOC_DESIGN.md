# The battery SOC constraint — what it is, and how to make the model obey it

Goal: a constraint that is a **true statement about the Victorian grid**, which
the model's output obeys on every day, the same way the real data does.

Numbers verified on `data/preprocessed/hist/5min/net_dispatch_ren/table.parquet`.
Reproduce with `planB/soc_design_experiments.py`. The physics is derived in
`quick_findings/vic_battery_constraints/` — panels B and C of the figure there
show the daily SOC curve and the drift problem described in §3.

---

## 1. A battery is a bucket

Charging puts energy in, discharging takes it out. Call the amount in the bucket
right now `E(t)`, in MWh. Every 5 minutes:

```
E(t) = E(t−1) + ( chg(t)·η − dis(t)/η ) · Δt

  Δt = 5/60 h        one timestep
  η  = 0.916         one-way efficiency
```

The `η` is losses. Multiply going in, divide coming out, so putting in 100 MWh
gets you about 84 MWh back — the round trip is η² = 0.839. That figure is not
assumed, it is measured: over Dec 2025 – Jul 2026 the fleet drew 1,296,899 MWh
and returned 1,075,382 MWh, and 0.839 is the ratio. It is also the only value
that makes the bucket neither fill nor drain over the window.

The constraint is just: **the bucket can't overflow and can't go below empty.**

```
0 ≤ E(t) ≤ E_max
```

## 2. Why this is a different *kind* of constraint from the ones you already have

| constraint | how many timesteps it looks at |
| --- | --- |
| cap: `chg ≤ 1611.5 MW` | one |
| ramp: `\|chg(t) − chg(t−1)\| ≤ 804.6 MW` | two |
| **SOC** | **all of them, from the start of the day to now** |

This is the whole difficulty. The cap and the ramp can be checked on a value in
isolation. SOC cannot — it constrains the *accumulation*.

Concretely: discharging at the full 1,687 MW for one 5-minute step is completely
fine. Doing it for three hours straight is not, because that is
1,687 × 3 = 5,061 MWh and the bucket only holds 3,966. **The constraint forbids
no single number. It forbids a pattern.** That is why it can't be written as a
box on the outputs, and why it needs a running state.

## 3. The hard part: you can never see how full the bucket is

The data gives you power in and power out. It never gives you `E`. So you can
compute how much the level *changed*, but not what it *was*.

You might think: start at `E = 0` on the first day and add up. That fails. `η` is
only constant on average (it varies week to week, 0.72–0.94), so a tiny error
each step accumulates like a random walk. Run it across 217 days and the computed
level wanders over a range of 5,960 MWh — wider than the bucket it is supposed to
live in. No choice of `η` fixes this; changing it just trades a positive drift for
a negative one.

**So the absolute level is not a usable quantity.** What *is* usable is the
difference between the highest and lowest point over a short period — call it the
**swing**. The swing needs no starting value, because the unknown constant
cancels when you subtract.

```
swing over a period  =  max E(t)  −  min E(t)
```

And the physical statement survives the rewrite: if the bucket holds `E_max`, the
level cannot move more than `E_max` from its low to its high.

```
swing ≤ E_max
```

## 4. Why "per day" — and why your instinct is the right test

The period has to be short enough that drift hasn't accumulated, and long enough
to contain the behaviour we care about. **One calendar day** is both, and not for
convenience: the VIC fleet runs exactly one cycle a day. It charges on midday
solar (~470 MW mean, 11:00–14:00, at $14–16/MWh) and discharges into the evening
peak (~410 MW mean, 18:00–19:00, at $79/MWh). SOC bottoms out around 08:00 and
peaks around 16:00. The daily cycle is a real feature of this grid.

**Your point is exactly the acceptance test.** Real dispatch satisfies the
constraint necessarily — it physically happened. So any constraint we write must
admit every real day, or the constraint is wrong, not the grid. Checked:

| window | days | median daily swing | max daily swing | days violating `E_max` = 3,966 |
| --- | ---: | ---: | ---: | ---: |
| 2026 | 186 | 2,656 MWh | 3,775 MWh | **0** |
| Dec 25 – Jul 26 | 217 | 2,727 MWh | 3,966 MWh | **0** |

That is how `E_max = 3,966 MWh` is chosen: it is the largest daily swing the
fleet has ever produced. Set it lower — say 3,000 — and you would be declaring 58
real days physically impossible. Set it to the registry nameplate of 4,735.75 and
it admits everything but never constrains anything, because the fleet has never
used more than 84% of its registered storage.

## 5. The specification

```
η      = 0.916                one-way (round trip 0.839)
E_max  = 3,966 MWh            largest observed daily swing
period = calendar day         reset the accounting at midnight

for each day d, over the steps t in that day:
    E(t)  =  E(t−1) + ( chg(t)·η − dis(t)/η ) · Δt      E at the day's start = 0
    max E(t) − min E(t)  ≤  E_max
```

`E` starting at 0 each day is not a claim that the battery is empty at midnight.
It is just fixing the unobservable constant. Only differences matter, and the
swing is a difference.

## 6. How to make the model's output obey it

This is the part that turns the idea into code, and it is simpler than it looks.

You process the model's output one step at a time, carrying the running level `E`
and the running min and max seen so far today. Before accepting step `t`, work
out how much room is left:

```
highest level still allowed :  U = min_so_far + E_max
lowest  level still allowed :  L = max_so_far − E_max
```

`E(t)` must land inside `[L, U]`. Since `E(t)` is just `E(t−1)` plus this step's
energy, that turns straight into limits on **this step's power**:

```
chg(t)  ≤  ( U − E(t−1) ) / (η · Δt)          can't charge past the overflow
dis(t)  ≤  ( E(t−1) − L ) · η / Δt            can't discharge past empty
```

So: **a constraint about accumulated energy becomes a shrinking box on this
step's power.** And a box on this step's power is exactly what the existing
projection head already handles, alongside the cap and the ramp. Nothing about
the head changes — you just hand it tighter bounds each step.

`planB/heads.py::_Soc.tighten` already implements this, including the conservative
"assume the other channel is zero" step that keeps the two limits independent.
The work is wiring it in and choosing the period, not writing new maths.

## 7. What to expect when you turn it on

Be ready for it to do nothing visible on the 3-hour task. Over 400 test gaps, the
worst SOC swing is 2,365 MWh for the real data and 2,106 MWh for the model —
53% of the bucket. The constraint is satisfied before you impose it, so it will
not change a single prediction and will not move MAE.

That is not a failure. It is a speed limit on an empty road: correct, and quiet.
The reason is that this is *aggregate* fleet data — 15 units at different states
of charge, so the total is never near its edge, even though individual 2-hour
units hit their own limits constantly. Only per-unit telemetry would change that,
and there is none in this repo.

It starts to matter in three places, and these are the honest reasons to build it:

1. **Whole-day imputation** (the 288-step `blackout` arm). The worst real 24-hour
   swing is 3,978 MWh against `E_max` 3,966, so it is genuinely on the boundary.
2. **Counterfactuals and demand-shift rollouts**, where the model is pushed
   outside observed behaviour and can produce a battery that discharges for three
   hours from an empty bucket. Nothing else in the constraint set catches that.
3. **It is the only hard constraint that does not depend on `nd`.** Per
   `PUBLICATION_REVIEW.md`, the balance plane relies on a net demand computed from
   the labels and collapses when the observable one is substituted. SOC is a
   statement about the battery's own energy and is untouched by that problem.

## 8. Separate point, kept for later

While measuring the above I found that `chg` and `dis` are nearly redundant as
model outputs. With `n = dis − chg` and `m = min(chg, dis)`:

```
chg = m + max(0, −n)          dis = m + max(0, n)
```

which is exact, and makes non-negativity automatic. `m` is small (mean 7.7 MW,
zero on 13% of steps), so predicting `m = 0` and only getting `n` right already
gives 7.7 MW MAE on both channels, against the current model's 42.7 / 51.4.

This is an accuracy idea, not a constraint idea, and it is independent of
everything above. Do not let it complicate the SOC work — noted here so it is not
lost.
