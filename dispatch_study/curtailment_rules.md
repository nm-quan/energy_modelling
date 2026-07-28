# When is curtailment released, and when is it not

Measured on Victoria 2022-01-01 → 2026-07-05, 474,336 five-minute intervals,
7,681,136 MWh of total curtailment. Reproduce with `dispatch_study/response.py`
and the queries in this file's companion scripts.

Curtailment here = `wind_curtailment + solar_curtailment`.

---

## Rule 1 — It is a low-demand phenomenon, and the scaling is steep

| net-demand decile | mean curtailment | active (>10 MW) |
|---|---:|---:|
| D1 (−3,559–1,534 MW) | **1,333 MW** | 99.9% |
| D2 (1,534–2,427) | 327 MW | 91.9% |
| D3 (2,427–3,051) | 117 MW | 63.0% |
| D4 (3,051–3,499) | 56 MW | 39.6% |
| D5 (3,499–3,830) | 33 MW | 25.7% |
| D6 (3,830–4,126) | 24 MW | 19.1% |
| D7 (4,126–4,432) | 19 MW | 15.9% |
| D8 (4,432–4,787) | 14 MW | 12.9% |
| D9 (4,787–5,329) | 11 MW | 10.8% |
| D10 (5,329–8,086) | **8 MW** | 9.1% |

**167× from bottom decile to top.** 85.4% of all curtailment MWh happens in D1
alone. If net demand is already high there is almost nothing left to release —
which is why raising demand at the evening peak frees very little.

---

## Rule 2 — The marginal release rate

MW of curtailment released per MW of net-demand increase (OLS slope of Δcurtailment
on Δnet-demand; negative means curtailment falls as demand rises):

| horizon | all steps | curtailment active | curtailment > 200 MW | price < 0 |
|---|---:|---:|---:|---:|
| 5 min | −0.263 | −0.439 | −0.560 | −0.573 |
| 30 min | −0.204 | −0.359 | −0.506 | −0.542 |
| 1 hour | −0.190 | −0.325 | −0.466 | −0.501 |
| **3 hours** | **−0.204** | **−0.306** | **−0.406** | **−0.434** |

Read the 3-hour row as: **if curtailment is already large and priced below zero,
about 43% of a demand increase is met by releasing spilled renewables rather than
by ramping fuels.** With no conditioning it is 20%.

The rate is highest at 5 minutes and settles by an hour. Fast demand changes are
absorbed disproportionately by releasing curtailment, because releasing costs
nothing and takes no ramping.

---

## Rule 3 — There is a floor, and it is network-bound

At the top demand decile curtailment still averages 8 MW and appears in 9.1% of
steps — and **9.1% of the non-negative-price steps too**. At high demand, price
has stopped mattering entirely: everything still curtailed is curtailed for a
reason demand cannot fix.

By volume across the whole record:

| price | share of all curtailment MWh | interpretation |
|---|---:|---|
| below zero | **84.7%** | economic — the generator stopped because running lost money. Releasable. |
| zero or above | **15.3%** | transmission congestion, system strength, local limits. **Not releasable at any demand.** |

More load in Melbourne does not add transmission capacity out of western Victoria.

---

## Rule 4 — Price predicts *how much*, not *whether*

This is the trap. 84.7% of curtailment **MWh** sits at negative prices, but
curtailment **occurs** at non-negative prices almost as often as at negative ones —
at D1, 99.9% of all steps are curtailing and 97.3% of the non-negative-price steps
are too.

So a step-level rule of the form "price < 0 ⇒ releasable, else not" gets the
aggregate volume roughly right and the individual steps wrong. Use price to scale
the *quantity* released, not as an on/off switch.

---

## Rule 5 — Practical formula

For a counterfactual that raises net demand by Δ at a step where the current
curtailment is `C` and the price is `p`:

```
released  =  min( C,  k · Δ )

k = 0.43   if C > 200 MW and p < 0        strong release
k = 0.31   if C >  10 MW                  curtailment present
k = 0.20   otherwise                      unconditional average
```

then floor it:

```
retained  ≥  0.153 · C          the network-bound share never goes
```

Both bounds matter. Without the cap you release energy that isn't there; without
the floor you predict curtailment reaching zero, which it never does.

---

## Where the current baseline sits

`imputation/nem_baseline.py` releases **4,511 MWh against a 37,558 MWh demand rise
= 12%**. The measured expectation for those midday windows — curtailment active,
mostly large — is 30–43%.

So the baseline **under-releases by roughly a factor of two**, and correspondingly
over-works the fuels. The cause is its per-step cap `min(curtailment, demand rise
at that step)` combined with an all-or-nothing price switch, which Rule 4 says is
the wrong shape. Replacing that switch with the `k` factors above is the obvious
correction and would move roughly 6,000–11,000 MWh from fuel ramping to released
renewables.
