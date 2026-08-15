# What blocks publication — BiLSTM + hardnet, 3h imputation

Adversarial read of the current arm, done by running the ablations rather than
inspecting the code. Scripts: `planB/ablation_head_vs_network.py`,
`planB/ablation_observable_nd.py`. All numbers are 400 random test 3h gaps,
seed 123, checkpoints from `colab/planB_results`.

---

## 1. Fatal — the head is given a linear functional of the labels

`fit.py::_supply_side_nd` overwrites the `net_demand` feature with
`SIGN · dispatch`, computed **before** the dispatch columns are masked. The
dispatch columns are then zeroed (`fit.py:189`) but `ND_COL` is not. So at every
masked step, both the network input and the constraint head receive

```
nd(t) = SIGN · P_true(t)
```

which `gap_data.py` verifies is an exact identity against the very six channels
being predicted. Six unknowns, one exact equation, at every step of the gap.

The defence in the docstring is that net demand "is known at every step even
inside the gap — only the BREAKDOWN is unknown". Regional demand *is* metered
independently, so the argument is not absurd. But the quantity fed to the model
is not the metered one. The same docstring records that the observable
(`demand − wind − solar − curtailment`) differs from `SIGN · dispatch` by "mean
391 MW and up to 4,974 MW". Measured on the test gaps here: **mean 505 MW, max
4,303 MW.**

Substituting the observable quantity at inference — which is what deploying on a
real outage forces — gives this:

| configuration | disp agg MAE | balance residual |
| --- | ---: | ---: |
| as published (`nd = SIGN·truth`) | **41.2 MW** | **0.0 MW** |
| observable `nd`, head only | 98.3 MW | 3,271.6 MW |
| observable `nd`, features and head | 98.7 MW | 3,271.6 MW |
| linear interpolation, no `nd` at all | 50.6 MW | 1,502.2 MW |

Two things break at once, and they are the two headline claims:

- accuracy degrades 2.4×, to **worse than linear interpolation**
- *"balance exact by construction"* becomes a 3,272 MW violation — the head
  projects exactly onto the wrong plane

A reviewer only has to ask "where does `nd` come from at inference?" to get here.
This is not a leakage technicality to be argued down; it is the difference
between a result and no result.

## 2. The reported baseline is the wrong one

`interp` is not the right reference, because it is the only row denied `nd`. The
honest control is the same interpolation pushed through the same head with the
same `nd` — **zero learned parameters**:

| arm | params | disp agg MAE |
| --- | ---: | ---: |
| interp (no nd) | 0 | 50.6 |
| **interp + head** | **0** | **49.2** |
| persistence + head | 0 | 82.4 |
| zeros + head | 0 | 467.9 |
| hardnet (trained) | 1,222,664 | **41.2** |
| rayen (trained) | 1,223,049 | 66.4 |
| unconstrained (trained) | 1,222,664 | 80.7 |

hardnet is genuinely better than the parameter-free control — by **16.2%**, not
the 18.6% implied by quoting against raw interp. That is a real but modest return
on 1.2M parameters, and it is the number that should be reported.

Everything else in the table loses to a projection with no network in it. `rayen`
is 35% worse than `interp + head`; `unconstrained` 64% worse. Only one arm
survives its own control.

## 3. A SOC constraint will not rescue this

Adding the reservoir constraint is a no-op at a 3h gap. Swing over each gap
against `E_max` = 3,966 MWh (see `quick_findings/vic_battery_constraints/`):

| | median | p99 | max | windows over E_max |
| --- | ---: | ---: | ---: | ---: |
| truth | 495 | 2,069 | 2,365 | **0 / 400** |
| hardnet | 452 | 1,736 | 2,106 | **0 / 400** |

The worst model window reaches 53% of the reservoir. The constraint is inactive
on every window, for truth and model alike, so implementing it changes no
prediction and no metric. It would begin to bind on the 1-day (`blackout`, 288
step) task — that is where it belongs, and only as a plausibility guarantee, not
an accuracy gain: projecting onto a set that already contains the truth can never
hurt, but here it also never fires.

## 4. Everything else a reviewer will raise

- **Baselines are a strawman.** BRITS at 113.1 MW aggregate is not a credible
  BRITS; it reads as untuned. No MICE, kNN, matrix factorisation, or a modern
  diffusion/attention imputer at parity. `planA/benchmark.md` has the same shape
  — every learned arm loses to `interp+map`.
- **One region, one split, one seed.** VIC only, seed 123, no repeats, no
  confidence intervals. NSW data is already pulled and unused for this.
- **Gaps are synthetic and MCAR.** Uniform random 3h windows. Real telemetry
  outages cluster and correlate with the events that make dispatch unusual, which
  is the regime that matters.
- **Aggregate MAE is a coal metric.** Channels span 4,000 MW (coal) to ~0
  (gas_steam, 96% zeros). The aggregate is dominated by the easiest channel;
  `gas_steam` shows 5.9 MW MAE at 65% MRE. `EVALUATION_PLAN.md` P3 already calls
  for on/off F1 and MAE-given-on; it is still not built.
- **Documentation drift.** `planB/results/eval.md` still says "1 epoch" and
  "p99 ramp envelope" while `CONSTRAINT.md` is p99.9 and the colab numbers are
  from a longer run. `planA/benchmark.md` reports arms that memory records as
  being the projected interpolation rather than model output.

## 5. What is actually publishable here

**The curtailment result.** Curtailment is the only channel the head never
touches, so it carries no `nd` leak. SAITS reaches 40.2 MW against
interpolation's 43.0 — a clean 6.5% win on an untainted channel. Small, but it is
the one number in the project that survives every ablation above.

**The constraint machinery**, if it is given a legitimate `nd`. The projection is
exact on balance, box, ramp and both seams, idempotent to 1.3e-9, and
differentiable. That is a real methods contribution; it is currently attached to
an illegitimate input.

## 6. The two ways out

**(a) Close the balance on observables.** Use metered regional demand and carry
the interconnector — or an explicit residual channel — as a predicted quantity,
so the identity the head enforces is built from things that still exist during an
outage. This keeps it an imputation paper. It is the harder path, and the 505 MW
mean discrepancy is exactly the thing that has to be modelled rather than
assumed away.

**(b) Stop calling it imputation.** In a counterfactual or scenario setting, `nd`
is *prescribed by the scenario*, so feeding it to the head is correct by
definition and the leak disappears. The constraint layer, the feasibility
guarantees and the dispatch-response work all transfer intact, and
`optimization/design.md` is already headed this way. The cost is that the
reconstruction benchmark stops being the headline claim.

(b) is the honest framing of what has actually been built. (a) is the paper the
current write-up claims to be.
