# Literature review — constrained neural forecasting/dispatch (Stage 5)

Status: EXPANDED + VERIFIED by web search 2026-07-14 (was: anchor list from
prior knowledge). Each theme maps to our constraint families (box / ramp /
demand / SOC), to a mechanism column of the methods matrix (soft /
hard-inference / hard-trained), and — new — to the bottleneck-study problems
(P1 WAPE, P2 response, P3 battery; see study_summary.md).

## 1. Hard-constraint layers (hard-trained column; P2 mechanism)

- **RAYEN** — Tordesillas, How, Hutter, arXiv:2307.08336. Interior-anchor +
  ray scaling for convex sets; what our RayenHead implements. Verified; the
  follow-up literature now benchmarks against it.
- **HardNet** — Min & Azizan, arXiv:2410.10807 (rev. 2025). Closed-form
  differentiable *projection* layer for input-dependent affine/convex
  constraints, trained through, with universal-approximation guarantees.
  Directly fits our setting (ramp RHS depends on P_{t-1} = input-dependent).
  Key contrast with our retrofit finding: training THROUGH the enforcement
  layer lets the backbone learn around the constraint — candidate replacement
  for the alpha*-ray step if the rayenfd retrain still under-delivers.
  Extensions: HardNet++/KKT-Hardnet (arXiv:2507.08124) for nonlinear
  equality+inequality via a KKT solve; T-SKM-Net (arXiv:2512.10461)
  sampling-Kaczmarz-Motzkin linear satisfaction; star-shaped sets
  (MDPI Mathematics 12(23):3788, 2024).
- **DC3** — Donti, Rolnick, Kolter, ICLR 2021. Equality completion +
  inequality gradient correction; our DemandAnchoredHead is a completion
  layer. (Unchanged anchor.)
- **Gauge map / LOOP-LC 2.0** — arXiv:2311.04838. Closed-form gauge
  (Minkowski) mapping onto the constraint polytope from an interior point; no
  iterative solves; built FOR power dispatch. The generalized gauge map is
  less input-variance-sensitive than the original — alternative to RAYEN's
  ray scaling with the same zero-solve property.
- **OptNet / cvxpylayers** — Amos & Kolter 2017; Agrawal et al. 2019. Exact
  differentiable QP/convex layers; expensive per step but the fallback if
  closed forms run out.

## 2. Soft / Lagrangian (soft column)

- **Lagrangian duality for constrained learning** — Fioretto, Van Hentenryck
  et al. (2020-21), AC-OPF applications. Adaptive multipliers = our Stage 2
  adaptive arm. Our own evidence (findings.md): fixed-lambda EB loss pays
  WAPE without reaching 0 violations.

## 3. Power-system ML with feasibility guarantees (domain; P2+P3)

- **E2ELR — end-to-end feasible optimization proxies for economic dispatch**
  — Chen, Tanneau, Van Hentenryck, arXiv:2304.11726 / IEEE TPS. Closed-form
  differentiable REPAIR layers (hypersimplex projection) that guarantee power
  balance + reserves; trained SELF-SUPERVISED (minimize dispatch cost, no
  labeled targets). Two imports for us: (a) their balance repair is a
  battle-tested alternative to our reprojection; (b) self-supervised training
  breaks the "imitate history" ceiling that pins us to persistence (P1) —
  the model learns to *dispatch*, not to mimic. Follow-up: self-certifying
  primal-dual proxies (arXiv:2510.15850).
- **DeepOPF** — Pan et al. Predict-then-reconstruct completion. (Anchor kept.)
- **Data-driven merit order** — arXiv:2501.02963 (Energy Economics 2025):
  learn a fundamental merit-order price/dispatch model with plant parameters
  estimated from data — the "model the market mechanism, not the time series"
  reframe, mid-way between our forecasting approach and full market sim.

## 4. Forecast reconciliation (hard-inference column, theory; P1+P3 fix)

- **MinT** — Wickramasuriya, Athanasopoulos, Hyndman, JASA 2019. Projection
  onto linear constraints provably does not increase expected squared error —
  why our anchor was WAPE-free. Refinement we currently ignore: use
  error-COVARIANCE (GLS/WLS) weights instead of proportional rescale.
  **Direct fix candidate for the study's P3 finding**: rayenfd's forced move
  allocates by ramp headroom (battery-heavy -> SOC 0/15 in CL stress);
  allocating by inverse validation-error variance (MinT-style, then ramp-
  clipped) would push the correction into channels that actually absorb it
  in reality.
- **Immutable-forecast reconciliation** — Zhang et al., arXiv:2204.09231.
  Reconciliation holding chosen series fixed — exactly our gas_steam
  passthrough, now with an optimality theory.
- **Non-negative reconciliation** — Forecasting 7(4):64 (2025) review +
  iterative pivoting (fix negatives at 0, move to immutable set) = a
  principled version of our relu/floor handling. Nonlinear constraints:
  arXiv:2510.21249. Review of the field: IJF 2023 (S0169207023001097).
- **Compositional (CoDa) fuel-mix forecasting** — Shang & Han,
  arXiv:2510.25185 (forecasting AUSTRALIAN generation by fuel mix).
  Model SHARES on the simplex (softmax head = balance for free) x a total —
  an alternative head design: mix shares x pinned nd(t-1) gives exact
  balance + structural response with a bounded, interpretable mix. Batteries
  need a signed extension (charging is a load), e.g. completion for batt_chg.

## 5. Safety layers / action projection (SOC, closed loop; P3-B)

- **Safety layers** — Dalal et al. 2018 (per-step analytic projection);
  provably-safe action projection via reachability (arXiv:2210.10691).
- **Physics-shielded DRL for microgrids w/ battery health** (ScienceDirect
  2025, S0016003225006040): stateful SOC shield with dynamically adjusted
  thresholds — the published template for our P3-B stateful clip.
- Safe-RL-for-power-systems reviews: arXiv:2407.00304, arXiv:2409.16256.

## 6. Batteries as optimizers, not time series (NEW; P3 + battery WAPE)

Batteries dispatch by price arbitrage optimization, not by autocorrelation —
which is why every model (and persistence) floors at WAPE ~0.25-0.27 on the
battery channels. Modeling routes:

- **Opportunity value function prediction** — Zheng & Xu et al.,
  arXiv:2211.07797: predict the SOC opportunity-cost function, dispatch by
  maximizing it; ~90% of perfect-foresight profit. Battery head = tiny
  value-based controller (price + SOC in, charge/discharge out) with SOC
  dynamics INTERNAL -> SOC feasible by construction.
- **Learning reachability of storage arbitrage** — arXiv:2512.06600: learn
  reachable SOC sets, constrain dispatch to them.
- **DRL + price forecasting for arbitrage** — arXiv:2410.20005: forecasts
  materially improve arbitrage control even when noisy.
- **Differentiable MPC** — Amos et al. 2018 (arXiv:1810.13400), PNNL
  differentiable predictive control: embed a small storage MPC as the battery
  head, train through it.

## 7. Closed-loop stability / exposure bias (NEW; P1 closed-loop, seam snap)

Teacher-forced h=1 MSE trains "noisy persistence" that collapses in closed
loop (our behaviour/report.md and the rayenfd base_WAPE 2.32 finding). The
sequence-model literature calls this exposure bias:

- **Scheduled sampling** — Bengio et al. 2015: mix own predictions into
  training inputs on a curriculum.
- **Flipped Classroom** — Teutsch & Maeder, arXiv:2210.08959: curriculum
  teacher-forcing schedules for time series specifically.
- **Soft-token trajectory forecasting** — arXiv:2512.10056 (2025):
  propagate distributions instead of point feedbacks; differentiable.
- Import for us: fine-tune the rayenfd backbone with scheduled sampling over
  3h windows (matching the free-window protocol) so the closed-loop mix stops
  drifting — attacks the SAME failure as the seam-snap ramp violations.

## 8. Counterfactual / what-if trajectory estimation (NEW — the simulation
##    track's true task family; see study discussion 2026-07-15)

Our scenario runs are formally: estimate the dispatch trajectory under a
MODIFIED exogenous driver (demand +g%, price 0) inside a window whose
surroundings are observed. Fields that do this successfully:

- **Causal ML for counterfactual outcomes over time** (medicine/econ):
  RMSN (Lim, NeurIPS 2018), Counterfactual Recurrent Network (Bica, ICLR
  2020), Causal Transformer (Melnychuk, ICML 2022), G-Transformer (2024).
  Treatment sequence = our demand modification. Validated on SEMI-SYNTHETIC
  worlds where counterfactual truth is known by construction — the formal
  version of the masked-gap + perturbation training idea.
- **Demand-response baseline estimation** (the closest applied cousin,
  deployed commercially): baseline load = "what consumption WOULD have been
  without the DR event" — the mirror image of our task (they subtract an
  intervention, we inject one). Settlement money is paid on these
  counterfactuals daily. Validation: accuracy on NON-event days = exactly
  our reconstruction-WAPE gate. Reviews: Energies 15:5259 (2022); physics-
  informed baselines (Applied Energy 2025); generalized synthetic control
  for DR (arXiv:2604.18469).
- **Synthetic control / CausalImpact** (policy evaluation; energy
  applications incl. market liberalization, renewable action plans):
  counterfactual trajectory from a weighted donor pool. Borrow the
  **placebo test**: run the full pipeline on windows with NO modification —
  the estimated "response" must be ~0. Cheap, strong validity check we can
  run verbatim.
- **Building-energy M&V "avoided energy use"** (IPMVP Option C, ASHRAE
  Guideline 14): an INSTITUTIONALIZED counterfactual with uncertainty
  standards; LBNL evaluation of savings-uncertainty methods; NeurIPS 2025
  interpretable counterfactual M&V. Borrow: report prediction intervals on
  counterfactual fills (fractional-uncertainty discipline).
- **Conditional scenario generation for grids** (cGAN/diffusion:
  operation-adversarial cGAN internalizing DC-OPF, arXiv:2110.02152;
  DiffCharge 2308.09857; production-simulation augmentation 2412.12146):
  generative models conditioned on drivers/risk labels, some with physics
  embedded — the generative sibling of our hard-constraint head.
- **Industry practice**: resource-adequacy and DOE reliability studies
  re-run 12+ historical weather years hour-by-hour with modified load
  traces; production-cost models (PLEXOS-class) are the optimizer-based
  standard for exactly our question. Our learned simulator is the ML
  surrogate of that workflow.

Cross-field validation toolbox for counterfactuals (no ground truth needed):
(1) held-out reconstruction on UNMODIFIED data [we have: reconstruction
WAPE]; (2) placebo runs, effect ~0 on null scenarios [cheap, to add];
(3) semi-synthetic worlds with known counterfactual truth [= perturbation
training data]; (4) physics/constraint compliance [we have: hard head];
(5) uncertainty quantification on the fill [optional, M&V-style].

## 9. Constrained subspace imputation — the professor's reframe, per pipeline
##    stage (2026-07-15; supervisor ref: GRIN, arXiv:2108.00298)

The simulation track recast as: impute the missing BREAKDOWN subspace of the
free window (totals known every step, carrying the scenario), bidirectionally
(both boundaries observed), under hard physics. Literature per stage:

- **Stage 0/1 — task = disaggregation (known total -> unknown split)**:
  NILM/energy disaggregation: seq2point review (IEEE 10775789), attention
  NILM (arXiv:1912.00759), VAE disaggregation (arXiv:2103.12177),
  multi-task disaggregation w/ injection identification (arXiv:2508.14600).
  Our balance identity = their aggregate-consistency, but ours is exact.
- **Stage 2 — bidirectional imputation engines**: GRIN (arXiv:2108.00298,
  ICLR'22 — bidirectional GRU + message-passing GNN across channels; the
  supervisor's reference; our 6 sources + drivers form exactly such a small
  relational graph), BRITS (1805.10572), SAITS (2202.08516), diffusion:
  CSDI (2107.03502), SSSD (2208.09399), MTSCI consistent conditional
  diffusion (2408.05740), ImputeFormer (KDD'24 3637528.3671751).
- **Stage 3 — hard constraints on the fill**: our themes 1/3/4 (RAYEN,
  HardNet, DC3, E2ELR, reconciliation) + the two-sided RAMP TUBE
  (per-step band = intersection of forward cone from p_L and backward cone
  from p_R; feasible iff |p_R - p_L| <= (N+1) r). Published combination of
  generation+hard constraints: **Constrained Posterior Sampling**
  (arXiv:2410.12652) — projection into the constraint set after each
  diffusion denoising step; and Chronos-based imputation with feasibility
  projection for energy (ScienceDirect S2666546826001060). These confirm
  the pairing exists but is recent/rare — combined with ramp+balance+SOC
  and counterfactual conditioning it is the project's gap/novelty.
- **Stage 4 — training so the filler READS demand (not boundary-blending)**:
  masked-block reconstruction + perturbed known-subspace/boundaries
  (professor's synthetic augmentation); formal cousin = semi-synthetic
  counterfactual benchmarks of theme 8 (CRN/Causal Transformer).
- **Stage 5 — evaluation ladder**: dumb baseline = per-source linear
  interpolation + balance rescale + ramp-tube clip (the imputation-track
  persistence analogue); reconstruction WAPE on unperturbed held-out gaps;
  placebo (g=0 -> capture ~0); violations = 0 incl. BOTH seams; optional
  M&V-style uncertainty intervals (theme 8 toolbox).

## Priority shortlist (mapped to the study ladders)

1. MinT-weighted delta allocation in RayenHeadFixedD (theme 4) — cheap,
   targets P3 battery overuse + P1 ocgt regression simultaneously.
2. Scheduled-sampling fine-tune of the rayenfd backbone (theme 7) — targets
   closed-loop mix quality (base_WAPE 2.32 -> ?) and the seam snap.
3. Battery-as-optimizer head (theme 6) — the only route found that could
   beat persistence on the battery channels; medium effort.
4. E2ELR-style self-supervised objective (theme 3) — escapes the imitate-
   history ceiling entirely; largest reframe, largest potential.
5. HardNet / gauge-map layer swap (theme 1) — only if the rayenfd retrain
   (study ladder B) still fails P1/P2.
