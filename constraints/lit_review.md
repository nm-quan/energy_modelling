# Literature review — constrained neural forecasting/dispatch (Stage 5)

Status: ANCHOR LIST from prior knowledge, not yet verified/expanded by search.
Each entry maps to our constraint families (box / ramp / demand / SOC) and to a
mechanism column of the methods matrix (soft / hard-inference / hard-trained).
TODO next pass: verify citations, add bibtex, 2-3 sentence annotations each,
expand each theme via targeted search (keywords listed per theme).

## 1. Hard-constraint layers (hard-trained column)

- **DC3: Deep Constraint Completion and Correction** — Donti, Rolnick, Kolter,
  ICLR 2021. Equality constraints by *completion* (predict a subset of
  variables, solve for the rest — our DemandAnchoredHead is exactly a
  completion layer for the demand identity), inequalities by gradient
  *correction* steps. Directly relevant to: demand, box, ramp.
- **OptNet** — Amos & Kolter, ICML 2017. Differentiable QP layer; exact
  projection inside the network, expensive per-step.
- **cvxpylayers** — Agrawal et al., NeurIPS 2019. General differentiable convex
  layers; candidate for the per-step box-intersect-hyperplane projection QP.
- **Gauge/ray-mapping networks (LOOP-LC line)** — map free outputs onto a
  polytope interior via gauge functions; hard feasibility without iterative
  solves. Keywords: "gauge map hard feasibility neural", "LOOP-LC".
- Search also: "hard linear constraint output layer neural network",
  "KKT-informed layers".

## 2. Soft / Lagrangian (soft column)

- **Lagrangian duality for constrained deep learning** — Fioretto, Van
  Hentenryck et al. (~2020-21), incl. AC-OPF applications. Adaptive multipliers
  (dual ascent) instead of fixed lambda — our Stage 2 adaptive arm.
- PINN-style penalty weighting literature (loss balancing, e.g. GradNorm-style)
  for multi-constraint lambda scheduling.
- Our own evidence: fixed-lambda EB loss pays WAPE without reaching 0
  violations and compliance decays with convergence
  (demand_simulation/findings.md).

## 3. Power-system ML with feasibility guarantees (domain)

- **DeepOPF** — Pan et al. Predict-then-reconstruct for OPF (predict
  independent variables, recover the rest from power-flow equations) —
  completion again, plus post-hoc feasibility restoration.
- Feasibility repair / projection post-processing for OPF proxies; warm-start
  ML + exact solver. Keywords: "DNN OPF feasibility guarantee", "E2ELR
  end-to-end learning repair", "projection layer optimal power flow".
- Dispatch/unit-commitment ML with ramp constraints. Keywords: "learning unit
  commitment ramp constraints neural".

## 4. Forecast reconciliation (hard-inference column, theory)

- **MinT / trace minimization** — Wickramasuriya, Athanasopoulos, Hyndman
  (JASA 2019); earlier GLS reconciliation (Hyndman et al. 2011). Projecting
  forecasts onto a linear-constraint subspace and *provably* not increasing
  (often reducing) expected squared error — the theoretical backing for why our
  demand projection was WAPE-free. Our demand identity = a 1-constraint
  reconciliation; worth citing the optimal-projection weights (we use
  proportional, MinT uses error-covariance GLS — a possible Stage 1 refinement).
- Keywords: "coherent probabilistic forecast reconciliation", "immutable
  series reconciliation" (we hold gas_steam/batt_chg fixed = immutable-series
  variant).

## 5. Safety layers / action projection (SOC, closed loop)

- **Safe exploration via safety layers** — Dalal et al., 2018. Per-step
  analytic projection of actions onto safety constraints — the template for
  the stateful closed-loop SOC swing-budget clip.
- Action masking / shielded RL for storage arbitrage & microgrid control.
  Keywords: "battery arbitrage RL state of charge constraint projection",
  "shielding reinforcement learning energy storage".
