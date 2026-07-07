First with the machine learning approach:
Top down from constraints 
-> we need net Dt ( predicted net demand) to be equal to sum of predicted energy dispatch
-> we need to follow ramp rate 

To do this we implement a machine learning model to predict these dispatch and demand 
For the constraint parts we impose this by 

Output vector: y = (Dₜ, P₁, ..., Pₙ), where Dₜ = predicted net demand at time t, Pᵢ = predicted dispatch of generator i, n = number of generators.
The layer (replaces your network's final layer):
Anchor point p = (ΣPᵢ,ₜ₋₁, P₁,ₜ₋₁, ..., Pₙ,ₜ₋₁), where Pᵢ,ₜ₋₁ = generator i's actual output at the previous step. This point satisfies everything simultaneously: its sum equals its demand entry, and zero change respects every ramp limit.
Network produces a raw direction vector r (n+1 numbers) and one scalar s.
Stay on the equality plane: remove from r its component along the vector a = (−1, 1, ..., 1) (a = the normal of the plane ΣPᵢ − Dₜ = 0): r ← r − (a·r / a·a)·a, where a·r = dot product. Any movement along the corrected r now preserves ΣPᵢ = Dₜ exactly.
Respect all ramps: α* = minᵢ ( Rᵢ / |rᵢ| ), where Rᵢ = max allowed change of generator i per step, rᵢ = the entry of r for generator i, α* = the largest step from p before any ramp wall is hit.
Output: y = p + σ(s)·α*·r, where σ(s) = sigmoid of s, a number between 0 and 1.
Every prediction satisfies ΣPᵢ = Dₜ and every ramp limit jointly, by construction, with exact gradients — this is exactly the Konstantinov/RAYEN  you can check the pdf 
Evaluation : 
Demand Balance: Regular Test Set 
Ramp Rate: The simulated Test set 

Metrics: WAPE (for energy prediction), n_violations ramp, n_violations demand, % violations ramp ( raw number mwh to percentage), % violation demand, battery feasibility 

Models to test with : transformer RevIn and base LSTM 

---

## Resolutions & implementation notes (2026-07-07)

1. SIGN vector: battery_charging is a load -> plane normal a = (-1, 1,1,1,1,-1,1)
   (identity: SIGN.P - D = 0), TARGETS order [hydro, coal, gas_steam, gas_ocgt,
   batt_chg, batt_dis].
2. Asymmetric ramps: alpha* uses R_up_i when r_i>0, R_dn_i when r_i<0 (data
   ramps from CONSTRAINT.md / check_caps.RAMPS incl. battery empirical bounds).
3. Floors added (deviation from spec, flag include_floor=True): walls P_i >= 0
   in alpha* with eps-interior anchor p = max(P_{t-1}, 0.5 MW), plus pre-zeroing
   of descent components on floored channels before the plane projection.
   Without this, gas_steam (at 0, ramp-down 500) can emit -500 MW per step.
   Ramp guarantee then holds within +-0.5 MW (the eps), covered by metric tol.
4. D_t is the model's own demand forecast: balance is guaranteed vs itself
   (@own = 0 by construction). Demand metrics get three refs: @own (mechanism
   check), @in (vs nd(t-1)), @act (vs actual nd(t) = D_t forecast quality).
   Responsiveness to exogenous demand shifts is NOT structural here (unlike the
   inference anchor) -- measured separately, may be low.
5. Backbones emit RAW directions (8 outputs: 7-dim r + scalar s):
   lstm_rayen = plain LSTM; itransformer_rayen = iTransformer with input RevIN
   but output denorm OFF (new revin_out flag). RevIN output denorm would add
   per-window means onto direction components.
6. Zero-direction = persistence: the layer's null output is the persistence
   baseline, so the network learns bounded constrained deviations from the
   strongest baseline -- the right inductive bias given stage-0 findings.
7. Protocol: strided pilot (--stride) before full runs (trained-through layers
   have failed here before); bare backbones trained alongside for the Pareto
   comparison; training on Colab via ml/train_hist.py + colab/train_hist.ipynb.

---

## Approach 2 — decision rules (hardlinearconstraints.pdf), resolutions (2026-07-07)

1. POST-HOC ONLY (user decision): task net = 7-output backbone (D_t, P_1..P_6)
   trained with plain MSE on the same y7 targets as the rayen arm
   (--arch {lstm,itransformer}_task7); equality projection (paper eq. 16) +
   safe blend (eq. 4) applied at inference. The bare task7 row doubles as the
   unconstrained Pareto baseline. itransformer_task7 keeps FULL RevIN (outputs
   are levels, not directions).
2. Feasible set includes floors P_i >= 0 (user decision) — identical guarantee
   set to the RAYEN layer.
3. Safe net input x = [1, P_prev(6), nd_prev]; input polytope X = box
   {0 <= P_prev_i <= 1.3*CAPS, nd in train range +-25%}. LP (eq. 14, per-row
   Prop.-2 duals, scipy/HiGHS) gives t* = 29.45 MW (= gas_steam R_up/2, the
   binding wall); solution is interpretable: persistence + fixed positive
   offsets, D = its own signed sum. Inputs escaping X are detectable at
   runtime (s_SN < 0, logged as n_outside_X — 0 on the test set).
4. Evaluation protocol (user decision): demand balance teacher-forced on the
   regular test set; RAMP RATE on the "simulated test set" = closed-loop
   autoregressive rollout over the 15 TEST-split stress episodes
   (constraints/mine_episodes.py -> stress_episodes.json), models feeding back
   their own dispatch, exogenous features actual.
5. Percent conventions: ramp_excess_pct = 100*sum(MW beyond limit)/sum(|pred|);
   mismatch_act_pct = 100*sum(|SIGN.pred - nd_act|)/sum(|nd_act|). Battery
   feasibility = per-day (TF) / per-episode (CL) SOC-window existence,
   eta_rt 0.834 (whole-window SOC is meaningless on hist — tracker finding).

Implementation map: lib/decision_rule.py (LP + DecisionRuleHead),
lib/models.py::make_task7, ml/train_hist.py (_task7 suffix + post-hoc DR row),
constraints/eval_hist_models.py (TF + CL tables),
constraints/test_constraint_layers.py (feasibility-at-any-weights tests, 6/6
pass), colab/train_hist.ipynb (full design + runbook).
