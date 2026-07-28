"""How much of the counterfactual response is the MODEL, and how much is the
constraint layer's weight rule?

THE DECOMPOSITION (exact algebra, not a fit). The head's output is, by
construction,  P = A + alpha* r.  Running it twice -- base drivers and shifted
drivers -- and subtracting:

    P_cf - P_base = (A_cf - A_base)                        <- ANCHOR term
                  + (alpha*_cf r_cf - alpha*_base r_base)  <- RAY term

The two terms sum to the total identically. What separates them:

  ANCHOR term  A = cyclic_project(interp, pL, pR, nd). Between the runs nd moves
               (nd_cf = nd_base + d_demand) and pL/pR move (context is the scaled
               dispatch), so A moves. cyclic_project NEVER sees the network, so
               every MWh here is the balance weight rule + the ramp/box operators.

  RAY term     r = F - A, and F is the BiLSTM's fill. This is the ONLY place the
               network enters the output. Its share is the honest answer to
               "did the model decide the allocation, or did I?".

Reported: per-channel MWh for each term over the free window, and the global
ray share  sum|ray| / (sum|anchor| + sum|ray|).

A, r and alpha* are read out of sizeaware_constraints_reference.rayen_shot
(return_debug=True) because constraint_layers.rayen_traj_project does not expose
its internals. That reference is verified bit-identical to the deployed function
(see its _verify(): max |diff| = 0.0 MW), so these are the deployed head's numbers.

SCOPE: free window (11:00-14:00) only, first --days full test days, soc=False
(matching plan1_figures.rayen's default). Does NOT include
plan1_figures.project_full_day, so this measures the HEAD's response, not the
final planA/energy_change_size_aware.md totals.

    python3 imputation/plan1_attribution.py                    # size_aware, 40 days
    python3 imputation/plan1_attribution.py --arm baseline --days 20
"""
import argparse
import sys, numpy as np, pandas as pd, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/"imputation")); sys.path.insert(0, str(ROOT/"lib"))
import plan1_figures as F
from gap_data import TARGETS, SIGN
from model import BiLSTMImputer
from shift_model import FixedPercentageShift
from sizeaware_constraints_reference import rayen_shot
DT = 5.0/60.0

ap = argparse.ArgumentParser()
ap.add_argument("--arm", default="size_aware", choices=["baseline", "cost", "size_aware"])
ap.add_argument("--days", type=int, default=40, help="first N full test days")
args = ap.parse_args()
SA = args.arm == "size_aware"

te, feat_cols, xm, xs_, ym, ys_ = F.load_test()
CTX, GAP, FREE, Q = F.CTX, F.GAP, F.FREE, F.Q
hour = te.index.hour; free_all = (hour>=FREE[0]) & (hour<FREE[1])
sh = FixedPercentageShift(Q, Q, free_hours=FREE).transform(
    pd.DataFrame({"demand_mw": te["demand_mw"].values,
                  "price_aud_per_mwh": te["price_aud_per_mwh"].values}, index=te.index))
d_dem = sh["demand_mw"].values - te["demand_mw"].values
nd_base = te[TARGETS].values @ SIGN; nd_cf = nd_base + d_dem
nd_feat_cf = te["net_demand"].values + d_dem
m = BiLSTMImputer(n_features=len(feat_cols))
m.load_state_dict(torch.load(F.WEIGHTS/f"{args.arm}.pt", map_location="cpu", weights_only=True)); m.eval()
days = te.index.normalize(); full=[d for d in pd.unique(days) if (days==d).sum()==288]
g0 = FREE[0]*12
t_ = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64)

A_term = np.zeros(6); R_term = np.zeros(6); n=0
for d in full[:args.days]:
    dr = np.where(days==d)[0]; rows = dr[g0:g0+GAP]
    span = np.arange(rows[0]-CTX, rows[-1]+1+CTX)
    if span[0]<0 or span[-1]>=len(te): continue
    truth = te.iloc[dr][TARGETS].values.astype(np.float64)
    fb,pL,pR = F.model_fill(m,te,feat_cols,xm,xs_,ym,ys_,rows,None,None)
    _,db = rayen_shot(t_(fb),t_(pL),t_(pR),t_(nd_base[rows]),size_aware=SA,
                      anchor_iters=400,return_debug=True)
    dd = truth*(nd_cf[dr]/nd_base[dr])[:,None]
    ctx=np.zeros((len(span),6))
    for k,r in enumerate(span):
        p=np.where(dr==r)[0]; ctx[k]= dd[p[0]] if len(p) else te.iloc[r][TARGETS].values
    ov={"demand_mw":sh["demand_mw"].values[span],"net_demand":nd_feat_cf[span],
        "price_aud_per_mwh":np.where(free_all[span],0.0,te.iloc[span]["price_aud_per_mwh"].values)}
    fc,pL2,pR2 = F.model_fill(m,te,feat_cols,xm,xs_,ym,ys_,rows,ov,ctx)
    _,dc = rayen_shot(t_(fc),t_(pL2),t_(pR2),t_(nd_cf[rows]),size_aware=SA,
                      anchor_iters=400,return_debug=True)
    A_term += (dc["anchor"].numpy() - db["anchor"].numpy()).sum(0)*DT
    R_term += (dc["alpha"]*dc["direction"].numpy()
               - db["alpha"]*db["direction"].numpy()).sum(0)*DT
    n+=1

tot = A_term + R_term
print(f"Counterfactual response attribution, free window, {n} test days, {args.arm}.pt\n")
print(f"  {'channel':22s} {'anchor term':>13s} {'ray term':>12s} {'total':>13s} {'ray share':>10s}")
print("  " + "-"*74)
for i,t in enumerate(TARGETS):
    sh_ = 100*abs(R_term[i])/(abs(A_term[i])+abs(R_term[i])+1e-9)
    print(f"  {t:22s} {A_term[i]:+13,.0f} {R_term[i]:+12,.0f} {tot[i]:+13,.0f} {sh_:9.1f}%")
print("  " + "-"*74)
gA, gR = np.abs(A_term).sum(), np.abs(R_term).sum()
print(f"  {'TOTAL |MWh|':22s} {gA:13,.0f} {gR:12,.0f} {'':13s} {100*gR/(gA+gR):9.1f}%")
print(f"\n  => {100*gR/(gA+gR):.1f}% of the counterfactual response is attributable to the")
print(f"     learned component. The rest is the anchor's weight rule.")
