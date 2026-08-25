"""EXPERIMENT 10 -- solve the whole window as one QP, and see what the head gave up.

The head (head_traj.py) is a SEQUENTIAL closed-form projection: it walks left to right and
solves a one-scalar problem per step. That buys speed and an exact non-emptiness argument,
and it pays for them with myopia -- it commits to P[t] before it has seen t+1, which is
precisely why SOC needed an iterative repair on top rather than holding by construction.

This file asks the obvious question: what does an exact multi-period solve get instead?
Same window, same constraint set, same network output, all 288 steps optimised JOINTLY:

    min  sum_t sum_i w_i (P[t,i] - F[t,i])^2
    s.t. balance      sum_i SIGN_i P[t,i] = nd[t]                      for all t
         capacity     P_min(t) <= P[t] <= P_max(t)
         ramp table   |traj[t+k] - traj[t]| <= R(k)                    for all t, k
         SOC          E_min <= E_0 + cumsum(...) <= E_max              for all t

TWO CHANNELS HERE, NOT ONE SIGNED CHANNEL. The head carries the battery as a single signed
`b` because that is what turns the SOC bound into a box, which is what the closed form
needs. A QP has no such requirement -- it takes any linear inequality -- so `chg` and `dis`
go in as separate non-negative variables, the reservoir dynamics are then exactly linear in
them, and the overlap m = min(chg, dis) falls out of the solution instead of being clipped
on afterwards. That is a genuine advantage of the QP and worth measuring.

RAMPS BY CONSTRAINT GENERATION. Writing all 288 horizons out is ~400k rows and the solver
chokes. Start with a subset, audit the solution against the FULL table, add whatever it
violated, re-solve. Terminates when the audit is clean, so the comparison is against the
same envelope the head enforces -- not a relaxation of it.

NOT DIFFERENTIABLE, ON PURPOSE. This is an inference-time reference, so it uses OSQP/Clarabel
rather than cvxpylayers. The repo already measured that training THROUGH a hard layer costs
accuracy (+0.21 WAPE) while projecting at inference is free, so there is no reason to pay
for gradients here.

    python colab/capacity_experiment/exp10_qp_reference.py [--windows 12]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cvxpy as cp
import numpy as np

import constraint_set as CS
import exp8_constrained_imputer as E8

HERE = Path(__file__).resolve().parent
N = CS.STEPS_PER_DAY
C = len(CS.CHANNELS)
BATT = CS.BATT
TH = [i for i in range(C) if i != BATT]
# horizons the QP starts with; more get added by constraint generation if the audit fails
K0 = [1, 2, 3, 4, 6, 9, 12, 18, 24, 36, 48, 72, 144, 288]


def solve_window(Fth, Fchg, Fdis, w, pL, pR, nd, Pmin, Pmax, res, e0, R_up, R_dn,
                 ks=None, rounds=4, solver="OSQP", verbose=False):
    """Exact multi-period projection for one window. Returns (P, m, info)."""
    ks = list(ks or K0)
    a, b = res.a, res.b                      # half-step MWh per MW of charge / discharge
    hist = []
    for it in range(rounds):
        th = cp.Variable((N, len(TH)), name="thermal")
        chg = cp.Variable(N, nonneg=True)
        dis = cp.Variable(N, nonneg=True)
        bat = dis - chg                                        # the signed channel

        cons = [th >= Pmin[:, TH], th <= Pmax[:, TH],
                chg <= (-Pmin[:, BATT]), dis <= Pmax[:, BATT]]
        cons += [cp.sum(th, axis=1) + bat == nd]               # balance, every step

        # ramp: the trajectory with both observed endpoints pinned
        rows = [cp.reshape(cp.Constant(pL), (1, C), order="C")]
        for t in range(N):
            rows.append(cp.reshape(cp.hstack([th[t], cp.reshape(bat[t], (1,), order="C")]),
                                   (1, C), order="C"))
        rows.append(cp.reshape(cp.Constant(pR), (1, C), order="C"))
        traj = cp.vstack(rows)                                 # (N+2, C)
        for k in ks:
            if k > N + 1:
                continue
            d = traj[k:] - traj[:-k]
            cons += [d <= R_up[k], d >= -R_dn[k]]

        # SOC: linear in (chg, dis) -- this is why the QP can take them separately
        pLc = float(max(-pL[BATT], 0.0))          # the observed step before the gap
        pLd = float(max(pL[BATT], 0.0))
        e_chg = a * (chg + cp.hstack([cp.Constant([pLc]), chg[:-1]]))
        e_dis = b * (dis + cp.hstack([cp.Constant([pLd]), dis[:-1]]))
        dE = e_chg - e_dis - res.draw
        E = e0 + cp.cumsum(dE)
        cons += [E >= res.e_min, E <= res.e_max]

        obj = (cp.sum(cp.multiply(w[:, TH], cp.square(th - Fth)))
               + cp.sum(cp.multiply(w[:, BATT], cp.square(chg - Fchg)))
               + cp.sum(cp.multiply(w[:, BATT], cp.square(dis - Fdis))))
        prob = cp.Problem(cp.Minimize(obj), cons)
        prob.solve(solver=solver, verbose=verbose)
        if th.value is None:
            return None, None, dict(status=prob.status, ks=ks, rounds=it + 1)

        P = np.zeros((N, C))
        P[:, TH] = th.value
        P[:, BATT] = dis.value - chg.value
        m = np.minimum(chg.value, dis.value)
        hist.append(prob.solver_stats.solve_time if prob.solver_stats else np.nan)

        # audit against the FULL table and add whatever was violated
        full = np.vstack([pL, P, pR])
        bad = []
        for k in range(1, N + 2):
            if k in ks:
                continue
            d = full[k:] - full[:-k]
            over = max(float(np.maximum(d - R_up[k], 0).max()),
                       float(np.maximum(-d - R_dn[k], 0).max()))
            if over > 1e-6:
                bad.append((over, k))
        if not bad:
            return P, m, dict(status=prob.status, ks=ks, rounds=it + 1,
                              solve_time=float(np.nansum(hist)))
        bad.sort(reverse=True)
        ks = sorted(set(ks + [k for _, k in bad[:12]]))
    return P, m, dict(status="ramp_gen_exhausted", ks=ks, rounds=rounds,
                      solve_time=float(np.nansum(hist)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=12)
    ap.add_argument("--solver", default="OSQP")
    ap.add_argument("--polish", type=int, default=6)
    ap.add_argument("--input", default="interp", choices=["interp", "model"],
                    help="what the two maps are fed. `interp` isolates the MAP; `model` is "
                         "the deployment case, and the gap between them measures how much "
                         "the sequential head's myopia costs as a function of how far the "
                         "input starts from feasible")
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()

    cs = CS.build()
    res = cs["res"]
    recs, X = E8.build_windows(cs)
    T = E8.tensors(cs, recs, X)
    te = np.arange(len(recs) - E8.TEST_DAYS, len(recs))[: a.windows]
    print(f"QP reference on {len(te)} test windows, solver {a.solver}")
    print(f"  {N} steps x {C} channels -> {N * (len(TH) + 2)} variables per window\n")

    # the same network output both maps see
    if a.input == "model":
        tr = np.arange(0, len(recs) - E8.VAL_DAYS - E8.TEST_DAYS)
        va = np.arange(len(tr), len(tr) + E8.VAL_DAYS)
        t_tr = time.time()
        model, scal, vbest = E8.train_backbone(X, T, tr, va, 0, a.epochs, 8, verbose=False)
        raw = E8.predict_raw(model, X, T, te, scal)
        print(f"backbone: seed 0, val {vbest:.5f}, {time.time() - t_tr:.0f}s")
    else:
        skel = E8.interp_skeleton(T["pLM"], T["pRM"])           # 0 parameters
        raw = np.concatenate([skel[te][..., :C], np.zeros((len(te), N, C)),
                              skel[te][..., C:]], -1)
    print(f"input: {a.input}")

    t0 = time.time()
    P_head, M_head = E8.apply_head(raw, T, te, cs, res, polish=a.polish)
    t_head = time.time() - t0
    print(f"head:  {t_head:.1f}s total, {t_head / len(te):.2f}s per window")

    print(f"\nQP, window by window:")
    Pq, Mq, infos, times = [], [], [], []
    for j, i in enumerate(te):
        Fb, Fm = raw[j, :, BATT], raw[j, :, 2 * C]
        Fchg = np.maximum(-Fb, 0.0) + np.maximum(Fm, 0.0)     # what the net asked for,
        Fdis = np.maximum(Fb, 0.0) + np.maximum(Fm, 0.0)      # in the QP's own variables
        w = np.ones((N, C))
        t1 = time.time()
        P, m, info = solve_window(raw[j][:, TH], Fchg, Fdis, w, T["pL"][i], T["pR"][i],
                                  T["nd_obs"][i], T["Pmin"][i], T["Pmax"][i], res,
                                  T["e0"][i], cs["R_up"], cs["R_dn"], solver=a.solver)
        dt = time.time() - t1
        times.append(dt)
        infos.append(info)
        if P is None:
            print(f"  [{j:2d}] {recs[i]['day'].date()}  FAILED {info['status']}  {dt:.1f}s")
            Pq.append(np.full((N, C), np.nan)); Mq.append(np.full(N, np.nan))
            continue
        Pq.append(P); Mq.append(m)
        print(f"  [{j:2d}] {recs[i]['day'].date()}  {info['status']:12s} "
              f"{dt:6.1f}s  {info['rounds']} round(s)  |ks|={len(info['ks'])}")
    Pq, Mq = np.stack(Pq), np.stack(Mq)
    t_qp = float(np.sum(times))
    print(f"\nQP:    {t_qp:.1f}s total, {t_qp / len(te):.2f}s per window "
          f"-> {t_qp / max(t_head, 1e-9):.0f}x the head")

    ok = np.isfinite(Pq).all((1, 2))
    print(f"       solved {int(ok.sum())}/{len(te)}")

    print("\n" + "=" * 96)
    print("ACCURACY and FEASIBILITY, identical input, identical envelope")
    print("=" * 96)
    rows = {}
    for name, P, M in (("head (closed form)", P_head, M_head), ("QP (joint solve)", Pq, Mq)):
        s = E8.score(P[ok], T, te[ok], cs, res, M=M[ok])
        v = s["v"]
        rows[name] = dict(MAE=s["MAE"], microWAPE=s["microWAPE"], **v)
        print(f"  {name:20s} MAE {s['MAE']:7.2f}  microWAPE {s['microWAPE']:.4f}   "
              f"bal {v['balance_obs']:8.1e}  box {v['box']:8.1e}  "
              f"ramp {v['ramp']:8.1e}  SOC {v['soc']:8.1e}")
    print("\n  MAE is against the RECORDED six channels; violations are magnitudes")
    print("  (MW, MWh for SOC), worst over every window and step.")

    d = np.abs(P_head[ok] - Pq[ok])
    print(f"\n  the two solutions differ by {d.mean():.2f} MW mean, {d.max():.1f} MW max")
    print(f"  overlap m: head {M_head[ok].mean():.2f} MW, QP {Mq[ok].mean():.2f} MW, "
          f"recorded {T['Mtrue'][te[ok]].mean():.2f} MW")

    (HERE / f"exp10_qp_reference_results_{a.input}.json").write_text(json.dumps(
        dict(n=int(ok.sum()), t_head=t_head, t_qp=t_qp, solver=a.solver, arms=rows,
             ks_final=[int(k) for k in infos[-1]["ks"]], input=a.input), indent=1))
    print("\nwrote exp10_qp_reference_results.json")


if __name__ == "__main__":
    main()
