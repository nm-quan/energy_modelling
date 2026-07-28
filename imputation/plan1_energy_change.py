"""Per-fuel total-energy % change under the counterfactual (size_aware model).

For every full test day, both counterfactual modes (scaled / masked), sums energy
(MWh) per source and vs the actual dispatch reports the % change -- plus total
demand and net-demand % change. Base = actual (no shift); counterfactual = the
mode's dispatch (rebound 2% / reduce 2%, free window 11-14; nd = nd + Delta total
demand, renewables + curtailment fixed). Writes planA/energy_change_size_aware.md.

    python3 imputation/plan1_energy_change.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "lib"))
import plan1_figures as F                                             # reuse helpers/constants
from gap_data import TARGETS, SIGN                                    # noqa: E402
from model import BiLSTMImputer                                      # noqa: E402
from shift_model import FixedPercentageShift                          # noqa: E402

DT = 5.0 / 60.0
OUT = ROOT / "planA"


def main():
    te, feat_cols, xm, xs_, ym, ys_ = F.load_test()
    CTX, GAP, FREE, Q = F.CTX, F.GAP, F.FREE, F.Q
    hour = te.index.hour
    free_all = (hour >= FREE[0]) & (hour < FREE[1])
    sh = FixedPercentageShift(Q, Q, free_hours=FREE).transform(
        pd.DataFrame({"demand_mw": te["demand_mw"].values,
                      "price_aud_per_mwh": te["price_aud_per_mwh"].values}, index=te.index))
    dem_s = sh["demand_mw"].values
    d_dem = dem_s - te["demand_mw"].values
    nd_bal_base = te[TARGETS].values @ SIGN
    nd_bal_cf = nd_bal_base + d_dem                                   # nd + Delta(total demand), renewables fixed
    nd_feat_base = te["net_demand"].values                            # demand-side (actual)
    nd_feat_cf = nd_feat_base + d_dem

    def overrides(span_idx):
        rows = te.iloc[span_idx]; fr = free_all[span_idx]
        return {"demand_mw": dem_s[span_idx], "net_demand": nd_feat_cf[span_idx],
                "price_aud_per_mwh": np.where(fr, 0.0, rows["price_aud_per_mwh"].values)}
        # wind / solar / curtailment stay at ACTUAL (renewables fixed)

    model = BiLSTMImputer(n_features=len(feat_cols))
    model.load_state_dict(torch.load(F.WEIGHTS / "size_aware.pt", map_location="cpu",
                                     weights_only=True)); model.eval()

    days = te.index.normalize()
    full = [d for d in pd.unique(days) if (days == d).sum() == 288]
    g0 = FREE[0] * 12
    res = {}                                                          # mode -> (E_base, E_cf, n_days, bal_resid)
    dem_b = dem_cf = ndb = ndcf = 0.0; n_used = 0
    for mode in ("scaled",):     # scaled = full-day counterfactual = the total delta
        E_base = np.zeros(6); E_cf = np.zeros(6); nd_used = 0; worst = 0.0; n_infeas = 0
        for d in full:
            day_rows = np.where(days == d)[0]
            rows = day_rows[g0:g0 + GAP]
            span = np.arange(rows[0] - CTX, rows[-1] + 1 + CTX)
            if span[0] < 0 or span[-1] >= len(te):
                continue                                              # edge day (no full context)
            truth_day = te.iloc[day_rows][TARGETS].values.astype(np.float64)
            if mode == "scaled":
                day_disp = truth_day * (nd_bal_cf[day_rows] / nd_bal_base[day_rows])[:, None]
            else:
                day_disp = truth_day.copy()
            ctx = np.zeros((len(span), 6))
            for k, r in enumerate(span):
                pos = np.where(day_rows == r)[0]
                ctx[k] = day_disp[pos[0]] if len(pos) else te.iloc[r][TARGETS].values
            fill, _, _ = F.model_fill(model, te, feat_cols, xm, xs_, ym, ys_,
                                      rows, overrides(span), ctx)
            day_disp[g0:g0 + GAP] = fill
            day_disp = F.project_full_day(day_disp, day_disp[0].copy(),
                                          day_disp[-1].copy(), nd_bal_cf[day_rows], True)
            worst = max(worst, float(np.abs((day_disp * SIGN).sum(-1) - nd_bal_cf[day_rows]).max()))
            E_base += truth_day.sum(0) * DT
            E_cf += day_disp.sum(0) * DT
            nd_used += 1
            if mode == "scaled":                                      # demand/nd identical across modes; sum once
                dem_b += te.iloc[day_rows]["demand_mw"].sum() * DT
                dem_cf += dem_s[day_rows].sum() * DT
                ndb += nd_feat_base[day_rows].sum() * DT
                ndcf += nd_feat_cf[day_rows].sum() * DT
                n_used += 1
        res[mode] = (E_base, E_cf, nd_used, worst)
        print(f"{mode}: {nd_used} days, worst balance resid {worst:.1f} MW")

    pct_dem = 100 * (dem_cf - dem_b) / dem_b
    pct_nd = 100 * (ndcf - ndb) / ndb
    L = [f"# plan1 — counterfactual energy change, size_aware model",
         "",
         f"Rebound {Q:g}% / reduce {Q:g}% (max feasible over the WHOLE day, balance+SOC across "
         f"288 steps; q_max 2.41%), free window "
         f"{FREE[0]}:00–{FREE[1]}:00. nd = nd + Δ(total demand), renewables + curtailment fixed. "
         f"Totals over {res['scaled'][2]} full test days; % change vs the actual (no-shift) "
         f"dispatch. Energy in MWh (= Σ MW × 5/60 h).",
         "",
         "## Drivers", "",
         "| quantity | actual (MWh) | counterfactual (MWh) | Δ energy | % change |",
         "| --- | --- | --- | --- | --- |",
         f"| total demand | {dem_b:,.0f} | {dem_cf:,.0f} | {dem_cf-dem_b:+,.0f} | {pct_dem:+.2f}% |",
         f"| net demand (demand − renewables − curtailment) | {ndb:,.0f} | {ndcf:,.0f} | {ndcf-ndb:+,.0f} | {pct_nd:+.2f}% |",
         "",
         "## Per-source total energy change (full-day counterfactual)", "",
         "| source | actual (MWh) | counterfactual (MWh) | Δ energy (MWh) | Δ % |",
         "| --- | --- | --- | --- | --- |"]
    Eb, Ecf = res["scaled"][0], res["scaled"][1]
    for i, t in enumerate(TARGETS):
        pct = 100 * (Ecf[i] - Eb[i]) / Eb[i]
        L.append(f"| {t} | {Eb[i]:,.0f} | {Ecf[i]:,.0f} | {Ecf[i]-Eb[i]:+,.0f} | {pct:+.2f}% |")
    L += ["",
          "_Counterfactual = full-day shifted scenario (off-window dispatch × nd_after/nd_before "
          "+ model-filled free window); the single total delta of the whole scenario vs actual._"]
    OUT.mkdir(exist_ok=True)
    (OUT / "energy_change_size_aware.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print("\nwrote", OUT / "energy_change_size_aware.md")


if __name__ == "__main__":
    main()
