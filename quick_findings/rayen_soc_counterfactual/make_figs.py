"""Feasible counterfactual (q*=3.699%) through itr_T4_rayen + rayen(soc=True) map.

Protocol = D3 (itr_figures.d3), with the guarantee map swapped from free-endpoint
posthoc to the SOC-aware RAYEN ray-shoot, Option-A pins = the model's own fill
endpoints (clipped to the box), matching how the free map floats its endpoints.

Outputs (quick_findings/rayen_soc_counterfactual/):
  stack_4day.png            base vs counterfactual all-energy stack incl. wind,
                            solar and curtailment (repo stack_plots convention)
  line_<fuel>.png x6        per-fuel model-output lines, base vs counterfactual
  README.md                 violations + per-fuel energy shift for the span
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path("/Users/nguyenminhquan/Downloads/energy_modelling")
sys.path.insert(0, str(ROOT / "imputation")); sys.path.insert(0, str(ROOT / "lib"))
from gap_data import load_flats, TARGETS, SIGN, ND_COL, DEM_COL      # noqa: E402
import constraints as C                                              # noqa: E402
import itr_data as D                                                 # noqa: E402
from itr_model import ITransformerImputer                            # noqa: E402
from itr_figures import impute_day                                   # noqa: E402
from scenario_eval import build_shift_scenario                       # noqa: E402
from constraint_layers import rayen_traj_project                     # noqa: E402
import stack_plots as sp                                             # noqa: E402

OUT = ROOT / "quick_findings" / "rayen_soc_counterfactual"
OUT.mkdir(parents=True, exist_ok=True)
Q = 3.699
FREE_LO, FREE_HI = D.FREE_W
N_DAYS = 4
COVER_END = pd.Timestamp("2026-05-18").date()   # renewables/curtailment data ends here

f = load_flats()
model = ITransformerImputer()
model.load_state_dict(torch.load(ROOT / "imputation/results/itr/itr_T4_rayen.pt",
                                 map_location="cpu", weights_only=True))
model.eval()

dem_b = f.col_mw(f.Xte, DEM_COL).astype(np.float64)
nd_b = f.col_mw(f.Xte, ND_COL).astype(np.float64)
dem_s, nd_s, P_ref = build_shift_scenario(f, Q)   # P_ref: actual dispatch free-projected onto the shifted set
P_act = f.y_to_mw(f.Yte).astype(np.float64)

# highest-median-demand 4 consecutive full days WITHIN renewables coverage
rec, days = D.day_windows(f)
best, best_v = None, -1.0
for i in range(len(days) - N_DAYS + 1):
    if (days[i + N_DAYS - 1] - days[i]).days != N_DAYS - 1:
        continue
    if days[i + N_DAYS - 1].date() > COVER_END:
        continue
    v = float(np.median([dem_b[rec["s"][i + k]: rec["s"][i + k] + D.T].mean()
                         for k in range(N_DAYS)]))
    if v > best_v:
        best, best_v = i, v
ss = rec["s"][best:best + N_DAYS]
span = [days[best + k] for k in range(N_DAYS)]
print(f"span: {span[0].date()} .. {span[-1].date()}  (median demand {best_v:.0f} MW)")

hour = np.arange(D.T) // 12
free = (hour >= FREE_LO) & (hour < FREE_HI)


def rayen_map(P, nd, A):
    """(T,6) MW fill -> feasible via rayen(soc=True). Anchor A = the Option-A
    projected reference for this day (P_ref for the shifted run, the actual
    dispatch for the base run); pins = the anchor's own endpoints, so the seams
    are trivially satisfied and no dispatch outside the day is pinned. The interp
    anchor stalls at the free-window nd discontinuity (~2000 MW/step); P_ref is
    the certificate's own feasible point for the shifted nd."""
    Pt = torch.tensor(P[None])
    At = torch.tensor(A[None])
    with torch.no_grad():
        return rayen_traj_project(Pt, At[:, 0], At[:, -1], torch.tensor(nd[None]),
                                  soc=True, anchor=At)[0].numpy()


def violations(P, nd):
    bal = np.abs((P * SIGN).sum(-1) - nd)
    d = np.diff(P, axis=0)
    ramp = int(((d > C.R_UP + 0.6) | (d < -(C.R_DN + 0.6))).sum())
    return {"bal>1MW": int((bal > 1.0).sum()), "bal_max_mw": float(bal.max()),
            "ramp": ramp, "neg": int((P < -0.1).sum()),
            "soc_over": int(C._soc_swing_mwh(P) > C.BATT_CAP_MWH + 1e-6)}


Pb_l, Ps_l, viol_b, viol_s = [], [], [], []
for s in ss:
    rows = np.arange(s, s + D.T)
    Pb = rayen_map(impute_day(model, f, int(s)), nd_b[rows], P_act[rows])
    Ps = rayen_map(impute_day(model, f, int(s), drivers=(nd_s[rows], dem_s[rows], free)),
                   nd_s[rows], P_ref[rows])
    viol_b.append(violations(Pb, nd_b[rows])); viol_s.append(violations(Ps, nd_s[rows]))
    Pb_l.append(Pb); Ps_l.append(Ps)
Pb = np.concatenate(Pb_l); Ps = np.concatenate(Ps_l)

lb = f.Xte.shape[0] - len(f.test_index)
pos = np.concatenate([np.arange(s - lb, s - lb + D.T) for s in ss])
idx = pd.DatetimeIndex(f.test_index[pos]).tz_localize(None)
rows_all = np.concatenate([np.arange(s, s + D.T) for s in ss])
ren = sp.load_renewables(f.test_index[pos])
free_all = np.tile(free, N_DAYS)

for tag, v in (("base", viol_b), ("counterfactual", viol_s)):
    agg = {k: sum(d[k] for d in v) for k in v[0] if k != "bal_max_mw"}
    agg["bal_max_mw"] = max(d["bal_max_mw"] for d in v)
    print(f"violations {tag}: {agg}")

# ---------------- figure 1: 4-day all-energy stack ----------------
ti = {t: i for i, t in enumerate(TARGETS)}


def draw(ax, disp, dem_line, title):
    sp._draw_full_stack(ax, idx, disp, ren, ti)
    ax.plot(idx, dem_line, "k--", lw=1.1, label="demand")
    for k in range(N_DAYS):
        d0 = idx[k * D.T].normalize()
        ax.axvspan(d0 + pd.Timedelta(hours=FREE_LO), d0 + pd.Timedelta(hours=FREE_HI),
                   color="gold", alpha=0.15)
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_ylabel("MW")


fig, axes = plt.subplots(2, 1, figsize=(16, 9.5), sharex=True, sharey=True)
draw(axes[0], Pb, dem_b[rows_all], "base — blackout fill, real drivers, rayen(soc=True) map")
draw(axes[1], Ps, dem_s[rows_all],
     f"counterfactual — rebound = {Q:g}%, reduction = {Q:g}%")
axes[1].xaxis.set_major_locator(mdates.DayLocator())
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
axes[1].xaxis.set_minor_locator(mdates.HourLocator(byhour=(6, 12, 18)))
axes[1].set_xlabel("date (free window 11:00–14:00 shaded)")
axes[0].legend(loc="upper left", ncol=5, fontsize=7, framealpha=0.9)
fig.suptitle(f"iTransformer+RAYEN(soc) — feasible counterfactual q={Q:g}% — "
             f"{span[0].date()} to {span[-1].date()}\n"
             "wind/solar/curtailment actual (renewables fixed under the shift); "
             "battery charging drawn negative", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "stack_4day.png", dpi=140); plt.close(fig)
print("wrote", OUT / "stack_4day.png")

# ---------------- figures 2-7: per-fuel model-output lines ----------------
for i, tg in enumerate(TARGETS):
    figl, ax = plt.subplots(figsize=(14, 4.2))
    c = sp.COLORS[tg]
    ax.plot(idx, Pb[:, i], color=c, lw=1.2, ls="--", alpha=0.75, label="base fill")
    ax.plot(idx, Ps[:, i], color=c, lw=1.6, label=f"counterfactual fill (q={Q:g}%)")
    for k in range(N_DAYS):
        d0 = idx[k * D.T].normalize()
        ax.axvspan(d0 + pd.Timedelta(hours=FREE_LO), d0 + pd.Timedelta(hours=FREE_HI),
                   color="gold", alpha=0.15)
    ax.axhline(0, color="k", lw=0.5)
    ax.margins(x=0); ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylabel("MW")
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=(6, 12, 18)))
    ax.set_title(f"{tg} — model output through rayen(soc=True), base vs counterfactual "
                 f"({span[0].date()}–{span[-1].date()}, free window shaded)",
                 loc="left", fontsize=10)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    figl.tight_layout()
    p = OUT / f"line_{tg}.png"
    figl.savefig(p, dpi=140); plt.close(figl)
    print("wrote", p)

# ---------------- README ----------------
dE_free = (Ps[free_all] - Pb[free_all]).sum(0) / 12.0
dE_off = (Ps[~free_all] - Pb[~free_all]).sum(0) / 12.0
dnd_free = (nd_s[rows_all][free_all] - nd_b[rows_all][free_all]).sum() / 12.0
dnd_off = (nd_s[rows_all][~free_all] - nd_b[rows_all][~free_all]).sum() / 12.0
vb = {k: sum(d[k] for d in viol_b) for k in viol_b[0] if k != "bal_max_mw"}
vs = {k: sum(d[k] for d in viol_s) for k in viol_s[0] if k != "bal_max_mw"}
lines = [
    f"# Feasible counterfactual (q={Q:g}%) — iTransformer T4_rayen + RAYEN(soc=True)",
    "",
    f"Span: {span[0].date()}..{span[-1].date()} (highest-median-demand 4 consecutive test "
    "days with real wind/solar/curtailment coverage; the last365 renewables extract ends "
    "2026-05-18, so the usual Jun 2-5 D2/D3 span has no curtailment data).",
    "",
    "Protocol: whole-day blackout imputation, drivers overridden with "
    f"FixedPercentageShift(q={Q:g}%) (off-window demand cut, 11:00-14:00 rebound, "
    "price->0 in the window, renewables fixed so dnd = ddemand). Option A: no dispatch "
    "outside the day is pinned. Guarantee map = rayen_traj_project(soc=True) with the "
    "anchor overridden to the Option-A projected reference (P_ref = actual dispatch "
    "free-endpoint-projected onto the shifted set; actual dispatch for the base run) "
    "and pins = the anchor's own endpoints -- the default interp anchor stalls at the "
    "~2000 MW one-step nd discontinuity at the free-window edges (total 1-step ramp "
    "capacity is 2891 MW up / 3369 down). Base and counterfactual differ only in the "
    "drivers and their anchors.",
    "",
    f"Violations, base run (4 days x 288 steps): {vb}",
    f"Violations, counterfactual run:            {vs}",
    "",
    "| fuel | dE free-window (MWh) | dE off-window (MWh) |",
    "| --- | --- | --- |",
]
for i, t in enumerate(TARGETS):
    lines.append(f"| {t} | {dE_free[i]:+.0f} | {dE_off[i]:+.0f} |")
lines += [
    "",
    f"Consistency: SIGN.dE free {float(dE_free @ SIGN):+.0f} MWh vs demand shift "
    f"{dnd_free:+.0f} MWh; off {float(dE_off @ SIGN):+.0f} vs {dnd_off:+.0f} MWh.",
    "",
    "Figures: stack_4day.png (all-energy stack incl. wind, solar, curtailment hatched, "
    "battery charging negative, demand dashed); line_<fuel>.png x6 (model-output lines, "
    "base vs counterfactual).",
]
(OUT / "README.md").write_text("\n".join(lines) + "\n")
print("wrote", OUT / "README.md")
