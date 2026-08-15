"""Test 2 of EVALUATION_PLAN: peak filling, 18:00-21:00.

The counterfactual window is 11:00-14:00, the midday TROUGH -- the one part of
the day where neither peak's dispatch logic applies. This script instead masks
the EVENING PEAK on every test day and asks each arm to fill it.

    gap    18:00 -> 20:55  (36 steps), pR pinned at 21:00
    days   every test day the split allows (186 in 2026-01-01..2026-07-05)
    task   pure imputation. The true net demand is given, as always; only the
           six dispatch channels and the two curtailment channels are hidden.

Two outputs, and they answer different questions:

  the FIGURE   day-averaged stacked profile -- mean over days at each of the 36
               steps. It shows whether the SHAPE and the MIX are right.
  the TABLE    per-channel MAE / MRE over all days x 36 steps. It shows whether
               the LEVEL is right.

A day-average can look correct while every individual day is wrong, so the
figure is never the result on its own -- read it with the table.

    python3 planB/peak_eval.py --arms rayen
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "imputation"))
sys.path.insert(0, str(ROOT / "dispatch_study"))
sys.path.insert(0, str(HERE))

import gap_data as GD                                                  # noqa: E402
GD.NPZ = ROOT / "data/preprocessed/hist/5min/net_dispatch_ren/prepared.npz"
from gap_data import load_flats, TARGETS, SIGN, TARGET_FEAT_IDX        # noqa: E402
from common import COLORS, LABEL, INK, MUTED, FUELS                    # noqa: E402
from nets import build_delta                                           # noqa: E402
import heads as HD                                                     # noqa: E402
from heads import HEADS                                                # noqa: E402
from fit import (build, curt_activation, CURT_COLS, CTX, GAP, apply_residual,          # noqa: E402
                 ARMS as ARM_SPEC, BACKBONES)

OUT = HERE / "results"
WIND_COL, SOLAR_COL = 17, 18
CHANNELS = TARGETS + ["wind_curtailment", "solar_curtailment"]
SHORT = {"hydro": "hydro", "coal_brown": "coal", "gas_steam": "steam",
         "gas_ocgt": "ocgt", "battery_charging": "bat_chg",
         "battery_discharging": "bat_dis", "wind_curtailment": "wind_cu",
         "solar_curtailment": "sol_cu"}


def peak_starts(f, hour: int):
    """Xte rows whose gap OPENS at `hour`:00 local, one per test day.

    test_index aligns with Xte[lb_carry:], so an index j into test_index is row
    j + lb_carry of Xte. The gap opens at start + CTX, hence the shift.
    """
    ti, W = f.test_index, 2 * CTX + GAP
    j = np.where((ti.hour == hour) & (ti.minute == 0))[0]
    s = j + f.lb_carry - CTX
    keep = (s >= 0) & (s + W <= f.Xte.shape[0]) & (s + CTX >= f.lb_carry)
    return s[keep], ti[j[keep]]


def run_arm(arm, f, b, nfeat):
    """Forward pass -> (n, GAP, 8): six dispatch channels in MW, then the two
    curtailment channels. Mirrors planB/eval.py exactly."""
    ck = torch.load(OUT / f"{arm}.pt", map_location="cpu", weights_only=False)
    head_fn, n_disp = HEADS[ck["head"]]
    bb = ck.get("backbone", "brits" if arm.startswith("brits") else "bilstm")
    m = (BACKBONES[bb](nfeat, n_disp, hidden=ck["hidden"]) if bb == "bilstm"
         else BACKBONES[bb](nfeat, TARGET_FEAT_IDX, CURT_COLS, n_disp,
                            hidden=ck["hidden"]))
    m.load_state_dict(ck["state"]); m.eval()
    # Checkpoints without a "residual" key predate the interp-residual restore
    # and were trained WITHOUT it, so False is the right default -- measured:
    # forcing it on costs unconstrained 106 -> 242 MW and hardnet 120 -> 141.
    # (rayen is residual-free either way; its head takes a direction, not a level.)
    resid = ck.get("residual", False)
    HD.set_soc(ck.get("soc"))          # reproduce the head the checkpoint trained with
    c_m, c_s = torch.tensor(b["c_mean"]), torch.tensor(b["c_scale"])
    ys_m = torch.tensor(f.y_mean, dtype=torch.float32)
    ys_s = torch.tensor(f.y_scale, dtype=torch.float32)

    out = []
    for i in range(0, len(b["X"]), 64):
        sl = slice(i, i + 64)
        x = torch.from_numpy(b["X"][sl]); mk = torch.from_numpy(b["mask"][sl])
        with torch.no_grad():
            dl = build_delta(mk, nfeat) if bb == "brits" else None
            d_raw, c_raw = m(x, mk, dl)
            d_raw, c_raw = d_raw[:, CTX:CTX + GAP], c_raw[:, CTX:CTX + GAP]
            curt = curt_activation(c_raw, c_m, c_s)
            if resid:
                d_raw = apply_residual(d_raw, torch.from_numpy(b["interp"][sl]))
            if head_fn is None:
                P = d_raw[..., :6] * ys_s + ys_m
            else:
                F_ = (d_raw * ys_s + ys_m if n_disp == 6 else
                      torch.cat([d_raw[..., :6] * ys_s + ys_m, d_raw[..., 6:]], -1))
                P = head_fn(F_, torch.from_numpy(b["pL"][sl]),
                            torch.from_numpy(b["pR"][sl]),
                            torch.from_numpy(b["nd"][sl]))
        out.append(np.concatenate([P.numpy(), curt.numpy()], -1))
    return np.concatenate(out, 0), ck


def draw(ax, disp, wind, solar, curt, nd, title, hour, legend=False):
    """One day-averaged stacked panel over the 36 gap steps."""
    x = np.arange(len(disp)) * 5.0 / 60.0 + hour
    ti = {t: i for i, t in enumerate(TARGETS)}
    base = np.zeros(len(x))
    for f_ in FUELS:                                   # excludes battery_charging
        top = base + disp[:, ti[f_]]
        ax.fill_between(x, base, top, color=COLORS[f_], lw=0.3, edgecolor="white",
                        zorder=2, label=LABEL[f_])
        base = top
    for arr, key in ((wind, "wind"), (solar, "solar_utility")):
        top = base + arr
        ax.fill_between(x, base, top, color=COLORS[key], lw=0.3, edgecolor="white",
                        zorder=2, label=LABEL[key])
        base = top
    for k, key in enumerate(("wind", "solar_utility")):
        top = base + curt[:, k]
        ax.fill_between(x, base, top, color=COLORS[key], alpha=0.35, lw=0,
                        hatch="///", zorder=2, label=f"{LABEL[key]} curtailed")
        base = top
    ax.fill_between(x, 0, -disp[:, ti["battery_charging"]],
                    color=COLORS["battery_charging"], lw=0.3, edgecolor="white",
                    zorder=2, label=LABEL["battery_charging"])
    ax.plot(x, nd, color="#B3261E", lw=1.3, zorder=4, label="net demand")
    ax.axhline(0, color=MUTED, lw=0.5); ax.margins(x=0)
    ax.grid(True, axis="y", alpha=0.15, lw=0.5, zorder=0)
    ax.tick_params(labelsize=8, colors=MUTED, length=0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK, pad=3)
    ax.set_ylabel("MW", fontsize=8, color=MUTED)
    if legend:
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=6,
                  frameon=False, fontsize=7.5)


def strip(panels, wind, solar, nd, days, hour, n_show, out, note):
    """10-day strip: the fill windows laid end to end, one row per arm.

    Full days would spend 87% of the ink on data no model touched, so only the
    filled windows are drawn, separated by day. x is step index, not clock time.
    """
    G = GAP
    fig, axes = plt.subplots(len(panels), 1, figsize=(15, 2.7 * len(panels)),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for k, (name, arr) in enumerate(panels):
        ax = axes[k]
        flat = arr[:n_show].reshape(-1, 8)
        draw(ax, flat[:, :6], wind[:n_show].ravel(), solar[:n_show].ravel(),
             flat[:, 6:], nd[:n_show].ravel(), name, 0,
             legend=(k == len(panels) - 1))
        for d in range(1, n_show):
            ax.axvline(d * G * 5 / 60.0, color="white", lw=1.4, zorder=6)
        ax.set_xticks([(d + 0.5) * G * 5 / 60.0 for d in range(n_show)])
        ax.set_xticklabels([str(x.date())[5:] for x in days[:n_show]], fontsize=8)
    end_h = hour + GAP * 5 // 60
    fig.suptitle(f"{hour:02d}:00-{end_h:02d}:00 filled on {n_show} consecutive test "
                 f"days — each panel is the {n_show} windows laid end to end{note}",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0.02, 1, 0.965])
    fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["rayen"])
    ap.add_argument("--hour", type=int, default=18, help="hour the gap opens")
    ap.add_argument("--days", type=int, default=0,
                    help="also draw a strip of this many consecutive days")
    ap.add_argument("--from-day", default=None, help="YYYY-MM-DD the strip starts")
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    f = load_flats()
    nfeat = len(f.feat_cols)
    starts, days = peak_starts(f, args.hour)
    b = build(f, starts, "test")
    n = len(starts)
    end_h = args.hour + GAP * 5 // 60
    print(f"{n} test days, gap {args.hour:02d}:00-{end_h:02d}:00 "
          f"({days[0].date()} .. {days[-1].date()})")

    truth = np.concatenate([b["Y"] * f.y_scale + f.y_mean, b["C"]], -1)   # (n,G,8)
    nd = b["nd"]                                                          # (n,G)
    Xg = b["X"][:, CTX:CTX + GAP]
    wind = Xg[:, :, WIND_COL] * f.x_scale[WIND_COL] + f.x_mean[WIND_COL]
    solar = Xg[:, :, SOLAR_COL] * f.x_scale[SOLAR_COL] + f.x_mean[SOLAR_COL]

    tt = (np.arange(1, GAP + 1) / (GAP + 1))[None, :, None]
    cL = f.Xte[starts + CTX - 1][:, CURT_COLS] * b["c_scale"] + b["c_mean"]
    cR = f.Xte[starts + CTX + GAP][:, CURT_COLS] * b["c_scale"] + b["c_mean"]
    preds = {"interp": np.concatenate(
        [b["pL"][:, None] + tt * (b["pR"] - b["pL"])[:, None],
         cL[:, None] + tt * (cR - cL)[:, None]], -1)}
    meta = {}
    for a in args.arms:
        if a == "anchor":
            # the rayen head with the NETWORK DELETED: a zero direction means the
            # ray cannot move, so P == A exactly. This isolates how much of the
            # arm's answer is structure (glide + balance snap) rather than
            # learning. It has no curtailment model, so it borrows interp's.
            from heads import rayen_head
            tp = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32)
            A = rayen_head(torch.zeros(n, GAP, 7), tp(b["pL"]), tp(b["pR"]),
                           tp(b["nd"])).numpy()
            preds[a] = np.concatenate([A, preds["interp"][..., 6:]], -1)
            continue
        if not (OUT / f"{a}.pt").exists():
            print(f"  [skip] {a}.pt missing"); continue
        preds[a], meta[a] = run_arm(a, f, b, nfeat)

    rows = [r for r in (["interp"] + args.arms) if r in preds]
    NAME = {"interp": "plain linear interpolation (no model)",
            "anchor": "anchor only — rayen head with the network deleted"}
    label = lambda r: NAME.get(r, f"{r}  ({meta[r].get('backbone', 'bilstm')} + "
                                  f"{meta[r]['head']} head)" if r in meta else r)

    # ---- figure: day-averaged stack, actual on top ------------------------
    panels = [("actual", truth)] + [(r, preds[r]) for r in rows]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 3.0 * len(panels)),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for k, (name, arr) in enumerate(panels):
        lbl = "actual" if name == "actual" else label(name)
        draw(axes[k], arr[..., :6].mean(0), wind.mean(0), solar.mean(0),
             arr[..., 6:].mean(0), nd.mean(0), lbl, args.hour,
             legend=(k == len(panels) - 1))
    fig.suptitle(f"Evening peak {args.hour:02d}:00-{end_h:02d}:00 — day-averaged "
                 f"dispatch over {n} test days", x=0.02, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0.03, 1, 0.985])
    png = OUT / f"peak_{args.hour:02d}_{end_h:02d}_dayavg{args.suffix}.png"
    fig.savefig(png, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {png}")

    # ---- optional: N-day strip ---------------------------------------------
    if args.days:
        off = 0
        if args.from_day:
            hit = np.where(np.array([str(d.date()) for d in days]) == args.from_day)[0]
            off = int(hit[0]) if len(hit) else 0
        sl = slice(off, off + args.days)
        strip([("actual", truth[sl])] + [(label(r), preds[r][sl]) for r in rows],
              wind[sl], solar[sl], nd[sl], days[sl], args.hour, args.days,
              OUT / f"peak_{args.hour:02d}_{end_h:02d}_{args.days}day{args.suffix}.png",
              "" if not args.from_day else f", from {args.from_day}")

    # ---- companion: per-channel lines --------------------------------------
    # The stack above is 77% coal, so every panel looks the same while the MAE
    # table says they are not. One small panel per channel, shared x, own y.
    ORDER = ["coal_brown", "hydro", "battery_discharging", "battery_charging",
             "gas_ocgt", "gas_steam", "wind_curtailment", "solar_curtailment"]
    idx = {c: k for k, c in enumerate(CHANNELS)}
    fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharex=True)
    hx = np.arange(GAP) * 5.0 / 60.0 + args.hour
    for ax, ch in zip(axes.ravel(), ORDER):
        k = idx[ch]
        ax.plot(hx, truth[..., k].mean(0), color=INK, lw=2.0, label="actual", zorder=4)
        ax.plot(hx, preds["interp"][..., k].mean(0), color=MUTED, lw=1.3,
                ls=":", label="interp", zorder=3)
        for r in [r for r in rows if r != "interp"]:
            ax.plot(hx, preds[r][..., k].mean(0),
                    color=COLORS.get(ch, "#B3261E") if ch in COLORS else "#B3261E",
                    lw=1.6, ls="--", label=r, zorder=3)
        nm = LABEL.get(ch, ch.replace("_", " "))
        bits = " | ".join(f"{r} {np.abs(preds[r][..., k] - truth[..., k]).mean():.0f}"
                          for r in rows)
        ax.set_title(f"{nm}   MAE  {bits}  MW", loc="left",
                     fontsize=8.5, color=INK, pad=3)
        ax.grid(True, alpha=0.15, lw=0.5); ax.margins(x=0)
        ax.tick_params(labelsize=7.5, colors=MUTED, length=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#D6D8DC")
    axes[0, 0].legend(loc="upper left", fontsize=7.5, frameon=False)
    fig.suptitle(f"Same window, per channel — day-averaged over {n} test days "
                 f"(note the independent y scales)", x=0.01, ha="left",
                 fontsize=11.5, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png2 = OUT / f"peak_{args.hour:02d}_{end_h:02d}_channels{args.suffix}.png"
    fig.savefig(png2, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {png2}")

    # ---- table ------------------------------------------------------------
    L = [f"# Peak filling {args.hour:02d}:00-{end_h:02d}:00 — {n} test days", "",
         f"Gap = {GAP} steps opening at {args.hour:02d}:00, right edge pinned at "
         f"{end_h:02d}:00. Every test day in {days[0].date()}..{days[-1].date()}. "
         f"Statistics are over all {n} x {GAP} = {n*GAP:,} cells per channel; the "
         "figure is the day-average of the same arrays.", "",
         "## MAE (MW)", "",
         "| arm | " + " | ".join(SHORT[c] for c in CHANNELS) + " | disp agg | curt agg |",
         "| --- |" + " ---: |" * (len(CHANNELS) + 2)]
    mae = {}
    for r in rows:
        e = np.abs(preds[r] - truth).reshape(-1, 8)
        mae[r] = e.mean(0)
        L.append(f"| {r} | " + " | ".join(f"{v:,.1f}" for v in mae[r]) +
                 f" | **{mae[r][:6].mean():,.1f}** | **{mae[r][6:].mean():,.1f}** |")
    L += ["", "## MRE (%)", "",
          "| arm | " + " | ".join(SHORT[c] for c in CHANNELS) + " |",
          "| --- |" + " ---: |" * len(CHANNELS)]
    T = np.abs(truth).reshape(-1, 8).sum(0)
    for r in rows:
        e = np.abs(preds[r] - truth).reshape(-1, 8).sum(0)
        L.append(f"| {r} | " + " | ".join(f"{v:,.1f}" for v in 100 * e / T) + " |")
    L += ["", "_Share of window cells at or below 1 MW: " + ", ".join(
        f"{SHORT[c]} {100*(np.abs(truth.reshape(-1,8)[:,k])<=1).mean():.0f}%"
        for k, c in enumerate(CHANNELS)) + "._", ""]

    # composition of the window's energy, which is what the stack shows
    L += ["## Window energy mix (% of gross generation, day-averaged)", "",
          "| arm | " + " | ".join(LABEL[f_] for f_ in FUELS) + " |",
          "| --- |" + " ---: |" * len(FUELS)]
    ti = {t: i for i, t in enumerate(TARGETS)}
    for name, arr in [("actual", truth)] + [(r, preds[r]) for r in rows]:
        e = np.array([arr[..., ti[f_]].sum() for f_ in FUELS])
        L.append(f"| {name} | " + " | ".join(f"{v:.1f}" for v in 100 * e / e.sum()) + " |")
    L += ["", "_Gross generation excludes battery charging, which is a load._", ""]

    md = OUT / f"peak_{args.hour:02d}_{end_h:02d}{args.suffix}.md"
    md.write_text("\n".join(L))
    print(f"wrote {md}\n")
    print("\n".join(L[L.index("## MAE (MW)"):]))


if __name__ == "__main__":
    main()
