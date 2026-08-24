"""Rebuild the battery channels and the fleet reservoir from per-unit telemetry.

WHY NOT THE FUELTECH ROLLUP. `vic_generation`'s battery_charging / battery_discharging
are a rollup over every VIC BESS, and the rollup double-counts a unit for a period after
it commissions. Measured against the 12-unit sum below the difference has std 56.7 MW and
reaches 470 MW. That matters here in a way it did not before: this is the first model to
attach a SOC constraint to those channels, and a channel must drive the same reservoir
whose level we observe. So both come from the same unit set.

THE UNIT SET IS FIXED, NOT MAXIMAL. A fleet total is only well posed if the membership is
held constant -- otherwise a commissioning event reads as a reservoir jump. MLB01 (frozen
telemetry, 66% missing), MRNBESS1 and TRGBESS1 (enter the feed 2026-07-30, 99% missing)
are excluded, leaving 12 units present on >90% of the 5-minute grid.

SIGN CONVENTION, verified against the SOC series rather than assumed: p < 0 is charging.
Regressing dSOC on p at high power gives a negative slope on every unit, and the implied
one-way efficiency lands at 0.86-0.95 where the telemetry is clean.

    python colab/capacity_experiment/battery_reconstruct.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import _naive, data_dir  # noqa: E402

# 12 units with telemetry on >90% of the grid over 2025-06-30..2026-08-13.
STABLE_UNITS = ["BALB1", "BULBES1", "GANNB1", "HBESS1", "KESSB1", "LVES1",
                "MREHA1", "MREHA2", "MREHA3", "PIBESS1", "RANGEB1", "VBB1"]

SOC_CSV = "vic_battery_soc.csv"
PWR_CSV = "vic_battery_power.csv"

# The level series before this date does not reconcile with the power series -- the
# repo records the same thing ("only trustworthy from 2026-02", KESSB1 R2 collapsing to
# 0.198 on the full window). Fitting through it drags eta down to 0.889; from here on the
# estimate is stable at 0.914-0.924 across every later start date, so the early stretch
# is misalignment, not physics. Levels are still USED before this date (a seed is a
# reading, not a fit) -- only the loss PARAMETERS are estimated after it.
FIT_SINCE = "2025-10-01"


def load_units(dd: Path | None = None, units: list[str] | None = None) -> pd.DataFrame:
    """5-min frame: b_mw, battery_charging, battery_discharging, battery_overlap, soc_mwh.

    THE CHANNELS ARE AGGREGATED PER UNIT, NOT FROM THE NET. Those are different numbers.
    On 54% of intervals some units in the fleet charge while others discharge, so

        battery_charging     = sum_u max(-p_u, 0)          NOT max(-sum_u p_u, 0)
        battery_discharging  = sum_u max( p_u, 0)          NOT max( sum_u p_u, 0)

    and the two differ by exactly the OVERLAP m = min(chg, dis): chg = max(-b,0) + m and
    dis = max(b,0) + m. Measured over the span, m averages 6.3 MW, reaches 454 MW, and
    holds 6.4% of all discharge energy. Collapsing it away is a real loss of information
    about the fleet, and it is invisible to any check whose reference has already been
    collapsed.

    It is NOT a loss for the reservoir. The tank responds only to the net, so `b_mw` alone
    drives the SOC dynamics exactly. The overlap costs round-trip losses and nothing else,
    which is why the constraint layer can carry `b` as its decision variable and reattach
    `m` afterwards.

    The all-or-nothing rule on the sums is deliberate. A partial sum silently changes both
    the fleet and the reservoir's size, which is the failure mode this module exists to
    avoid.
    """
    dd = dd or data_dir()
    units = list(units or STABLE_UNITS)

    pw = pd.read_csv(dd / PWR_CSV, parse_dates=["interval"])
    pw["interval"] = _naive(pw["interval"])
    pw = pw.set_index("interval").sort_index()

    soc = pd.read_csv(dd / SOC_CSV, parse_dates=["interval"])
    soc["interval"] = _naive(soc["interval"])
    soc = soc.set_index("interval").sort_index()

    missing = ([u for u in units if f"{u}_mw" not in pw.columns]
               + [u for u in units if f"{u}_mwh" not in soc.columns])
    if missing:
        raise KeyError(f"units absent from the telemetry: {sorted(set(missing))}")

    U = pw[[f"{u}_mw" for u in units]]
    full = U.notna().all(axis=1)
    b = U.sum(axis=1).where(full)
    chg = (-U).clip(lower=0.0).sum(axis=1).where(full)
    dis = U.clip(lower=0.0).sum(axis=1).where(full)
    e = soc[[f"{u}_mwh" for u in units]].sum(axis=1, min_count=len(units))

    out = pd.DataFrame({"b_mw": b, "battery_charging": chg,
                        "battery_discharging": dis}).join(e.rename("soc_mwh"), how="outer")
    out["battery_overlap"] = np.minimum(out["battery_charging"],
                                        out["battery_discharging"])
    return out[["b_mw", "battery_charging", "battery_discharging",
                "battery_overlap", "soc_mwh"]].sort_index()


def reservoir_max(bat: pd.DataFrame, quantile: float | None = None) -> float:
    """Observed reservoir size in MWh.

    The default is the observed maximum, matching the repo's policy for caps: an envelope
    that forbids a level the fleet demonstrably reached is a different and unjustified
    claim. `quantile` is offered for sensitivity analysis only.
    """
    s = bat["soc_mwh"].dropna()
    return float(s.max() if quantile is None else s.quantile(quantile))


def fit_loss(bat: pd.DataFrame, dt_h: float = 5.0 / 60.0, since: str | None = FIT_SINCE):
    """(eta_one_way, draw_mwh_per_step) from the measured level.

    THREE DETAILS, each verified before being relied on (see
    constraints/hard_constraints_with_soc.md section 1):
      - power LEADS soc by one interval (power is stamped at the end of its dispatch
        interval, soc is an instantaneous snapshot), so the driver is b.shift(-1)
      - energy is the TRAPEZOID, not P*dt, because the power series is instantaneous
      - the loss has TWO terms. An efficiency alone and a constant draw alone fit almost
        equally well per step because each absorbs the other's mean; fitting both is what
        separates them. Fitting eta alone is how the repo canon got 0.834 -- one
        parameter doing two jobs.
    """
    d = bat[["b_mw", "soc_mwh"]].dropna().copy()
    if since is not None:
        d = d[d.index >= pd.Timestamp(since)]
    d = d[d.index.to_series().diff().dt.total_seconds().fillna(300) == 300]
    lead = d["b_mw"].shift(-1)
    chg = (-lead).clip(lower=0.0)
    dis = lead.clip(lower=0.0)
    # trapezoid over the step the level change spans
    chg_e = 0.5 * (chg + chg.shift(1)) * dt_h
    dis_e = 0.5 * (dis + dis.shift(1)) * dt_h
    dE = d["soc_mwh"].shift(-1) - d["soc_mwh"]
    m = dE.notna() & chg_e.notna() & dis_e.notna()
    # dE = eta*chg_e - dis_e/eta - draw  ->  solve for (eta, 1/eta, draw) unconstrained,
    # then report the charge-side coefficient as eta (the discharge side is the noisier
    # of the two: the telemetry's discharge coefficient comes back slightly above 1).
    A = np.column_stack([chg_e[m], -dis_e[m], -np.ones(int(m.sum()))])
    coef, *_ = np.linalg.lstsq(A, dE[m].values, rcond=None)
    eta = float(np.clip(coef[0], 0.5, 1.0))
    draw = float(max(coef[2], 0.0))
    pred = A @ coef
    r2 = 1.0 - ((dE[m].values - pred) ** 2).sum() / ((dE[m].values - dE[m].mean()) ** 2).sum()
    return eta, draw, float(r2)


def main() -> None:
    dd = data_dir()
    bat = load_units(dd)
    print(f"source {dd}\nunits ({len(STABLE_UNITS)}): {', '.join(STABLE_UNITS)}\n")
    print(f"rows {len(bat):,}   {bat.index.min()} .. {bat.index.max()}")
    cov = bat["soc_mwh"].notna().mean()
    print(f"aggregate SOC present on {100 * cov:.1f}% of the 5-min grid "
          f"(all {len(STABLE_UNITS)} units reporting)")

    emax = reservoir_max(bat)
    s = bat["soc_mwh"].dropna()
    print(f"\nE_max (observed)     {emax:8.0f} MWh")
    print(f"  level range        {100 * s.min() / emax:6.1f}% .. {100 * s.max() / emax:.1f}% of E_max")
    print(f"  within 5% of edge  {100 * ((s < 0.05 * emax) | (s > 0.95 * emax)).mean():6.2f}% of intervals")

    print("\nloss model, by fit start (eta and draw are the two terms that must be fitted")
    print("TOGETHER -- alone each absorbs the other's mean, which is how the repo canon")
    print("got eta_rt 0.834 with one parameter doing two jobs):")
    for since in (None, "2025-10-01", "2026-01-01", "2026-04-01"):
        e_, d_, r_ = fit_loss(bat, since=since)
        mark = "  <- used" if since == FIT_SINCE else ""
        print(f"  since {str(since):10s}  eta {e_:.3f} one-way (eta_rt {e_ ** 2:.3f})   "
              f"draw {d_ * 288:5.0f} MWh/day   step R2 {r_:.3f}{mark}")
    eta, draw, r2 = fit_loss(bat)

    # does the reservoir constraint actually bind on a whole day? -- the question that
    # decides whether enforcing it is doing work or is decoration
    d = bat.dropna(subset=["soc_mwh"])
    rows = []
    for day, gr in d.groupby(d.index.normalize()):
        if len(gr) < 270:
            continue
        e0 = gr["soc_mwh"].iloc[0]
        rows.append((e0, gr["soc_mwh"].max() - e0, emax - e0,
                     e0 - gr["soc_mwh"].min(), e0,
                     gr["soc_mwh"].max() - gr["soc_mwh"].min()))
    r = pd.DataFrame(rows, columns=["E0", "up", "up_room", "dn", "dn_room", "swing"])
    fc, fd = r.up / r.up_room.clip(lower=1), r.dn / r.dn_room.clip(lower=1)
    print(f"\nover {len(r)} clean days:")
    print(f"  charge headroom used     median {100 * fc.median():5.1f}%   p90 {100 * fc.quantile(.9):5.1f}%"
          f"   >95% on {100 * (fc > .95).mean():4.1f}% of days")
    print(f"  discharge headroom used  median {100 * fd.median():5.1f}%   p90 {100 * fd.quantile(.9):5.1f}%"
          f"   >95% on {100 * (fd > .95).mean():4.1f}% of days")
    print(f"  daily swing              median {r.swing.median():5.0f}   max {r.swing.max():5.0f} MWh"
          f"   ({100 * r.swing.max() / emax:.0f}% of E_max)")
    print("\n  the LEVEL form binds (charge side); the SWING form does not -- which is why\n"
          "  the swing budget in the existing heads never fires.")

    # how far the fueltech rollup is from this
    g = pd.read_parquet(dd / "vic_generation_20250101_20260814.parquet")
    g["interval"] = _naive(g["interval"])
    w = g.pivot_table(index="interval", columns="fueltech", values="power_mw")
    ft = w["battery_discharging"].fillna(0.0) - w["battery_charging"].fillna(0.0)
    diff = (ft - bat["b_mw"]).dropna()
    print(f"\nfueltech rollup minus this 12-unit sum: mean {diff.mean():.1f}  "
          f"std {diff.std():.1f}  max |.| {diff.abs().max():.1f} MW  "
          f"-- the reason the channels are rebuilt")

    m = bat["battery_overlap"].dropna()
    print(f"\nsimultaneous charge AND discharge across the fleet (the overlap m):")
    print(f"  present on {100 * (m > 1).mean():.1f}% of intervals (>1 MW), "
          f"{100 * (m > 20).mean():.1f}% above 20 MW")
    print(f"  mean {m.mean():.1f}  p90 {m.quantile(.9):.1f}  p99 {m.quantile(.99):.1f}  "
          f"max {m.max():.1f} MW")
    print(f"  holds {100 * m.sum() / bat['battery_discharging'].sum():.1f}% of all "
          f"discharge energy")
    print("  a single signed channel cannot represent this, so the head predicts it as a\n"
          "  separate output and reattaches it -- see head_traj.hardnet_traj(overlap=True)")


if __name__ == "__main__":
    main()
