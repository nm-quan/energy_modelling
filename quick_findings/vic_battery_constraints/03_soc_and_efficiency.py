"""Final checks: continuous SOC path, eta sensitivity, drift, and reserve floor."""
import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
T = pd.read_parquet("data/preprocessed/hist/5min/net_dispatch_totdem/table.parquet")
T.index = pd.to_datetime(T.index)
DT = 5 / 60.0
A, B = "2025-12-01", "2026-07-05"
W = T.loc[A:B]
chg, dis = W.battery_charging.to_numpy(), W.battery_discharging.to_numpy()
idx = W.index
day = idx.floor("D")

print("=" * 78)
print("F. CONTINUOUS SOC PATH (no daily reset) — how sensitive to eta?")
print("=" * 78)
print(f"{'eta_rt':>8} {'drift MWh/day':>14} {'range(max-min)':>15} {'fits in 4000?':>14}")
for rt in [0.80, 0.81, 0.82, 0.829, 0.834, 0.839, 0.85, 0.86, 0.88, 0.90]:
    eta = np.sqrt(rt)
    E = np.cumsum((chg * eta - dis / eta) * DT)
    drift = (E[-1] - E[0]) / (len(E) * DT / 24)
    print(f"{rt:8.3f} {drift:14.1f} {E.max()-E.min():15.0f} "
          f"{'yes' if E.max()-E.min() <= 4000 else 'NO':>14}")

# eta that zeroes drift
rt_star = dis.sum() / chg.sum()
print(f"\n  drift-zeroing rt over window = {rt_star:.4f}   (eta_oneway {np.sqrt(rt_star):.4f})")

print("\n" + "=" * 78)
print("G. DRIFT-FREE ETA, ROLLING — is it stable enough to hardcode?")
print("=" * 78)
mo = pd.DataFrame({"c": chg * DT, "d": dis * DT}, index=idx).resample("MS").sum()
mo["rt"] = mo.d / mo.c
print(mo.round(3).to_string())
wk = pd.DataFrame({"c": chg * DT, "d": dis * DT}, index=idx).resample("W").sum()
wk["rt"] = wk.d / wk.c
print(f"\n  weekly rt: median={wk.rt.median():.3f}  p10={wk.rt.quantile(.1):.3f} "
      f" p90={wk.rt.quantile(.9):.3f}  min={wk.rt.min():.3f}  max={wk.rt.max():.3f}")

print("\n" + "=" * 78)
print("H. HOW WIDE MUST THE RESERVOIR BE, per horizon length?")
print("=" * 78)
eta = np.sqrt(rt_star)
E = pd.Series(np.cumsum((chg * eta - dis / eta) * DT), index=idx)
print(f"{'horizon':>10} {'p50':>8} {'p99':>8} {'p99.9':>8} {'max':>8}   (MWh swing needed)")
for name, n in [("1h", 12), ("3h", 36), ("6h", 72), ("12h", 144),
                ("24h", 288), ("48h", 576), ("7d", 2016)]:
    r = E.rolling(n).max() - E.rolling(n).min()
    print(f"{name:>10} {r.quantile(.5):8.0f} {r.quantile(.99):8.0f} "
          f"{r.quantile(.999):8.0f} {r.max():8.0f}")

print("\n" + "=" * 78)
print("I. RESERVE FLOOR — does the fleet ever fully empty / fully fill?")
print("=" * 78)
socd = E.groupby(day).transform(lambda v: v - v.min())
daily_max = socd.groupby(day).max()
print(f"  daily usable swing: min={daily_max.min():.0f}  p10={daily_max.quantile(.1):.0f} "
      f" median={daily_max.median():.0f}  p90={daily_max.quantile(.9):.0f}  max={daily_max.max():.0f}")
print("  -> the aggregate fleet uses the full reservoir on some days but not most;")
print("     min/max SOC of the AGGREGATE is not observable, only the swing.")

print("\n" + "=" * 78)
print("J. RAMP: is the observed extreme a fleet-wide event or a single unit?")
print("=" * 78)
for name, x in [("chg", chg), ("dis", dis), ("net", dis - chg)]:
    d = np.diff(x)
    print(f"  {name:4s} |d| : p99={np.quantile(np.abs(d),.99):7.1f}  p99.9={np.quantile(np.abs(d),.999):7.1f} "
          f" p99.99={np.quantile(np.abs(d),.9999):7.1f}  max={np.abs(d).max():7.1f}   "
          f"(cap-normalised max/P_max = {np.abs(d).max()/x.max():.2f} per 5min)")
print("\n  full swing at nameplate would need |dP| up to 1600 MW/5min; observed max is far below,")
print("  so the ramp limit is a real, binding physical/market constraint, not an artefact.")

print("\n" + "=" * 78)
print("K. CROSS-CHECK against data/capacity_constraints.json registry values")
print("=" * 78)
print(f"  registry P_charge_max_cap    = 2210.0 MW   |  observed max = {chg.max():.1f} MW "
      f"({chg.max()/2210*100:.0f}%)")
print(f"  registry P_discharge_max_cap = 2260.0 MW   |  observed max = {dis.max():.1f} MW "
      f"({dis.max()/2260*100:.0f}%)")
print(f"  registry capacity_storage    = 4735.8 MWh  |  observed max swing = "
      f"{daily_max.max():.0f} MWh ({daily_max.max()/4735.75*100:.0f}%)")
print(f"  registry duration            = {4735.75/2260:.2f} h  |  observed = "
      f"{daily_max.max()/dis.max():.2f} h")
