"""Infer VIC battery constraints from 2025-2026 5-min dispatch."""
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)
T = pd.read_parquet("data/preprocessed/hist/5min/net_dispatch_totdem/table.parquet")
T.index = pd.to_datetime(T.index)
DT = 5 / 60.0  # hours per step

chg = T["battery_charging"]
dis = T["battery_discharging"]

print("=" * 78)
print("0. FLEET GROWTH — why the window matters")
print("=" * 78)
m = pd.DataFrame({
    "chg_max": chg.resample("MS").max(),
    "dis_max": dis.resample("MS").max(),
    "chg_p99.9": chg.resample("MS").quantile(0.999),
    "dis_p99.9": dis.resample("MS").quantile(0.999),
    "chg_TWh": chg.resample("MS").sum() * DT / 1e6,
    "dis_TWh": dis.resample("MS").sum() * DT / 1e6,
})
print(m.round(1).to_string())

# ---- restrict to the recent fleet -------------------------------------------
W = T.loc["2025-01-01":]
chg = W["battery_charging"].to_numpy()
dis = W["battery_discharging"].to_numpy()
idx = W.index
print(f"\nWINDOW {idx[0]} -> {idx[-1]}  n={len(W)} steps "
      f"({len(W)*5/60/24:.0f} days)")

print("\n" + "=" * 78)
print("1. POWER CAPS")
print("=" * 78)
for name, x in [("charging", chg), ("discharging", dis)]:
    qs = np.quantile(x, [0.5, 0.9, 0.99, 0.999, 0.9999, 1.0])
    print(f"{name:12s} max={x.max():8.1f}  p99.99={qs[4]:8.1f}  p99.9={qs[3]:8.1f} "
          f" p99={qs[2]:8.1f}  p90={qs[1]:7.1f}  median={qs[0]:6.1f} "
          f" min={x.min():8.3f}  frac>0={np.mean(x>0):.3f}")
# year-on-year
for yr in [2025, 2026]:
    s = W.loc[str(yr)]
    print(f"  {yr}: chg max {s.battery_charging.max():7.1f}   "
          f"dis max {s.battery_discharging.max():7.1f}")

print("\n" + "=" * 78)
print("2. RAMPS (MW / 5 min)")
print("=" * 78)
for name, x in [("charging", chg), ("discharging", dis), ("net(dis-chg)", dis - chg)]:
    d = np.diff(x)
    print(f"{name:14s} down: max={d.min():9.1f}  p0.1={np.quantile(d,0.001):8.1f} "
          f" p1={np.quantile(d,0.01):8.1f}   |  up: p99={np.quantile(d,0.99):8.1f} "
          f" p99.9={np.quantile(d,0.999):8.1f}  max={d.max():8.1f}")

print("\n" + "=" * 78)
print("3. SIMULTANEITY — is charge/discharge mutually exclusive?")
print("=" * 78)
both = (chg > 1) & (dis > 1)
print(f"both channels > 1 MW  : {both.mean()*100:.1f}% of steps")
for thr in [5, 20, 50, 100]:
    b = (chg > thr) & (dis > thr)
    print(f"  both > {thr:3d} MW      : {b.mean()*100:5.2f}%   "
          f"min(chg,dis) mean when so = {np.minimum(chg,dis)[b].mean() if b.any() else 0:6.1f} MW")
mn = np.minimum(chg, dis)
print(f"overlap min(chg,dis): mean={mn.mean():.1f}  p99={np.quantile(mn,0.99):.1f}  max={mn.max():.1f} MW")
print(f"net = dis-chg        : min={(dis-chg).min():.1f}  max={(dis-chg).max():.1f} MW")

print("\n" + "=" * 78)
print("4. ROUND-TRIP EFFICIENCY")
print("=" * 78)
day = idx.floor("D")
dchg = pd.Series(chg * DT, index=day).groupby(level=0).sum()
ddis = pd.Series(dis * DT, index=day).groupby(level=0).sum()
r = (ddis / dchg).replace([np.inf, -np.inf], np.nan).dropna()
print(f"whole window   : charge {dchg.sum():,.0f} MWh   discharge {ddis.sum():,.0f} MWh"
      f"   ratio = {ddis.sum()/dchg.sum():.4f}")
print(f"daily ratio    : median={r.median():.3f}  mean={r.mean():.3f} "
      f" p10={r.quantile(.1):.3f}  p90={r.quantile(.9):.3f}  n={len(r)}")
mo = pd.DataFrame({"chg": dchg, "dis": ddis}).resample("MS").sum()
mo["ratio"] = mo.dis / mo.chg
print(mo.round(3).to_string())

print("\n" + "=" * 78)
print("5. IMPLIED SOC / RESERVOIR")
print("=" * 78)
for eta in [1.0, np.sqrt(0.834), np.sqrt(ddis.sum()/dchg.sum())]:
    dE = (chg * eta - dis / eta) * DT
    E = np.concatenate([[0.0], np.cumsum(dE)])
    s = pd.Series(E[1:], index=idx)
    swing_day = s.groupby(day).agg(lambda v: v.max() - v.min())
    print(f"\n  eta_oneway={eta:.4f} (rt={eta**2:.3f})")
    print(f"    whole-window drift  : {E[-1]-E[0]:+,.0f} MWh over {len(r)} days "
          f"({(E[-1]-E[0])/len(r):+.1f} MWh/day)")
    print(f"    daily swing (MWh)   : median={swing_day.median():.0f} "
          f" p90={swing_day.quantile(.9):.0f}  p99={swing_day.quantile(.99):.0f} "
          f" max={swing_day.max():.0f}")
    for wname, wsteps in [("6h", 72), ("12h", 144), ("24h", 288), ("48h", 576)]:
        ser = pd.Series(E[1:])
        rng = ser.rolling(wsteps).max() - ser.rolling(wsteps).min()
        print(f"    rolling {wname:>3s} swing  : p99={rng.quantile(.99):7.0f} "
              f" max={rng.max():7.0f} MWh")

print("\n" + "=" * 78)
print("6. DURATION & CYCLES  (using eta = sqrt(measured rt))")
print("=" * 78)
eta = np.sqrt(ddis.sum() / dchg.sum())
dE = (chg * eta - dis / eta) * DT
E = pd.Series(np.cumsum(dE), index=idx)
swing_day = E.groupby(day).agg(lambda v: v.max() - v.min())
Emax = swing_day.quantile(0.99)
Pdis = np.quantile(dis, 0.9999)
print(f"E_max (p99 of daily swing) = {Emax:,.0f} MWh")
print(f"P_dis (p99.99)             = {Pdis:,.0f} MW")
print(f"implied duration           = {Emax/Pdis:.2f} h")
print(f"daily discharge energy     : median={ddis.median():,.0f} MWh "
      f"=> {ddis.median()/Emax:.2f} equivalent cycles/day")

print("\n" + "=" * 78)
print("7. WHEN does it charge / discharge (hour-of-day, and vs price)")
print("=" * 78)
hod = pd.DataFrame({"chg": chg, "dis": dis, "price": W["price_aud_per_mwh"].to_numpy()},
                   index=idx).groupby(idx.hour).mean()
print(hod.round(1).to_string())
