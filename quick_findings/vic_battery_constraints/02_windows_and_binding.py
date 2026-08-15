"""Refine: pick a stationary fleet window, then re-derive every battery constraint."""
import numpy as np
import pandas as pd

pd.set_option("display.width", 220)
T = pd.read_parquet("data/preprocessed/hist/5min/net_dispatch_totdem/table.parquet")
T.index = pd.to_datetime(T.index)
DT = 5 / 60.0

print("=" * 78)
print("A. WHEN DID THE FLEET STOP GROWING?  (90-day trailing max, monthly samples)")
print("=" * 78)
roll = pd.DataFrame({
    "chg_90d_max": T.battery_charging.rolling("90D").max(),
    "dis_90d_max": T.battery_discharging.rolling("90D").max(),
}).loc["2025-01-01":].resample("MS").last()
print(roll.round(0).to_string())

WINDOWS = {
    "2025H1  (Jan-Jun 25)": ("2025-01-01", "2025-06-30"),
    "2025H2  (Jul-Dec 25)": ("2025-07-01", "2025-12-31"),
    "2026    (Jan-Jul 26)": ("2026-01-01", "2026-07-05"),
    "last12m (Jul25-Jul26)": ("2025-07-01", "2026-07-05"),
    "stable  (Dec25-Jul26)": ("2025-12-01", "2026-07-05"),
}

rows = []
for label, (a, b) in WINDOWS.items():
    W = T.loc[a:b]
    chg, dis = W.battery_charging.to_numpy(), W.battery_discharging.to_numpy()
    idx = W.index
    day = idx.floor("D")
    Echg = pd.Series(chg * DT, index=day).groupby(level=0).sum()
    Edis = pd.Series(dis * DT, index=day).groupby(level=0).sum()
    rt = Edis.sum() / Echg.sum()
    eta = np.sqrt(rt)
    E = pd.Series(np.cumsum((chg * eta - dis / eta) * DT), index=idx)
    swing = E.groupby(day).agg(lambda v: v.max() - v.min())
    r24 = (E.rolling(288).max() - E.rolling(288).min())
    dchg, ddis = np.diff(chg), np.diff(dis)
    rows.append(dict(
        window=label, days=len(swing),
        P_chg_max=chg.max(), P_dis_max=dis.max(),
        P_chg_p9999=np.quantile(chg, .9999), P_dis_p9999=np.quantile(dis, .9999),
        rt_eff=rt,
        E_p99=swing.quantile(.99), E_max=swing.max(), E_roll24_max=r24.max(),
        ramp_chg_dn=dchg.min(), ramp_chg_up=dchg.max(),
        ramp_dis_dn=ddis.min(), ramp_dis_up=ddis.max(),
        cyc_per_day=Edis.median() / swing.quantile(.99),
        overlap_p99=np.quantile(np.minimum(chg, dis), .99),
        overlap_max=np.minimum(chg, dis).max(),
    ))
D = pd.DataFrame(rows).set_index("window")
print("\n" + "=" * 78)
print("B. CONSTRAINTS BY WINDOW")
print("=" * 78)
print(D[["days", "P_chg_max", "P_dis_max", "P_chg_p9999", "P_dis_p9999", "rt_eff"]].round(3).to_string())
print()
print(D[["E_p99", "E_max", "E_roll24_max", "cyc_per_day"]].round(1).to_string())
print()
print(D[["ramp_chg_dn", "ramp_chg_up", "ramp_dis_dn", "ramp_dis_up", "overlap_p99", "overlap_max"]].round(1).to_string())

# ---------------------------------------------------------------- stable window
A, B = "2025-12-01", "2026-07-05"
W = T.loc[A:B]
chg, dis = W.battery_charging.to_numpy(), W.battery_discharging.to_numpy()
idx = W.index
day = idx.floor("D")
Echg = pd.Series(chg * DT, index=day).groupby(level=0).sum()
Edis = pd.Series(dis * DT, index=day).groupby(level=0).sum()
rt = Edis.sum() / Echg.sum(); eta = np.sqrt(rt)

print("\n" + "=" * 78)
print(f"C. IS THE ENERGY CONSTRAINT BINDING?   window {A}..{B}, eta_rt={rt:.4f}")
print("=" * 78)
E = pd.Series(np.cumsum((chg * eta - dis / eta) * DT), index=idx)
# per-day SOC path anchored so that min = 0
soc = E.groupby(day).transform(lambda v: v - v.min())
for cap in [4735.75, 4000, 3500, 3000, 2500, 2000]:
    frac = (soc > cap).mean()
    ndays = (soc.groupby(day).max() > cap).sum()
    print(f"  E_cap={cap:8.0f} MWh -> steps over {frac*100:6.3f}%   days violating {ndays:3d}/{len(Echg)}")
print(f"  daily SOC range used: median={soc.groupby(day).max().median():.0f}  "
      f"p90={soc.groupby(day).max().quantile(.9):.0f}  max={soc.groupby(day).max().max():.0f} MWh")

print("\n  --- how often is POWER at the cap? ---")
for name, x, cap in [("chg", chg, chg.max()), ("dis", dis, dis.max())]:
    for f in [0.95, 0.90, 0.80]:
        print(f"    {name}: steps within {(1-f)*100:.0f}% of max ({cap*f:.0f} MW): "
              f"{(x >= cap*f).mean()*100:.3f}%")

print("\n" + "=" * 78)
print("D. HOUR-OF-DAY PROFILE + PRICE  (stable window)")
print("=" * 78)
hod = pd.DataFrame({"chg": chg, "dis": dis, "net_dis": dis - chg,
                    "price": W.price_aud_per_mwh.to_numpy(),
                    "soc": soc.to_numpy()}, index=idx).groupby(idx.hour).mean()
print(hod.round(1).to_string())

print("\n" + "=" * 78)
print("E. SIMULTANEITY DETAIL (stable window) — does a net-power model lose anything?")
print("=" * 78)
mn = np.minimum(chg, dis)
print(f"  steps with min(chg,dis) > 20 MW : {(mn>20).mean()*100:.2f}%")
print(f"  energy in the overlap           : {mn.sum()*DT:,.0f} MWh "
      f"= {mn.sum()/dis.sum()*100:.2f}% of discharge energy")
print(f"  rt efficiency if computed on NET only: ", end="")
net = dis - chg
print(f"{(np.clip(net,0,None).sum())/(np.clip(-net,0,None).sum()):.4f}")
