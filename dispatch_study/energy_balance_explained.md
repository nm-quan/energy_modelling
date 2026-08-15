# Why generation ≠ demand, and what to do about it

Reproduce: the block at the end of this file. VIC, 2026, 5-minute data.

---

## 1. It does balance. Three terms were missing

A region's energy balance is not "generation = demand". It is

```
generation  +  net imports  =  demand  +  losses
```

and on top of that, "demand" has more than one definition. Test all four
combinations and only one closes:

| candidate identity | mean gap | **std** | as % of demand |
| --- | ---: | ---: | ---: |
| **gen + imports − demand** | **197 MW** | **136** | **3.9%** |
| gen + rooftop + imports − demand | 888 MW | 1,008 | 17.7% |
| gen − demand (ignore interconnector) | 747 MW | 570 | 15.9% |
| gen + rooftop − demand | 1,438 MW | 990 | 29.1% |

**Read the std column, not the mean.** A small mean with a *small* std means you
have the right physics and one systematic term left over. A big std means the
identity itself is wrong. Row 1 has a std of 136 MW; the others are 570–1,008.

So two rules fall out:

**Include the interconnector.** VIC is a net exporter — 550 MW on average in
2026. Generation must cover local demand *plus* what leaves the state. Ignoring
it puts you 747 MW out.

**Exclude rooftop solar.** The `demand` series is *operational demand*, which is
already measured net of rooftop PV — rooftop sits behind the customer's meter and
shows up as demand that never appears. Adding it to generation double-counts and
costs you 691 MW. (Note also that rooftop is an AEMO *estimate*, not a
measurement, so it carries its own error.)

## 2. The remaining 197 MW is transmission losses

Not an error. It has all three signatures of a loss term:

- **The sign is right.** Generation exceeds demand + exports on **96%** of
  intervals. It must — you always have to generate more than you deliver.
- **It scales with throughput.** Regressing the residual on generation and
  interconnector flow gives `197 ≈ 1.87% × generation + 15.2% × |net import|`.
  Losses are resistive, so they rise with how much power is moving.
- **It is stable as a fraction**, 3.1–3.6% across generation quintiles:

| generation quintile | mean generation | residual | as % |
| --- | ---: | ---: | ---: |
| Q1 | 4,099 MW | 129 MW | 3.1% |
| Q2 | 5,282 MW | 183 MW | 3.5% |
| Q3 | 5,791 MW | 210 MW | 3.6% |
| Q4 | 6,311 MW | 225 MW | 3.6% |
| Q5 | 7,318 MW | 237 MW | 3.2% |

3–4% is a normal intra-regional loss figure for the NEM. The fit only explains
27% of the *variance*, so losses are the level and the sign, not the whole story —
the rest is small non-scheduled generation not in the fueltech list, rooftop
estimation error, and 5-minute alignment between separately-published feeds.

## 3. Why this does not threaten the model

The model does not use this identity. Its balance is

```
net_demand(t)  =  Σ SIGN_i · P_i(t)
                = coal + gas_ocgt + gas_steam + hydro + batt_dis − batt_chg
```

which is **exact to 0.0 MW by construction** — `net_demand` is *defined* as that
sum in `lib/pipeline.py`, and `gap_data.py` verifies it. There is no internal
inconsistency to justify.

The 197 MW is not a gap *inside* the model. It is the distance between the
model's world and the physical grid — a **boundary** question. The right response
is to state the boundary, not to try to make it vanish.

## 4. How to say it in the write-up

State the modelling boundary explicitly, once, near the data description:

> We model the Victorian dispatchable fleet as a closed system. The balance
> enforced is `net_demand = Σ SIGN · P` over the six dispatchable channels, which
> holds exactly by construction. Against metered operational demand the fleet
> balance closes to 197 MW (3.9% of demand, std 136 MW), attributable to
> transmission losses — the residual is positive on 96% of intervals and scales
> at 1.87% of generation. Rooftop PV is excluded throughout, since operational
> demand is already net of it. Interconnector flows are netted into demand
> (`lib/pipeline.py:27`), so Victoria is treated as an island and the fleet serves
> demand-plus-net-exports.

That is a defensible, checkable statement. Reviewers object to unexplained
residuals, not to explained ones.

## 5. The one place it actually bites: defining a scenario

For a counterfactual you prescribe `nd`, so you need to translate "demand rises
by X" into "`nd` rises by Y". The chain is

```
Δnd  =  Δdemand · (1 + loss rate)  −  Δwind  −  Δsolar  −  Δnet_imports
```

Two decisions to make and to state:

1. **Which demand are you shifting?** Operational demand (what the fleet sees) or
   underlying demand (including rooftop)? If the latter, a 5% rise on underlying
   demand is a *larger* percentage rise on operational demand, because rooftop is
   unchanged. These are different experiments.
2. **Is the interconnector allowed to respond?** Under the island assumption it
   is not, so the whole rise falls on the VIC fleet. That is a legitimate and
   conservative choice, but it is a choice — the measured behaviour is that
   Victoria meets ~28% of a real demand rise by exporting less
   (`dispatch_study/midday_response_2026.md`, section B). Say which you are doing,
   because it changes every response share you report.

---

```python
import pandas as pd, numpy as np
T = pd.read_parquet("data/preprocessed/hist/5min/net_dispatch_ren/table.parquet")
T.index = pd.to_datetime(T.index)
ic = pd.read_parquet("data/vic_interconnector_20211001_20260706.parquet").set_index("interval")
T = T.join(ic["net_import_mw"], how="left")
W = T[T.index.year == 2026].dropna(subset=["net_import_mw"])
raw_demand = W.demand_mw + W.net_import_mw          # undo lib/pipeline.py:27
gen = W[["coal_brown","gas_ocgt","gas_steam","hydro","battery_discharging",
         "wind","solar_utility"]].sum(1) - W.battery_charging
r = gen + W.net_import_mw - raw_demand
print(r.mean(), r.std(), (r > 0).mean())
```
