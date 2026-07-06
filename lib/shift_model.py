"""Fixed-percentage demand-shift simulation model (first approach).

A deliberately simple, transparent transform of the test feature table:

  - During the daily free window (default 11:00-15:00 local), demand is
    increased by rebound_pct and the price is forced to free_price (0).
  - In every other interval (uniform load shift), demand is reduced by
    reduction_pct.

The load shift conserves energy: the raw MWh removed from the non-free
intervals (reduction_pct) is added back into the same day's free window. On top
of that the free window gets a rebound (original * rebound_pct) of induced
demand, so the only net change to daily energy is +rebound_pct of the
free-window energy -- a deliberate, controlled "increased demand", independent
of the reduction.

Only the demand feature and the price are modified. The dispatchable energy
histories are left untouched; the trained iTransformer then predicts how the
dispatch mix responds to the shifted demand + free price.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FixedPercentageShift:
    rebound_pct: float                 # +% demand inside the free window
    reduction_pct: float               # -% demand outside it (uniform load shift)
    free_hours: tuple[int, int] = (11, 13)   # [start, end) local hour -> 11:00-13:00
    free_price: float = 0.0            # price forced to this inside the free window
    uniform: bool = True              # reduce every non-free interval uniformly
    demand_col: str = "demand_mw"     # which demand series to shift
    price_col: str = "price_aud_per_mwh"

    def free_mask(self, index: pd.DatetimeIndex) -> pd.Series:
        """Boolean mask: True for intervals inside the daily free window."""
        h = index.hour
        return (h >= self.free_hours[0]) & (h < self.free_hours[1])

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of df with demand load-shifted and free-window price zeroed.

        Per calendar day: cut the non-free intervals by reduction_pct, then add
        the raw MWh removed back evenly across that day's free intervals (the
        energy-conserving shift), plus original*rebound_pct of induced demand.
        """
        out = df.copy()
        idx = out.index
        free = np.asarray(self.free_mask(idx))
        red = self.reduction_pct / 100.0
        reb = self.rebound_pct / 100.0
        d0 = out[self.demand_col].to_numpy(dtype=float, copy=True)

        new = d0.copy()
        add = np.zeros_like(d0)
        if self.uniform:
            new[~free] = d0[~free] * (1.0 - red)
            removed = np.where(~free, d0 * red, 0.0)
            agg = pd.DataFrame({"day": idx.normalize(),
                                "removed": removed, "free": free.astype(float)})
            shifted = agg.groupby("day")["removed"].transform("sum").to_numpy()
            n_free = agg.groupby("day")["free"].transform("sum").to_numpy()
            add = np.where(free & (n_free > 0), shifted / np.where(n_free > 0, n_free, 1.0), 0.0)
        new[free] = d0[free] * (1.0 + reb) + add[free]

        out[self.demand_col] = new
        out.loc[free, self.price_col] = self.free_price
        return out
