"""Does the random window sampler actually miss peaks?

fit.sample draws `n` start positions uniformly WITH REPLACEMENT:
    rng.integers(0, Xflat.shape[0] - (2*CTX+gap), size=n)

A timestep t lands in the masked GAP of any window starting in
[t-CTX-gap+1, t-CTX], i.e. `gap` = 36 of the possible starts cover it. So

    P(t never masked) = (1 - gap/N_starts)^n

Measured here on the ACTUAL seed-0 draws, overall and by hour of day.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/nguyenminhquan/Downloads/energy_modelling")
sys.path.insert(0, str(ROOT / "imputation"))
sys.path.insert(0, str(ROOT / "planB"))

import gap_data as GD                                                  # noqa: E402
GD.NPZ = ROOT / "data/preprocessed/hist/5min/net_dispatch_ren/prepared.npz"
from gap_data import load_flats                                        # noqa: E402
from fit import CTX, GAP, flats_by_date                                # noqa: E402

f = load_flats()
N_TRAIN, SEED = 40000, 0


def report(name, n_rows, index=None):
    n_starts = n_rows - (2 * CTX + GAP)
    rng = np.random.default_rng(SEED)
    starts = rng.integers(0, n_starts, size=N_TRAIN)
    hits = np.zeros(n_rows, dtype=np.int32)
    for s in starts:                              # gap = [s+CTX, s+CTX+GAP)
        hits[s + CTX:s + CTX + GAP] += 1
    body = hits[CTX:n_rows - CTX - GAP]           # positions that CAN be masked
    print(f"\n{name}")
    print(f"  rows {n_rows:,}   distinct start positions {n_starts:,}   "
          f"windows drawn {N_TRAIN:,}")
    print(f"  unique starts used      {len(np.unique(starts)):,} "
          f"({100*len(np.unique(starts))/n_starts:.1f}% of possible)")
    print(f"  timesteps NEVER masked  {int((body == 0).sum()):,} "
          f"({100*(body == 0).mean():.3f}%)   predicted "
          f"{100*(1-GAP/n_starts)**N_TRAIN:.3f}%")
    print(f"  times masked per step   mean {body.mean():.1f}  "
          f"p1 {np.percentile(body,1):.0f}  median {np.median(body):.0f}")
    if index is not None:
        hr = index.hour.values[:n_rows]
        print("  coverage by hour of day (mean times masked):")
        line = "    "
        for h in range(24):
            m = (hr[CTX:n_rows - CTX - GAP] == h)
            line += f"{h:02d}h {body[m].mean():5.1f}  "
            if h % 6 == 5:
                print(line); line = "    "
        pk = np.isin(hr[CTX:n_rows - CTX - GAP], [18, 19, 20])
        print(f"  evening peak 18-21h: {body[pk].mean():.1f} times/step vs "
              f"{body[~pk].mean():.1f} elsewhere  -> ratio {body[pk].mean()/body[~pk].mean():.3f}")
        print(f"  share of all masked cells falling in 18-21h: "
              f"{100*body[pk].sum()/body.sum():.1f}%  (3h/24h = 12.5% if uniform)")


Xtr, _, idx_tr = flats_by_date(f, "2021-01-01", "2025-06-30")
report("DEFAULT npz train split (used by every headline planB arm)",
       f.Xtr.shape[0], idx_tr if len(idx_tr) == f.Xtr.shape[0] else None)

X25, _, idx25 = flats_by_date(f, "2025-01-01", "2026-03-31")
report("--train-range 2025-01-01 2026-03-31 (hardnet_2025 / unconstrained_2025)",
       X25.shape[0], idx25)
