"""The four constraints, as data. One module, imported by both the head and the scorer.

Anything that both a projection and its own audit read from the same place cannot be
proved by that audit -- so the head (`head_traj.py`) and the independent scorer both
import their numbers from HERE, and only their numbers. The mechanisms stay separate.

    1. BALANCE     sum_i SIGN_i P[t,i] = nd[t]          nd from OBSERVABLES, not from truth
    2. SOC         0 <= E[t] <= E_max,  E linear in P    level form, measured seed
    3. CAPACITY    P_min(t) <= P[t] <= P_max(t)          TIME-VARYING, monthly C(t)
    4. RAMP        |P[t+k] - P[t]| <= R(k)               a TABLE over k, not one number

FIVE CHANNELS, NOT SIX. The battery is carried as one SIGNED channel b = dis - chg.
That is not cosmetic -- it is what makes constraint 2 exactly representable. dE is a
strictly decreasing piecewise-linear function of b, so a bound on dE inverts to a plain
INTERVAL on b, and the per-step feasible set stays {box} inter {hyperplane}: the shape
the exact closed-form projection needs. Split into (chg, dis) on the way out.
Carried per-channel, dE's bound is not a box at all, and the conservative separable
surrogate the repo uses today is simultaneously too loose to guarantee the reservoir and
too tight to stay compatible with balance -- measured, it overflows by 310-5,700 MWh.

RAMPS AT THE EMPIRICAL MAX, NOT A PERCENTILE. The repo tightened its per-step ramp to a
percentile because a single-step max is set by unit trips and is therefore inert. A
duration TABLE removes that reason: R(288)/(288*R(1)) is 0.007-0.030, i.e. the per-step
box is 33-140x looser than the real envelope over a day, and the table recovers all of
it while still admitting every recorded step. Measured, the percentile costs what the max
does not: at p99.9 the balance target becomes unreachable inside the tightened box on
real windows (worst shortfall 443 MW), so "balance exact by construction" would degrade
to "exact except where the envelope forbids it". At the max the shortfall is 0.00 MW.

    python colab/capacity_experiment/constraint_set.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from battery_reconstruct import fit_loss, load_units, reservoir_max  # noqa: E402
from pipeline import _naive, data_dir  # noqa: E402

# ---------------------------------------------------------------- channels + geometry
CHANNELS = ["hydro", "coal_brown", "gas_steam", "gas_ocgt", "battery"]
SIGN = np.array([1.0, 1.0, 1.0, 1.0, 1.0])          # battery is signed, so all +1
BATT = CHANNELS.index("battery")
REPORT = ["hydro", "coal_brown", "gas_steam", "gas_ocgt",
          "battery_charging", "battery_discharging"]   # what accuracy is scored on
# The reported battery channels are NOT a function of the signed one. Across the fleet,
# some units charge while others discharge on 54% of intervals, and the two views differ by
# the OVERLAP m = min(chg, dis). So the head carries `b` (which is all the reservoir sees)
# and predicts `m` alongside it; REPORT is reconstructed from both. See to_report().
OVERLAP = "battery_overlap"

DT_H = 5.0 / 60.0
STEPS_PER_DAY = 288

# observables: known everywhere, including inside the gap
OBSERVED = ["demand", "price", "wind", "solar_utility", "solar_rooftop",
            "curtailment_wind", "curtailment_solar", "net_import"]
ND_FEATURES = ["demand", "wind", "solar_utility", "net_import"]

CAP_OF = {"hydro": "cap_hydro", "coal_brown": "cap_coal_brown",
          "gas_steam": "cap_gas_steam", "gas_ocgt": "cap_gas_ocgt",
          "battery": "cap_battery"}

INTERCONNECTOR = ["vic_interconnector_20211001_20260706.parquet",
                  "vic_interconnector_20260529_20260713.parquet"]


# ------------------------------------------------------------------------ data loading
def _interconnector(root: Path) -> pd.Series:
    """VIC net imports. Not in data/updata -- script/pull_updata.py never saved the stream
    even though script/pull_data_hist.py fetches it. Until it is re-pulled this comes from
    the tracked parquets, which END 2026-07-12 while generation runs to 2026-08-13; the
    joined table is truncated accordingly and load_table reports the loss."""
    parts = []
    for name in INTERCONNECTOR:
        p = root / "data" / name
        if p.exists():
            d = pd.read_parquet(p)
            d["interval"] = _naive(d["interval"])
            parts.append(d.set_index("interval")["net_import_mw"])
    if not parts:
        raise FileNotFoundError(f"no interconnector parquet under {root / 'data'}")
    s = pd.concat(parts)
    return s[~s.index.duplicated()].sort_index().rename("net_import")


def load_table(dd: Path | None = None, root: Path | None = None) -> pd.DataFrame:
    """5-min table: the 5 constrained channels in MW, the observables, cap_* and soc_mwh.

    Battery power and level come from battery_reconstruct (the 12-unit set), NOT from the
    fueltech rollup -- the channels have to drive the reservoir we measure.
    """
    dd = dd or data_dir()
    root = root or Path(__file__).resolve().parents[2]

    g = pd.read_parquet(dd / "vic_generation_20250101_20260814.parquet")
    g["interval"] = _naive(g["interval"])
    w = g.pivot_table(index="interval", columns="fueltech", values="power_mw",
                      aggfunc="first")

    m = pd.read_parquet(dd / "vic_market_20250101_20260814.parquet")
    m["interval"] = _naive(m["interval"])
    m = m.set_index("interval").rename(columns={"demand_mw": "demand",
                                                "price_aud_per_mwh": "price"})

    c = pd.read_parquet(dd / "vic_curtailment_20250101_20260814.parquet")
    c["interval"] = _naive(c["interval"])
    c = c.set_index("interval").rename(columns={"curtailment_wind_mw": "curtailment_wind",
                                                "curtailment_solar_mw": "curtailment_solar"})

    bat = load_units(dd)

    df = pd.DataFrame(index=w.index)
    for ch in CHANNELS[:BATT]:
        df[ch] = w[ch].clip(lower=0.0)
    df["battery"] = bat["b_mw"]
    # the TRUE reported channels, aggregated per unit -- not derived from the net
    df["battery_charging"] = bat["battery_charging"]
    df["battery_discharging"] = bat["battery_discharging"]
    df[OVERLAP] = bat[OVERLAP]
    df["soc_mwh"] = bat["soc_mwh"]
    for col in ("wind", "solar_utility", "solar_rooftop"):
        df[col] = w[col]
    df = df.join(m[["demand", "price"]]).join(
        c[["curtailment_wind", "curtailment_solar"]]).join(_interconnector(root))

    h = df.index.hour + df.index.minute / 60.0
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    dow = df.index.dayofweek
    df["dow_sin"], df["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)

    cap = pd.read_parquet(dd / "capacity_VIC1_by_fueltech_MS.parquet")
    cap["interval"] = _naive(cap["interval"])
    cw = cap.pivot_table(index="interval", columns="fueltech", values="capacity_mw",
                         aggfunc="first").reindex(df.index, method="ffill")
    cw.columns = [f"cap_{k}" for k in cw.columns]
    df = df.join(cw)

    need = CHANNELS + REPORT[BATT:] + [OVERLAP] + OBSERVED + list(CAP_OF.values())
    keep = df[need].notna().all(axis=1)
    return df.loc[keep].sort_index()


def segments(index: pd.DatetimeIndex) -> np.ndarray:
    """Contiguity label -- ramps may only be measured across adjacent 5-min steps."""
    return (pd.Series(index).diff() != pd.Timedelta("5min")).cumsum().values


# --------------------------------------------------------------------- 4. the ramp table
def ramp_table(X: np.ndarray, seg: np.ndarray, kmax: int = STEPS_PER_DAY + 1):
    """(R_up, R_dn), each (kmax+1, n_ch) MW: the largest move ever made over k steps.

    Three passes, each load-bearing:

    1. EMPIRICAL MAX per k, rounded outward to 0.1 MW -> 0 historical violations by
       construction, which is the repo's stated policy for anything that validates
       recorded dispatch.
    2. MONOTONE in k by a running max. A displacement reachable in k steps is reachable in
       k+1 (move, then hold), so the true R is non-decreasing; where the raw estimate dips
       it is a sampling artifact, and taking it would assert that a big jump MUST reverse.
    3. SUBADDITIVE CLOSURE, R(a+b) <- min(R(a+b), R(a)+R(b)), by DP.

    Pass 3 is what makes the head PROVABLE. In the forward sweep the step-t interval is
    non-empty by induction exactly when R(m+1) <= R(m) + R(1) -- the forward cone from the
    fixed past and the backward cone to the pinned right endpoint then always intersect.
    The empirical max already satisfies it (|x_{t+a+b} - x_t| <= R(a) + R(b) by the
    triangle inequality, so the max of the left side cannot exceed the right); the closure
    is computed anyway so the guarantee does not rest on that argument holding for whatever
    estimator a future caller substitutes.
    """
    n = X.shape[1]
    ru = np.zeros((kmax + 1, n))
    rd = np.zeros((kmax + 1, n))
    for k in range(1, kmax + 1):
        d = X[k:] - X[:-k]
        d = d[seg[k:] == seg[:-k]]
        if len(d) == 0:
            ru[k], rd[k] = ru[k - 1], rd[k - 1]
            continue
        ru[k] = np.ceil(d.max(0) * 10) / 10
        rd[k] = np.ceil(-d.min(0) * 10) / 10
    for k in range(2, kmax + 1):                                    # monotone
        ru[k] = np.maximum(ru[k], ru[k - 1])
        rd[k] = np.maximum(rd[k], rd[k - 1])
    for k in range(2, kmax + 1):                                    # subadditive closure
        for j in range(1, k // 2 + 1):
            ru[k] = np.minimum(ru[k], ru[j] + ru[k - j])
            rd[k] = np.minimum(rd[k], rd[j] + rd[k - j])
    return ru, rd


def check_subadditive(R: np.ndarray) -> int:
    """Number of (a,b) pairs violating R(a+b) <= R(a)+R(b). Must be 0."""
    K = len(R) - 1
    bad = 0
    for a in range(1, K + 1):
        for b in range(a, K + 1 - a):
            bad += int((R[a + b] > R[a] + R[b] + 1e-9).any())
    return bad


# ------------------------------------------------------------------- 3. time-varying cap
def capacity_bounds(df: pd.DataFrame, lift_to: np.ndarray | None = None):
    """(P_min, P_max), each (T, n_ch) MW, from the MONTHLY installed capacity.

    The battery is signed, so its bounds are [-C, +C]: charging is a load. Everything else
    floors at 0.

    `lift_to` (usually the recorded dispatch) widens the bound where the monthly registry
    sits below output the fleet demonstrably produced -- 0.05% of channel-steps. Declaring
    recorded dispatch infeasible would make every violation table unreadable, so the
    validation envelope lifts and both numbers get reported.
    """
    C = np.column_stack([df[CAP_OF[c]].values for c in CHANNELS])
    P_max = C.copy()
    P_min = np.zeros_like(C)
    P_min[:, BATT] = -C[:, BATT]
    if lift_to is not None:
        P_max = np.maximum(P_max, lift_to)
        P_min = np.minimum(P_min, lift_to)
    return P_min, P_max


# -------------------------------------------------------------------------- 2. reservoir
class Reservoir:
    """SOC as an exact interval on the signed battery channel.

    E is LINEAR in the dispatch, so 0 <= E[t] <= E_max is just more linear inequalities and
    the feasible set stays a polytope. The work is turning a bound on the ACCUMULATION into
    a bound on the CURRENT STEP without conservatism, which is what `b_window` does.

    Integration is the trapezoid, E_t - E_{t-1} = c(b_t) + c(b_{t-1}) - draw, because the
    power series is instantaneous; the repo measured trapezoid + the one-interval power
    lead as R2 0.944 -> 0.986 against P*dt. The b_{t-1} term is already fixed when step t
    is projected, so it only shifts the interval.
    """

    def __init__(self, eta: float, draw_per_step: float, e_max: float, dt_h: float = DT_H,
                 e_min: float = 0.0):
        self.eta, self.draw, self.e_max, self.dt = eta, draw_per_step, e_max, dt_h
        self.e_min = e_min
        self.a = eta * dt_h / 2.0                 # MWh per MW of charge, half-step
        self.b = dt_h / (2.0 * eta)               # MWh per MW of discharge, half-step

    @property
    def delta(self):
        """MWh burned per MW of OVERLAP, per half step. Positive because eta < 1: pushing
        m MW through the fleet in both directions at once charges m*a and discharges m*b,
        netting -m*(b - a) -- pure round-trip loss, no change in net power."""
        return self.b - self.a

    def contrib(self, b, m=0.0):
        """Half-step energy from signed power b (and overlap m). STRICTLY DECREASING in b --
        the property the whole reformulation rests on (slope -a below 0, -b above, a < b).

        The overlap term is subtracted, so m can only LOWER the level. That is what makes
        the overlap safe to bolt on afterwards: the binding SOC bound here is the ceiling
        (the median day uses 78% of its charging headroom and the floor never binds), and
        nothing that only lowers the level can breach a ceiling."""
        return (self.a * np.maximum(-b, 0.0) - self.b * np.maximum(b, 0.0)
                - self.delta * m)

    def contrib_inv(self, v):
        """The inverse. Decreasing, so it maps [lo,hi] to [inv(hi), inv(lo)]."""
        return np.where(v >= 0.0, -v / self.a, -v / self.b)

    def tube(self, b_lo: np.ndarray, b_hi: np.ndarray,
             e_target: float | None = None, tol: float = 0.0):
        """(E_lo, E_hi) each (n+1,). Mirrors head_traj.SocTorch.tube -- see that docstring
        for why the rates are not clamped at zero."""
        n = len(b_lo)
        dE_min = 2.0 * self.contrib(b_hi) - self.draw
        dE_max = 2.0 * self.contrib(b_lo) - self.draw
        lo = np.full(n + 1, self.e_min)
        hi = np.full(n + 1, self.e_max)
        if e_target is not None:
            lo[n] = max(self.e_min, e_target - tol)
            hi[n] = min(self.e_max, e_target + tol)
        for j in range(n - 1, -1, -1):
            hi[j] = min(self.e_max, hi[j + 1] - dE_min[j])
            lo[j] = min(max(self.e_min, lo[j + 1] - dE_max[j]), hi[j])
        return lo, hi

    def b_window(self, e_prev, b_prev, e_lo_next, e_hi_next):
        """The step's admissible signed power, EXACTLY.

            dE = contrib(b) + contrib(b_prev) - draw   and   E_prev + dE in [lo, hi]
          =>  contrib(b) in [lo - E_prev - contrib(b_prev) + draw, hi - ... ]
          =>  b in [contrib_inv(hi'), contrib_inv(lo')]     (contrib is decreasing)

        No conservatism anywhere: this is an equivalence, not a sufficient condition.
        """
        k = self.contrib(b_prev) - self.draw
        lo_v = e_lo_next - e_prev - k
        hi_v = e_hi_next - e_prev - k
        return self.contrib_inv(hi_v), self.contrib_inv(lo_v)

    def advance(self, e_prev, b_prev, b, m_prev=0.0, m=0.0):
        return (e_prev + self.contrib(b, m) + self.contrib(b_prev, m_prev) - self.draw)

    def trajectory(self, B: np.ndarray, e0, b_prev, M=None, m_prev=0.0):
        """(T,) or (B,T) levels from a signed-power path, for scoring."""
        B = np.asarray(B, dtype=float)
        M = np.zeros_like(B) if M is None else np.asarray(M, dtype=float)
        prev = np.broadcast_to(np.asarray(b_prev, dtype=float), B.shape[:-1])
        mprev = np.broadcast_to(np.asarray(m_prev, dtype=float), B.shape[:-1])
        e = np.broadcast_to(np.asarray(e0, dtype=float), B.shape[:-1]).astype(float).copy()
        out = np.empty_like(B)
        for t in range(B.shape[-1]):
            e = self.advance(e, prev, B[..., t], mprev, M[..., t])
            out[..., t] = e
            prev, mprev = B[..., t], M[..., t]
        return out


# ---------------------------------------------------------------------------- 1. balance
def fit_nd_map(df: pd.DataFrame, train: np.ndarray):
    """Least-squares map from OBSERVABLES to the net demand the 5 channels must serve.

    THE POINT OF THIS FUNCTION. Every constrained model in the repo is fed
    nd = SIGN . truth, which makes "balance exact by construction" circular -- and when
    PUBLICATION_REVIEW.md swapped in an observable nd the residual went to 3,272 MW and
    accuracy fell below linear interpolation. The identity
    demand - wind - solar - net_import does not close either (194 MW MAE) because rooftop
    PV, distribution losses and the non-scheduled fueltechs sit outside the six channels.
    Fitted, it closes to ~53 MW out of sample against a 3,872 MW level, which is what makes
    constraint 1 a real constraint rather than a restatement of the answer.
    """
    A = np.column_stack([df.loc[train, ND_FEATURES].values,
                         np.ones(int(train.sum()))])
    y = (df.loc[train, CHANNELS].values * SIGN).sum(-1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def nd_observable(df: pd.DataFrame, coef: np.ndarray) -> np.ndarray:
    A = np.column_stack([df[ND_FEATURES].values, np.ones(len(df))])
    return A @ coef


def nd_truth(df: pd.DataFrame) -> np.ndarray:
    return (df[CHANNELS].values * SIGN).sum(-1)


def split_battery(B: np.ndarray, M: np.ndarray | float = 0.0):
    """Signed b (+ overlap m) -> (charging, discharging).

        chg = m + max(-b, 0)        dis = m + max(b, 0)

    m = 0 recovers the net-only view, which is what a single signed channel can express and
    is wrong by exactly m on both channels.
    """
    return M + np.maximum(-B, 0.0), M + np.maximum(B, 0.0)


def to_report(P: np.ndarray, M: np.ndarray | float = 0.0) -> np.ndarray:
    """(..., 5) signed [+ (...,) overlap] -> (..., 6) in REPORT order, for scoring."""
    chg, dis = split_battery(P[..., BATT], M)
    return np.concatenate([P[..., :BATT], chg[..., None], dis[..., None]], axis=-1)


def truth_report(df: pd.DataFrame) -> np.ndarray:
    """The 6 reported channels as RECORDED -- overlap included. This, not to_report(truth),
    is the reference an accuracy or feasibility-floor claim has to be made against; a
    reference built by collapsing the truth is blind to the very thing being tested."""
    return df[REPORT].values


def calibrate_floor(df: pd.DataFrame, res: "Reservoir", round_to: float = 10.0) -> float:
    """How far below zero the MODELLED level runs when driven by the RECORDED dispatch.

    The reservoir recursion is open loop: it is seeded on a measured level and then
    integrates predicted power, so it inherits a drift the telemetry does not have. Over a
    whole day that drift is large -- measured here, the modelled level departs the measured
    one by up to 44% of E_max, which is the same order the repo reports (24 h p90 17.7%).

    Enforcing a hard floor of exactly 0 against a drifting estimate would declare recorded
    dispatch infeasible, which is the one thing CONSTRAINT.md says an envelope must never
    do. So the floor is lowered to admit every recorded day, and the amount is reported
    rather than chosen. The CEILING needs no such margin -- the modelled level never
    exceeds E_max on recorded dispatch -- and the ceiling is the side the data says
    actually binds (median day uses 78% of its charging headroom).
    """
    idx, pos = df.index, pd.Series(np.arange(len(df)), index=df.index)
    B = df["battery"].values
    worst = 0.0
    for day in pd.DatetimeIndex(sorted(set(idx.normalize()))):
        lo_t, hi_t = day - pd.Timedelta("5min"), day + pd.Timedelta(days=1)
        if lo_t not in pos.index or hi_t not in pos.index:
            continue
        i0 = int(pos[lo_t])
        if int(pos[hi_t]) - i0 != STEPS_PER_DAY + 1:
            continue
        e0 = df["soc_mwh"].iloc[i0]
        if not np.isfinite(e0):
            continue
        E = res.trajectory(B[i0 + 1:i0 + 1 + STEPS_PER_DAY], float(e0), B[i0])
        worst = min(worst, float(E.min()))
    return float(np.floor(worst / round_to) * round_to)


def build(dd: Path | None = None, root: Path | None = None, kmax: int = STEPS_PER_DAY + 1,
          train_end: str = "2026-02-28"):
    """Everything the head and the scorer need, from one call."""
    df = load_table(dd, root)
    seg = segments(df.index)
    ru, rd = ramp_table(df[CHANNELS].values, seg, kmax=kmax)
    bat = load_units(dd or data_dir())
    # the overlap gets its own R(k), on the same footing as the dispatch channels -- it is
    # a physical MW quantity with its own dynamics, not a free residual
    rmu, rmd = ramp_table(df[[OVERLAP]].values, seg, kmax=kmax)
    eta, draw, _ = fit_loss(bat)
    res = Reservoir(eta, draw, reservoir_max(bat))
    res.e_min = calibrate_floor(df, res)
    train = np.asarray(df.index <= pd.Timestamp(train_end))
    return dict(df=df, seg=seg, R_up=ru, R_dn=rd, Rm_up=rmu, Rm_dn=rmd, res=res,
                nd_coef=fit_nd_map(df, train), train_mask=train)


def main() -> None:
    cs = build()
    df, ru, rd, res = cs["df"], cs["R_up"], cs["R_dn"], cs["res"]
    print(f"rows {len(df):,}   {df.index.min()} .. {df.index.max()}   "
          f"{cs['seg'].max() + 1} contiguous segments")
    print(f"  (generation runs to 2026-08-13; the table stops where the interconnector "
          f"series does)\n")

    ks = [1, 3, 6, 12, 24, 36, 72, 144, 288]
    print("=" * 100)
    print("4. RAMP TABLE R(k) -- MW over k steps of 5 min, empirical max")
    print("=" * 100)
    print(f"  {'channel':14s} " + "".join(f"{f'{k * 5}m':>9s}" for k in ks))
    for i, c in enumerate(CHANNELS):
        print(f"  {c:12s} up " + "".join(f"{ru[k][i]:9.0f}" for k in ks))
        print(f"  {'':12s} dn " + "".join(f"{rd[k][i]:9.0f}" for k in ks))
    print(f"\n  R(k) / [k * R(1)] -- 1.00 would mean the per-step box was already right")
    print(f"  {'channel':14s} " + "".join(f"{f'{k * 5}m':>9s}" for k in ks[1:]))
    for i, c in enumerate(CHANNELS):
        r1 = max(ru[1][i], rd[1][i])
        print(f"  {c:14s} " + "".join(
            f"{max(ru[k][i], rd[k][i]) / (k * r1):9.3f}" for k in ks[1:]))
    print("\n  THE INVERSE VIEW -- shortest time to move X% of installed capacity:")
    caps = df[[CAP_OF[c] for c in CHANNELS]].iloc[-1].values
    pcts = [5, 10, 20, 30, 50, 75]
    print(f"  {'channel':14s} " + "".join(f"{f'{p}%':>9s}" for p in pcts))
    for i, c in enumerate(CHANNELS):
        cells = []
        for p in pcts:
            need = p / 100.0 * caps[i]
            k = np.searchsorted(np.maximum(ru[1:, i], rd[1:, i]), need) + 1
            cells.append(f"{k * 5:6d}min" if k <= len(ru) - 1 else "   never")
        print(f"  {c:14s} " + "".join(f"{x:>9s}" for x in cells))

    print("\n" + "=" * 100)
    print("3. CAPACITY C(t) -- monthly, MW")
    print("=" * 100)
    C = df[[CAP_OF[c] for c in CHANNELS]]
    print(f"  {'channel':14s} {'first':>10s} {'last':>10s} {'growth':>8s} "
          f"{'truth over C(t)':>16s}")
    X = df[CHANNELS].values
    _, pmax = capacity_bounds(df)
    pmin, _ = capacity_bounds(df)
    for i, c in enumerate(CHANNELS):
        over = ((X[:, i] > pmax[:, i] + 1e-6) | (X[:, i] < pmin[:, i] - 1e-6)).mean()
        print(f"  {c:14s} {C.iloc[0, i]:10.0f} {C.iloc[-1, i]:10.0f} "
              f"{C.iloc[-1, i] / C.iloc[0, i]:7.2f}x {100 * over:15.3f}%")

    print("\n" + "=" * 100)
    print("2. RESERVOIR")
    print("=" * 100)
    print(f"  E_max {res.e_max:.0f} MWh   E_min {res.e_min:.0f} MWh (calibrated so the "
          f"recorded dispatch stays feasible)")
    print(f"  eta {res.eta:.3f} one-way   draw {res.draw * 288:.0f} MWh/day")
    print(f"  contrib is strictly decreasing: contrib(-1000)={res.contrib(-1000.0):+.2f}  "
          f"contrib(0)={res.contrib(0.0):+.2f}  contrib(+1000)={res.contrib(1000.0):+.2f} MWh")
    v = np.array([-40.0, -5.0, 0.0, 5.0, 40.0])
    print(f"  contrib_inv round-trips to {np.abs(res.contrib(res.contrib_inv(v)) - v).max():.2e} MWh")

    print("\n" + "=" * 100)
    print("1. BALANCE -- net demand from observables")
    print("=" * 100)
    tr, te = cs["train_mask"], ~cs["train_mask"]
    y = nd_truth(df)
    for name, pred in (("fitted map", nd_observable(df, cs["nd_coef"])),
                       ("naive identity", (df["demand"] - df["wind"]
                                           - df["solar_utility"] - df["net_import"]).values)):
        r = y[te] - pred[te]
        print(f"  {name:16s} out-of-sample  MAE {np.abs(r).mean():7.1f}  std {r.std():7.1f}  "
              f"max |r| {np.abs(r).max():8.1f} MW")
    print(f"  nd level: mean {y.mean():.0f}  std {y.std():.0f} MW   "
          f"(train <= 2026-02-28, {100 * tr.mean():.0f}% of rows)")
    print("  coef " + ", ".join(f"{k} {v:+.3f}" for k, v in
                                zip(ND_FEATURES + ["const"], cs["nd_coef"])))

    print("\n" + "=" * 100)
    print("CHECKS")
    print("=" * 100)
    X, seg = df[CHANNELS].values, cs["seg"]
    over = 0
    for k in (1, 3, 12, 36, 144, 288):
        d = X[k:] - X[:-k]
        d = d[seg[k:] == seg[:-k]]
        over += int(((d > ru[k] + 1e-6) | (d < -rd[k] - 1e-6)).sum())
    print(f"  recorded dispatch outside R(k)          {over} cells (must be 0)")
    print(f"  R_up subadditivity failures             {check_subadditive(ru)} (must be 0)")
    print(f"  R_dn subadditivity failures             {check_subadditive(rd)} (must be 0)")
    mono_u = int((np.diff(ru, axis=0) < -1e-9).sum())
    mono_d = int((np.diff(rd, axis=0) < -1e-9).sum())
    print(f"  R monotone in k                         {mono_u + mono_d} failures (must be 0)")


if __name__ == "__main__":
    main()
