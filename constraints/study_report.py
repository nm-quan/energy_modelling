"""Bottleneck-study gates, summary and trade-off figure.

Reads whatever result JSONs exist (TF/CL from eval_hist_models.py, scenario
rollouts from demand_simulation/study_shift.py) and decides each problem's
A->B->C ladder (colab/study_bottlenecks.ipynb):

  P1 accuracy        rayenfd macro WAPE within +0.005 of persistence AND every
                     non-battery channel within +0.01 -> the gas_steam
                     passthrough (A) suffices; else retrain (B).
  P2 responsiveness  rayenfd response_capture in [0.8, 1.2], free-window
                     tracking p50 <= 100 MW, 0 free-window ramp violations,
                     0 negatives -> pinning D to nd(t-1) (A) works; else
                     retrain with passthrough (B); anchor(persistence) is the
                     always-available C fallback, reported alongside.
  P3 battery         per-day SOC feasible on >= 95% of days and worst day
                     <= 100% of nameplate (TF + scenario rollout) -> SOC stays
                     a reported diagnostic (A); else a stateful closed-loop
                     clip (B) is actually needed.

Outputs: constraints/results/study_summary.md, study_verdicts.json (read by the
notebook to run conditional cells), figure/study_tradeoff.png.

    python3 constraints/study_report.py
    python3 constraints/study_report.py --scenario increase --g 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
STUDY = ROOT / "demand_simulation" / "sweep_eqnd" / "study"
TARGETS = ["hydro", "coal_brown", "gas_steam", "gas_ocgt",
           "battery_charging", "battery_discharging"]
NON_BATT = TARGETS[:4]

P1_MACRO_TOL = 0.005
P1_CHANNEL_TOL = 0.010
P2_CAPTURE = (0.8, 1.2)
P2_TRACK_P50 = 100.0
P3_DAY_PCT = 95.0


def _load(pattern: str, base: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(base.glob(pattern))]


def _by_model(rows):
    return {r["model"]: r for r in rows}


def pick_candidate(models: dict, sub: str = "rayenfd"):
    """Prefer the most-fixed rayenfd variant: SOC shield > weighted alloc >
    baseline; itransformer > lstm; +spt > bare (suffixes from study_shift)."""
    hits = [(m, r) for m, r in models.items() if sub in m]
    if not hits:
        return None
    hits.sort(key=lambda mr: ("[soc]" in mr[0], "[" in mr[0],
                              mr[0].startswith("itransformer"), "+spt" in mr[0], mr[0]),
              reverse=True)
    return hits[0][1]


def gate_p1(tf: dict) -> dict:
    pers = tf.get("persistence")
    cand = pick_candidate(tf)
    if not (pers and cand and cand.get("per_target_WAPE") and pers.get("per_target_WAPE")):
        return {"pass": None, "why": "need TF rows with per_target_WAPE "
                                     "(rerun constraints/eval_hist_models.py)"}
    macro_gap = cand["WAPE"] - pers["WAPE"]
    ch_gaps = {t: cand["per_target_WAPE"][t] - pers["per_target_WAPE"][t] for t in NON_BATT}
    batt_gap = float(np.mean([cand["per_target_WAPE"][t] - pers["per_target_WAPE"][t]
                              for t in TARGETS[4:]]))
    ok = macro_gap <= P1_MACRO_TOL and all(v <= P1_CHANNEL_TOL for v in ch_gaps.values())
    return {"pass": bool(ok), "candidate": cand["model"],
            "macro_WAPE": cand["WAPE"], "persistence_WAPE": pers["WAPE"],
            "macro_gap": macro_gap, "non_battery_gaps": ch_gaps,
            "battery_mean_gap": batt_gap,
            "why": f"macro gap {macro_gap:+.4f} (tol +{P1_MACRO_TOL}), worst non-battery "
                   f"channel gap {max(ch_gaps.values()):+.4f} (tol +{P1_CHANNEL_TOL}); "
                   f"battery gap {batt_gap:+.4f} is the accepted irreducible part"}


def gate_p2(study: dict) -> dict:
    cand = pick_candidate(study)
    if cand is None:
        return {"pass": None, "why": "no rayenfd scenario row "
                                     "(run demand_simulation/study_shift.py)"}
    cap = cand.get("response_capture")
    ok = (cap is not None and P2_CAPTURE[0] <= cap <= P2_CAPTURE[1]
          and cand["track_free_p50_mw"] <= P2_TRACK_P50
          and cand["n_ramp_free"] == 0 and cand["n_neg"] == 0)
    contrast = {m: r.get("response_capture") for m, r in study.items()
                if "rayen" in m and "rayenfd" not in m}
    fallback = study.get("anchor_persistence", {}).get("response_capture")
    return {"pass": bool(ok), "candidate": cand["model"], "capture": cap,
            "track_free_p50_mw": cand["track_free_p50_mw"],
            "n_ramp_free": cand["n_ramp_free"], "n_ramp_seam": cand["n_ramp_seam"],
            "n_neg": cand["n_neg"], "rayen_capture_contrast": contrast,
            "anchor_fallback_capture": fallback,
            "why": f"capture {cap if cap is None else round(cap, 3)} "
                   f"(need {P2_CAPTURE[0]}-{P2_CAPTURE[1]}), track p50 "
                   f"{cand['track_free_p50_mw']:.0f} MW (<= {P2_TRACK_P50:.0f}), "
                   f"free-window ramps {cand['n_ramp_free']}, neg {cand['n_neg']}"}


def gate_p3(tf: dict, cl: dict, study: dict) -> dict:
    checks, worst = [], []
    for src, rows in (("tf", tf), ("scenario", study)):
        cand = pick_candidate(rows)
        if cand and "soc_day_feasible_pct" in cand:
            checks.append((src, cand["soc_day_feasible_pct"], cand["soc_worst_day_pct"]))
            worst.append(cand["soc_worst_day_pct"])
    cand_cl = pick_candidate(cl)
    if cand_cl and "soc_feasible_eps" in cand_cl:
        # stress episodes are ~24h windows, so per-episode == per-day physics
        pct = 100.0 * cand_cl["soc_feasible_eps"] / cand_cl["n_episodes"]
        checks.append(("cl_stress", pct, cand_cl["soc_worst_ep_pct"]))
        worst.append(cand_cl["soc_worst_ep_pct"])
    if not checks:
        return {"pass": None, "why": "no rayenfd rows with per-day SOC yet"}
    ok = all(day >= P3_DAY_PCT and w <= 100.0 for _, day, w in checks)
    legacy = pick_candidate(study)
    return {"pass": bool(ok),
            "checks": [{"source": s, "day_feasible_pct": d, "worst_day_pct": w}
                       for s, d, w in checks],
            "legacy_window_swing_pct": (legacy or {}).get("soc_window_swing_pct"),
            "why": f"per-day feasible {[round(d, 1) for _, d, _ in checks]}% "
                   f"(need >= {P3_DAY_PCT}), worst {max(worst):.1f}% of nameplate; "
                   "whole-window swing is reported for transparency only (drift artifact)"}


def tradeoff_figure(tf: dict, study: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = []
    for m, r in study.items():
        cap = r.get("response_capture")
        if cap is None:
            continue
        viol = r["n_neg"] + r["n_ramp_free"] + r["n_ramp_tf"]  # seam excluded: reconnection artifact
        pts.append((m, cap, r["base_WAPE"], viol, tf.get(m, {}).get("WAPE")))
    if not pts:
        return False
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for m, cap, wape, viol, tfw in pts:
        clean = viol == 0
        ax.scatter(cap, wape, s=90, marker="o" if clean else "X",
                   color="#2a9d8f" if clean else "#e76f51", zorder=3)
        label = m + (f"\nTF WAPE {tfw:.3f}" if tfw else "")
        ax.annotate(label, (cap, wape), textcoords="offset points",
                    xytext=(8, 4), fontsize=8)
    ax.axvline(1.0, ls="--", lw=0.8, color="grey")
    ax.set_xlabel("response capture (fraction of simulated load increase delivered)")
    ax.set_ylabel("closed-loop WAPE, response region (no-shift rollout)")
    ax.set_title("Feasibility–accuracy–responsiveness trade-off\n"
                 "(o = constraint-clean, X = has violations; goal: bottom-right, on the line)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None, help="which study scenario to gate on "
                    "(default: increase with the largest g present, else reshape)")
    ap.add_argument("--g", type=float, default=None)
    args = ap.parse_args()

    tf = _by_model(_load("hist_tf_*.json", RESULTS))
    cl = _by_model(_load("hist_cl_*.json", RESULTS))
    tags = sorted({p.name.split("_", 2)[:2] and "_".join(p.name.split("_")[:2])
                   for p in STUDY.glob("*_g*_*.json")}) if STUDY.exists() else []
    if args.scenario and args.g is not None:
        tag = f"{args.scenario}_g{args.g:g}"
    else:
        inc = sorted([t for t in tags if t.startswith("increase")],
                     key=lambda t: float(t.split("_g")[1]))
        tag = inc[-1] if inc else (tags[-1] if tags else None)
    study = _by_model(_load(f"{tag}_*.json", STUDY)) if tag else {}

    p1, p2, p3 = gate_p1(tf), gate_p2(study), gate_p3(tf, cl, study)
    verdicts = {
        "scenario_tag": tag,
        "p1_accuracy": p1, "p2_responsiveness": p2, "p3_battery": p3,
        "run_retrain": bool((p1["pass"] is False) or (p2["pass"] is False)),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "study_verdicts.json").write_text(json.dumps(verdicts, indent=2))

    def line(name, g, a_txt, b_txt):
        s = {None: "…  (inputs missing)", True: "PASS -> A suffices",
             False: "FAIL -> escalate to B"}[g["pass"]]
        return f"| {name} | {a_txt} | {b_txt} | **{s}** | {g['why']} |"

    md = ["# Bottleneck study — ladder verdicts\n",
          f"Scenario gated on: `{tag}`. Thresholds: P1 macro +{P1_MACRO_TOL} / channel "
          f"+{P1_CHANNEL_TOL} vs persistence; P2 capture {P2_CAPTURE[0]}-{P2_CAPTURE[1]}, "
          f"track p50 <= {P2_TRACK_P50:.0f} MW, 0 free ramps/negs; P3 >= {P3_DAY_PCT}% "
          "days feasible, worst day <= 100%.\n",
          "| problem | A | B (if A fails) | verdict | detail |",
          "| --- | --- | --- | --- | --- |",
          line("P1 WAPE vs persistence", p1, "gas_steam passthrough",
               "retrain rayenfd (passthrough on)"),
          line("P2 demand response", p2, "pin D=nd(t-1) + scenario nd in free window",
               "retrain rayenfd; C = inference anchor"),
          line("P3 battery constraint", p3, "per-day SOC as reported diagnostic",
               "stateful closed-loop SOC clip"),
          ""]
    if study:
        md += ["## Scenario rows (gated tag)\n",
               "| model | capture | track p50 MW | n_ramp free/seam/tf | n_neg | soc day % |",
               "| --- | --- | --- | --- | --- | --- |"]
        for m, r in study.items():
            cap = r.get("response_capture")
            md.append(f"| {m} | {'—' if cap is None else f'{cap:+.3f}'} | "
                      f"{r['track_free_p50_mw']:.0f} | {r['n_ramp_free']}/{r['n_ramp_seam']}"
                      f"/{r['n_ramp_tf']} | {r['n_neg']} | {r['soc_day_feasible_pct']:.0f} |")
        md.append("")
    curve = []
    for t in tags:
        if t.startswith("increase"):
            for r in _load(f"{t}_*.json", STUDY):
                if "rayenfd" in r["model"]:
                    curve.append((float(t.split("_g")[1]), r))
    if curve:
        md += ["## Response vs g (increase scenario, rayenfd)\n",
               "| g % | capture | track p50 MW | track p95 MW | n_ramp | soc worst day % |",
               "| --- | --- | --- | --- | --- | --- |"]
        for g, r in sorted(curve):
            md.append(f"| {g:g} | {r['response_capture']:+.3f} | "
                      f"{r['track_free_p50_mw']:.0f} | {r['track_free_p95_mw']:.0f} | "
                      f"{r['n_ramp']} | {r['soc_worst_day_pct']:.1f} |")
        md.append("")
    if tradeoff_figure(tf, study, RESULTS / "figure" / "study_tradeoff.png"):
        md.append("![trade-off](figure/study_tradeoff.png)\n")
    (RESULTS / "study_summary.md").write_text("\n".join(md))
    print("\n".join(md[:12]))
    print("wrote", RESULTS / "study_summary.md", "and study_verdicts.json")


if __name__ == "__main__":
    main()
