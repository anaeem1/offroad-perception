"""Conformal gate on GOOSE held-out recordings (second benchmark): does a monitor calibrated in clear weather keep its
risk bound in real rain and snow, and on injected fault families it was not calibrated on?

Runs come from scripts/slurm/goose_faults.sbatch (pipeline/goose_monitor_eval.py). Loss and gate as in
pipeline/conformal_gate.py: per-scan share of truly lethal cells reported free, 0 for a rejected scan; alpha = the
mean clean loss of the calibration recordings (clear recordings for test 1, all held-out recordings otherwise).
  0 per run: mean per-scan loss; AUROC of each signal for unsafe scans (loss above the clear clean 95th pct)
  1 weather: calibrate on clear recordings (clean + every fault run), test on the weather recordings, clean and
    with faults, pooled and per recording
  2 unseen fault family: calibrate on every recording without one family, test on that family
  3 leave one recording out: calibrate on the other six recordings, test on it (clean + every fault run)
  4 rain vs overcast: same area (neubiberg), clean, mean loss and monitor signals

  python -m pipeline.goose_gate --out docs/measurements/goose_gate.json
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np

from pipeline.conformal_gate import evaluate, threshold
from pipeline.pilot_monitors import auroc, fit_logistic
from pipeline.rellis import DATA_ROOT

RUNS = {"clean": "none", "drop25": "returns lost", "drop50": "returns lost", "drop75": "returns lost",
        "rings64": "fewer rings", "rings32": "fewer rings", "noise10": "range noise", "noise30": "range noise",
        "sector60": "sector blocked", "sector120": "sector blocked"}
SIGNALS = {"density": -1, "near_coverage": -1, "lidar_entropy": 1}  # higher score = more suspicious
WEATHER = {"2023-05-15_neubiberg_rain", "2022-11-29_campus_rain_2", "2022-11-04_campus_rain", "2023-01-20_campus_snow"}


def load():
    rows = []
    for name, fam in RUNS.items():
        path = DATA_ROOT / f"runs/goose_heldout_{name}/frames.jsonl"
        if not path.exists():
            continue
        for line in open(path):
            f = json.loads(line)
            if f["err"]["gt_lethal"] == 0:
                continue
            rows.append({"run": name, "family": fam, "seq": f["sequence"],
                         "err": f["err"]["missed"] / f["err"]["gt_lethal"], **f["monitor"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = load()
    run = np.array([r["run"] for r in rows])
    fam = np.array([r["family"] for r in rows])
    seq = np.array([r["seq"] for r in rows])
    err = np.array([r["err"] for r in rows])
    X = np.stack([SIGNALS[k] * np.array([r[k] for r in rows], float) for k in SIGNALS], 1)
    weather = np.isin(seq, list(WEATHER))
    clean = run == "clean"
    alpha = float(err[clean & ~weather].mean())
    alpha_all = float(err[clean].mean())
    unsafe = err > np.percentile(err[clean & ~weather], 95)

    def scores(train):
        mu, sd = X[train].mean(0), X[train].std(0) + 1e-9
        w = fit_logistic((X[train] - mu) / sd, unsafe[train].astype(float))
        out = {k: X[:, i] for i, k in enumerate(SIGNALS)}
        out["learned (calibration scans only)"] = np.hstack([(X - mu) / sd, np.ones((len(X), 1))]) @ w
        out["oracle"] = err
        out["accept all"] = np.zeros(len(err))
        return out

    def gate(cal, test, a):
        res = {}
        for k, s in scores(cal).items():
            t = threshold(s[cal], err[cal], a) if k != "accept all" else np.inf
            risk, cov = evaluate(s[test], err[test], t)
            res[k] = {"risk": round(risk, 5), "coverage": round(cov, 4), "above_alpha": bool(risk > a)}
        return res

    out = {"date": datetime.date.today().isoformat(), "alpha": round(alpha, 5),
           "alpha_definition": "mean clean per-scan loss on the clear-weather held-out recordings (test 1)",
           "alpha_all": round(alpha_all, 5),
           "alpha_all_definition": "mean clean per-scan loss on all held-out recordings (tests 2 and 3)",
           "runs": sorted(set(run)), "scans": len(rows)}
    # 0: per run, and how well each signal ranks unsafe scans
    out["per_run_mean_loss"] = {r: {"clear": round(float(err[(run == r) & ~weather].mean()), 5),
                                    "weather": round(float(err[(run == r) & weather].mean()), 5),
                                    "all": round(float(err[run == r].mean()), 5)} for r in RUNS if r in set(run)}
    out["auroc_unsafe"] = {k: {"clean only": round(auroc(X[clean, i], unsafe[clean]), 3),
                               **{f: round(auroc(X[(fam == f) | clean, i], unsafe[(fam == f) | clean]), 3)
                                  for f in sorted(set(fam) - {"none"})}} for i, k in enumerate(SIGNALS)}
    # 1: clear -> real weather
    cal = ~weather
    out["weather"] = {"clean weather scans": gate(cal, weather & clean, alpha),
                      "weather scans with injected faults": gate(cal, weather & ~clean, alpha),
                      "per recording": {s: {"clean": gate(cal, (seq == s) & clean, alpha),
                                            "with faults": gate(cal, (seq == s) & ~clean, alpha)}
                                        for s in sorted(WEATHER)}}
    # 2: unseen fault family (all recordings)
    out["unseen_fault_family"] = {f: gate(fam != f, fam == f, alpha_all) for f in sorted(set(fam) - {"none"})}
    # 3: unseen recording
    out["leave_one_recording_out"] = {s: gate(seq != s, seq == s, alpha_all) for s in sorted(set(seq))}
    # 4: rain vs overcast, same area
    pair = {}
    for s in ("2023-05-15_neubiberg_rain", "2023-05-24_neubiberg_cloudy"):
        m = clean & (seq == s)
        pair[s] = {"scans": int(m.sum()), "mean_loss": round(float(err[m].mean()), 5),
                   **{k: round(float(np.median([r[k] for r, mm in zip(rows, m) if mm])), 4) for k in SIGNALS}}
    out["rain_vs_overcast"] = pair
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
