"""Monitor pilot: do cheap runtime signals predict, per scan, how much truly lethal terrain the grid reports free?

Reads the per-scan `err` and `monitor` fields that run_replay logs, from fault-injected replays of the 00001
test block (scripts/slurm/robustness.sbatch with PREFIX=pilot). For each signal, and for a small logistic
combination scored leave-one-fault-family-out:
  - AUROC for flagging unsafe scans (safety error above the clean run's 95th percentile)
  - the same AUROC within each run (does it pick the bad scans at a fixed fault level?)
  - risk-coverage: aggregate safety error among the scans a monitor lets through, by the share it lets through
  - availability kept: the largest share of scans it can let through with the aggregate error at or below the
    clean run's level

  python -m pipeline.pilot_monitors --out docs/measurements/monitor_pilot_00001.json
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np

from pipeline.rellis import DATA_ROOT

RUNS = {"baseline": "none", "drop25": "returns lost", "drop50": "returns lost", "drop75": "returns lost",
        "drop90": "returns lost", "beams32": "fewer rings", "noise10": "range noise", "noise30": "range noise",
        "sector60": "front sector blocked", "sector120": "front sector blocked", "camyaw4": "camera",
        "camoff200": "camera"}
# signal -> sign so that a higher score means more suspicious; NaN camera disagreement = no lidar in the camera view
SIGNALS = {"density": -1, "near_coverage": -1, "lidar_entropy": 1, "cross_disagree": 1, "memory_flip": 1}
COVERAGE = np.round(np.arange(0.50, 1.0001, 0.05), 2)


def load(name):
    rows = []
    for line in open(DATA_ROOT / f"runs/replay_00001_pilot_{name}/frames.jsonl"):
        f = json.loads(line)
        e = f["err"]
        if e["gt_lethal"] == 0:
            continue
        m = f["monitor"]
        rows.append({"run": name, "family": RUNS[name], "scan": f["scan"], "missed": e["missed"],
                     "gt": e["gt_lethal"], **{k: (1.0 if k == "cross_disagree" and m[k] != m[k] else m[k]) for k in SIGNALS}})
    return rows


def auroc(score, label):
    """P(score of a random unsafe scan > score of a random safe one), ties counted half."""
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    s = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(s):  # average ranks over ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def risk_coverage(score, missed, gt):
    """Aggregate safety error among the lowest-scoring (accepted) scans, for each accepted share."""
    order = np.argsort(score, kind="mergesort")
    cm, cg = np.cumsum(missed[order]), np.cumsum(gt[order])
    out = []
    for c in COVERAGE:
        k = max(1, int(round(c * len(order))))
        out.append(float(cm[k - 1] / cg[k - 1]))
    return out, cm / cg


def kept_at(curve_full, target):
    ok = np.flatnonzero(curve_full <= target)
    return float((ok.max() + 1) / len(curve_full)) if len(ok) else 0.0


def fit_logistic(X, y, l2=1e-2, iters=50):
    w = np.zeros(X.shape[1] + 1)
    A = np.hstack([X, np.ones((len(X), 1))])
    for _ in range(iters):  # Newton / IRLS
        p = 1 / (1 + np.exp(-A @ w))
        g = A.T @ (p - y) + l2 * np.r_[w[:-1], 0]
        H = (A * (p * (1 - p))[:, None]).T @ A + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, g)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = [r for n in RUNS if (DATA_ROOT / f"runs/replay_00001_pilot_{n}/frames.jsonl").exists() for r in load(n)]
    missing = [n for n in RUNS if not (DATA_ROOT / f"runs/replay_00001_pilot_{n}/frames.jsonl").exists()]
    run = np.array([r["run"] for r in rows])
    fam = np.array([r["family"] for r in rows])
    missed = np.array([r["missed"] for r in rows], float)
    gt = np.array([r["gt"] for r in rows], float)
    err = missed / gt
    base = run == "baseline"
    tau = float(np.percentile(err[base], 95))
    unsafe = err > tau
    clean_rate = float(missed[base].sum() / gt[base].sum())
    S = {k: SIGNALS[k] * np.array([r[k] for r in rows], float) for k in SIGNALS}

    # Learned combination, leave one fault family out; the clean run is split in time between train and test.
    scans = np.array([r["scan"] for r in rows])
    half = base & (scans >= np.median(scans[base]))
    X = np.stack([S[k] for k in SIGNALS], 1)
    learned = np.full(len(rows), np.nan)
    for f in sorted(set(fam) - {"none"}):
        test = (fam == f) | half
        train = ~test & ~(base & half)
        mu, sd = X[train].mean(0), X[train].std(0) + 1e-9
        w = fit_logistic((X[train] - mu) / sd, unsafe[train].astype(float))
        z = np.hstack([(X[test] - mu) / sd, np.ones((test.sum(), 1))]) @ w
        learned[test & ~base] = z[~base[test]]
        learned[test & base] = z[base[test]]  # clean half scored by every fold; last one kept (all similar)
    S["learned (leave one family out)"] = learned

    rng = np.random.default_rng(0)
    evaluated = ~base | half  # every monitor is scored on the same scans; the first clean half only trains
    out_signals = {}
    for k, s in list(S.items()) + [("oracle", err), ("random", rng.random(len(err)))]:
        ok = evaluated & ~np.isnan(s)
        within = [auroc(s[ok & (run == n)], unsafe[ok & (run == n)]) for n in RUNS if n != "baseline"]
        rc, full = risk_coverage(s[ok], missed[ok], gt[ok])
        out_signals[k] = {"auroc_pooled": round(auroc(s[ok], unsafe[ok]), 3),
                          "auroc_within_run_mean": round(float(np.nanmean(within)), 3),
                          "risk_at_coverage": [round(v, 4) for v in rc],
                          "coverage_at_clean_risk": round(kept_at(full, clean_rate), 3)}
    # The deployed health check: lidar DEGRADED when density < 0.2 (health_monitor_node.cpp, gen_ros_params.py).
    dens = np.array([r["density"] for r in rows])
    floor = evaluated & (dens >= 0.2)
    deployed = {"rule": "density >= 0.2", "coverage": round(float(floor.sum() / evaluated.sum()), 3),
                "risk": round(float(missed[floor].sum() / gt[floor].sum()), 4)}
    per_run = {n: {"aggregate_error": round(float(missed[run == n].sum() / gt[run == n].sum()), 4),
                   "density_median": round(float(np.median(dens[run == n])), 3),
                   "entropy_median": round(float(np.median([r["lidar_entropy"] for r in rows if r["run"] == n])), 4)}
               for n in RUNS if (run == n).any()}
    out = {"date": datetime.date.today().isoformat(), "seq": "00001", "scans": "0:1047 (test block) per run",
           "runs": {n: RUNS[n] for n in RUNS if n not in missing}, "missing_runs": missing, "n_scans": len(rows),
           "n_scans_evaluated": int(evaluated.sum()),
           "safety_error": "per scan: truly lethal cells reported free / truly lethal cells (ground truth from the clean sweep)",
           "unsafe_threshold": round(tau, 4), "unsafe_share": round(float(unsafe.mean()), 3),
           "clean_aggregate_error": round(clean_rate, 4), "coverage_grid": COVERAGE.tolist(),
           "deployed_density_floor": deployed, "per_run": per_run, "signals": out_signals}
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print("deployed density floor:", deployed)
    print(f"{len(rows)} scans from {len(RUNS) - len(missing)} runs (missing {missing}); unsafe > {tau:.4f} "
          f"({unsafe.mean():.1%}); clean aggregate {clean_rate:.4f}")
    for k, v in out_signals.items():
        print(f"{k:32s} AUROC {v['auroc_pooled']:.3f}  within-run {v['auroc_within_run_mean']:.3f}  "
              f"kept at clean risk {v['coverage_at_clean_risk']:.1%}  risk@80% {v['risk_at_coverage'][6]:.4f}")


if __name__ == "__main__":
    main()
