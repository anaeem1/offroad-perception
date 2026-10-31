"""Conformal risk control for a per-scan gate: accept a scan (let the planner act on it) when its monitor score is
at most a threshold calibrated so that the expected per-scan safety loss stays under a target.

Per-scan loss L_i(t) = e_i * 1[score_i <= t], with e_i the share of truly lethal cells reported free. A rejected
scan costs nothing in safety (the machine does not act on it) but costs availability. L is bounded by 1 and
non-decreasing in t, so conformal risk control (Angelopoulos et al., ICLR 2024) gives
    E[L_test(t_hat)] <= alpha,  with  t_hat = largest t such that (sum_i L_i(t) + 1) / (n + 1) <= alpha
when calibration and test scans are exchangeable. Scans are split in 5 s blocks (50 scans) because neighbouring
scans are strongly correlated.

Two tests on the pilot replays (pipeline/pilot_monitors.py):
  A  exchangeable: blocks from all runs split at random into calibration and test, repeated
  B  unseen fault family: calibrate on the clean run's first half + every other family, test on the held-out
     family + the clean run's second half

  python -m pipeline.conformal_gate --out docs/measurements/conformal_gate_pilot.json
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np

from pipeline.pilot_monitors import RUNS, SIGNALS, fit_logistic, load

BLOCK = 50


def threshold(score, loss, alpha):
    """Largest score threshold whose adjusted calibration risk is <= alpha (-inf: reject everything)."""
    order = np.argsort(score, kind="mergesort")
    s, cum = score[order], np.cumsum(loss[order])
    n = len(s)
    ok = (cum + 1.0) / (n + 1) <= alpha
    ok &= np.r_[s[1:] != s[:-1], True]  # a threshold must accept every tied score
    idx = np.flatnonzero(ok)
    return float(s[idx.max()]) if len(idx) else -np.inf


def evaluate(score, err, t):
    acc = score <= t
    return float(np.mean(err * acc)), float(acc.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=200)
    args = ap.parse_args()
    rows = [r for n in RUNS for r in load(n)]
    run = np.array([r["run"] for r in rows])
    fam = np.array([r["family"] for r in rows])
    scan = np.array([r["scan"] for r in rows])
    err = np.array([r["missed"] / r["gt"] for r in rows])
    X = np.stack([SIGNALS[k] * np.array([r[k] for r in rows], float) for k in SIGNALS], 1)
    block = np.array([f"{r}:{s // BLOCK}" for r, s in zip(run, scan)])
    clean = run == "baseline"
    alpha = float(err[clean].mean())  # target: the fault-free mean per-scan error
    unsafe = err > np.percentile(err[clean], 95)

    def scores(train):
        """Monitor scores (higher = more suspicious); the learned one is fit on the calibration scans only."""
        mu, sd = X[train].mean(0), X[train].std(0) + 1e-9
        w = fit_logistic((X[train] - mu) / sd, unsafe[train].astype(float))
        out = {"point density": X[:, 0], "lidar entropy": X[:, 2],
               "learned (calibration scans only)": np.hstack([(X - mu) / sd, np.ones((len(X), 1))]) @ w,
               "oracle": err, "accept all": np.zeros(len(err))}
        return out

    # A: exchangeable block splits
    rng = np.random.default_rng(0)
    blocks = np.unique(block)
    res_a = {}
    for _ in range(args.trials):
        cal_blocks = set(rng.choice(blocks, len(blocks) // 2, replace=False))
        cal = np.isin(block, list(cal_blocks))
        for k, s in scores(cal).items():
            t = threshold(s[cal], err[cal], alpha) if k != "accept all" else np.inf
            risk, cov = evaluate(s[~cal], err[~cal], t)
            res_a.setdefault(k, []).append((risk, cov))
    out_a = {k: {"mean_test_risk": round(float(np.mean([r for r, _ in v])), 5),
                 "share_of_trials_above_alpha": round(float(np.mean([r > alpha for r, _ in v])), 3),
                 "mean_coverage": round(float(np.mean([c for _, c in v])), 4)} for k, v in res_a.items()}

    # B: one fault family unseen in calibration
    first_half = clean & (scan < np.median(scan[clean]))
    out_b = {}
    for f in sorted(set(fam) - {"none"}):
        cal = (fam != f) & ~clean | first_half
        test = (fam == f) | (clean & ~first_half)
        fam_only = fam == f
        out_b[f] = {}
        for k, s in scores(cal).items():
            t = threshold(s[cal], err[cal], alpha) if k != "accept all" else np.inf
            risk, cov = evaluate(s[test], err[test], t)
            risk_f, cov_f = evaluate(s[fam_only], err[fam_only], t)
            out_b[f][k] = {"test_risk": round(risk, 5), "coverage": round(cov, 4), "above_alpha": bool(risk > alpha),
                           "risk_on_faulted_scans": round(risk_f, 5), "coverage_on_faulted_scans": round(cov_f, 4)}

    out = {"date": datetime.date.today().isoformat(), "data": "pilot replays of the 00001 test block (12 runs)",
           "loss": "per-scan share of truly lethal cells reported free; 0 for a rejected scan",
           "alpha": round(alpha, 5), "alpha_definition": "mean per-scan loss of the fault-free run",
           "block_scans": BLOCK, "trials": args.trials,
           "A_exchangeable": out_a, "B_unseen_fault_family": out_b}
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"alpha = {alpha:.5f}")
    print("A (exchangeable):")
    for k, v in out_a.items():
        print(f"  {k:34s} risk {v['mean_test_risk']:.5f}  above alpha {v['share_of_trials_above_alpha']:.0%}  "
              f"coverage {v['mean_coverage']:.1%}")
    print("B (unseen family): risk on held-out family scans / coverage there")
    for f, d in out_b.items():
        print(f"  {f:22s} " + "  ".join(f"{k.split(' ')[0]} {v['risk_on_faulted_scans']:.4f}/{v['coverage_on_faulted_scans']:.0%}"
                                        for k, v in d.items()))


if __name__ == "__main__":
    main()
