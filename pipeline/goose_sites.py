"""Cross-fitted GOOSE study over all 23 recordings: each scan's loss and monitor signals come from the model
that never saw its recording (folds a, b, c of config/goose_split.yaml, one SalsaNext per fold; runs from
scripts/slurm/goose_faults.sbatch with SPLIT=heldout:<fold>). Loss and gate as in pipeline/conformal_gate.py:
per-scan share of truly lethal cells reported free, 0 for a rejected scan; a scan's rows under every run are one
exchangeable unit (threshold units=). alpha is a required level, swept (--alphas), not the clean mean: on GOOSE
the faults lower the mean loss (fewer rings and blocked sectors turn lethal cells UNKNOWN), so with the clean
mean the fault mix already meets alpha and every gate accepts everything (first run, 2026-11-02).
"learned" is a logistic combination of the signals fitted on the calibration scans only.
  0 per recording: fold, scans, clean loss; per model: best val mIoU
  1 AUROC of each signal for unsafe scans (loss above the pooled clean 95th percentile)
  2 exchangeable: a random half of the scans calibrates, the other half tests
  3 leave one recording out: calibrate on the other recordings, test on it
  4 a new site with k labelled scans (its first k, or k at random): calibrate on those alone (site-only) or
    add them to the other recordings (pooled + site); test on the rest of the recording. Score functions
    (and the learned weights) come from the other recordings, only the threshold uses the site scans.
  5 fold a only (five-member ensemble): deep-ensemble mutual information next to the other signals

  python -m pipeline.goose_sites --out docs/measurements/goose_sites.json
"""
import argparse
import datetime
import json

import numpy as np

from pipeline.conformal_gate import evaluate, threshold
from pipeline.goose_gate import RUNS
from pipeline.pilot_monitors import auroc, fit_logistic
from pipeline.rellis import DATA_ROOT

FOLDS = ("a", "b", "c")
# higher score = more suspicious; a signal is used only if every loaded scan has it
SIGNALS = {"density": -1, "near_coverage": -1, "lidar_entropy": 1, "entropy_grid": 1, "mc_mi": 1}
ENSEMBLE = "ens_mi"
K_SITE = (25, 50, 100, 200)


def load(folds, pattern):
    rows = []
    for fold in folds:
        for name, fam in RUNS.items():
            for line in open(DATA_ROOT / "runs" / pattern.format(fold=fold, name=name) / "frames.jsonl"):
                f = json.loads(line)
                if f["err"]["gt_lethal"] == 0:
                    continue
                rows.append({"fold": fold, "run": name, "family": fam, "seq": f["sequence"], "name": f["name"],
                             "err": f["err"]["missed"] / f["err"]["gt_lethal"], **f["monitor"]})
    return rows


def best_miou(run_dir):
    p = run_dir / "train_log.jsonl"
    if not p.exists():
        return None
    return max((json.loads(line).get("val_miou_every4", -1) for line in open(p)), default=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", default="abc")
    ap.add_argument("--pattern", default="goose_heldout_{fold}_{name}", help="run directory under <data root>/runs")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--alphas", default="0.05,0.075", help="required levels of the per-scan loss")
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = load(tuple(args.folds), args.pattern)
    col = lambda k: np.array([r.get(k, np.nan) for r in rows], float)
    run, seq, name, fold = (np.array([r[k] for r in rows]) for k in ("run", "seq", "name", "fold"))
    err = col("err")
    keys = [k for k in SIGNALS if not np.isnan(col(k)).any()]
    sig = {k: SIGNALS[k] * col(k) for k in keys}
    ens = col(ENSEMBLE)
    if not np.isnan(ens[fold == "a"]).any():
        sig[ENSEMBLE] = ens
    clean = run == "clean"
    alphas = [float(v) for v in args.alphas.split(",")]
    unsafe = err > np.percentile(err[clean], 95)
    recs = sorted(set(seq))
    scan_id = np.char.add(np.char.add(seq.astype(str), "/"), name.astype(str))
    scan_ix = np.unique(scan_id, return_inverse=True)[1]  # a scan's rows (one per run) are one exchangeable unit

    def score_set(cal, ks):
        Z = np.stack([sig[k] for k in ks], 1)
        mu, sd = Z[cal].mean(0), Z[cal].std(0) + 1e-9
        w = fit_logistic((Z[cal] - mu) / sd, unsafe[cal].astype(float))
        out = {k: sig[k] for k in ks}
        out["learned"] = np.hstack([(Z - mu) / sd, np.ones((len(Z), 1))]) @ w
        out["oracle"] = err
        out["accept all"] = np.zeros(len(err))
        return out

    def gate(S, cal, test, a):
        res = {}
        for k, s in S.items():
            t = np.inf if k == "accept all" else threshold(s[cal], err[cal], a, units=len(np.unique(scan_ix[cal])))
            risk, cov = evaluate(s[test], err[test], t)
            res[k] = {"risk": round(risk, 5), "coverage": round(cov, 4)}
        return res

    def exchangeable(universe, ks, a):
        ids = np.unique(scan_id[universe])
        acc = {}
        for _ in range(args.trials):
            half = set(rng.choice(ids, len(ids) // 2, replace=False))
            in_cal = np.isin(scan_id, list(half))
            cal, test = universe & in_cal, universe & ~in_cal
            for k, v in gate(score_set(cal, ks), cal, test, a).items():
                acc.setdefault(k, []).append((v["risk"], v["coverage"]))
        return {k: {"mean_risk": round(float(np.mean([r for r, _ in v])), 5),
                    "mean_coverage": round(float(np.mean([c for _, c in v])), 4),
                    "share_above_alpha": round(float(np.mean([r > a for r, _ in v])), 3)} for k, v in acc.items()}

    def loro(universe, ks, a):
        per = {s: gate(score_set(universe & (seq != s), ks), universe & (seq != s), universe & (seq == s), a)
               for s in sorted(set(seq[universe]))}
        summary = {}
        for k in next(iter(per.values())):
            ratio = np.array([per[s][k]["risk"] / a for s in per])
            summary[k] = {"recordings_above_alpha": int((ratio > 1).sum()), "of": len(per),
                          "median_risk_over_alpha": round(float(np.median(ratio)), 3),
                          "max_risk_over_alpha": round(float(ratio.max()), 3),
                          "mean_coverage": round(float(np.mean([per[s][k]["coverage"] for s in per])), 4)}
        return summary, per

    out = {"date": datetime.date.today().isoformat(), "folds": args.folds, "alphas": alphas,
           "clean_mean_loss": round(float(err[clean].mean()), 5),
           "fault_mix_mean_loss": round(float(err.mean()), 5),
           "signals": list(sig), "runs": sorted(set(run)), "rows": len(rows), "recordings": len(recs)}
    # 0
    w = DATA_ROOT / "weights"
    out["models_best_val_miou"] = {d.name: best_miou(d) for d in sorted(w.glob("salsanext_goose_*_s*"))}
    out["per_recording"] = {s: {"fold": str(fold[seq == s][0]), "scans": int((clean & (seq == s)).sum()),
                                "clean_loss": round(float(err[clean & (seq == s)].mean()), 5),
                                "fault_mix_loss": round(float(err[seq == s].mean()), 5)} for s in recs}
    out["per_run_mean_loss"] = {r: round(float(err[run == r].mean()), 5) for r in RUNS if r in set(run)}
    everything = np.ones(len(err), bool)
    # 1
    out["auroc_unsafe"] = {k: {"clean": round(auroc(sig[k][clean], unsafe[clean]), 3),
                               "all runs": round(auroc(sig[k], unsafe), 3)} for k in keys}
    s_rec = {s: score_set(seq != s, keys) for s in recs}
    fa = fold == "a"
    if ENSEMBLE in sig:
        ks_a = keys + [ENSEMBLE]
        out["fold_a"] = {"clean_mean_loss": round(float(err[clean & fa].mean()), 5),
                         "fault_mix_mean_loss": round(float(err[fa].mean()), 5),
                         "auroc_unsafe": {k: {"clean": round(auroc(sig[k][clean & fa], unsafe[clean & fa]), 3),
                                              "all runs": round(auroc(sig[k][fa], unsafe[fa]), 3)} for k in ks_a},
                         "by_alpha": {}}
    out["by_alpha"] = {}
    for alpha in alphas:
        res = {}
        # 2, 3
        res["exchangeable"] = exchangeable(everything, keys, alpha)
        res["leave_one_recording_out"], res["leave_one_recording_out_per_recording"] = loro(everything, keys, alpha)
        # 4
        few = {}
        for k_site in K_SITE:
            for mode in ("first", "random"):
                acc = {}
                for s in recs:
                    ids = np.unique(scan_id[seq == s])  # sorted names = time order within a recording
                    if len(ids) < k_site + 50:
                        continue
                    picks = [ids[:k_site]] if mode == "first" else \
                        [rng.choice(ids, k_site, replace=False) for _ in range(max(1, args.trials // 5))]
                    for pick in picks:
                        site = np.isin(scan_id, pick)
                        test = (seq == s) & ~site
                        for method, cal in (("site only", site), ("pooled + site", (seq != s) | site)):
                            for k, v in gate(s_rec[s], cal, test, alpha).items():
                                acc.setdefault((method, k), {}).setdefault(s, []).append((v["risk"], v["coverage"]))
                for (method, k), by_rec in acc.items():
                    risk = np.array([np.mean([r for r, _ in v]) for v in by_rec.values()])
                    above = np.array([np.mean([r > alpha for r, _ in v]) for v in by_rec.values()])
                    few.setdefault(f"k={k_site} {mode}", {}).setdefault(method, {})[k] = {
                        "recordings": len(by_rec), "recordings_above_alpha": round(float(above.sum()), 2),
                        "median_risk_over_alpha": round(float(np.median(risk / alpha)), 3),
                        "mean_coverage": round(float(np.mean([np.mean([c for _, c in v])
                                                              for v in by_rec.values()])), 4)}
        res["labelled_site_scans"] = few
        out["by_alpha"][str(alpha)] = res
        # 5
        if ENSEMBLE in sig:
            out["fold_a"]["by_alpha"][str(alpha)] = {"exchangeable": exchangeable(fa, ks_a, alpha),
                                                     "leave_one_recording_out": loro(fa, ks_a, alpha)[0]}
    with open(args.out, "w") as f:
        f.write(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"alphas": alphas, "auroc_unsafe": out["auroc_unsafe"],
                      "leave_one_recording_out": {a: v["leave_one_recording_out"] for a, v in out["by_alpha"].items()}},
                     indent=1))

if __name__ == "__main__":
    main()
