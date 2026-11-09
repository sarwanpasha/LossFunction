"""Compact summary of the threshold-policy runs and the probe ablation.
usage: python rev_summary.py results_es results_thr  -> rev_summary.json (and printed)"""
import glob, json, os, sys
import numpy as np
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

RE = sys.argv[1] if len(sys.argv) > 1 else "results_es"
RT = sys.argv[2] if len(sys.argv) > 2 else "results_thr"
out = {}

# ---- threshold policies ----
T = {}
for f in sorted(glob.glob(os.path.join(RT, "*.json"))):
    j = json.load(open(f))
    for r in j["runs"]:
        for pol in ("default", "val", "plugin"):
            T.setdefault((r["loss"], pol), {}).setdefault(j["dataset"], []).append(r["f1_" + pol])
dsets = sorted({d for v in T.values() for d in v})
M = {k: np.array([np.mean(v[d]) for d in dsets]) for k, v in T.items() if len(v) == len(dsets)}
out["thr_n_datasets"] = len(dsets)
out["thr_mean_f1"] = {f"{l}|{p}": round(float(x.mean()), 4) for (l, p), x in M.items()}
ref = M.get(("paf", "default"))
if ref is not None:
    out["paf_default_vs"] = {f"{l}|{p}": dict(win=round(float((ref > x).mean()), 3), diff=round(float((ref - x).mean()), 4),
                                              p=float(wilcoxon(ref, x).pvalue) if np.any(ref != x) else 1.0)
                             for (l, p), x in M.items() if (l, p) != ("paf", "default")}
out["thr_datasets"] = dsets

# ---- probe ablation ----
obs, coll = {}, {}
P = {}
for f in sorted(glob.glob(os.path.join(RE, "*", "_probes.json"))):
    ds = os.path.basename(os.path.dirname(f))
    runs = [json.load(open(g)) for g in sorted(glob.glob(os.path.join(RE, ds, "real_s*.json")))]
    runs = [r for r in runs if "error" not in r]
    pj = json.load(open(f))["pred"]
    for p, pr in pj.items():
        if "error" in pr:
            continue
        for ln, v in pr.items():
            f1 = [r["losses"][ln]["f1"] for r in runs]; cl = [r["losses"][ln]["collapsed"] for r in runs]
            P.setdefault(p, []).append((ds, ln, v, float(np.mean(f1)), cl))
pa = {}
for p, rows in P.items():
    pred = np.array([r[2] for r in rows]); f1 = np.array([r[3] for r in rows])
    per = []
    for ds in sorted({r[0] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r[0] == ds]
        if f1[idx].std() > 0 and pred[idx].std() > 0:
            per.append(spearmanr(pred[idx], f1[idx]).statistic)
    yc = np.concatenate([np.array(r[4], float) for r in rows]); sc = np.concatenate([[-r[2]] * len(r[4]) for r in rows])
    pa[p] = dict(n_datasets=len({r[0] for r in rows}), rank=round(float(np.nanmean(per)), 3),
                 mae=round(float(np.abs(pred - f1).mean()), 3), bias=round(float((pred - f1).mean()), 3),
                 collapse_auroc=round(float(roc_auc_score(yc, sc)), 3))
out["probe_ablation"] = pa
json.dump(out, open("rev_summary.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "thr_datasets"}, indent=1))
