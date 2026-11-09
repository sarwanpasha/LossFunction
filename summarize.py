"""Summary statistics for figures and tables: per-dataset means, corrected theory
predictions, and paired Wilcoxon tests with Holm correction.
usage: python summarize.py results_es  -> summary.json.gz"""
import glob, gzip, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

R = sys.argv[1] if len(sys.argv) > 1 else "results_es"
ALL = pd.read_csv(os.path.join(R, "summary", "all_runs.csv"))
real = ALL[ALL.gen == "real"]
out = dict(meta={}, real=[], rt=pd.read_csv(os.path.join(R, "summary", "ranking_transfer.csv")).round(4).to_dict("list"),
           tv=None, hist={}, stats={})
rows = []
for f in sorted(glob.glob(os.path.join(R, "*", "_theory_fs.json"))):
    j = json.load(open(f)); ds = j["dataset"]
    out["meta"][ds] = {k: (round(v, 5) if isinstance(v, float) else v) for k, v in j.items() if k != "runs"}
    for s, r in j["runs"].items():
        for ln, m in r.items():
            rows.append(dict(dataset=ds, seed=int(s), loss=ln, **m))
    h = json.load(open(os.path.join(os.path.dirname(f), "real_s0.json")))
    out["hist"][ds] = h["profile_hist"]
fs = pd.DataFrame(rows)
fs.collapsed = fs.collapsed.astype(float)
au = real.groupby(["dataset", "loss"])[["auprc", "ece"]].mean()
g = fs.groupby(["dataset", "loss"]).agg(f1=("f1", "mean"), sd=("f1", "std"), pred0=("pred0", "mean"),
                                          pred_fs=("pred_fs", "mean"), bound=("bound", "mean"),
                                          coll=("collapsed", "mean"), t=("t", "mean")).join(au).reset_index()
out["real"] = g.round(4).to_dict("list")
# synthetic worlds: per dataset x gen x loss means (for theory validation across worlds)
tv = ALL.groupby(["dataset", "gen", "loss"]).agg(f1=("f1", "mean"), pred=("pred_f1", "mean"),
                                                coll=("collapsed", "mean")).reset_index()
out["tv"] = tv.round(4).to_dict("list")
# theory stats: uncorrected vs corrected
st = {}
for name, sub in [("all", g), ("mosaic", g[g.dataset.str.contains("mosaic")]),
                  ("giab", g[g.dataset.str.contains("giab")])]:
    for col in ("pred0", "pred_fs"):
        e = sub.f1 - sub[col]
        per = [spearmanr(s.f1, s[col]).statistic for _, s in sub.groupby("dataset") if s.f1.std() > 0]
        st[f"{name}_{col}"] = dict(pearson=pearsonr(sub.f1, sub[col]).statistic, mae=float(np.abs(e).mean()),
                                   bias=float((sub[col] - sub.f1).mean()), rank=float(np.nanmean(per)),
                                   n=len(sub))
st["auroc_pred0"] = roc_auc_score(fs.collapsed, -fs.pred0)
st["auroc_pred_fs"] = roc_auc_score(fs.collapsed, -fs.pred_fs)
st["gap_within_bound"] = float((np.abs(fs.pred_fs - fs.pred0) <= fs.bound + 1e-9).mean())
# Wilcoxon (paired over datasets on mean F1) + Holm
M = g.pivot(index="dataset", columns="loss", values="f1")
BASE = ["bce", "wbce3", "wbce_bal", "focal_a25_g2", "focal_a50_g2", "focal_a75_g2", "focal_a50_g1", "la_t1", "la_t05"]
for o in ("paf", "pa_bce", "scfl"):
    ps = {}
    for b in BASE:
        d = M[o] - M[b]
        ps[b] = dict(p=float(wilcoxon(M[o], M[b]).pvalue), win=float((d > 0).mean()),
                     tie=float((d == 0).mean()), mean_diff=float(d.mean()))
    order = sorted(ps, key=lambda b: ps[b]["p"]); k = len(order); run = 0
    for i, b in enumerate(order):
        run = max(run, min(1.0, (k - i) * ps[b]["p"])); ps[b]["p_holm"] = run
    st["wilcoxon_" + o] = ps
out["stats"] = st
json.dump(out, gzip.open("summary.json.gz", "wt"), default=float)
print(json.dumps(st, indent=1, default=float)[:3000])
print("bytes", os.path.getsize("summary.json.gz"))
