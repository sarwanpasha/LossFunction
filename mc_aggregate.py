"""Summarize multi-class long-tailed runs and compare with profile predictions.
usage: python mc_aggregate.py results_mc  -> results_mc/summary_mc.json"""
import glob, json, os, sys
import numpy as np
from scipy.stats import spearmanr

R = sys.argv[1] if len(sys.argv) > 1 else "results_mc"
LOSSES = ["ce", "focal", "wce_drw", "cb_drw", "la", "posthoc_la", "ldam_drw", "pa_ce_drw", "paf_drw"]


def metrics(Z, y, counts):
    C = Z.shape[1]; h = Z.argmax(1)
    acc = float((h == y).mean())
    f1 = []
    for c in range(C):
        tp = ((h == c) & (y == c)).sum(); p = (h == c).sum(); t = (y == c).sum()
        f1.append(2 * tp / max(p + t, 1))
    pc = np.array([(h[y == c] == c).mean() for c in range(C)])
    many, few = counts > 100, counts < 20
    return dict(acc=acc, macro_f1=float(np.mean(f1)), many=float(pc[many].mean()) if many.any() else None,
                medium=float(pc[~many & ~few].mean()) if (~many & ~few).any() else None,
                few=float(pc[few].mean()) if few.any() else None)


def cb_w(counts, beta=0.9999):
    w = (1 - beta) / (1 - np.power(beta, counts)); return w / w.sum() * len(counts)


def main():
    import mc_profile as P
    out = {}
    for ds in ["cifar10", "cifar100"]:
        pj = os.path.join(R, f"{ds}_profile.json")
        if not os.path.exists(pj):
            continue
        prof = json.load(open(pj)); counts = np.array(prof["counts"]); pi = counts / counts.sum()
        eta = np.load(os.path.join(R, f"{ds}_profile.npz"))["eta_te_bal"]
        imp = {"ce": np.ones_like(pi), "focal": np.ones_like(pi), "wce_drw": 1 / counts, "cb_drw": cb_w(counts),
               "la": 1 / pi, "posthoc_la": 1 / pi, "pa_ce_drw": np.array(prof["pa_weights"]),
               "paf_drw": np.array(prof["pa_weights"])}
        rows = {}
        for ln in LOSSES:
            src = "ce" if ln == "posthoc_la" else ln
            ms = []
            for f in sorted(glob.glob(os.path.join(R, f"{ds}_{src}_s*.npz"))):
                d = np.load(f); Z = d["logits"].astype(np.float64); y = d["y"].astype(int)
                if ln == "posthoc_la":
                    Z = Z - np.log(pi)[None, :]
                ms.append(metrics(Z, y, counts))
            if not ms:
                continue
            r = {k: float(np.mean([m[k] for m in ms])) for k in ms[0] if ms[0][k] is not None}
            r.update({k + "_sd": float(np.std([m[k] for m in ms])) for k in ("acc", "macro_f1")})
            r["n"] = len(ms)
            if ln in imp:
                pa, pf = P.pred_metrics(eta, imp[ln] * pi)
                r["pred_acc"], r["pred_macro_f1"] = pa, pf
            rows[ln] = r
        ks = [k for k in rows if "pred_acc" in rows[k]]
        corr = {}
        if len(ks) > 3:
            for m in ("acc", "macro_f1"):
                corr[m] = float(spearmanr([rows[k][m] for k in ks], [rows[k]["pred_" + m] for k in ks]).statistic)
        out[ds] = dict(rows=rows, rank_corr=corr, probe=dict(acc=prof["probe_test_acc"], bal_acc=prof["probe_test_bal_acc"]))
    json.dump(out, open(os.path.join(R, "summary_mc.json"), "w"), indent=1)
    for ds, v in out.items():
        print(ds, "rank corr", v["rank_corr"], "probe", v["probe"])
        for ln, r in v["rows"].items():
            print(f"  {ln:11s} acc={r['acc']:.4f} f1={r['macro_f1']:.4f} n={r['n']}"
                  + (f" pred_acc={r['pred_acc']:.4f} pred_f1={r['pred_macro_f1']:.4f}" if "pred_acc" in r else ""))


if __name__ == "__main__":
    main()
