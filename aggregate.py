"""Aggregate results/<dataset>/<gen>_s<seed>.json into paper tables.

Outputs (results/summary/):
  ranking_transfer.csv   per dataset x generator: Kendall tau, Spearman, top-1 regret,
                         W1 distance between world and real difficulty profiles
  theory_validation.csv  per dataset x world x loss: predicted vs empirical F1
  methods.csv            per dataset: real-world F1 of every loss (mean, sd, collapse rate)
  summary.txt            headline numbers
"""
import glob, json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, pearsonr, wilcoxon

R = sys.argv[1] if len(sys.argv) > 1 else "results"
MINPOS = int(sys.argv[2]) if len(sys.argv) > 2 else 0   # keep datasets with >= MINPOS train positives
S = os.path.join(R, "summary"); os.makedirs(S, exist_ok=True)
rows, prof = [], {}
errs = 0
for f in glob.glob(os.path.join(R, "*", "*_s*.json")):
    d = json.load(open(f))
    ds = os.path.basename(os.path.dirname(f))
    if "error" in d:
        errs += 1; continue
    prof.setdefault((ds, d["gen"]), []).append(np.array(d["profile_hist"], float))
    for ln, m in d["losses"].items():
        rows.append(dict(dataset=ds, gen=d["gen"], seed=d["seed"], loss=ln, f1=m["f1"],
                         auprc=m.get("auprc"), ece=m.get("ece"), collapsed=m["collapsed"],
                         pred_f1=m["pred_f1"], implied_t=m["implied_t"], t_star=d["t_star"],
                         pi=d["pi_train"]))
df = pd.DataFrame(rows)
npos = {}
for f in glob.glob(os.path.join(R, "*", "real_s0.json")):
    d = json.load(open(f))
    if "error" not in d:
        npos[os.path.basename(os.path.dirname(f))] = d["pi_train"] * d["n_train"]
if MINPOS:
    keep = [k for k, v in npos.items() if v >= MINPOS]
    df = df[df.dataset.isin(keep)]
    print(f"MINPOS={MINPOS}: keeping {len(keep)} datasets")
df.to_csv(os.path.join(S, "all_runs.csv"), index=False)
print("runs:", len(df), "errors:", errs)

BASELINES = ["bce", "wbce3", "wbce_bal", "focal_a25_g2", "focal_a50_g2", "focal_a75_g2",
             "focal_a50_g1", "la_t1", "la_t05"]
OURS = ["paf", "pa_bce", "scfl"]
ALL = BASELINES + OURS

mean = df.groupby(["dataset", "gen", "loss"]).f1.mean().unstack("loss")


def w1(h1, h2):
    c1, c2 = np.cumsum(h1 / h1.sum()), np.cumsum(h2 / h2.sum())
    return float(np.abs(c1 - c2).sum() * 0.01)


# ---------------- ranking transfer ----------------
rt = []
for ds in df.dataset.unique():
    if (ds, "real") not in mean.index:
        continue
    real = mean.loc[(ds, "real")][ALL].dropna()
    for g in df[df.dataset == ds].gen.unique():
        if g == "real" or (ds, g) not in mean.index:
            continue
        syn = mean.loc[(ds, g)][real.index].dropna()
        idx = syn.index
        if len(idx) < 4:
            continue
        tau = kendalltau(real[idx], syn[idx]).statistic
        rho = spearmanr(real[idx], syn[idx]).statistic
        pick = syn[idx].idxmax()
        regret = real[idx].max() - real[pick]
        # flips among loss pairs whose real F1 differ by > 0.02
        flips = tot = 0
        for i in idx:
            for j in idx:
                if i < j and abs(real[i] - real[j]) > 0.02:
                    tot += 1
                    flips += np.sign(real[i] - real[j]) != np.sign(syn[i] - syn[j])
        W = w1(np.mean(prof[(ds, g)], 0), np.mean(prof[(ds, "real")], 0)) if (ds, "real") in prof else np.nan
        rt.append(dict(dataset=ds, gen=g, kendall=tau, spearman=rho, top1_regret=regret,
                       top1_hit=float(pick == real[idx].idxmax()),
                       flip_rate=flips / max(tot, 1), W1_profile=W))
rt = pd.DataFrame(rt)
rt.to_csv(os.path.join(S, "ranking_transfer.csv"), index=False)

# ---------------- theory validation ----------------
tv = df.groupby(["dataset", "gen", "loss"]).agg(f1=("f1", "mean"), pred_f1=("pred_f1", "mean"),
                                                collapse_rate=("collapsed", "mean")).reset_index()
tv.to_csv(os.path.join(S, "theory_validation.csv"), index=False)

# ---------------- methods on real data ----------------
real = df[df.gen == "real"]
meth = real.groupby(["dataset", "loss"]).agg(f1=("f1", "mean"), sd=("f1", "std"),
                                            auprc=("auprc", "mean"), ece=("ece", "mean"),
                                            collapse=("collapsed", "mean")).reset_index()
meth.to_csv(os.path.join(S, "methods.csv"), index=False)

with open(os.path.join(S, "summary.txt"), "w") as fh:
    def P(*a):
        print(*a); print(*a, file=fh)
    P("=== Ranking transfer (mean over datasets) ===")
    if len(rt):
        P(rt.groupby("gen")[["kendall", "spearman", "top1_regret", "top1_hit", "flip_rate", "W1_profile"]]
          .mean().sort_values("kendall", ascending=False).round(3).to_string())
        ok = rt.dropna(subset=["W1_profile", "kendall"])
        if len(ok) > 5:
            P("corr(W1_profile, flip_rate) =", round(spearmanr(ok.W1_profile, ok.flip_rate).statistic, 3),
              " corr(W1_profile, kendall) =", round(spearmanr(ok.W1_profile, ok.kendall).statistic, 3))
    P("\n=== Theory: predicted vs empirical F1 ===")
    for g, sub in tv.groupby("gen"):
        sub = sub.dropna()
        if len(sub) > 5:
            r = pearsonr(sub.f1, sub.pred_f1).statistic
            mae = np.abs(sub.f1 - sub.pred_f1).mean()
            per = [spearmanr(s.f1, s.pred_f1).statistic for _, s in sub.groupby("dataset") if s.f1.std() > 0]
            P(f"{g:12s} pearson={r:.3f} MAE={mae:.3f} within-dataset rank corr={np.nanmean(per):.3f}")
    rs = tv[tv.gen == "real"]
    pc = rs.pred_f1 < 0.05
    if pc.any():
        P(f"collapse predicted (pred_f1<0.05): {pc.sum()} cells; empirical collapse rate there "
          f"{rs[pc].collapse_rate.mean():.3f} vs elsewhere {rs[~pc].collapse_rate.mean():.3f}")
    # collapse prediction quality (real world, per run)
    rr = df[df.gen == "real"]
    if len(rr) and rr.collapsed.any():
        from sklearn.metrics import roc_auc_score
        pc = rr.pred_f1 < 0.05
        prec = rr[pc].collapsed.mean() if pc.any() else float("nan")
        rec = (pc & rr.collapsed).sum() / max(rr.collapsed.sum(), 1)
        P(f"collapse (per run): n_collapsed={int(rr.collapsed.sum())}/{len(rr)}; rule pred_f1<0.05 "
          f"precision={prec:.3f} recall={rec:.3f}; AUROC(-pred_f1 -> collapsed)="
          f"{roc_auc_score(rr.collapsed, -rr.pred_f1):.3f}")
    P("\n=== Methods on real data (mean F1 over datasets; win-rate vs best fixed baseline) ===")
    M = meth.pivot(index="dataset", columns="loss", values="f1")
    P(M.mean().sort_values(ascending=False).round(4).to_string())
    best_base = M[[c for c in BASELINES if c in M]].max(1)
    for o in OURS:
        if o in M:
            diff = M[o] - best_base
            P(f"{o}: mean F1 {M[o].mean():.4f}; vs best-in-hindsight baseline {diff.mean():+.4f}; "
              f"vs each baseline win-rate " +
              ", ".join(f"{b}:{(M[o] > M[b]).mean():.2f}" for b in BASELINES if b in M))
    rank = M.rank(axis=1, ascending=False)
    P("\nmean rank (1=best):\n" + rank.mean().sort_values().round(2).to_string())
    C = meth.pivot(index="dataset", columns="loss", values="collapse")
    P("\ncollapse rate (mean over datasets):\n" + C.mean().sort_values().round(3).to_string())
