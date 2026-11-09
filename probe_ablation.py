"""Probe ablation: re-estimate the test profile with different probes and recompute the
profile predictions F_0(t_l) for the 12 losses of each dataset.

probes: hgb (calibrated gradient boosting, the default), hgb_raw (no calibration),
        logreg (calibrated logistic regression), knn (calibrated k-nearest neighbors)
usage: python probe_ablation.py results_es  -> results_es/<ds>/_probes.json
"""
import glob, json, os, sys
import numpy as np


def make(name, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name == "hgb_raw":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, early_stopping=True, random_state=seed)
    if name == "logreg":
        base = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    elif name == "knn":
        base = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=50, n_jobs=2))
    else:
        raise ValueError(name)
    return CalibratedClassifierCV(base, method="isotonic", cv=3)


def main():
    import lf_core as C
    from run_dataset import load_split
    R = sys.argv[1] if len(sys.argv) > 1 else "results_es"
    rng = np.random.default_rng(0)
    for d in sorted(glob.glob(os.path.join(R, "*", "_profile.npz"))):
        ds = os.path.dirname(d); name = os.path.basename(ds)
        out = os.path.join(ds, "_probes.json")
        runs = sorted(glob.glob(os.path.join(ds, "real_s*.json")))
        if os.path.exists(out) or not runs:
            continue
        Xtr, ytr, Xte, yte = load_split(os.path.join("data", name + ".npz"))
        ts = {ln: m["implied_t"] for ln, m in json.load(open(runs[0]))["losses"].items() if ln != "scfl"}
        res = dict(dataset=name, pred={})
        res["pred"]["hgb"] = {ln: C.f1_from_profile(np.load(d)["eta_te_hat"], t) for ln, t in ts.items()}
        for p in ["hgb_raw", "logreg", "knn"]:
            Xf, yf = Xtr, ytr
            if p == "knn" and len(ytr) > 20000:
                i = rng.choice(len(ytr), 20000, replace=False); Xf, yf = Xtr[i], ytr[i]
            try:
                eta = make(p).fit(Xf, yf).predict_proba(Xte)[:, 1]
                res["pred"][p] = {ln: C.f1_from_profile(eta, t) for ln, t in ts.items()}
            except Exception as e:
                res["pred"][p] = {"error": repr(e)}
        json.dump(res, open(out, "w"))
        print(name, flush=True)


if __name__ == "__main__":
    main()
