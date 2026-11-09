"""Finite-sample (noisy-threshold) correction of profile-predicted F1.

A trained network does not threshold eta exactly: its score is a noisy estimate of a
monotone transform of eta. Modeling the logit error as N(0, sigma^2) gives the
smoothed prediction

    F1_sigma(t) = 2 sum_i eta_i p_i / (sum_i eta_i + sum_i p_i),
    p_i = Phi((logit eta_i - logit t) / sigma).

sigma is estimated from the data alone: two probes fitted on disjoint halves of the
training split are evaluated on the test split; Var(z1 - z2)/2 estimates the logit
error variance of a probe fitted on n/2 points, which is rescaled to the training
size used by the network (0.8 n, after the validation split) with a 1/sqrt(n) rate.

usage: python theory_fs.py results_es
writes <resultsdir>/<dataset>/_theory_fs.json
"""
import glob, json, os, sys
import numpy as np
from scipy.special import logit
from scipy.stats import norm

import lf_core as C
from run_dataset import load_split

CLIP = 1e-4


def f1_smooth(eta, t, sigma):
    eta = np.asarray(eta, float)
    if sigma <= 0:
        p = (eta > t).astype(float)
    else:
        z = logit(np.clip(eta, CLIP, 1 - CLIP)) - logit(np.clip(t, CLIP, 1 - CLIP))
        p = norm.cdf(z / sigma)
    s = eta.sum()
    return float(2 * (eta * p).sum() / (s + p.sum())) if s + p.sum() > 0 else 0.0


def gap_bound(eta, t, sigma):
    """Proposition 2: |F1_sigma(t) - F1_0(t)| <= 3 ||p - q||_1 / sum(eta)."""
    eta = np.asarray(eta, float)
    if sigma <= 0:
        return 0.0
    d = np.abs(logit(np.clip(eta, CLIP, 1 - CLIP)) - logit(np.clip(t, CLIP, 1 - CLIP)))
    return float(min(3 * norm.cdf(-d / sigma).sum() / eta.sum(), 1.0))


def estimate_sigma(Xtr, ytr, Xte, seed=0):
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(2, shuffle=True, random_state=seed)
    (a, b), _ = list(skf.split(Xtr, ytr))
    z = []
    for idx in (a, b):
        p = C.fit_probe(Xtr[idx], ytr[idx], seed).predict_proba(Xte)[:, 1]
        z.append(logit(np.clip(p, CLIP, 1 - CLIP)))
    s_half = float(np.std(z[0] - z[1]) / np.sqrt(2))
    return s_half * np.sqrt(0.5 / 0.8), s_half


def main():
    R = sys.argv[1] if len(sys.argv) > 1 else "results_es"
    for d in sorted(glob.glob(os.path.join(R, "*", "_profile.npz"))):
        ds = os.path.dirname(d)
        name = os.path.basename(ds)
        out = os.path.join(ds, "_theory_fs.json")
        runs = sorted(glob.glob(os.path.join(ds, "real_s*.json")))
        if os.path.exists(out) or not runs:
            continue
        data = os.path.join("data", name + ".npz")
        Xtr, ytr, Xte, yte = load_split(data)
        eta = np.load(d)["eta_te_hat"]
        sigma, s_half = estimate_sigma(Xtr, ytr, Xte)
        res = dict(dataset=name, sigma=sigma, sigma_half=s_half, n_train=len(ytr),
                   n_test=len(yte), dim=Xtr.shape[1], pi_train=float(ytr.mean()),
                   pi_test=float(yte.mean()), mean_eta_test=float(eta.mean()), runs={})
        for f in runs:
            j = json.load(open(f))
            if "error" in j:
                continue
            r = {}
            for ln, m in j["losses"].items():
                t = m["implied_t"]
                r[ln] = dict(f1=m["f1"], collapsed=m["collapsed"], t=t,
                             pred0=f1_smooth(eta, t, 0.0), pred_fs=f1_smooth(eta, t, sigma),
                             bound=gap_bound(eta, t, sigma))
            res["runs"][str(j["seed"])] = r
        json.dump(res, open(out, "w"))
        print(name, f"sigma={sigma:.3f}", flush=True)


if __name__ == "__main__":
    main()
