"""Run all (world x loss x seed) cells for one dataset.

worlds: real + synthetic generators. In every world each loss is trained on the
world's train split and scored on the world's test split; loss rankings in each
synthetic world are compared to the real ranking downstream (aggregate.py).

usage: python run_dataset.py --data data/ecoli.npz --out results --seeds 5 --workers 6
"""
import os
import argparse, json, os, time, traceback
import numpy as np

GENS = ["real", "gauss", "copula", "gmm", "dms_t", "dms_l",
        "dms_l_dp0.1", "dms_l_dp1", "dms_l_dp10"]
MAX_TR, MAX_TE = 60000, 40000


def load_split(path):
    from sklearn.model_selection import train_test_split
    d = np.load(path, allow_pickle=True)
    X, y = d["X"].astype(np.float64), d["y"].astype(np.float64)
    keep = X.std(0) > 0
    X = X[:, keep]
    if "split" in d.files:  # predefined split (e.g. GIAB: train/test by chromosome region)
        sp = d["split"]
        Xtr, ytr, Xte, yte = X[sp == 0], y[sp == 0], X[sp == 1], y[sp == 1]
    else:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    rng = np.random.default_rng(0)
    if len(ytr) > MAX_TR:
        i = rng.choice(len(ytr), MAX_TR, replace=False); Xtr, ytr = Xtr[i], ytr[i]
    if len(yte) > MAX_TE:
        i = rng.choice(len(yte), MAX_TE, replace=False); Xte, yte = Xte[i], yte[i]
    return Xtr, ytr, Xte, yte


def world(gen, seed, Xtr, ytr, Xte, yte, prof_tr, probe):
    """returns (Xtr, ytr, Xte, yte, eta_te_true or None)."""
    import lf_core as C
    rng = np.random.default_rng(1000 + seed)
    ntr, nte = len(ytr), len(yte)
    if gen == "real":
        return Xtr, ytr, Xte, yte, None
    if gen in ("gauss", "copula", "gmm"):
        f = dict(gauss=C.gen_gauss, copula=C.gen_copula, gmm=C.gen_gmm)[gen]
        a, b = f(Xtr, ytr, ntr, rng); c, d = f(Xtr, ytr, nte, rng)
        return a, b, c, d, None
    if gen == "dms_t":
        a, b, _ = C.gen_dms_transport(Xtr, ytr, ntr, rng, prof_tr, probe)
        c, d, e = C.gen_dms_transport(Xtr, ytr, nte, rng, prof_tr, probe)
        return a, b, c, d, e
    if gen.startswith("dms_l"):
        eps = float(gen.split("dp")[1]) if "dp" in gen else None
        edges, probs = C.profile_histogram(prof_tr, eps=eps, rng=rng)
        a, b, _ = C.gen_dms_latent(Xtr.shape[1], ntr, rng, edges, probs)
        c, d, e = C.gen_dms_latent(Xtr.shape[1], nte, rng, edges, probs)
        return a, b, c, d, e
    raise ValueError(gen)


def run_task(args):
    gen, seed, data, outdir, steps, es = args
    out = os.path.join(outdir, f"{gen}_s{seed}.json")
    if os.path.exists(out):
        return out
    import torch
    torch.set_num_threads(2)
    import lf_core as C, lf_train as T
    t0 = time.time()
    try:
        Xtr, ytr, Xte, yte = load_split(data)
        cache = os.path.join(outdir, "_profile.npz")
        pc = np.load(cache)
        prof_tr, eta_te_hat = pc["prof_tr"], pc["eta_te_hat"]
        probe = None
        if gen == "dms_t":
            probe = C.fit_probe(Xtr, ytr)
        A, b, Cx, d, eta_true = world(gen, seed, Xtr, ytr, Xte, yte, prof_tr, probe)
        if b.sum() < 2 or d.sum() < 1:
            raise RuntimeError("too few positives in synthetic world")
        pi = float(b.mean())
        # world-internal difficulty profile (what a practitioner could estimate)
        eta_w_tr = C.crossfit_eta(A, b, k=3, seed=seed)
        t_star, f1_star = C.optimal_threshold(eta_w_tr)
        # profile used for theory predictions on this world's test set
        if eta_true is not None:
            eta_pred_te = eta_true
        elif gen == "real":
            eta_pred_te = eta_te_hat
        else:
            eta_pred_te = C.fit_probe(A, b, seed).predict_proba(Cx)[:, 1]
        pool = C.make_loss_pool(pi)
        pool["paf"] = dict(kind="focal", alpha=1 - t_star, gamma=2.0)      # ours
        pool["pa_bce"] = dict(kind="wbce", w=(1 - t_star) / max(t_star, 1e-3))  # ours
        res = dict(gen=gen, seed=seed, data=data, pi_train=pi, n_train=len(b), n_test=len(d),
                   t_star=t_star, f1_star_profile=f1_star, losses={})
        for name, spec in pool.items():
            th = C.implied_threshold(spec)
            m = T.train_eval(A, b, Cx, d, spec, seed, steps=steps, es=es)
            m["implied_t"] = th
            m["pred_f1"] = C.f1_from_profile(eta_pred_te, th)
            m["spec"] = {k: (float(v) if not isinstance(v, str) else v) for k, v in spec.items()}
            res["losses"][name] = m
        m = T.train_eval(A, b, Cx, d, dict(kind="focal", gamma=2.0), seed, steps=steps, scfl=True, es=es)
        m["implied_t"] = m["scfl_traj"][-1] if m["scfl_traj"] else 0.5
        m["pred_f1"] = C.f1_from_profile(eta_pred_te, m["implied_t"])
        res["losses"]["scfl"] = m                                          # ours
        # profile summary of the world test set
        q = np.quantile(eta_pred_te, [0.5, 0.9, 0.99])
        res["profile_q"] = q.tolist()
        res["profile_hist"] = np.histogram(eta_pred_te, np.linspace(0, 1, 101))[0].tolist()
        res["profile_pos_mass_above"] = {str(t): float(eta_pred_te[eta_pred_te > t].sum() / eta_pred_te.sum())
                                         for t in (0.05, 0.25, 0.5, 0.75)}
        res["secs"] = time.time() - t0
    except Exception as e:
        res = dict(gen=gen, seed=seed, data=data, error=repr(e), tb=traceback.format_exc())
    with open(out, "w") as fh:
        json.dump(res, fh)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--es", action="store_true", help="validation early stopping")
    ap.add_argument("--minpos", type=int, default=0, help="skip datasets with fewer train positives")
    ap.add_argument("--gens", default=",".join(GENS))
    a = ap.parse_args()
    import lf_core as C
    name = os.path.splitext(os.path.basename(a.data))[0]
    if a.minpos:
        _, ytr0, _, _ = load_split(a.data)
        if ytr0.sum() < a.minpos:
            print(f"[{name}] skip: {int(ytr0.sum())} train positives < {a.minpos}", flush=True)
            return
    outdir = os.path.join(a.out, name)
    os.makedirs(outdir, exist_ok=True)
    cache = os.path.join(outdir, "_profile.npz")
    if not os.path.exists(cache):
        Xtr, ytr, Xte, yte = load_split(a.data)
        prof_tr = C.crossfit_eta(Xtr, ytr, k=5)
        eta_te_hat = C.fit_probe(Xtr, ytr).predict_proba(Xte)[:, 1]
        tmp = cache + ".tmp.npz"
        np.savez(tmp, prof_tr=prof_tr, eta_te_hat=eta_te_hat)
        os.replace(tmp, cache)  # atomic: concurrent runners never read a partial cache
        print(f"[{name}] profile cached; n_tr={len(ytr)} pi={ytr.mean():.4f}", flush=True)
    tasks = [(g, s, a.data, outdir, a.steps, a.es) for g in a.gens.split(",") for s in range(a.seeds)]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    with ctx.Pool(a.workers) as pool:
        for i, o in enumerate(pool.imap_unordered(run_task, tasks)):
            print(f"[{name}] {i+1}/{len(tasks)} {o}", flush=True)


if __name__ == "__main__":
    main()
