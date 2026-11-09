"""Decision-threshold baselines on the real data.

For each dataset and seed, the losses bce, focal (alpha=0.5, gamma=2), la_t05 and PAF are
trained once (same network and early stopping as run_dataset.py), and the test F1 is
reported under 3 threshold policies applied to the network output p(x):
  default : p > 0.5
  val     : threshold maximizing F1 on the validation split
  plugin  : p > t_hat*, the F1-optimal threshold of the estimated training profile

usage: python run_thresh.py --data data/x.npz --out results_thr --seeds 5
"""
import argparse, json, os, time
import numpy as np


def f1_at(y, p, t):
    pred = p > t
    tp = float((pred & (y == 1)).sum()); fp = float((pred & (y == 0)).sum()); fn = float((~pred & (y == 1)).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1e-12)


def best_val_threshold(y, p):
    q = np.unique(np.quantile(p, np.linspace(0, 1, 1001)))
    cand = np.concatenate([[0.0], (q[:-1] + q[1:]) / 2])
    f = [f1_at(y, p, t) for t in cand]
    return float(cand[int(np.argmax(f))])


def train_probs(Xtr, ytr, Xte, spec, seed, steps=3000, bs=512, lr=1e-3, es_every=100, patience=8):
    """Same model, split, and early stopping as lf_train.train_eval(es=True); returns
    validation labels/probabilities and test probabilities."""
    import torch
    import lf_train as T
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    pos, neg = np.where(ytr == 1)[0], np.where(ytr == 0)[0]
    pos, neg = rng.permutation(pos), rng.permutation(neg)
    npv, nnv = max(int(0.2 * len(pos)), 1), max(int(0.2 * len(neg)), 1)
    vi = np.r_[pos[:npv], neg[:nnv]]; ti = np.r_[pos[npv:], neg[nnv:]]
    Xv, yv = Xtr[vi], ytr[vi]; Xa, ya = Xtr[ti], ytr[ti]
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-6
    f = lambda A: torch.tensor((A - mu) / sd, dtype=torch.float32, device=T.DEV)
    Xt, yt, Xe, Xvt = f(Xa), torch.tensor(ya, dtype=torch.float32, device=T.DEV), f(Xte), f(Xv)
    yvt = torch.tensor(yv, dtype=torch.float32, device=T.DEV)
    model = T.MLP(Xt.shape[1]).to(T.DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    g = torch.Generator(device=T.DEV); g.manual_seed(seed)
    n = len(yt); best, best_state, bad = float("inf"), None, 0
    for s in range(steps):
        ib = torch.randint(0, n, (min(bs, n),), device=T.DEV, generator=g)
        loss = T.loss_fn(spec, model(Xt[ib]), yt[ib])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (s + 1) % es_every == 0:
            model.eval()
            with torch.no_grad():
                vl = float(T.loss_fn(spec, model(Xvt), yvt))
            model.train()
            if vl < best - 1e-6:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return yv, T._predict(model, Xvt), T._predict(model, Xe)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results_thr")
    ap.add_argument("--profiles", default="results_es")
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    import lf_core as C
    from run_dataset import load_split
    name = os.path.splitext(os.path.basename(a.data))[0]
    cache = os.path.join(a.profiles, name, "_profile.npz")
    if not os.path.exists(cache):
        return
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, name + ".json")
    if os.path.exists(out):
        return
    Xtr, ytr, Xte, yte = load_split(a.data)
    t_star, _ = C.optimal_threshold(np.load(cache)["prof_tr"])
    pi = float(ytr.mean())
    specs = {"bce": dict(kind="wbce", w=1.0), "focal_a50_g2": dict(kind="focal", alpha=0.5, gamma=2.0),
             "la_t05": dict(kind="la", tau=0.5, pi=pi), "paf": dict(kind="focal", alpha=1 - t_star, gamma=2.0)}
    res = dict(dataset=name, t_star=t_star, runs=[])
    t0 = time.time()
    for seed in range(a.seeds):
        for ln, spec in specs.items():
            yv, pv, pt = train_probs(Xtr, ytr, Xte, spec, seed)
            tv = best_val_threshold(yv, pv)
            res["runs"].append(dict(seed=seed, loss=ln, f1_default=f1_at(yte, pt, 0.5),
                                    f1_val=f1_at(yte, pt, tv), f1_plugin=f1_at(yte, pt, t_star),
                                    t_val=tv))
    res["secs"] = time.time() - t0
    json.dump(res, open(out, "w"))
    print(name, f"{res['secs']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
