"""Torch training of an MLP under a given loss spec; returns test metrics."""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loss_fn(spec, z, y):
    k = spec["kind"]
    if k == "wbce":
        return F.binary_cross_entropy_with_logits(
            z, y, pos_weight=torch.tensor(spec["w"], device=z.device))
    if k == "focal":
        a, g = spec["alpha"], spec["gamma"]
        ce = F.binary_cross_entropy_with_logits(z, y, reduction="none")
        p = torch.sigmoid(z)
        pt = y * p + (1 - y) * (1 - p)
        at = y * a + (1 - y) * (1 - a)
        return (at * (1 - pt).clamp_min(0) ** g * ce).mean()
    if k == "la":
        off = spec["tau"] * float(np.log(spec["pi"] / (1 - spec["pi"])))
        return F.binary_cross_entropy_with_logits(z + off, y)
    raise ValueError(k)


class MLP(nn.Module):
    def __init__(self, d, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def metrics(y, prob, thr=0.5):
    from sklearn.metrics import average_precision_score, roc_auc_score
    pred = prob > thr
    tp = float((pred & (y == 1)).sum()); fp = float((pred & (y == 0)).sum())
    fn = float((~pred & (y == 1)).sum())
    f1 = 2 * tp / max(2 * tp + fp + fn, 1e-12)
    out = dict(f1=f1, recall=tp / max(tp + fn, 1e-12),
               precision=tp / max(tp + fp, 1e-12) if tp + fp > 0 else 0.0,
               pos_rate=float(pred.mean()), collapsed=bool(pred.sum() == 0 or f1 < 0.05))
    try:
        out["auprc"] = float(average_precision_score(y, prob))
        out["auroc"] = float(roc_auc_score(y, prob))
    except Exception:
        out["auprc"] = out["auroc"] = float("nan")
    # ECE (10 bins) of raw probabilities
    bins = np.clip((prob * 10).astype(int), 0, 9)
    ece = 0.0
    for b in range(10):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - prob[m].mean())
    out["ece"] = float(ece)
    return out


def _predict(model, X, bs=8192):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            out.append(torch.sigmoid(model(X[i:i + bs])).float().cpu().numpy())
    model.train()
    return np.concatenate(out)


def train_eval(Xtr, ytr, Xte, yte, spec, seed, steps=3000, bs=512, lr=1e-3,
               scfl=False, scfl_every=250, t0=0.5, es=False, es_every=100, patience=8):
    """spec: loss spec dict. scfl=True => self-calibrating focal (ours): alpha is
    re-set every `scfl_every` steps to 1 - t, with t <- F1_val(current)/2."""
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    if scfl or es:  # hold out a stratified 20% validation split (SCFL update / early stopping)
        pos, neg = np.where(ytr == 1)[0], np.where(ytr == 0)[0]
        pos, neg = rng.permutation(pos), rng.permutation(neg)
        npv, nnv = max(int(0.2 * len(pos)), 1), max(int(0.2 * len(neg)), 1)
        vi = np.r_[pos[:npv], neg[:nnv]]; ti = np.r_[pos[npv:], neg[nnv:]]
        Xv, yv = Xtr[vi], ytr[vi]; Xtr, ytr = Xtr[ti], ytr[ti]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    f = lambda A: torch.tensor((A - mu) / sd, dtype=torch.float32, device=DEV)
    Xt, yt, Xe = f(Xtr), torch.tensor(ytr, dtype=torch.float32, device=DEV), f(Xte)
    if scfl or es:
        Xvt = f(Xv); yvt = torch.tensor(yv, dtype=torch.float32, device=DEV)
    model = MLP(Xt.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    n = len(yt)
    g = torch.Generator(device=DEV); g.manual_seed(seed)
    t = t0
    traj = []
    cur = dict(spec)
    if scfl:
        cur = dict(kind="focal", alpha=1 - t, gamma=spec.get("gamma", 2.0))
    best, best_state, bad, stop_step = float("inf"), None, 0, steps
    for s in range(steps):
        ib = torch.randint(0, n, (min(bs, n),), device=DEV, generator=g)
        loss = loss_fn(cur, model(Xt[ib]), yt[ib])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if scfl and (s + 1) % scfl_every == 0:
            pv = _predict(model, Xvt)
            f1v = metrics(yv, pv)["f1"]
            t = float(np.clip(f1v / 2, 0.02, 0.5))
            cur["alpha"] = 1 - t
            traj.append(t)
        if es and (s + 1) % es_every == 0:
            model.eval()
            with torch.no_grad():
                vl = float(loss_fn(cur, model(Xvt), yvt))
            model.train()
            if vl < best - 1e-6:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    stop_step = s + 1
                    break
    if es and best_state is not None:
        model.load_state_dict(best_state)
    prob = _predict(model, Xe)
    m = metrics(yte, prob)
    m["stop_step"] = stop_step
    if scfl:
        m["scfl_traj"] = traj
    return m
