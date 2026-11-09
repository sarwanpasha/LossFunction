"""Multi-class long-tailed experiments: CIFAR-10-LT and CIFAR-100-LT (imbalance ratio 100),
ResNet-32 trained end to end. Data augmentation runs on the GPU, so the CPU load is low.

losses: ce, focal (gamma=1), wce_drw (1/n weights, deferred), cb_drw (class-balanced,
beta=0.9999, deferred), la (logit-adjusted loss, tau=1), ldam_drw, pa_ce_drw and paf_drw
(profile-adapted weights from mc_profile.py, deferred). Post-hoc logit adjustment is
evaluated from the ce runs.

usage: python mc_lt.py --dataset cifar10 --loss ce --seed 0 --out results_mc
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)


def lt_indices(y, C, ir=100, seed=0):
    nmax = int(np.bincount(y).max())
    counts = [int(nmax * (1 / ir) ** (c / (C - 1))) for c in range(C)]
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.where(y == c)[0], counts[c], replace=False) for c in range(C)])
    return np.sort(idx), np.array(counts)


def load(dataset):
    import torchvision
    cls = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
    tr = cls("dl_cache/" + dataset, train=True, download=True)
    te = cls("dl_cache/" + dataset, train=False, download=True)
    C = 10 if dataset == "cifar10" else 100
    ytr = np.array(tr.targets); idx, counts = lt_indices(ytr, C)
    return tr.data[idx], ytr[idx], te.data, np.array(te.targets), counts, C


# ---------------- ResNet-32 (He et al., 2016, CIFAR variant) ----------------
class Block(nn.Module):
    def __init__(self, i, o, s):
        super().__init__()
        self.c1 = nn.Conv2d(i, o, 3, s, 1, bias=False); self.b1 = nn.BatchNorm2d(o)
        self.c2 = nn.Conv2d(o, o, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(o)
        self.sh = None
        if s != 1 or i != o:
            self.sh = nn.Sequential(nn.Conv2d(i, o, 1, s, bias=False), nn.BatchNorm2d(o))

    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        return F.relu(out + (x if self.sh is None else self.sh(x)))


class NormedLinear(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.w = nn.Parameter(torch.empty(i, o).uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5))

    def forward(self, x):
        return F.normalize(x, dim=1) @ F.normalize(self.w, dim=0)


class ResNet32(nn.Module):
    def __init__(self, C, normed=False):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, 1, 1, bias=False); self.bn = nn.BatchNorm2d(16)
        layers, i = [], 16
        for o, s in [(16, 1), (32, 2), (64, 2)]:
            for k in range(5):
                layers.append(Block(i, o, s if k == 0 else 1)); i = o
        self.layers = nn.Sequential(*layers)
        self.fc = NormedLinear(64, C) if normed else nn.Linear(64, C)

    def forward(self, x):
        x = self.layers(F.relu(self.bn(self.conv(x))))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


# ---------------- losses ----------------
def cb_weights(counts, beta=0.9999):
    w = (1 - beta) / (1 - np.power(beta, counts))
    return w / w.sum() * len(counts)


def make_loss(name, counts, pa_w=None):
    C = len(counts); pi = counts / counts.sum()
    logpi = torch.tensor(np.log(pi), dtype=torch.float32, device=DEV)
    inv = 1.0 / counts; inv = inv / inv.sum() * C
    m = 1.0 / np.power(counts, 0.25); m = torch.tensor(m * (0.5 / m.max()), dtype=torch.float32, device=DEV)

    def wt(w):
        return torch.tensor(w, dtype=torch.float32, device=DEV)

    def focal(z, y, gamma, w=None):
        logp = F.log_softmax(z, 1).gather(1, y[:, None]).squeeze(1)
        l = -((1 - logp.exp()) ** gamma) * logp
        if w is not None:
            l = l * w[y]
        return l.mean()

    def fn(z, y, epoch, drw_epoch):
        late = epoch >= drw_epoch
        if name == "ce":
            return F.cross_entropy(z, y)
        if name == "focal":
            return focal(z, y, 1.0)
        if name == "wce_drw":
            return F.cross_entropy(z, y, weight=wt(inv) if late else None)
        if name == "cb_drw":
            return F.cross_entropy(z, y, weight=wt(cb_weights(counts)) if late else None)
        if name == "la":
            return F.cross_entropy(z + logpi, y)
        if name == "ldam_drw":
            zm = z - F.one_hot(y, C) * m[None, :]
            return F.cross_entropy(30.0 * zm, y, weight=wt(cb_weights(counts)) if late else None)
        if name == "pa_ce_drw":
            return F.cross_entropy(z, y, weight=wt(pa_w) if late else None)
        if name == "paf_drw":
            return focal(z, y, 1.0, wt(pa_w) if late else None)
        raise ValueError(name)
    return fn


def augment(xb):
    """xb: uint8 NHWC on GPU -> normalized float NCHW with random crop (pad 4) and flip."""
    x = xb.permute(0, 3, 1, 2).float() / 255.0
    n = x.shape[0]
    x = F.pad(x, (4, 4, 4, 4))
    i = torch.randint(0, 9, (n,), device=x.device); j = torch.randint(0, 9, (n,), device=x.device)
    ar = torch.arange(32, device=x.device)
    rows = (i[:, None] + ar[None, :])[:, None, :, None].expand(n, 3, 32, 40)
    cols = (j[:, None] + ar[None, :])[:, None, None, :].expand(n, 3, 32, 32)
    x = x.gather(2, rows).gather(3, cols)
    flip = torch.rand(n, device=x.device) < 0.5
    x[flip] = x[flip].flip(3)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def prep_test(xb):
    x = xb.permute(0, 3, 1, 2).float() / 255.0
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--loss", default="ce")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--out", default="results_mc")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    out = os.path.join(a.out, f"{a.dataset}_{a.loss}_s{a.seed}.npz")
    if os.path.exists(out):
        return
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    Xtr, ytr, Xte, yte, counts, C = load(a.dataset)
    pa_w = None
    if a.loss.startswith("pa"):
        pa_w = np.array(json.load(open(os.path.join(a.out, f"{a.dataset}_profile.json")))["pa_weights"])
        pa_w = pa_w / pa_w.sum() * C
    lossf = make_loss(a.loss, counts, pa_w)
    model = ResNet32(C, normed=(a.loss == "ldam_drw")).to(DEV).to(memory_format=torch.channels_last)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4, nesterov=False)
    Xg = torch.tensor(Xtr, device=DEV); yg = torch.tensor(ytr, device=DEV, dtype=torch.long)
    n = len(yg); bs = 128; steps = (n + bs - 1) // bs
    drw = int(0.8 * a.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=DEV.type == "cuda")
    t0 = time.time()
    for ep in range(a.epochs):
        lr = 0.1 * (ep + 1) / 5 if ep < 5 else 0.1 * (0.01 if ep >= int(0.9 * a.epochs) else (0.1 if ep >= drw else 1.0))
        for gq in opt.param_groups:
            gq["lr"] = lr
        model.train()
        perm = torch.randperm(n, device=DEV)
        for k in range(steps):
            b = perm[k * bs:(k + 1) * bs]
            with torch.autocast(device_type=DEV.type, dtype=torch.float16, enabled=DEV.type == "cuda"):
                z = model(augment(Xg[b]).contiguous(memory_format=torch.channels_last))
            loss = lossf(z.float(), yg[b], ep, drw)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    model.eval(); Z = []
    Xt = torch.tensor(Xte, device=DEV)
    with torch.no_grad():
        for k in range(0, len(Xt), 1000):
            Z.append(model(prep_test(Xt[k:k + 1000]).contiguous(memory_format=torch.channels_last)).float().cpu().numpy())
    Z = np.concatenate(Z)
    np.savez_compressed(out, logits=Z.astype(np.float16), y=yte.astype(np.int16), counts=counts, secs=time.time() - t0)
    print(out, f"acc={(Z.argmax(1) == yte).mean():.4f}", f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
