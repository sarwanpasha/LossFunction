"""Multi-class difficulty profile for CIFAR-LT.

1. Frozen ImageNet ResNet-18 features of the long-tailed training set and the test set.
2. Calibrated multinomial logistic-regression probe: 5-fold cross-fitted on the training
   set (training profile) and fit on the full training set (test profile).
3. Profile-adapted class weights: coordinate ascent on the predicted macro-F1 of the rule
   argmax_y a_y eta_y under the balanced test prior, started from a_y = 1/pi_y.
writes results_mc/<dataset>_profile.json and <dataset>_profile.npz

usage: python mc_profile.py --dataset cifar10
"""
import argparse, json, os, time
import numpy as np


def feats(X, bs=64, res=160, sleep=0.02):
    import torch, torchvision
    import torchvision.transforms.functional as TF
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity(); m = m.to(dev).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for k in range(0, len(X), bs):
            x = torch.tensor(X[k:k + bs], device=dev).permute(0, 3, 1, 2).float() / 255.0
            x = torch.nn.functional.interpolate(x, size=res, mode="bilinear", align_corners=False)
            out.append(m((x - mean) / std).float().cpu().numpy())
            if sleep:
                time.sleep(sleep)
    return np.concatenate(out)


def probe():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))


def rule(eta_bal, b):
    return np.argmax(eta_bal * b[None, :], 1)


def pred_metrics(eta_bal, b):
    """Predicted balanced accuracy and macro-F1 of argmax_y b_y eta_bal_y."""
    n, C = eta_bal.shape
    h = rule(eta_bal, b)
    acc = float(eta_bal[np.arange(n), h].mean())
    f1 = []
    for y in range(C):
        tp = eta_bal[h == y, y].sum(); p = (h == y).sum(); t = eta_bal[:, y].sum()
        f1.append(2 * tp / max(p + t, 1e-12))
    return acc, float(np.mean(f1))


def pa_weights(eta_bal, sweeps=3, grid=(0.5, 0.7, 0.85, 1.0, 1.2, 1.4, 2.0)):
    C = eta_bal.shape[1]
    b = np.ones(C); best = pred_metrics(eta_bal, b)[1]
    for _ in range(sweeps):
        for y in range(C):
            cur = b[y]; bb, bv = cur, best
            for g in grid:
                b[y] = cur * g
                v = pred_metrics(eta_bal, b)[1]
                if v > bv + 1e-9:
                    bb, bv = b[y], v
            b[y] = bb; best = bv
    return b, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--out", default="results_mc")
    a = ap.parse_args()
    from mc_lt import load
    from sklearn.model_selection import StratifiedKFold
    os.makedirs(a.out, exist_ok=True)
    Xtr, ytr, Xte, yte, counts, C = load(a.dataset)
    pi = counts / counts.sum()
    fc = os.path.join(a.out, f"{a.dataset}_feats.npz")
    if not os.path.exists(fc):
        np.savez(fc, Ftr=feats(Xtr), Fte=feats(Xte))
    d = np.load(fc); Ftr, Fte = d["Ftr"], d["Fte"]
    eta_tr = np.zeros((len(ytr), C))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Ftr, ytr):
        eta_tr[te] = probe().fit(Ftr[tr], ytr[tr]).predict_proba(Ftr[te])
    eta_te = probe().fit(Ftr, ytr).predict_proba(Fte)
    bal = lambda e: (e / pi[None, :]) / (e / pi[None, :]).sum(1, keepdims=True)
    etr_b, ete_b = bal(eta_tr), bal(eta_te)
    b, f1b = pa_weights(etr_b)
    a_w = b / pi                       # weights on the training-prior posterior
    res = dict(dataset=a.dataset, counts=counts.tolist(), pa_weights=(a_w / a_w.sum() * C).tolist(),
               pa_b=b.tolist(), pred_train_macro_f1=f1b,
               probe_test_acc=float((eta_te.argmax(1) == yte).mean()),
               probe_test_bal_acc=float((ete_b.argmax(1) == yte).mean()))
    json.dump(res, open(os.path.join(a.out, f"{a.dataset}_profile.json"), "w"))
    np.savez_compressed(os.path.join(a.out, f"{a.dataset}_profile.npz"), eta_te_bal=ete_b.astype(np.float32))
    print(json.dumps({k: v for k, v in res.items() if k not in ("counts", "pa_weights", "pa_b")}), flush=True)


if __name__ == "__main__":
    main()
