"""CIFAR-10 one-vs-rest long-tailed binary tasks on frozen ImageNet ResNet-18
features -> data/cifar_lt_<class>.npz with a predefined train/test split
(positives subsampled to prevalence PI in both splits)."""
import os
import numpy as np
import torch, torchvision
import torchvision.transforms as T

PI = 0.02
os.makedirs("data", exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
RES = int(os.environ.get("LF_RES", "224"))
BATCH = int(os.environ.get("LF_BATCH", "512"))
SLEEP = float(os.environ.get("LF_SLEEP", "0"))   # optional pause between batches
torch.set_num_threads(int(os.environ.get("LF_THREADS", "4")))
tf = T.Compose([T.Resize(RES), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
m.fc = torch.nn.Identity(); m = m.to(dev).eval()


def feats(train):
    ds = torchvision.datasets.CIFAR10("dl_cache/cifar", train=train, download=True, transform=tf)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH, num_workers=(0 if os.name == 'nt' else 8))
    F, Y = [], []
    with torch.no_grad():
        for x, y in dl:
            F.append(m(x.to(dev)).float().cpu().numpy()); Y.append(y.numpy())
            if SLEEP:
                import time; time.sleep(SLEEP)
    return np.concatenate(F), np.concatenate(Y)


Ftr, Ytr = feats(True)
Fte, Yte = feats(False)
rng = np.random.default_rng(0)
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
for c, cname in enumerate(classes):
    parts = []
    for F, Y, sp in ((Ftr, Ytr, 0), (Fte, Yte, 1)):
        neg = np.where(Y != c)[0]; pos = np.where(Y == c)[0]
        npos = int(round(PI / (1 - PI) * len(neg)))
        pos = rng.choice(pos, min(npos, len(pos)), replace=False)
        idx = np.r_[neg, pos]
        parts.append((F[idx], (Y[idx] == c).astype(np.int8), np.full(len(idx), sp, np.int8)))
    X = np.vstack([p[0] for p in parts]); y = np.r_[parts[0][1], parts[1][1]]
    split = np.r_[parts[0][2], parts[1][2]]
    np.savez_compressed(f"data/cifar_lt_{cname}.npz", X=X, y=y, split=split)
    print(cname, X.shape, y.mean(), flush=True)
