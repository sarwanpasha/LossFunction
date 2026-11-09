"""Download imbalanced tabular benchmarks -> data/<name>.npz (X, y in {0,1})."""
import os, sys
import numpy as np

os.makedirs("data", exist_ok=True)
home = os.path.join(os.getcwd(), "dl_cache")
os.makedirs(home, exist_ok=True)

from imblearn.datasets import fetch_datasets
ds = fetch_datasets(data_home=home, verbose=True)
for name, b in ds.items():
    X = np.asarray(b.data, dtype=np.float64)
    y = (np.asarray(b.target) == 1).astype(np.int8)
    np.savez_compressed(f"data/imb_{name}.npz", X=X, y=y)
    print(name, X.shape, round(y.mean(), 4), flush=True)

# credit-card fraud (OpenML 1597)
try:
    from sklearn.datasets import fetch_openml
    b = fetch_openml(data_id=1597, as_frame=False, data_home=home, parser="auto")
    X = np.asarray(b.data, dtype=np.float64)
    y = (np.asarray(b.target).astype(str) == "1").astype(np.int8)
    np.savez_compressed("data/creditcard.npz", X=X, y=y)
    print("creditcard", X.shape, y.mean(), flush=True)
except Exception as e:
    print("creditcard failed", e, file=sys.stderr)
