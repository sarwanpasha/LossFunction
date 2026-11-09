"""
lf_core.py -- Difficulty-profile theory of loss rankings.

Components
  * LOSSES: a pool of imbalance-aware binary losses (weighted BCE, focal, logit
    adjustment) + our profile-adapted losses (PAF, PA-BCE) and the online
    self-calibrating focal loss (SCFL).
  * implied_threshold(loss): eta-threshold t_l at which the population minimiser of
    the loss switches sign (computed numerically for any loss).
  * f1_from_profile(eta, t): F1 of the Bayes-l classifier predicted from a
    difficulty profile (sample of eta = P(y=1|x)).
  * Probe: cross-fitted, isotonic-calibrated gradient boosting estimate of eta.
  * Generators: gauss / copula / gmm (marginal-or-moment matched baselines),
    DMS-T (difficulty-matched synthesis by monotone transport, ours),
    DMS-L (latent difficulty-matched synthesis, needs only the 1-D profile, ours),
    DMS-L-DP (eps-DP histogram release of the profile, ours).
"""
import math
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit, logit
from scipy.stats import norm

# --------------------------------------------------------------------------
# Loss specifications (pure-python description; torch impl in train.py)
# --------------------------------------------------------------------------

def make_loss_pool(pi):
    """pi = positive prevalence of the training set."""
    pool = {
        "bce": dict(kind="wbce", w=1.0),
        "wbce3": dict(kind="wbce", w=3.0),
        "wbce_bal": dict(kind="wbce", w=(1 - pi) / pi),
        "focal_a25_g2": dict(kind="focal", alpha=0.25, gamma=2.0),
        "focal_a50_g2": dict(kind="focal", alpha=0.50, gamma=2.0),
        "focal_a75_g2": dict(kind="focal", alpha=0.75, gamma=2.0),
        "focal_a50_g1": dict(kind="focal", alpha=0.50, gamma=1.0),
        "la_t1": dict(kind="la", tau=1.0, pi=pi),
        "la_t05": dict(kind="la", tau=0.5, pi=pi),
    }
    return pool


def conditional_risk(kind, z, eta, **kw):
    """Population conditional risk C(z; eta) of a logit z when P(y=1|x)=eta."""
    p = expit(z)
    lp, lq = -np.log(max(p, 1e-300)), -np.log(max(1 - p, 1e-300))
    if kind == "wbce":
        return kw["w"] * eta * lp + (1 - eta) * lq
    if kind == "focal":
        a, g = kw["alpha"], kw["gamma"]
        return a * eta * (1 - p) ** g * lp + (1 - a) * (1 - eta) * p ** g * lq
    if kind == "la":
        za = z + kw["tau"] * math.log(kw["pi"] / (1 - kw["pi"]))
        pa = expit(za)
        return eta * -np.log(max(pa, 1e-300)) + (1 - eta) * -np.log(max(1 - pa, 1e-300))
    raise ValueError(kind)


def bayes_logit(kind, eta, **kw):
    r = minimize_scalar(lambda z: conditional_risk(kind, z, eta, **kw),
                        bounds=(-40, 40), method="bounded", options={"xatol": 1e-9})
    return r.x


def implied_threshold(spec):
    """t_l such that the population minimiser predicts positive iff eta > t_l."""
    kind = spec["kind"]
    kw = {k: v for k, v in spec.items() if k != "kind"}
    if kind == "wbce":
        return 1.0 / (1.0 + kw["w"])
    if kind == "focal":
        return 1.0 - kw["alpha"]
    if kind == "la":
        return float(expit(kw["tau"] * logit(kw["pi"])))
    f = lambda e: bayes_logit(kind, e, **kw)
    return brentq(f, 1e-6, 1 - 1e-6)


def implied_threshold_numeric(spec):
    kind = spec["kind"]
    kw = {k: v for k, v in spec.items() if k != "kind"}
    return brentq(lambda e: bayes_logit(kind, e, **kw), 1e-6, 1 - 1e-6, xtol=1e-7)

# --------------------------------------------------------------------------
# Profile -> predicted metrics
# --------------------------------------------------------------------------

def f1_from_profile(eta, t):
    eta = np.asarray(eta, float)
    pred = eta > t
    tp = eta[pred].sum()
    npos = eta.sum()
    npred = pred.sum()
    if npos + npred == 0:
        return 0.0
    return float(2 * tp / (npos + npred))


def recall_precision_from_profile(eta, t):
    eta = np.asarray(eta, float)
    pred = eta > t
    tp = eta[pred].sum()
    rec = tp / max(eta.sum(), 1e-12)
    prec = tp / max(pred.sum(), 1e-12) if pred.sum() > 0 else 0.0
    return float(rec), float(prec)


def optimal_threshold(eta, grid=None):
    """argmax_t F1(t; profile). Also returns the F1*/2 fixed-point check."""
    eta = np.sort(np.asarray(eta, float))
    if grid is None:
        grid = np.unique(np.quantile(eta, np.linspace(0, 1, 2001)))
        grid = np.concatenate([[0.0], grid[:-1]])
    vals = np.array([f1_from_profile(eta, t) for t in grid])
    i = int(np.argmax(vals))
    return float(grid[i]), float(vals[i])


def fixed_point_threshold(eta, t0=0.5, iters=100, tol=1e-7):
    """Iterate t <- F1(t)/2 (self-calibration map). Returns trajectory."""
    traj = [t0]
    t = t0
    for _ in range(iters):
        tn = f1_from_profile(eta, t) / 2.0
        traj.append(tn)
        if abs(tn - t) < tol:
            break
        t = tn
    return traj

# --------------------------------------------------------------------------
# Probe: cross-fitted calibrated estimate of eta
# --------------------------------------------------------------------------

def fit_probe(X, y, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.calibration import CalibratedClassifierCV
    base = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                          early_stopping=True, random_state=seed)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(X, y)
    return clf


def crossfit_eta(X, y, k=5, seed=0):
    from sklearn.model_selection import StratifiedKFold
    eta = np.zeros(len(y))
    skf = StratifiedKFold(k, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        eta[te] = fit_probe(X[tr], y[tr], seed).predict_proba(X[te])[:, 1]
    return np.clip(eta, 1e-6, 1 - 1e-6)

# --------------------------------------------------------------------------
# Generators. Each returns (X, y) with n rows.
# --------------------------------------------------------------------------

def _class_counts(n, pi, rng):
    n1 = rng.binomial(n, pi)
    return n - n1, n1


def gen_gauss(X, y, n, rng):
    """Class-conditional Gaussian matching first two moments (the 'calibrated
    to marginal statistics' synthetic benchmark)."""
    pi = y.mean()
    n0, n1 = _class_counts(n, pi, rng)
    out = []
    for c, nc in ((0, n0), (1, n1)):
        Xc = X[y == c]
        mu = Xc.mean(0)
        cov = np.cov(Xc, rowvar=False) + 1e-3 * np.eye(X.shape[1])
        out.append(rng.multivariate_normal(mu, cov, size=nc, method="eigh"))
    Xs = np.vstack(out)
    ys = np.r_[np.zeros(n0), np.ones(n1)]
    p = rng.permutation(n)
    return Xs[p], ys[p]


def _copula_fit(Xc):
    n, d = Xc.shape
    ranks = np.argsort(np.argsort(Xc, axis=0), axis=0)
    U = (ranks + 0.5) / n
    Z = norm.ppf(U)
    C = np.corrcoef(Z, rowvar=False) if d > 1 else np.ones((1, 1))
    C = np.nan_to_num(C) + 1e-4 * np.eye(d)
    np.fill_diagonal(C, 1.0)
    return np.sort(Xc, axis=0), C


def _copula_sample(fit, m, rng):
    Xsorted, C = fit
    n, d = Xsorted.shape
    Z = rng.multivariate_normal(np.zeros(d), C, size=m, method="eigh")
    U = norm.cdf(Z)
    idx = np.clip((U * n).astype(int), 0, n - 1)
    return np.take_along_axis(Xsorted, idx, axis=0)


def gen_copula(X, y, n, rng):
    """Class-conditional Gaussian copula with exact empirical marginals."""
    pi = y.mean()
    n0, n1 = _class_counts(n, pi, rng)
    Xs = np.vstack([_copula_sample(_copula_fit(X[y == 0]), n0, rng),
                    _copula_sample(_copula_fit(X[y == 1]), n1, rng)])
    ys = np.r_[np.zeros(n0), np.ones(n1)]
    p = rng.permutation(n)
    return Xs[p], ys[p]


def gen_gmm(X, y, n, rng):
    from sklearn.mixture import GaussianMixture
    pi = y.mean()
    n0, n1 = _class_counts(n, pi, rng)
    out = []
    for c, nc in ((0, n0), (1, n1)):
        Xc = X[y == c]
        k = int(max(1, min(5, len(Xc) // 50)))
        g = GaussianMixture(k, covariance_type="full", reg_covar=1e-3,
                            random_state=int(rng.integers(1 << 30))).fit(Xc)
        out.append(g.sample(nc)[0] if nc > 0 else np.zeros((0, X.shape[1])))
    Xs = np.vstack(out)
    ys = np.r_[np.zeros(n0), np.ones(n1)]
    p = rng.permutation(n)
    return Xs[p], ys[p]


def gen_dms_transport(X, y, n, rng, eta_profile, probe):
    """DMS-T. Features are drawn from a class-conditional Gaussian copula of X (its labels
    are discarded); the posterior is set by monotone (quantile) transport of probe scores
    onto the real difficulty profile, and labels are drawn from it. The synthetic data
    then has a posterior whose law equals the empirical real profile, while keeping
    realistic feature marginals and correlations."""
    Xs, _ = gen_copula(X, y, n, rng)
    s = probe.predict_proba(Xs)[:, 1] + 1e-9 * rng.standard_normal(n)
    r = np.argsort(np.argsort(s))
    q = np.sort(eta_profile)
    eta_s = q[np.clip(((r + 0.5) / n * len(q)).astype(int), 0, len(q) - 1)]
    ys = (rng.random(n) < eta_s).astype(float)
    return Xs, ys, eta_s


class LatentMap:
    """Random invertible nonlinear embedding R^d -> R^d."""
    def __init__(self, d, rng):
        self.Q1 = np.linalg.qr(rng.standard_normal((d, d)))[0]
        self.Q2 = np.linalg.qr(rng.standard_normal((d, d)))[0]
        self.b = 0.5 * rng.standard_normal(d)

    def __call__(self, H):
        U = H @ self.Q1 + self.b
        U = U + 0.5 * np.tanh(2 * U)          # strictly monotone => invertible
        return U @ self.Q2


def profile_histogram(eta, bins=50, eps=None, rng=None):
    """Histogram of eta on a logit-uniform grid; optional eps-DP Laplace release
    (add/remove-one neighbouring datasets => L1 sensitivity 1)."""
    edges = expit(np.linspace(-9, 9, bins + 1))
    edges[0], edges[-1] = 0.0, 1.0
    h = np.histogram(eta, edges)[0].astype(float)
    if eps is not None:
        h = h + rng.laplace(0, 1.0 / eps, size=h.shape)
        h = np.clip(h, 0, None)
        if h.sum() <= 0:
            h = np.ones_like(h)
    return edges, h / h.sum()


def sample_profile(edges, probs, n, rng):
    b = rng.choice(len(probs), size=n, p=probs)
    lo, hi = logit(np.clip(edges[b], 1e-6, 1 - 1e-6)), logit(np.clip(edges[b + 1], 1e-6, 1 - 1e-6))
    return expit(lo + (hi - lo) * rng.random(n))


def gen_dms_latent(d, n, rng, edges, probs, scale=1.0):
    """DMS-L (ours). Needs only the 1-D difficulty histogram. Bayes posterior of
    the generated data is exactly eta ~ histogram (x is an invertible function of
    (logit eta, nuisance z), and y is independent of z given eta)."""
    d = max(d, 4)
    eta = sample_profile(edges, probs, n, rng)
    H = np.c_[logit(np.clip(eta, 1e-6, 1 - 1e-6)) * scale, rng.standard_normal((n, d - 1))]
    Xs = LatentMap(d, np.random.default_rng(12345))(H)   # map fixed per dataset
    ys = (rng.random(n) < eta).astype(float)
    return Xs, ys, eta
