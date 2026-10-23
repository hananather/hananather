"""Design-based survey estimation toolkit for the embeddings-as-auxiliaries benchmark.

Everything is vectorised over several target variables at once: Y is an (N, q) matrix.

Designs
    stratified SRSWOR with optional take-all strata (SRSWOR = one stratum).
Estimators (all return totals; divide by N for means)
    ht            Horvitz-Thompson / expansion estimator
    greg          GREG = linear calibration on auxiliary matrix X (with known population totals)
    ridge_greg    GREG with ridge-penalised coefficients (penalised calibration); penalty only on
                  the "soft" columns so that the "hard" columns are still (essentially) calibrated
    difference    model-assisted difference estimator  sum_U m(x) + sum_s d (y - m(x)),
                  optionally with cross-fitted predictions for the sampled units
Variance
    residual-based linearisation variance for stratified SRSWOR, v = sum_h N_h^2 (1-f_h) s^2_{e,h} / n_h
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------------------
# Designs
# --------------------------------------------------------------------------------------


class StratifiedDesign:
    """Stratified SRSWOR. strata: int array (N,), n_h: allocation per stratum label 0..H-1."""

    def __init__(self, strata: np.ndarray, n_h: np.ndarray):
        self.strata = np.asarray(strata, dtype=int)
        self.H = int(self.strata.max()) + 1
        self.N_h = np.bincount(self.strata, minlength=self.H)
        self.n_h = np.asarray(n_h, dtype=int)
        assert self.n_h.shape == (self.H,)
        assert np.all(self.n_h <= self.N_h) and np.all(self.n_h >= 0)
        self.members = [np.flatnonzero(self.strata == h) for h in range(self.H)]
        self.pi = (self.n_h / np.maximum(self.N_h, 1))[self.strata]
        self.n = int(self.n_h.sum())

    def draw(self, rng: np.random.Generator) -> np.ndarray:
        idx = [rng.choice(m, size=k, replace=False) if k < len(m) else m
               for m, k in zip(self.members, self.n_h) if k > 0]
        return np.sort(np.concatenate(idx))


def srswor(N: int, n: int) -> StratifiedDesign:
    return StratifiedDesign(np.zeros(N, dtype=int), np.array([n]))


def neyman_allocation(strata: np.ndarray, x: np.ndarray, n: int, min_n: int = 2,
                      take_all: np.ndarray | None = None) -> np.ndarray:
    """Neyman allocation on a size variable x, with iterative take-all for strata whose
    allocation would exceed N_h. take_all: boolean mask over stratum labels forced to f=1."""
    strata = np.asarray(strata, dtype=int)
    H = strata.max() + 1
    N_h = np.bincount(strata, minlength=H).astype(float)
    S_h = np.array([x[strata == h].std(ddof=1) if N_h[h] > 1 else 0.0 for h in range(H)])
    fixed = np.zeros(H, dtype=bool) if take_all is None else take_all.copy()
    n_h = np.zeros(H)
    for _ in range(100):
        n_h[fixed] = N_h[fixed]
        rem = n - n_h[fixed].sum()
        free = ~fixed
        w = N_h[free] * S_h[free]
        alloc = rem * w / w.sum() if w.sum() > 0 else rem * N_h[free] / N_h[free].sum()
        over = alloc >= N_h[free]
        if not over.any():
            n_h[free] = alloc
            break
        fixed[np.flatnonzero(free)[over]] = True
    n_h = np.floor(n_h).astype(int)
    n_h = np.minimum(np.maximum(n_h, np.minimum(min_n, N_h.astype(int))), N_h.astype(int))
    # distribute rounding remainder to the largest fractional parts among free strata
    short = n - n_h.sum()
    order = np.argsort(-(N_h * S_h) * (n_h < N_h))
    i = 0
    while short > 0 and i < 10 * H:
        h = order[i % H]
        if n_h[h] < N_h[h]:
            n_h[h] += 1
            short -= 1
        i += 1
    while short < 0:  # min_n pushed us over budget: trim the largest free strata
        h = np.argmax(np.where(n_h > min_n, n_h * (n_h < N_h), -1))
        n_h[h] -= 1
        short += 1
    return n_h


def cum_sqrt_f_boundaries(z: np.ndarray, n_classes: int, bins: int = 200) -> np.ndarray:
    """Dalenius-Hodges cum-sqrt(f) class boundaries for stratifying on z."""
    hist, edges = np.histogram(z, bins=bins)
    c = np.cumsum(np.sqrt(hist))
    targets = c[-1] * np.arange(1, n_classes) / n_classes
    return edges[1:][np.searchsorted(c, targets)]


# --------------------------------------------------------------------------------------
# Estimators (one sample s; Y_s (n,q); d_s (n,))
# --------------------------------------------------------------------------------------


def ht_total(Y_s, d_s):
    return d_s @ Y_s


def strat_var_total(E_s, design: StratifiedDesign, s: np.ndarray):
    """Linearisation variance of a total from (residual) values E_s on the sample, stratified SRSWOR."""
    E_s = np.atleast_2d(E_s.T).T
    h_s = design.strata[s]
    v = np.zeros(E_s.shape[1])
    for h in np.unique(h_s):
        nh, Nh = design.n_h[h], design.N_h[h]
        if nh >= Nh or nh < 2:
            continue
        e = E_s[h_s == h]
        v += Nh ** 2 * (1 - nh / Nh) * e.var(axis=0, ddof=1) / nh
    return v


def _solve_weighted(X_s, d_s, Y_s, penalty=None, ridge_eps=1e-8):
    XtD = X_s.T * d_s
    A = XtD @ X_s
    if penalty is not None:
        A = A + np.diag(penalty)
    # tiny per-column (scale-invariant) ridge for numerical safety; a global trace-scaled
    # epsilon would swamp small-scale columns (dummies, PCs) next to dollar-valued ones
    A = A + ridge_eps * np.diag(np.maximum(np.diag(A), 1e-300))
    return np.linalg.solve(A, XtD @ Y_s)


def greg_total(Y_s, d_s, X_s, T_X, penalty=None):
    """GREG / linear calibration estimator. Returns (totals (q,), residuals (n,q), B)."""
    B = _solve_weighted(X_s, d_s, Y_s, penalty)
    t = d_s @ Y_s + (T_X - d_s @ X_s) @ B
    E = Y_s - X_s @ B
    return t, E, B


def calibration_weights(X_s, d_s, T_X, penalty=None):
    """Linear (chi-square distance) calibration weights; with penalty -> ridge calibration."""
    XtD = X_s.T * d_s
    A = XtD @ X_s
    if penalty is not None:
        A = A + np.diag(penalty)
    lam = np.linalg.solve(A + 1e-10 * np.eye(A.shape[0]), T_X - d_s @ X_s)
    return d_s * (1 + X_s @ lam)


def difference_total(Y_s, d_s, M_s, sum_M_U):
    """Difference estimator: sum_U m + sum_s d (y - m). M_s (n,q) predictions for sampled units,
    sum_M_U (q,) population sum of predictions. Returns (totals, residuals)."""
    E = Y_s - M_s
    return sum_M_U + d_s @ E, E


# --------------------------------------------------------------------------------------
# Ridge prediction with cross-fitting (closed form via SVD, many lambdas, many targets)
# --------------------------------------------------------------------------------------


def ridge_fit_predict(F_tr, Y_tr, w_tr, F_te_list, lam, unpen=0):
    """Weighted ridge with intercept. First `unpen` columns of F get (almost) no penalty.
    Returns list of predictions for each matrix in F_te_list."""
    mu = np.average(F_tr, axis=0, weights=w_tr)
    ymu = np.average(Y_tr, axis=0, weights=w_tr)
    Fc = (F_tr - mu) * np.sqrt(w_tr)[:, None]
    Yc = (Y_tr - ymu) * np.sqrt(w_tr)[:, None]
    p = F_tr.shape[1]
    pen = np.full(p, lam, dtype=float)
    pen[:unpen] = lam * 1e-6
    A = Fc.T @ Fc + np.diag(pen)
    B = np.linalg.solve(A, Fc.T @ Yc)
    return [ymu + (F - mu) @ B for F in F_te_list]


def crossfit_predictions(F_s, Y_s, w_s, F_U, lam, k_folds=5, rng=None, unpen=0, folds=None):
    """K-fold cross-fitting on the sample. Returns (M_s cross-fitted (n,q), M_U averaged over
    fold models (N,q)). If k_folds <= 1, uses in-sample fit for both (overfitting-prone)."""
    n = F_s.shape[0]
    if k_folds <= 1:
        M_s, M_U = ridge_fit_predict(F_s, Y_s, w_s, [F_s, F_U], lam, unpen)
        return M_s, M_U
    if folds is None:
        rng = rng or np.random.default_rng(0)
        folds = rng.permutation(n) % k_folds
    M_s = np.zeros((n, Y_s.shape[1]))
    M_U = np.zeros((F_U.shape[0], Y_s.shape[1]))
    for f in range(k_folds):
        tr, te = folds != f, folds == f
        p_te, p_U = ridge_fit_predict(F_s[tr], Y_s[tr], w_s[tr], [F_s[te], F_U], lam, unpen)
        M_s[te] = p_te
        M_U += p_U / k_folds
    return M_s, M_U


# --------------------------------------------------------------------------------------
# Monte Carlo summaries
# --------------------------------------------------------------------------------------


def summarise(est: np.ndarray, var: np.ndarray | None, truth: np.ndarray, z: float = 1.959964):
    """est, var: (R, q). truth: (q,). Returns dict of metric arrays (q,)."""
    err = est - truth
    mse = (err ** 2).mean(0)
    out = {
        "rel_bias_pct": 100 * err.mean(0) / truth,
        "rel_bias_mcse_pct": 100 * err.std(0, ddof=1) / np.sqrt(len(est)) / np.abs(truth),
        "rrmse_pct": 100 * np.sqrt(mse) / np.abs(truth),
        "mse": mse,
        "emp_var": est.var(0, ddof=1),
    }
    if var is not None:
        se = np.sqrt(np.maximum(var, 0))
        out["coverage_pct"] = 100 * (np.abs(err) <= z * se).mean(0)
        out["var_rel_bias_pct"] = 100 * (var.mean(0) / est.var(0, ddof=1) - 1)
        out["ci_rel_width_pct"] = 100 * 2 * z * se.mean(0) / np.abs(truth)
    return out
