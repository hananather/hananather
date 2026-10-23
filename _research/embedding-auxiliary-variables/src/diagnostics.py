"""Stage-1 population diagnostics: how much could each auxiliary set reduce variance, at best?

For a fixed design (stratified SRSWOR) and a model m fitted on the WHOLE population, the
difference estimator has anticipated variance
    AV(m) = sum_h N_h^2 (1 - f_h) / n_h * S^2_{y-m, h}.
AV(HT) / AV(m) is the large-sample relative efficiency a model-assisted estimator built on that
auxiliary set can approach. It is cheap (no Monte Carlo) and is used to (i) screen y variables and
feature sets before the expensive Monte Carlo, and (ii) pre-register expected gains: Monte Carlo
gains far above it indicate leakage or oracle tuning, far below it overfitting.

Models are fitted by weighted least squares with weights d_k - 1 (the weights that minimise AV),
with K-fold cross-fitting over the population so that p ~ 1000 features cannot overfit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def anticipated_var(E: np.ndarray, design) -> np.ndarray:
    E = np.atleast_2d(E.T).T
    v = np.zeros(E.shape[1])
    for h in range(design.H):
        nh, Nh = design.n_h[h], design.N_h[h]
        if nh >= Nh or nh == 0:
            continue
        e = E[design.strata == h]
        v += Nh ** 2 * (1 - nh / Nh) / nh * e.var(axis=0, ddof=1)
    return v


def wls_ridge_oof(F, Y, w, lam, unpen, folds):
    """Out-of-fold predictions of a weighted ridge (first `unpen` columns unpenalised)."""
    out = np.zeros_like(Y, dtype=float)
    for f in np.unique(folds):
        tr = (folds != f) & (w > 0)
        te = folds == f
        sw = np.sqrt(w[tr])[:, None]
        mu = np.average(F[tr], axis=0, weights=w[tr]); ymu = np.average(Y[tr], axis=0, weights=w[tr])
        Fc = (F[tr] - mu) * sw
        sd = Fc.std(0) + 1e-12
        Fc = Fc / sd
        pen = np.full(F.shape[1], lam * len(Fc))
        pen[:unpen] = 1e-8 * len(Fc)
        B = np.linalg.solve(Fc.T @ Fc + np.diag(pen), Fc.T @ ((Y[tr] - ymu) * sw))
        out[te] = ymu + ((F[te] - mu) / sd) @ B
    return out


def evaluate_feature_sets(Y: np.ndarray, ynames, design, feature_sets: dict, lams=(1e-4, 1e-3, 1e-2, 1e-1),
                          k_folds=5, seed=0, baseline="strong"):
    """feature_sets: name -> (F matrix, n_unpenalised_cols). Returns long DataFrame."""
    d = 1 / design.pi
    w = d - 1
    folds = np.random.default_rng(seed).integers(0, k_folds, len(Y))
    av_ht = anticipated_var(Y, design)
    rows, av = [], {}
    for name, (F, unpen) in feature_sets.items():
        best = None
        for lam in (lams if F.shape[1] > unpen else (0.0,)):
            M = wls_ridge_oof(F, Y, w, lam, unpen, folds)
            a = anticipated_var(Y - M, design)
            best = a if best is None else np.minimum(best, a)  # per-y best lambda (ceiling)
        av[name] = best
    for name, a in av.items():
        for j, yn in enumerate(ynames):
            rows.append({"y": yn, "features": name, "p": feature_sets[name][0].shape[1],
                         "re_vs_ht": av_ht[j] / a[j],
                         "re_vs_baseline": av[baseline][j] / a[j] if baseline in av else np.nan,
                         "cv_pct_at_design": 100 * np.sqrt(a[j]) / Y[:, j].sum()})
    return pd.DataFrame(rows)
