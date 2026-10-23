"""Frozen (previous-wave) prediction models: the realistic deployment pattern.

Wave t-1 (frame TY2021 -> survey TY2022): draw a production-design sample, fit a model of y on frame
variables (+ text embeddings) using only that sample, then freeze it. Wave t (frame TY2022 -> survey
TY2023): apply the frozen model to every frame unit. The difference estimator built on it is exactly
design-unbiased for wave t, because the model does not depend on the wave-t sample (it is PPI with
a model trained on independent labelled data).

Several independent previous-wave samples are drawn so that results average over the luck of the
previous wave's sample; the Monte Carlo cycles through them (spec kind 'frozen', M has shape (k, N, q)).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from frame import production_design
from mc import _ridge_cv_lambda, _ridge_fit


def _cats(a: pd.Series, b: pd.Series):
    return sorted(set(a.astype(str)) | set(b.astype(str)))


def frame_features(pop, cats):
    parts = []
    for col, levels in cats.items():
        v = pop[col].astype(str)
        parts.append(np.stack([(v == l).to_numpy(float) for l in levels[1:]], 1))
    lag = pop[["lag_revenue", "lag_expenses", "lag_assets", "lag_employees"]].clip(lower=0).to_numpy(float)
    parts.append(np.log1p(lag))
    return np.concatenate(parts, 1)


def pca_basis(E, k):
    mu = E.mean(0)
    _, _, Vt = np.linalg.svd(E - mu, full_matrices=False)
    Z = (E - mu) @ Vt[:k].T
    return mu, Vt[:k], Z.std(0)


def train_frozen(prev_pop, curr_pop, ynames, prev_E=None, curr_E=None, n=1000, k_draws=10, K=32, seed=0,
                 ratio=True):
    """Returns (N_curr, q) arrays stacked as (k_draws, N_curr, q) for a text-free and a text model."""
    cats = {c: _cats(prev_pop[c], curr_pop[c]) for c in ["ntee_major10", "state"]}
    cats["bmf_income_cd"] = _cats(prev_pop.bmf_income_cd.fillna(-1), curr_pop.bmf_income_cd.fillna(-1))
    pp, cp = prev_pop.copy(), curr_pop.copy()
    pp["bmf_income_cd"] = pp.bmf_income_cd.fillna(-1); cp["bmf_income_cd"] = cp.bmf_income_cd.fillna(-1)
    Fp, Fc = frame_features(pp, cats), frame_features(cp, cats)
    feats = {"frame": (Fp, Fc)}
    if prev_E is not None:
        mu, V, sd = pca_basis(np.asarray(prev_E, float), K)   # basis fixed on the previous wave
        Zp = (np.asarray(prev_E, float) - mu) @ V.T / sd
        Zc = (np.asarray(curr_E, float) - mu) @ V.T / sd
        feats["text"] = (np.c_[Fp, Zp], np.c_[Fc, Zc])
    des, _ = production_design(prev_pop, n)
    d = 1 / des.pi
    ta = (des.n_h >= des.N_h)[des.strata]
    Yp = prev_pop[ynames].to_numpy(float)
    xp = prev_pop.lag_expenses.clip(lower=0).to_numpy(float) + 1
    xc = curr_pop.lag_expenses.clip(lower=0).to_numpy(float) + 1
    is_bin = np.array([y.startswith("y_bin_") for y in ynames])
    rng = np.random.default_rng(seed)
    out = {k: np.zeros((k_draws, len(curr_pop), len(ynames))) for k in feats}
    for b in range(k_draws):
        s = des.draw(rng)
        s = s[~ta[s]]                     # model the take-some part, as in the current-wave estimators
        for name, (F_prev, F_curr) in feats.items():
            for j_set, use_ratio in ((~is_bin, ratio), (is_bin, False)):
                j = np.flatnonzero(j_set)
                if len(j) == 0:
                    continue
                if use_ratio:
                    T, w = Yp[s][:, j] / xp[s][:, None], d[s] * xp[s]
                else:
                    T, w = Yp[s][:, j], d[s]
                lam = _ridge_cv_lambda(F_prev[s], T, w, 0, rng=rng)
                b0, B = _ridge_fit(F_prev[s], T, w, lam, 0)
                pred = b0 + F_curr @ B
                out[name][b][:, j] = pred * xc[:, None] if use_ratio else pred
    return out
