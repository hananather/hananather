"""Frozen (previous-wave) prediction heads: the realistic deployment pattern (protocol §7.4).

Previous wave (frame TY2021 -> survey TY2022): draw J independent production-design samples, keep the
respondents (TY2022 return filed <= 300 days after FYE) with weighting-class (stratum) adjusted
weights, and fit the y-type learner on FC@T (+ Z_K of E with the PCA rotation fitted on the previous
wave's frame). Current wave (frame TY2022 -> survey TY2023): apply every head to every frame unit.
A difference estimator on a frozen head is exactly design-unbiased (PPI with independent labels).

Leak control (test B5): the 2023-08 BMF snapshot's amounts and income/asset codes describe TY2022 for
most units, i.e. the previous wave's *target* year. For both waves the BMF amounts and size codes used
here are therefore recomputed from that wave's own lagged return.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from frame import production_design
from mc import _ridge_fit, cv_select, make_folds
from protocol import TIER_LAGS, F1_LAGS, bmfmax_codes, _dum

IRS_BRACKETS = [1, 10_000, 25_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000, 50_000_000]


def income_class(v):
    return np.searchsorted(IRS_BRACKETS, np.asarray(v, float), side="right")


def _b5(pop):
    p = pop.copy()
    p["bmf_revenue_amt"] = p.lag_revenue.clip(lower=0)
    p["bmf_income_amt"] = p.lag_revenue.clip(lower=0)
    p["bmf_asset_amt"] = p.lag_assets.clip(lower=0)
    p["bmf_income_cd"] = income_class(p.lag_revenue.clip(lower=0))
    p["bmf_asset_cd"] = income_class(p.lag_assets.clip(lower=0))
    return p


def joint_features(prev, curr, T, prev_E=None, curr_E=None, K=32, min_count=20):
    """FC@T for both waves built on the concatenated frame (aligned dummy columns), plus Z_K."""
    both = pd.concat([_b5(prev), _b5(curr)], ignore_index=True)
    n0 = len(prev)
    av = both.avail_t1.to_numpy(float) if "avail_t1" in both else np.ones(len(both))
    lags = both[TIER_LAGS[T]].clip(lower=0).to_numpy(float).copy()
    for j, c in enumerate(TIER_LAGS[T]):
        if c not in F1_LAGS:
            lags[:, j] *= av
    cat_part = np.c_[_dum(both.ntee_major10), _dum(both.state), _dum(both.bmf_income_cd), 1 - av]
    C, Ccont = bmfmax_codes(both, min_count)
    F = np.c_[cat_part, C, np.log1p(lags), Ccont]
    cat = np.r_[np.ones(cat_part.shape[1] + C.shape[1], bool), np.zeros(lags.shape[1] + Ccont.shape[1], bool)]
    feats = {"twin": (F[:n0], F[n0:], cat)}
    if prev_E is not None:
        Ep, Ec = np.asarray(prev_E, float), np.asarray(curr_E, float)
        mu = Ep.mean(0)
        _, _, Vt = np.linalg.svd(Ep - mu, full_matrices=False)
        V = Vt[:K]
        sd = ((Ep - mu) @ V.T).std(0)
        Zp, Zc = (Ep - mu) @ V.T / sd, (Ec - mu) @ V.T / sd
        feats["text"] = (np.c_[F[:n0], Zp], np.c_[F[n0:], Zc], np.r_[cat, np.zeros(K, bool)])
    return feats


def train_frozen(prev, curr, ynames, T="F1p", prev_E=None, curr_E=None, n=1000, J=20, K=32, seed=0,
                 resp_days=300, respondents_only=True):
    """Returns ({'fz_twin': (J, N, q), 'fz_text': (J, N, q), 'fz_twin:<y>': (J, N, 1), ...}, log)."""
    feats = joint_features(prev, curr, T, prev_E, curr_E, K)
    des, _ = production_design(prev, n)
    d = 1 / des.pi
    ta = (des.n_h >= des.N_h)[des.strata]
    fac = np.zeros(des.H)
    ok = (des.n_h > 0) & (des.n_h < des.N_h)
    fac[ok] = des.N_h[ok] ** 2 * (1 - des.n_h[ok] / des.N_h[ok]) / des.n_h[ok]
    resp = (prev.days_to_file_t <= resp_days).fillna(False).to_numpy() if respondents_only else np.ones(len(prev), bool)
    Yp = prev[ynames].to_numpy(float)
    xp = prev.lag_expenses.clip(lower=0).to_numpy(float) + 1
    xc = curr.lag_expenses.clip(lower=0).to_numpy(float) + 1
    is_bin = np.array([y.startswith("y_bin_") for y in ynames])
    rng = np.random.default_rng(seed)
    out = {f"fz_{k}": np.zeros((J, len(curr), len(ynames))) for k in feats}
    log = []
    for b in range(J):
        s = des.draw(rng)
        s = s[~ta[s]]
        h = des.strata[s]
        r = s[resp[s]]
        hr = des.strata[r]
        w0 = d[r].copy()
        for hh in np.unique(h):            # weighting-class (stratum) nonresponse adjustment
            if (hr == hh).any():
                w0[hr == hh] *= d[s][h == hh].sum() / d[r][hr == hh].sum()
        log.append({"draw": b, "n_ts": len(s), "n_resp": len(r)})
        folds = make_folds(len(r), 5, rng, None, hr)
        for name, (F_prev, F_curr, cat) in feats.items():
            for j_set, use_ratio in ((~is_bin, True), (is_bin, False)):
                j = np.flatnonzero(j_set)
                if len(j) == 0:
                    continue
                if use_ratio:
                    Tt, w = Yp[r][:, j] / xp[r][:, None], w0 * xp[r]
                    lo, hi = np.percentile(Tt, [1, 99], axis=0)
                else:
                    Tt, w = Yp[r][:, j], w0
                    lo, hi = -np.inf, np.inf
                lam, lamc, _ = cv_select(F_prev[r], Tt, w, folds, hr, fac, back=xp[r] if use_ratio else None,
                                         cat=cat, lo=lo, hi=hi)
                b0, B = _ridge_fit(F_prev[r], Tt, w, lam, 0, cat, lamc)
                pred = np.clip(b0 + F_curr @ B, lo, hi)
                out[f"fz_{name}"][b][:, j] = pred * xc[:, None] if use_ratio else pred
    for k in [k for k in out if ":" not in k]:
        for jj, y in enumerate(ynames):
            out[f"{k}:{y}"] = out[k][:, :, [jj]]
    return out, pd.DataFrame(log)
