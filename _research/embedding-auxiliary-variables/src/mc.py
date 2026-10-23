"""Design-based Monte Carlo for the embeddings-as-auxiliaries benchmark.

One call = one (population, design, n) cell. All estimators see the same R samples (common random
numbers), so differences between estimators are estimated much more precisely than their levels.
Per-replicate estimates and variance estimates are returned in long format, so that every metric can
be recomputed (with Monte Carlo standard errors) in the notebook.

Take-all units (f_h = 1) are handled as in business-survey production: they enter the total with
weight 1 and are excluded from modelling and calibration, which act on the take-some part only:
    T_hat = sum_{TA} y_k + T_hat_TS .

Estimator specs (dicts) -- `kind` is one of
  ht     : expansion estimator
  greg   : linear calibration on all columns of X (OLS-GREG)
  ridge  : penalised calibration; first `hard` columns calibrated exactly, the rest ridge-penalised,
           penalty chosen by K-fold CV *inside the sample* (never tuned on the population)
  mcal   : model calibration (Wu & Sitter 2001): fit m_hat by weighted ridge on F (first `hard` cols
           unpenalised), cross-fitted (`cf`=True) or in-sample, optionally on the ratio scale y/x;
           then calibrate on [X_base, m_hat]
  syn    : synthetic  sum_TA y + sum_TS m_hat  (no design correction; shows model bias)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from survey import greg_total, ht_total, strat_var_total

LAMS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)


# ----------------------------------------------------------------------------------------------
# weighted ridge (standardised columns, intercept, per-target penalty)
def _ridge_fit(F, Y, w, lam, hard):
    mu = np.average(F, axis=0, weights=w)
    ymu = np.average(Y, axis=0, weights=w)
    sd = np.sqrt(np.average((F - mu) ** 2, axis=0, weights=w)) + 1e-12
    sw = np.sqrt(w)[:, None]
    Fc = (F - mu) / sd * sw
    Yc = (Y - ymu) * sw
    A = Fc.T @ Fc
    rhs = Fc.T @ Yc
    n = len(F)
    lam = np.broadcast_to(np.atleast_1d(lam), (Y.shape[1],))
    B = np.zeros((F.shape[1], Y.shape[1]))
    for lv in np.unique(lam):
        j = np.flatnonzero(lam == lv)
        pen = np.full(F.shape[1], lv * n)
        pen[:hard] = 1e-6 * n
        B[:, j] = np.linalg.solve(A + np.diag(pen), rhs[:, j])
    Bx = B / sd[:, None]
    b0 = ymu - mu @ Bx
    return b0, Bx


def _folds(n, k, rng, groups=None):
    """K folds; if groups are given, all members of a group share a fold (grouped cross-fitting)."""
    if groups is None:
        return rng.permutation(n) % k
    u, inv = np.unique(groups, return_inverse=True)
    return (rng.permutation(len(u)) % k)[inv]


def _ridge_cv_lambda(F, Y, w, hard, lams=LAMS, k=5, rng=None, groups=None):
    folds = _folds(len(F), k, rng or np.random.default_rng(0), groups)
    sse = np.zeros((len(lams), Y.shape[1]))
    for f in range(k):
        tr, te = folds != f, folds == f
        for i, lam in enumerate(lams):
            b0, Bx = _ridge_fit(F[tr], Y[tr], w[tr], lam, hard)
            sse[i] += (w[te][:, None] * (Y[te] - b0 - F[te] @ Bx) ** 2).sum(0)
    return np.asarray(lams)[sse.argmin(0)]


# ----------------------------------------------------------------------------------------------
class Cell:
    """Precomputes everything that depends on the population only."""

    def __init__(self, Y, ynames, design, mats: dict, size: np.ndarray, groups=None, resp=None,
                 frozen=None):
        """groups: population cluster ids for grouped cross-fitting folds (near-duplicate texts).
        resp: population boolean response indicator (deterministic response, e.g. filed by cutoff).
        frozen: name -> (N, q) predictions from a model trained on independent (previous-wave) data."""
        self.Y, self.ynames, self.design = Y, list(ynames), design
        self.d = 1 / design.pi
        ta_h = design.n_h >= design.N_h
        self.ta = ta_h[design.strata]
        self.ts = ~self.ta
        self.T_ta = Y[self.ta].sum(0)
        # take-some rows only; drop columns that are identically zero there (e.g. take-all dummies)
        self.mats = {}
        for k, v in mats.items():
            v = v[self.ts]
            self.mats[k] = v[:, np.abs(v).sum(0) > 0]
        self.tots = {k: v.sum(0) for k, v in self.mats.items()}
        self.ts_index = np.full(len(Y), -1)
        self.ts_index[self.ts] = np.arange(self.ts.sum())
        self.size = size[self.ts]
        self.N_ts = int(self.ts.sum())
        self._cache = {}
        self.groups = None if groups is None else np.asarray(groups)[self.ts]
        self.resp = None if resp is None else np.asarray(resp, bool)
        self.frozen = {k: (np.asarray(v, float)[..., self.ts, :] if np.ndim(v) == 3 else np.asarray(v, float)[self.ts])
                       for k, v in (frozen or {}).items()}
        self.rep = 0


def _estimate(spec, cell: Cell, s_ts, d_ts, Y_ts, rng):
    kind = spec["kind"]
    i = cell.ts_index[s_ts]
    if kind == "ht":
        return d_ts @ Y_ts, Y_ts
    if kind == "greg":
        X = cell.mats[spec["X"]]
        t, e, _ = greg_total(Y_ts, d_ts, X[i], cell.tots[spec["X"]])
        return t, e
    if kind == "ridge":
        X = cell.mats[spec["X"]]
        Xs = X[i]
        hard = spec["hard"]
        lam = _ridge_cv_lambda(Xs, Y_ts, d_ts, hard, rng=rng)
        mu = np.average(Xs, axis=0, weights=d_ts)
        col_var = np.average((Xs - mu) ** 2, axis=0, weights=d_ts)
        t, e = np.zeros(Y_ts.shape[1]), np.zeros_like(Y_ts)
        for lv in np.unique(lam):
            j = np.flatnonzero(lam == lv)
            pen = np.r_[np.zeros(hard), lv * d_ts.sum() * col_var[hard:]]
            t_, e_, _ = greg_total(Y_ts[:, j], d_ts, Xs, cell.tots[spec["X"]], penalty=pen)
            t[j], e[:, j] = t_, e_
        return t, e
    if kind == "frozen":
        M = cell.frozen[spec["m"]]
        if M.ndim == 3:                     # several previous-wave draws: cycle through them
            M = M[cell.rep % M.shape[0]]
        Ms = M[i]
        if spec.get("calibrate"):
            Xb = cell.mats[spec["X_base"]]
            t, e = np.zeros(Y_ts.shape[1]), np.zeros_like(Y_ts)
            for j in range(Y_ts.shape[1]):
                t_, e_, _ = greg_total(Y_ts[:, [j]], d_ts, np.c_[Xb[i], Ms[:, j]], np.r_[cell.tots[spec["X_base"]], M[:, j].sum()])
                t[j], e[:, j] = t_[0], e_[:, 0]
            return t, e
        E = Y_ts - Ms                       # exactly design-unbiased: m does not depend on s
        return M.sum(0) + d_ts @ E, E
    if kind in ("mcal", "syn"):
        F = cell.mats[spec["F"]]
        Fs = F[i]
        hard = spec["hard"]
        ratio = spec.get("ratio", False)
        sz = cell.size[i]
        if ratio:  # model y/x with weights d*x  (ratio-type heteroscedasticity, Var(y) ~ x)
            xt = sz + spec.get("ratio_offset", 1.0)
            T_target, w = Y_ts / xt[:, None], d_ts * xt
        else:
            T_target, w = Y_ts, d_ts
        lam = _ridge_cv_lambda(Fs, T_target, w, hard, rng=rng,
                               groups=None if cell.groups is None else cell.groups[i])
        sumF = cell.tots[spec["F"]]
        key = (spec["F"], spec.get("ratio_offset", 1.0))
        if ratio and key not in cell._cache:
            xt_U = cell.size + spec.get("ratio_offset", 1.0)
            cell._cache[key] = (xt_U.sum(), xt_U @ F)
        sumxF = cell._cache.get(key)
        def pop_sum(b0, Bx):
            if ratio:
                return b0 * sumxF[0] + sumxF[1] @ Bx
            return b0 * cell.N_ts + sumF @ Bx
        if spec.get("cf", True):
            k = spec.get("k", 5)
            folds = _folds(len(Fs), k, rng, None if cell.groups is None else cell.groups[i])
            M = np.zeros_like(Y_ts)
            sumM = np.zeros(Y_ts.shape[1])
            for f in range(k):
                tr, te = folds != f, folds == f
                b0, Bx = _ridge_fit(Fs[tr], T_target[tr], w[tr], lam, hard)
                M[te] = b0 + Fs[te] @ Bx
                sumM += pop_sum(b0, Bx) / k
        else:
            b0, Bx = _ridge_fit(Fs, T_target, w, lam, hard)
            M = b0 + Fs @ Bx
            sumM = pop_sum(b0, Bx)
        if ratio:
            M = M * xt[:, None]
        if kind == "syn":
            return sumM, np.full_like(Y_ts, np.nan)
        Xb = cell.mats[spec["X_base"]]
        Xb_s, Tb = Xb[i], cell.tots[spec["X_base"]]
        t, e = np.zeros(Y_ts.shape[1]), np.zeros_like(Y_ts)
        for j in range(Y_ts.shape[1]):
            t_, e_, _ = greg_total(Y_ts[:, [j]], d_ts, np.c_[Xb_s, M[:, j]], np.r_[Tb, sumM[j]])
            t[j], e[:, j] = t_[0], e_[:, 0]
        return t, e
    raise ValueError(kind)


def _nr_weights(spec, cell: Cell, s_ts, d_ts):
    """Respondent subset and nonresponse-adjusted weights (take-all units assumed to respond).
    nr_adj = 'none' : no adjustment (naive)
             'rhg'  : response homogeneity groups = design strata (weighting-class adjustment)
             'ps'   : logistic propensity on matrix spec['ps_F'], then rescaled within strata"""
    rmask = cell.resp[s_ts]
    r_ts, dr = s_ts[rmask], d_ts[rmask].copy()
    adj = spec.get("nr_adj", "rhg")
    h_s, h_r = cell.design.strata[s_ts], cell.design.strata[r_ts]
    if adj == "ps":
        from sklearn.linear_model import LogisticRegression
        F = cell.mats[spec["ps_F"]][cell.ts_index[s_ts]]
        mu, sd = F.mean(0), F.std(0) + 1e-9
        lr = LogisticRegression(C=spec.get("ps_C", 1.0), max_iter=500)
        lr.fit((F - mu) / sd, rmask.astype(int), sample_weight=d_ts / d_ts.mean())
        p = lr.predict_proba((F[rmask] - mu) / sd)[:, 1]
        dr = dr / np.clip(p, 0.02, 1.0)
    if adj in ("rhg", "ps"):
        for h in np.unique(h_s):
            num = d_ts[h_s == h].sum()
            den = dr[h_r == h].sum()
            if den > 0:
                dr[h_r == h] *= num / den
        dr *= d_ts.sum() / dr.sum()   # strata with no respondent: spread their weight globally
    return r_ts, dr


def _var_counts(E, design, r_ts):
    """Stratified linearisation variance using respondent counts r_h in place of n_h."""
    E = np.atleast_2d(E.T).T
    h_r = design.strata[r_ts]
    v = np.zeros(E.shape[1])
    for h in np.unique(h_r):
        rh, Nh = (h_r == h).sum(), design.N_h[h]
        if rh < 2 or rh >= Nh:
            continue
        v += Nh ** 2 * (1 - rh / Nh) * E[h_r == h].var(axis=0, ddof=1) / rh
    return v


def run_cell(cell: Cell, specs: dict, R=500, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    des = cell.design
    rows, timing = [], {k: 0.0 for k in specs}
    t0 = time.time()
    for r in range(R):
        s = des.draw(rng)
        cell.rep = r
        s_ts = s[cell.ts[s]]
        d_ts, Y_ts = cell.d[s_ts], cell.Y[s_ts]
        for name, spec in specs.items():
            t1 = time.time()
            try:
                if spec.get("nr"):
                    r_ts, dr = _nr_weights(spec, cell, s_ts, d_ts)
                    t, e = _estimate(spec, cell, r_ts, dr, cell.Y[r_ts], rng)
                    v = _var_counts(e, des, r_ts) if np.isfinite(e).all() else np.full(len(t), np.nan)
                else:
                    t, e = _estimate(spec, cell, s_ts, d_ts, Y_ts, rng)
                    v = strat_var_total(e, des, s_ts) if np.isfinite(e).all() else np.full(len(t), np.nan)
            except np.linalg.LinAlgError:
                t, v = np.full(Y_ts.shape[1], np.nan), np.full(Y_ts.shape[1], np.nan)
            timing[name] += time.time() - t1
            tot = cell.T_ta + t
            for j, yn in enumerate(cell.ynames):
                rows.append((r, name, yn, tot[j], v[j]))
        if verbose and (r + 1) % max(1, R // 5) == 0:
            print(f"  rep {r + 1}/{R}  {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows, columns=["rep", "estimator", "y", "est", "var"])
    df.attrs["timing_s"] = timing
    return df


# ----------------------------------------------------------------------------------------------
def metrics(df: pd.DataFrame, truth: dict, baseline: str, z=1.959964):
    """Tidy metrics with Monte Carlo standard errors. truth: y -> population total.
    RE is MSE(baseline)/MSE(estimator) on paired replicates, with a delta-method MC SE."""
    out = []
    piv = df.pivot_table(index=["rep", "y"], columns="estimator", values="est")
    for (est, y), g in df.groupby(["estimator", "y"]):
        T = truth[y]
        err = g.est.to_numpy() - T
        R = len(err)
        mse = np.mean(err ** 2)
        b = piv.loc[(slice(None), y), baseline].to_numpy() - T
        a1, a2 = err ** 2, b ** 2
        mse_b = a2.mean()
        re = mse_b / mse
        c = np.cov(a1, a2)
        var_log_re = c[0, 0] / (R * mse ** 2) + c[1, 1] / (R * mse_b ** 2) - 2 * c[0, 1] / (R * mse * mse_b)
        row = {"estimator": est, "y": y, "R": R,
               "rel_bias_pct": 100 * err.mean() / T,
               "rel_bias_mcse_pct": 100 * err.std(ddof=1) / np.sqrt(R) / abs(T),
               "rrmse_pct": 100 * np.sqrt(mse) / abs(T),
               "re": re, "re_mcse": re * np.sqrt(max(var_log_re, 0))}
        v = g["var"].to_numpy()
        if np.isfinite(v).all():
            se = np.sqrt(np.maximum(v, 0))
            cover = np.abs(err) <= z * se
            row.update({"coverage_pct": 100 * cover.mean(),
                        "coverage_mcse_pct": 100 * np.sqrt(cover.mean() * (1 - cover.mean()) / R),
                        "var_rel_bias_pct": 100 * (v.mean() / mse - 1),
                        "ci_rel_width_pct": 100 * 2 * z * se.mean() / abs(T)})
        out.append(row)
    return pd.DataFrame(out)
