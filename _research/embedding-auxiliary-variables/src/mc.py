"""Design-based Monte Carlo for the embeddings-as-auxiliaries benchmark.

One call = one (population, design, n) cell. All estimators see the same R samples (common random
numbers), and -- protocol rule T4 -- the same fold vector in each replicate, so a text arm and its
text-free twin differ only in the text columns. Per-replicate estimates and variance estimates are
returned in long format so every metric can be recomputed (with Monte Carlo SEs) in the notebooks.

Take-all units (f_h = 1) enter the total with weight 1 and are excluded from modelling and
calibration, which act on the take-some part only:  T_hat = sum_{TA} y_k + T_hat_TS.

Estimator specs (dicts) -- `kind` is one of
  ht      expansion estimator
  greg    linear calibration on all columns of X (OLS-GREG)
  ridge   ridge-assisted GREG: first `hard` columns unpenalised, the rest penalised; lambda by
          in-sample CV of the same ridge (never tuned on the population)
  mcal    model calibration (Wu & Sitter 2001): cross-fitted weighted ridge m_hat on F (continuous
          targets on the ratio scale y/x~ when ratio=True; binaries always additive), then
          calibrate on [X_base, m_hat], one m_hat per target
  mcalm   single-weight model calibration: bounded calibration on [X_base, m_hat_j for j in heads]
  syn     synthetic sum_TA y + sum_TS m_hat (no design correction; shows model bias)
  frozen  previous-wave model m (fixed, not fitted on this sample): difference estimator, or
          calibration on [X_base, m_j] (calibrate=True), or single-weight on several heads (heads=[...])

Ridge penalty (protocol §4.2, rule T2): weights normalised to mean 1; continuous columns standardised
by their weighted SD and penalised lam*n; categorical (0/1) columns kept on the raw scale and penalised
lam_c (a pseudo-count: a code with m sampled units is shrunk by about m/(m+lam_c)); intercept free.
CV loss (rule T3): cross-validated anticipated variance on the y scale,
  L = sum_h N_h^2 (1-f_h)/n_h * s^2_h(e_oof).
"""
from __future__ import annotations

import time
import zlib

import numpy as np
import pandas as pd

from survey import bounded_calibration, greg_total, strat_var_total

LAMS = tuple(10.0 ** np.arange(-4, 2.01, 0.5))     # 13 values, continuous blocks
LAMS_C = (1.0, 5.0, 20.0, 100.0, 500.0)            # pseudo-counts, categorical blocks


# ----------------------------------------------------------------------------------------------
# weighted ridge with separate continuous / categorical penalties
class RidgeGram:
    """Centred, weighted Gram matrix of one training set; solves for any (lam, lam_c) cheaply.

    Columns split into a fixed-penalty block f (categorical: lam_c; `hard`: ~0) and a lam-dependent
    continuous block v. For each lam_c the f block is factorised once and the Schur complement of the v
    block eigendecomposed, so every lam on the continuous path costs only a small matrix product:
      beta_v = V diag(1/(e + lam n)) V' r~,   beta_f = G (r_f - A_fv beta_v)."""

    def __init__(self, F, Y, w, cat=None, hard=0):
        w = w / w.mean()
        self.mu = np.average(F, axis=0, weights=w)
        self.ymu = np.average(Y, axis=0, weights=w)
        sd = np.sqrt(np.average((F - self.mu) ** 2, axis=0, weights=w))
        p = F.shape[1]
        cat = np.zeros(p, bool) if cat is None else np.asarray(cat, bool)
        sd = np.where(cat | (sd < 1e-12), 1.0, sd)        # categoricals stay on the 0/1 scale
        self.sd, self.cat, self.hard, self.n = sd, cat, hard, len(F)
        sw = np.sqrt(w)[:, None]
        Fc = (F - self.mu) / sd * sw
        self.A = Fc.T @ Fc
        self.rhs = Fc.T @ ((Y - self.ymu) * sw)
        hard_mask = np.zeros(p, bool)
        hard_mask[:hard] = True
        self.fix = cat | hard_mask
        self.var = ~self.fix
        self._cache = {}

    def _prep(self, lam_c):
        if lam_c in self._cache:
            return self._cache[lam_c]
        f, v = self.fix, self.var
        Afv = self.A[np.ix_(f, v)]
        if f.any() and self.cat[f].all():
            # all fixed columns categorical: one eigendecomposition serves every lam_c
            if "_eig_f" not in self.__dict__:
                a, Q = np.linalg.eigh(self.A[np.ix_(f, f)])
                self._eig_f = (np.maximum(a, 0), Q, Q.T @ Afv, Q.T @ self.rhs[f])
            a, Q, QA, Qr = self._eig_f
            inv = 1.0 / (a + lam_c)
            Gfv, Gr = Q @ (inv[:, None] * QA), Q @ (inv[:, None] * Qr)
        elif f.any():
            pen_f = np.where(self.cat[f], lam_c, 1e-6 * self.n)
            Aff = self.A[np.ix_(f, f)] + np.diag(pen_f)
            Gfv = np.linalg.solve(Aff, Afv)
            Gr = np.linalg.solve(Aff, self.rhs[f])
        if f.any():
            S = self.A[np.ix_(v, v)] - Afv.T @ Gfv
            rt = self.rhs[v] - Afv.T @ Gr
        else:
            Gfv = Gr = None
            S, rt = self.A, self.rhs
        e, V = np.linalg.eigh((S + S.T) / 2)
        out = (Gfv, Gr, np.maximum(e, 0), V, V.T @ rt)
        self._cache[lam_c] = out
        return out

    def coef(self, lam, lam_c, cols):
        Gfv, Gr, e, V, Vrt = self._prep(lam_c)
        p = self.A.shape[0]
        B = np.zeros((p, len(cols)))
        bv = V @ (Vrt[:, cols] / (e + lam * self.n)[:, None])
        B[self.var] = bv
        if Gr is not None:
            B[self.fix] = Gr[:, cols] - Gfv @ bv
        Bx = B / self.sd[:, None]
        return self.ymu[cols] - self.mu @ Bx, Bx


def _ridge_fit(F, Y, w, lam, hard, cat=None, lam_c=20.0):
    """Convenience wrapper (one lam / lam_c for all targets, or per-target arrays)."""
    g = RidgeGram(F, Y, w, cat, hard)
    q = Y.shape[1]
    lam = np.broadcast_to(np.atleast_1d(lam), (q,))
    lam_c = np.broadcast_to(np.atleast_1d(lam_c), (q,))
    b0, Bx = np.zeros(q), np.zeros((F.shape[1], q))
    for key in set(zip(lam, lam_c)):
        j = np.flatnonzero((lam == key[0]) & (lam_c == key[1]))
        b0[j], Bx[:, j] = g.coef(key[0], key[1], j)
    return b0, Bx


def make_folds(n, k, rng, groups=None, strata=None):
    """K folds. Groups (near-duplicate texts) share a fold; groups are dealt round-robin within
    design strata so that folds are balanced (protocol rule T4)."""
    groups = np.arange(n) if groups is None else np.asarray(groups)
    u, first, inv = np.unique(groups, return_index=True, return_inverse=True)
    st = np.zeros(len(u)) if strata is None else np.asarray(strata)[first]
    order = np.lexsort((rng.random(len(u)), st))
    fold_u = np.empty(len(u), int)
    fold_u[order] = np.arange(len(u)) % k
    return fold_u[inv]


def _av_loss(E, strata, fac):
    """sum_h fac_h * s^2_h(E) for every column of E (fac_h = N_h^2 (1-f_h)/n_h)."""
    out = np.zeros(E.shape[1])
    for h in np.unique(strata):
        m = strata == h
        if m.sum() > 1:
            out += fac[h] * E[m].var(axis=0, ddof=1)
    return out


def cv_select(F, T, w, folds, strata, fac, back=None, cat=None, hard=0, lams=LAMS, lams_c=LAMS_C,
              lo=None, hi=None):
    """Choose (lam, lam_c) per target by cross-validated AV loss on the y scale.
    T: fitting target (ratio scale if back is given); back: (n,) multiplier back to the y scale.
    Coordinate search: lam_c first (with lam = 1), then lam given lam_c. Returns (lam, lam_c, grams)."""
    k = folds.max() + 1
    grams = [RidgeGram(F[folds != f], T[folds != f], w[folds != f], cat, hard) for f in range(k)]
    q = T.shape[1]
    has_cat = cat is not None and np.any(cat)
    lo = -np.inf if lo is None else lo
    hi = np.inf if hi is None else hi

    def loss(lam_vec, lamc_vec):
        E = np.zeros_like(T, dtype=float)
        for f, g in enumerate(grams):
            te = folds == f
            for key in set(zip(lam_vec, lamc_vec)):
                j = np.flatnonzero((lam_vec == key[0]) & (lamc_vec == key[1]))
                b0, Bx = g.coef(key[0], key[1], j)
                P = np.clip(b0 + F[te] @ Bx, lo if np.isscalar(lo) else lo[j], hi if np.isscalar(hi) else hi[j])
                E[np.ix_(np.flatnonzero(te), j)] = T[te][:, j] - P
        if back is not None:
            E = E * back[:, None]
        return _av_loss(E, strata, fac)

    lamc = np.full(q, 20.0)
    if has_cat:
        Lc = np.array([loss(np.full(q, 1.0), np.full(q, c)) for c in lams_c])
        lamc = np.asarray(lams_c)[Lc.argmin(0)]
    L = np.array([loss(np.full(q, l), lamc) for l in lams])
    lam = np.asarray(lams)[L.argmin(0)]
    return lam, lamc, grams


# ----------------------------------------------------------------------------------------------
class Cell:
    """Precomputes everything that depends on the population only."""

    def __init__(self, Y, ynames, design, mats: dict, size: np.ndarray, groups=None, resp=None,
                 frozen=None, cat_masks=None):
        """mats: name -> (N, p) array. cat_masks: name -> (p,) bool, True for 0/1 categorical columns.
        groups: population cluster ids for grouped folds (near-duplicate texts).
        resp: population boolean response indicator (deterministic response, e.g. filed by cutoff).
        frozen: name -> (N, q) or (J, N, q) predictions from models trained on previous-wave data."""
        self.Y, self.ynames, self.design = Y, list(ynames), design
        self.is_bin = np.array([y.startswith("y_bin_") for y in self.ynames])
        self.d = 1 / design.pi
        ta_h = design.n_h >= design.N_h
        self.ta = ta_h[design.strata]
        self.ts = ~self.ta
        self.T_ta = Y[self.ta].sum(0)
        self.mats, self.cats = {}, {}
        cat_masks = cat_masks or {}
        for k, v in mats.items():           # take-some rows only; drop columns identically zero there
            v = v[self.ts]
            keep = np.abs(v).sum(0) > 0
            self.mats[k] = v[:, keep]
            cm = cat_masks.get(k)
            self.cats[k] = None if cm is None else np.asarray(cm, bool)[keep]
        self.tots = {k: v.sum(0) for k, v in self.mats.items()}
        self.ts_index = np.full(len(Y), -1)
        self.ts_index[self.ts] = np.arange(self.ts.sum())
        self.size = size[self.ts]
        self.N_ts = int(self.ts.sum())
        self.groups = None if groups is None else np.asarray(groups)[self.ts]
        self.resp = None if resp is None else np.asarray(resp, bool)
        self.frozen = {k: (np.asarray(v, float)[..., self.ts, :] if np.ndim(v) == 3 else np.asarray(v, float)[self.ts])
                       for k, v in (frozen or {}).items()}
        fac = np.zeros(design.H)
        ok = (design.n_h > 0) & (design.n_h < design.N_h)
        fac[ok] = design.N_h[ok] ** 2 * (1 - design.n_h[ok] / design.N_h[ok]) / design.n_h[ok]
        self.av_fac = fac
        self.rep = 0
        self.fold_of = np.full(self.N_ts, -1)   # shared fold of each sampled take-some unit (this rep)


def model_predict(cell, spec, F, i, Y_ts, d_ts, rng=None, groups=None, need_U=False):
    """Fit the prediction model of an mcal/mcalm/syn spec on the take-some sample (rows i).
    Returns (M_s (n,q) cross-fitted or in-sample, population sums of predictions (q,), M_U or None).
    Continuous targets on the ratio scale when spec['ratio'] (y/x~, weights d*x~, predicted ratios
    clipped to the sample's 1st-99th pct); binaries always additive (never the ratio learner)."""
    cat = cell.cats.get(spec["F"]) if isinstance(spec.get("F"), str) else None
    hard = spec.get("hard", 0)
    q = Y_ts.shape[1]
    ratio_cols = np.flatnonzero(~cell.is_bin) if spec.get("ratio", False) else np.array([], int)
    add_cols = np.setdiff1d(np.arange(q), ratio_cols)
    Fs = F[i]
    off = spec.get("ratio_offset", 1.0)
    xs, xU = cell.size[i] + off, cell.size + off
    folds = cell.fold_of[i]
    if (folds < 0).any():                    # fall back to private grouped folds
        folds = make_folds(len(i), 5, rng or np.random.default_rng(0), groups)
    strata = cell.design.strata[np.flatnonzero(cell.ts)[i]]
    cf = spec.get("cf", True)
    M = np.zeros_like(Y_ts, dtype=float)
    sumM = np.zeros(q)
    M_U = np.zeros((cell.N_ts, q)) if need_U else None
    for cols, ratio in ((ratio_cols, True), (add_cols, False)):
        if len(cols) == 0:
            continue
        T, w = (Y_ts[:, cols] / xs[:, None], d_ts * xs) if ratio else (Y_ts[:, cols], d_ts)
        lo, hi = (np.percentile(T, [1, 99], axis=0) if (ratio and spec.get("clip", True)) else (-np.inf, np.inf))
        lam, lamc, grams = cv_select(Fs, T, w, folds, strata, cell.av_fac, back=xs if ratio else None,
                                     cat=cat, hard=hard, lo=lo, hi=hi)
        fits = []
        if cf:
            for f, g in enumerate(grams):
                te = np.flatnonzero(folds == f)
                b0, Bx = np.zeros(len(cols)), np.zeros((F.shape[1], len(cols)))
                for key in set(zip(lam, lamc)):
                    j = np.flatnonzero((lam == key[0]) & (lamc == key[1]))
                    b0[j], Bx[:, j] = g.coef(key[0], key[1], j)
                M[np.ix_(te, cols)] = np.clip(b0 + Fs[te] @ Bx, lo, hi)
                fits.append((b0, Bx))
        else:
            b0, Bx = _ridge_fit(Fs, T, w, lam, hard, cat, lamc)
            M[:, cols] = np.clip(b0 + Fs @ Bx, lo, hi)
            fits.append((b0, Bx))
        # population predictions from the fold-averaged model (exact average of the linear fold
        # predictions; differs from averaging clipped predictions only through the clip) -- deviation D-6
        b0m = np.mean([f_[0] for f_ in fits], 0)
        Bxm = np.mean([f_[1] for f_ in fits], 0)
        P = np.clip(b0m + F @ Bxm, lo, hi)
        if ratio:
            P = P * xU[:, None]
        sumM[cols] = P.sum(0)
        if need_U:
            M_U[:, cols] = P
        if ratio:
            M[:, cols] = M[:, cols] * xs[:, None]
    return M, sumM, M_U


def _calibrate_each(Y_ts, d_ts, Xb_s, Tb, M_s, sumM):
    """Per-target model calibration on [X_base, m_j]."""
    t, e = np.zeros(Y_ts.shape[1]), np.zeros_like(Y_ts)
    for j in range(Y_ts.shape[1]):
        t_, e_, _ = greg_total(Y_ts[:, [j]], d_ts, np.c_[Xb_s, M_s[:, j]], np.r_[Tb, sumM[j]])
        t[j], e[:, j] = t_[0], e_[:, 0]
    return t, e


def _calibrate_single(Y_ts, d_ts, X_s, T_X, bounds=(0.2, 5.0)):
    """One weight set for every target: bounded linear calibration on X; residual-based variance."""
    w, conv = bounded_calibration(X_s, d_ts, T_X, *bounds)
    XtD = X_s.T * d_ts
    B = np.linalg.solve(XtD @ X_s + 1e-8 * np.diag(np.maximum(np.diag(XtD @ X_s), 1e-300)), XtD @ Y_ts)
    E = Y_ts - X_s @ B
    return w @ Y_ts, E, w / d_ts, conv


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
        folds = cell.fold_of[i]
        strata = cell.design.strata[s_ts]
        lam, lamc, _ = cv_select(Xs, Y_ts, d_ts, folds, strata, cell.av_fac, cat=cell.cats.get(spec["X"]), hard=hard)
        b0, Bx = _ridge_fit(Xs, Y_ts, d_ts, lam, hard, cell.cats.get(spec["X"]), lamc)
        E = Y_ts - (b0 + Xs @ Bx)
        return b0 * cell.N_ts + cell.tots[spec["X"]] @ Bx + d_ts @ E, E
    if kind == "frozen":
        Xb = cell.mats[spec["X_base"]] if "X_base" in spec else None
        if spec.get("heads"):                # single weight on several frozen heads (FROZEN_calm)
            cols = [cell.frozen[h] for h in spec["heads"]]
            cols = [c[cell.rep % c.shape[0]] if c.ndim == 3 else c for c in cols]
            H = np.column_stack(cols)
            X_U = np.c_[Xb, H]
            t, e, g, conv = _calibrate_single(Y_ts, d_ts, X_U[i], X_U.sum(0), spec.get("bounds", (0.2, 5.0)))
            return t, e
        M = cell.frozen[spec["m"]]
        if M.ndim == 3:                      # several previous-wave draws: cycle through them
            M = M[cell.rep % M.shape[0]]
        Ms = M[i]
        if spec.get("calibrate"):
            return _calibrate_each(Y_ts, d_ts, Xb[i], cell.tots[spec["X_base"]], Ms, M.sum(0))
        E = Y_ts - Ms                        # exactly design-unbiased: m does not depend on s
        return M.sum(0) + d_ts @ E, E
    if kind in ("mcal", "syn", "mcalm"):
        F = cell.mats[spec["F"]]
        grp = None if cell.groups is None else cell.groups[i]
        M, sumM, M_U = model_predict(cell, spec, F, i, Y_ts, d_ts, rng, grp, need_U=(kind == "mcalm"))
        if kind == "syn":
            return sumM, np.full_like(Y_ts, np.nan)
        Xb = cell.mats[spec["X_base"]]
        if kind == "mcalm":
            jj = [cell.ynames.index(h) for h in spec["heads"]]
            X_U = np.c_[Xb, M_U[:, jj]]
            X_s = np.c_[Xb[i], M[:, jj]]
            t, e, g, conv = _calibrate_single(Y_ts, d_ts, X_s, X_U.sum(0), spec.get("bounds", (0.2, 5.0)))
            return t, e
        return _calibrate_each(Y_ts, d_ts, Xb[i], cell.tots[spec["X_base"]], M, sumM)
    raise ValueError(kind)


def _stable_hash(name: str) -> int:
    return zlib.crc32(name.encode())


def _nr_weights(spec, cell: Cell, s_ts, d_ts):
    """Respondent subset and nonresponse-adjusted weights (take-all units handled by the caller).
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


def run_cell(cell: Cell, specs: dict, R=500, seed=0, verbose=True, ta_values=None, rep_range=None):
    """Streams: samples from default_rng(seed) (identical whatever estimators run); the shared fold
    vector of replicate r from default_rng([seed, r, 1]); estimator-internal randomness from
    default_rng([seed, r, hash(name)]).
    ta_values: optional callable(r) -> (q,) take-all total to use instead of the true one (take-all
    nonresponse with imputation, Stage 3)."""
    rng_draw = np.random.default_rng(seed)
    des = cell.design
    rows, timing, fails = [], {k: 0.0 for k in specs}, {k: 0 for k in specs}
    t0 = time.time()
    a, b = (0, R) if rep_range is None else rep_range
    for r in range(b):
        s = des.draw(rng_draw)              # draws for skipped replicates are made and discarded, so a
        if r < a:                           # replicate's sample never depends on how runs are split
            continue
        cell.rep = r
        s_ts = s[cell.ts[s]]
        d_ts, Y_ts = cell.d[s_ts], cell.Y[s_ts]
        cell.fold_of[:] = -1
        grp = None if cell.groups is None else cell.groups[cell.ts_index[s_ts]]
        cell.fold_of[cell.ts_index[s_ts]] = make_folds(len(s_ts), 5, np.random.default_rng([seed, r, 1]), grp,
                                                       des.strata[s_ts])
        T_ta = cell.T_ta if ta_values is None else ta_values(r)
        for name, spec in specs.items():
            t1 = time.time()
            rng = np.random.default_rng([seed, r, _stable_hash(name)])
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
                fails[name] += 1
            timing[name] += time.time() - t1
            tot = T_ta + t
            for j, yn in enumerate(cell.ynames):
                rows.append((r, name, yn, tot[j], v[j]))
        if verbose and (r + 1 - a) % max(1, (b - a) // 5) == 0:
            print(f"  rep {r + 1}/{b}  {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows, columns=["rep", "estimator", "y", "est", "var"])
    df.attrs["timing_s"], df.attrs["failures"] = timing, fails
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
                        "var_rel_bias_empvar_pct": 100 * (v.mean() / np.var(g.est.to_numpy(), ddof=1) - 1),
                        "ci_rel_width_pct": 100 * 2 * z * se.mean() / abs(T)})
        out.append(row)
    return pd.DataFrame(out)


# backwards-compatible alias used by older scripts
_folds = lambda n, k, rng, groups=None: make_folds(n, k, rng, groups)
_ridge_cv_lambda = None  # removed: use cv_select (AV-scale loss)
