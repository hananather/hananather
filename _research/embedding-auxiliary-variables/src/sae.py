"""Small area (domain) estimation arm.

Domains are unplanned (not design strata), e.g. state x NTEE group: realistic SAE with domain sample
sizes from 0 to ~50. For each Monte Carlo sample we fit a few global prediction models on the
take-some sample (same rules as the national estimators: in-sample CV only, cross-fitting), predict
every frame unit, and form for each domain d and target y:

  DIR    direct (HT)                         sum_{s_d} d y
  MA     model-assisted (GREG-type)          sum_{U_d} m + sum_{s_d} d (y - m)      approx. design-unbiased
  SYN    synthetic                           sum_{U_d} m                             biased, low variance
  FH     Fay-Herriot EBLUP on domain means with the synthetic mean as the single covariate:
         theta_d = g_d * DIR_d + (1-g_d) * (b * SYN_d),  g_d = s2u / (s2u + psi_d)
         (psi_d = MA/direct variance estimate; s2u by Prasad-Rao moments, truncated at 0)

Take-all units enter every domain total exactly. Variances: domain linearisation with
indicator-zeroed residuals.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from mc import model_predict
from survey import greg_total


def _dom_var(E, dom_s, D, design, s):
    """v_d = sum_h N_h^2 (1-f_h)/n_h * s^2_h(e * 1{d}) for every domain (vectorised over y)."""
    h_s = design.strata[s]
    q = E.shape[1]
    v = np.zeros((D, q))
    for h in np.unique(h_s):
        nh, Nh = design.n_h[h], design.N_h[h]
        if nh >= Nh or nh < 2:
            continue
        m = h_s == h
        Eh, dh = E[m], dom_s[m]
        Z = np.zeros((nh, D, q))
        Z[np.arange(nh), dh] = Eh
        v += Nh ** 2 * (1 - nh / Nh) / nh * Z.var(axis=0, ddof=1)
    return v


def _fh(direct_mean, psi, syn_mean, ok):
    """Fay-Herriot EBLUP, one covariate (synthetic mean), no intercept; Prasad-Rao s2u."""
    th, x, p = direct_mean[ok], syn_mean[ok], psi[ok]
    if ok.sum() < 5:
        return syn_mean.copy()
    b = (x @ th) / max(x @ x, 1e-300)
    resid = th - b * x
    m = ok.sum()
    h_lev = x ** 2 / max(x @ x, 1e-300)
    s2u = max((resid @ resid - (p * (1 - h_lev)).sum()) / (m - 1), 0.0)
    g = np.where(ok, s2u / (s2u + np.where(psi > 0, psi, np.inf)), 0.0)
    return g * np.nan_to_num(direct_mean) + (1 - g) * b * syn_mean


def run_sae(cell, domains, models: dict, R=200, seed=0, verbose=True):
    """cell: mc.Cell; domains: population int labels (N,); models: name -> spec with kind in
    {'greg' (X), 'mcal_ratio' (F, X_base), 'mcal_raw' (F, X_base)}. Returns long DataFrame of
    domain estimates and variances, and the domain truth table."""
    rng_draw = np.random.default_rng(seed)    # same sample stream as mc.run_cell with the same seed
    des = cell.design
    dom = pd.factorize(np.asarray(domains))[0]
    D = dom.max() + 1
    Y = cell.Y
    q = Y.shape[1]
    truth = np.zeros((D, q)); np.add.at(truth, dom, Y)
    N_d = np.bincount(dom, minlength=D)
    dom_ts = dom[cell.ts]
    T_ta_d = np.zeros((D, q)); np.add.at(T_ta_d, dom[cell.ta], Y[cell.ta])
    rows = []
    t0 = time.time()
    for r in range(R):
        s = des.draw(rng_draw)
        rng = np.random.default_rng([seed, r, 7])
        s_ts = s[cell.ts[s]]
        i = cell.ts_index[s_ts]
        d_ts, Y_ts = cell.d[s_ts], Y[s_ts]
        ds_dom = dom_ts[i]
        dir_t = T_ta_d.copy(); np.add.at(dir_t, ds_dom, d_ts[:, None] * Y_ts)
        dir_v = _dom_var(Y_ts, ds_dom, D, des, s_ts)
        n_d = np.bincount(ds_dom, minlength=D)
        out = {"DIR": (dir_t, dir_v)}
        for name, spec in models.items():
            if spec["kind"] == "greg":
                X = cell.mats[spec["X"]]
                _, _, B = greg_total(Y_ts, d_ts, X[i], cell.tots[spec["X"]])
                M_U = X @ B
                M_s = M_U[i]
            else:
                F = cell.mats[spec["F"]]
                grp = None if cell.groups is None else cell.groups[i]
                mspec = {"hard": 0, "ratio": spec["kind"] == "mcal_ratio", "cf": True}
                M_s, _, M_U = model_predict(cell, mspec, F, i, Y_ts, d_ts, rng, grp, need_U=True)
                # calibrate the prediction on the base matrix (model calibration, as nationally)
                Xb = cell.mats[spec["X_base"]]
                M_cal_U, M_cal_s = np.zeros_like(M_U), np.zeros_like(M_s)
                for j in range(q):
                    Xm_U = np.c_[Xb, M_U[:, j]]
                    _, _, Bj = greg_total(Y_ts[:, [j]], d_ts, np.c_[Xb[i], M_s[:, j]], Xm_U.sum(0))
                    M_cal_U[:, j] = Xm_U @ Bj[:, 0]
                    M_cal_s[:, j] = np.c_[Xb[i], M_s[:, j]] @ Bj[:, 0]
                M_U, M_s = M_cal_U, M_cal_s
            syn = T_ta_d.copy(); np.add.at(syn, dom_ts, M_U)
            E = Y_ts - M_s
            ma = syn.copy(); np.add.at(ma, ds_dom, d_ts[:, None] * E)
            ma_v = _dom_var(E, ds_dom, D, des, s_ts)
            out[f"MA_{name}"] = (ma, ma_v)
            out[f"SYN_{name}"] = (syn, np.full_like(syn, np.nan))
            fh = np.zeros_like(ma)
            for j in range(q):
                psi = ma_v[:, j] / N_d ** 2
                ok = (n_d >= 2) & (psi > 0)
                fh[:, j] = _fh(ma[:, j] / N_d, psi, syn[:, j] / N_d, ok) * N_d
            out[f"FH_{name}"] = (fh, np.full_like(fh, np.nan))
        for est, (t, v) in out.items():
            for j in range(q):
                rows.append(pd.DataFrame({"rep": r, "estimator": est, "y": cell.ynames[j], "domain": np.arange(D),
                                          "n_d": n_d, "est": t[:, j], "var": v[:, j]}))
        if verbose and (r + 1) % max(1, R // 5) == 0:
            print(f"  sae rep {r + 1}/{R}  {time.time() - t0:.0f}s", flush=True)
    df = pd.concat(rows, ignore_index=True)
    tt = pd.DataFrame([(d, cell.ynames[j], truth[d, j], N_d[d]) for d in range(D) for j in range(q)],
                      columns=["domain", "y", "truth", "N_d"])
    return df, tt


def sae_metrics(df, tt, z=1.959964):
    m = df.merge(tt, on=["domain", "y"])
    m = m[m.truth != 0]
    m["err"] = m.est - m.truth
    g = m.groupby(["estimator", "y", "domain"])
    per = g.agg(truth=("truth", "first"), N_d=("N_d", "first"), mean_n=("n_d", "mean"),
                bias=("err", "mean"), mse=("err", lambda e: np.mean(e ** 2))).reset_index()
    cov = m.dropna(subset=["var"]).assign(c=lambda x: np.abs(x.err) <= z * np.sqrt(np.maximum(x["var"], 0)))
    cov = cov.groupby(["estimator", "y", "domain"]).c.mean().rename("coverage").reset_index()
    per = per.merge(cov, how="left", on=["estimator", "y", "domain"])
    per["arb_pct"] = 100 * np.abs(per.bias) / np.abs(per.truth)
    per["rrmse_pct"] = 100 * np.sqrt(per.mse) / np.abs(per.truth)
    return per
