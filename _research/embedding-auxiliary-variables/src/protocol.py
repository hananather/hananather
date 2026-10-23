"""Benchmark protocol v1.0 (notes/benchmark-protocol.md) as code: register tiers, matrices, estimator
specs with protocol names, placebo and planted targets, E* selection, primary statistics and gates.

Matrix names (take-some rows are selected later by mc.Cell):
  size_cat@T     calibration matrix: strata dummies + tier-T lags ($) + NTEE10 + state + availability
  F@T            learner features: NTEE10, state, BMF income code, availability (categorical, raw 0/1)
                 + log1p(tier lags) (continuous)
  FC@T           F@T + BMF-max codes C (categorical) + log1p(BMF amounts)
  F@T+E, FC@T+E  ... + first K population PCs of embedding E (continuous)
  RC@T...        linear factorial arms (ridge-calibration): size_cat@T + soft blocks
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from frame import pcs
from mc import RidgeGram

H_D = ["y_gov_grants", "y_contrib", "y_prog_rev", "y_fundraising_exp", "y_officer_comp"]
H_S = ["y_volunteers_cap10k", "y_bin_has_volunteers", "y_bin_grants_to_indiv_us", "y_bin_foreign_activities",
       "y_bin_lobbying"]
H_SET = H_D + H_S
R_SET = ["y_revenue", "y_expenses", "y_employees", "y_salaries"]
J4 = ["y_gov_grants", "y_contrib", "y_fundraising_exp", "y_bin_grants_to_indiv_us"]
F1_LAGS = ["lag_revenue", "lag_expenses", "lag_assets", "lag_employees"]
TIER_LAGS = {
    "F0": [],
    "F1": F1_LAGS,
    "F1p": F1_LAGS + ["lag_salaries"],
    "F1c": F1_LAGS + ["lag_salaries", "lag_contrib", "lag_prog_rev"],
    "F2": F1_LAGS + ["lag_salaries", "lag_contrib", "lag_prog_rev", "oracle_lag_gov_grants",
                     "oracle_lag_func_exp_fundraising", "oracle_lag_officer_comp"],
}


def _dum(s, drop_first=True, prefix=None):
    return pd.get_dummies(pd.Series(s).astype(str), drop_first=drop_first, prefix=prefix, dtype=float).to_numpy()


def bmfmax_codes(pop: pd.DataFrame, min_count=20):
    """BMF-max code block (protocol §4.1 matrix C). Full NTEE code pooled to codes with >= min_count
    units, the rest pooled to their NTEE letter; NTEE letter; subsection; foundation; affiliation;
    classification; first activity code (pooled); deductibility; organization; ACCT_PD; GROUP != 0;
    ruling-year bins; BMF asset code; valid-URL flag. Pooling never looks at y.
    Returns (categorical 0/1 block, continuous block = log1p BMF amounts)."""
    blocks = []
    ntee = pop.bmf_ntee_cd.astype(str) if "bmf_ntee_cd" in pop else pop.ntee_code.fillna("NA").astype(str)
    letter = ntee.str[0].where(ntee.str[0].str.match(r"[A-Z]"), "NA")
    vc = ntee.value_counts()
    blocks.append(_dum(ntee.where(ntee.map(vc) >= min_count, "L_" + letter)))
    blocks.append(_dum(letter))
    for c in ["bmf_subsection", "bmf_foundation", "bmf_affiliation", "bmf_classification", "bmf_deductibility",
              "bmf_organization", "bmf_acct_pd"]:
        if c in pop:
            v = pop[c].astype(str)
            vc = v.value_counts()
            blocks.append(_dum(v.where(v.map(vc) >= min_count, "OTHER")))
    if "bmf_activity" in pop:
        a = pop.bmf_activity.astype(str).str[:3]
        vc = a.value_counts()
        blocks.append(_dum(a.where(a.map(vc) >= min_count, "OTHER")))
    if "bmf_group" in pop:
        blocks.append((~pop.bmf_group.astype(str).isin(["0", "0000", "NA"])).to_numpy(float)[:, None])
    ry = pop.ruling_year.fillna(0)
    blocks.append(_dum(pd.cut(ry, [-1, 1970, 1990, 2000, 2010, 2015, 3000], labels=False)))
    blocks.append(_dum(pop.bmf_asset_cd.fillna(-1)))
    blocks.append(pop.has_website.to_numpy(float)[:, None])
    C = np.concatenate(blocks, 1)
    C = C[:, C.std(0) > 0]
    cont = np.log1p(pop[[c for c in ["bmf_asset_amt", "bmf_income_amt", "bmf_revenue_amt"] if c in pop]]
                    .clip(lower=0).to_numpy(float))
    return C, cont


def tier_mats(pop: pd.DataFrame, design, tiers=("F1p",), min_count=20):
    """Matrices and categorical masks for the given register tiers."""
    S = _dum(design.strata, drop_first=False)
    av = pop.avail_t1.to_numpy(float) if "avail_t1" in pop else np.ones(len(pop))
    x = pop.lag_expenses.clip(lower=0).to_numpy(float)
    nt = _dum(pop.ntee_major10)
    st = _dum(pop.state)
    inc = _dum(pop.bmf_income_cd.fillna(-1))
    C, Ccont = bmfmax_codes(pop, min_count)
    mats, cats = {}, {}
    for T in tiers:
        lags = pop[TIER_LAGS[T]].clip(lower=0).to_numpy(float).copy() if TIER_LAGS[T] else np.zeros((len(pop), 0))
        # extras beyond F1 are unknown when the t-1 return is not available at D
        for j, c in enumerate(TIER_LAGS[T]):
            if c not in F1_LAGS:
                lags[:, j] *= av
        avb = np.c_[1 - av, (1 - av) * x]
        mats[f"size_cat@{T}"] = np.c_[S, lags, nt, st, avb]
        cat_part = np.c_[nt, st, inc, 1 - av]
        cont_part = np.log1p(lags)
        mats[f"F@{T}"] = np.c_[cat_part, cont_part]
        cats[f"F@{T}"] = np.r_[np.ones(cat_part.shape[1], bool), np.zeros(cont_part.shape[1], bool)]
        mats[f"FC@{T}"] = np.c_[cat_part, C, cont_part, Ccont]
        cats[f"FC@{T}"] = np.r_[np.ones(cat_part.shape[1] + C.shape[1], bool), np.zeros(cont_part.shape[1] + Ccont.shape[1], bool)]
    mats["_C"], cats["_C"] = C, np.ones(C.shape[1], bool)
    return mats, cats


def add_text(mats, cats, pop, tiers, name, E, K=32, linear_factorial=False):
    """Text features: Z_K(E) appended to F@T and FC@T; optionally the linear factorial matrices."""
    Z, _ = pcs(np.asarray(E, float), K)
    for T in tiers:
        for base in ("F", "FC"):
            mats[f"{base}@{T}+{name}"] = np.c_[mats[f"{base}@{T}"], Z]
            cats[f"{base}@{T}+{name}"] = np.r_[cats[f"{base}@{T}"], np.zeros(K, bool)]
        if linear_factorial:
            x = pop.lag_expenses.clip(lower=0).to_numpy(float)
            emp = pop.lag_employees.clip(lower=0).to_numpy(float)
            Zx = Z * x[:, None]
            Ze = Z * (emp * x.std() / max(emp.std(), 1e-9))[:, None]
            sc = mats[f"size_cat@{T}"]
            C = mats["_C"]
            Csmall = C[:, : min(C.shape[1], 40)] * x[:, None]
            mats[f"RC@{T}+{name}"] = np.c_[sc, Zx, Ze]
            cats[f"RC@{T}+{name}"] = np.zeros(mats[f"RC@{T}+{name}"].shape[1], bool)
            mats[f"RC@{T}+C"] = np.c_[sc, C, Csmall]
            cats[f"RC@{T}+C"] = np.r_[np.zeros(sc.shape[1], bool), np.ones(C.shape[1], bool), np.zeros(Csmall.shape[1], bool)]
            mats[f"RC@{T}+C+{name}"] = np.c_[sc, C, Csmall, Zx, Ze]
            cats[f"RC@{T}+C+{name}"] = np.r_[cats[f"RC@{T}+C"], np.zeros(2 * K, bool)]
    return mats, cats


# ----------------------------------------------------------------------------------------------
def specs_primary(T="F1p", estar="E", controls=(), extra_text=(), with_frozen=False, n_sc=None):
    """Protocol §4.3 names. estar: embedding key of E*. controls: embedding keys used as N1-N4.
    n_sc: number of columns of size_cat@T on the take-some part (hard columns of the ridge kind)."""
    sp = {
        "HT": {"kind": "ht"},
        f"GREG_size_cat@{T}": {"kind": "greg", "X": f"size_cat@{T}"},
        f"MCAL_frame@{T}": {"kind": "mcal", "F": f"F@{T}", "X_base": f"size_cat@{T}", "ratio": True},
        f"MCAL_bmfmax@{T}": {"kind": "mcal", "F": f"FC@{T}", "X_base": f"size_cat@{T}", "ratio": True},
        f"MCAL_frame@{T}+{estar}": {"kind": "mcal", "F": f"F@{T}+{estar}", "X_base": f"size_cat@{T}", "ratio": True},
        f"MCAL_bmfmax@{T}+{estar}": {"kind": "mcal", "F": f"FC@{T}+{estar}", "X_base": f"size_cat@{T}", "ratio": True},
        # single-weight current-sample system (P2')
        f"MCALm_bmfmax@{T}": {"kind": "mcalm", "F": f"FC@{T}", "X_base": f"size_cat@{T}", "ratio": True, "heads": J4},
        f"MCALm_bmfmax@{T}+{estar}": {"kind": "mcalm", "F": f"FC@{T}+{estar}", "X_base": f"size_cat@{T}", "ratio": True, "heads": J4},
        # linear factorial arms
        f"RIDGECAL@{T}+{estar}": {"kind": "ridge", "X": f"RC@{T}+{estar}", "hard": n_sc},
        f"RIDGECAL@{T}+C": {"kind": "ridge", "X": f"RC@{T}+C", "hard": n_sc},
        f"RIDGECAL@{T}+C+{estar}": {"kind": "ridge", "X": f"RC@{T}+C+{estar}", "hard": n_sc},
        # pitfall demonstrations
        f"MCAL_bmfmax@{T}+{estar}:nocf": {"kind": "mcal", "F": f"FC@{T}+{estar}", "X_base": f"size_cat@{T}", "ratio": True, "cf": False},
        f"SYN_bmfmax@{T}+{estar}": {"kind": "syn", "F": f"FC@{T}+{estar}", "ratio": True},
    }
    for c in controls:
        sp[f"MCAL_bmfmax@{T}+{c}"] = {"kind": "mcal", "F": f"FC@{T}+{c}", "X_base": f"size_cat@{T}", "ratio": True}
    for e in extra_text:
        sp[f"MCAL_bmfmax@{T}+{e}"] = {"kind": "mcal", "F": f"FC@{T}+{e}", "X_base": f"size_cat@{T}", "ratio": True}
    if with_frozen:
        heads_txt = [f"fz_text:{y}" for y in J4]
        heads_twin = [f"fz_twin:{y}" for y in J4]
        sp.update({
            f"FROZEN_calm_bmfmax@{T}+{estar}": {"kind": "frozen", "heads": heads_txt, "X_base": f"size_cat@{T}"},
            f"FROZEN_calm_bmfmax@{T}": {"kind": "frozen", "heads": heads_twin, "X_base": f"size_cat@{T}"},
            f"FROZEN_cal_bmfmax@{T}+{estar}": {"kind": "frozen", "m": "fz_text", "calibrate": True, "X_base": f"size_cat@{T}"},
            f"FROZEN_cal_bmfmax@{T}": {"kind": "frozen", "m": "fz_twin", "calibrate": True, "X_base": f"size_cat@{T}"},
            f"FROZEN_diff_bmfmax@{T}+{estar}": {"kind": "frozen", "m": "fz_text"},
            f"FROZEN_diff_bmfmax@{T}": {"kind": "frozen", "m": "fz_twin"},
        })
    return sp


def specs_tier(T, estar):
    """Twin and text at another tier (F1 realism check, F2 charity frame / oracle-lag control)."""
    return {
        f"GREG_size_cat@{T}": {"kind": "greg", "X": f"size_cat@{T}"},
        f"MCAL_bmfmax@{T}": {"kind": "mcal", "F": f"FC@{T}", "X_base": f"size_cat@{T}", "ratio": True},
        f"MCAL_bmfmax@{T}+{estar}": {"kind": "mcal", "F": f"FC@{T}+{estar}", "X_base": f"size_cat@{T}", "ratio": True},
    }


# ----------------------------------------------------------------------------------------------
# control embeddings and placebo / planted targets
def control_embeddings(E, design, seed=0):
    rng = np.random.default_rng(seed)
    N, D = E.shape
    perm = E[rng.permutation(N)]
    permh = E.copy()
    for h in np.unique(design.strata):
        idx = np.flatnonzero(design.strata == h)
        permh[idx] = E[rng.permutation(idx)]
    return {"N1random": rng.normal(size=(N, D)), "N2perm": perm, "N3permh": permh, "N4const": np.ones((N, D))}


def placebo_targets(pop, design, seed=0):
    """N5 placebos: PLC_perm (gov grants permuted within stratum x lag-size decile), PLC_noise
    (x~ * lognormal sigma 2), PLC_bern (Bernoulli(p_h) within strata, p_h = stratum share with gov grants)."""
    rng = np.random.default_rng(seed)
    x = pop.lag_expenses.clip(lower=0).to_numpy(float) + 1
    dec = pd.qcut(pd.Series(x).rank(method="first"), 10, labels=False).to_numpy()
    key = design.strata * 10 + dec
    g = pop.y_gov_grants.to_numpy(float)
    perm = g.copy()
    for k in np.unique(key):
        idx = np.flatnonzero(key == k)
        perm[idx] = g[rng.permutation(idx)]
    noise = x * np.exp(2.0 * rng.normal(size=len(x)) - 2.0)
    has = (g > 0).astype(float)
    p_h = pd.Series(has).groupby(design.strata).transform("mean").to_numpy()
    bern = (rng.random(len(x)) < p_h).astype(float)
    return {"y_plc_perm": perm, "y_plc_noise": noise, "y_bin_plc_bern": bern}


def _pop_av(Y, F, w, cat, design, ts, back, lam=1e-2, lam_c=20.0, k=5, seed=12345):
    """Anticipated variance (per column) of the difference estimator built on a population-fitted
    same-class ridge, using K-fold cross-fitting over the take-some population: without it the
    handful of very large units that dominate the AV are fitted in-sample by extra columns, which
    inflated ceilings by 3-5% even for a planted target with no text signal (deviation D-4)."""
    idx = np.flatnonzero(ts)
    folds = np.random.default_rng(seed).permutation(len(idx)) % k
    E = np.zeros((len(idx), Y.shape[1]))
    for f in range(k):
        tr, te = idx[folds != f], idx[folds == f]
        g = RidgeGram(F[tr], Y[tr], w[tr], cat, 0)
        b0, Bx = g.coef(lam, lam_c, np.arange(Y.shape[1]))
        E[folds == f] = (Y[te] - (b0 + F[te] @ Bx)) * back[te][:, None]
    fac = np.zeros(design.H)
    ok = (design.n_h > 0) & (design.n_h < design.N_h)
    fac[ok] = design.N_h[ok] ** 2 * (1 - design.n_h[ok] / design.N_h[ok]) / design.n_h[ok]
    out = np.zeros(Y.shape[1])
    st = design.strata[idx]
    for h in np.unique(st):
        m = st == h
        if m.sum() > 1:
            out += fac[h] * E[m].var(0, ddof=1)
    return out


def planted_targets(pop, design, E, F_twin, cat_twin, levels=(0.0, 0.05, 0.10, 0.20), plantings=3, K=32, seed=0,
                    sigma=0.5):
    """Planted-signal targets with known same-class partial rho^2 on the AV scale (protocol §5.2),
    rho^2 = 1 - AV(F+C+E)/AV(F+C), computed with the same ratio-ridge class as the MC estimators.

    u = w'Z_K(E) with w random, orthogonalised against the twin features and standardised.

    Family 'pr' (protocol form, diagnostic only): y = y0 exp(a u - a^2/2) on the gov-grants null base
      y0 = x~ (y/x~)_pi (pi within NTEE10 x size class x state). Its attainable rho^2 is capped near
      1/(1 + CV^2) of the zero-inflated, heavy-tailed base, so the achieved rho^2 at a in {0.5,1,2,3}
      is recorded instead of calibrating to the targets (deviation D-3).
    Family 'lgn' (used in the MC): y = x~ c exp(sigma (rho u + sqrt(1-rho^2) eps) - sigma^2/2), eps
      iid N(0,1), c = median positive gov-grant ratio; rho calibrated by bisection to each target."""
    rng = np.random.default_rng(seed)
    x = pop.lag_expenses.clip(lower=0).to_numpy(float) + 1
    cls = np.searchsorted(np.quantile(np.log(x), [1 / 3, 2 / 3]), np.log(x))
    key = pd.factorize(pop.ntee_major10.astype(str) + "_" + pop.state.astype(str) + "_" + cls.astype(str))[0]
    r = pop.y_gov_grants.to_numpy(float) / x
    r0 = r.copy()
    for k in np.unique(key):
        idx = np.flatnonzero(key == k)
        r0[idx] = r[rng.permutation(idx)]
    y0 = x * r0
    c = np.median(r[r > 0])
    Z, _ = pcs(np.asarray(E, float), K)
    ta = (design.n_h >= design.N_h)[design.strata]
    ts = ~ta
    w = (1 / design.pi) * x
    Fte = np.c_[F_twin, Z]
    cat_te = np.r_[cat_twin, np.zeros(K, bool)]

    def rho2_of(y):
        T = (y / x)[:, None]
        return 1 - _pop_av(T, Fte, w, cat_te, design, ts, x)[0] / _pop_av(T, F_twin, w, cat_twin, design, ts, x)[0]

    out, info = {}, []
    for p in range(plantings):
        u = Z @ rng.normal(size=K)
        A = F_twin - F_twin.mean(0)
        coef = np.linalg.lstsq(A[ts], (u - u.mean())[ts], rcond=None)[0]
        u = u - A @ coef
        u = (u - u.mean()) / u.std()
        eps = rng.normal(size=len(x))
        for a_ in (0.5, 1.0, 2.0, 3.0):                      # protocol form: attainable rho^2
            info.append({"family": "pr", "planting": p, "a": a_, "rho2_achieved": rho2_of(y0 * np.exp(a_ * u - a_ * a_ / 2))})
        for rho2 in levels:
            lo_r, hi_r = 0.0, 1.0
            rho = 0.0
            if rho2 > 0:
                for _ in range(16):
                    rho = 0.5 * (lo_r + hi_r)
                    yy = x * c * np.exp(sigma * (rho * u + np.sqrt(1 - rho * rho) * eps) - sigma ** 2 / 2)
                    if rho2_of(yy) < rho2:
                        lo_r = rho
                    else:
                        hi_r = rho
            yy = x * c * np.exp(sigma * (rho * u + np.sqrt(1 - rho * rho) * eps) - sigma ** 2 / 2)
            name = f"y_plant_r{int(round(100 * rho2)):02d}_p{p}"
            out[name] = yy
            info.append({"family": "lgn", "y": name, "planting": p, "rho2_target": rho2, "rho": rho,
                         "rho2_achieved": rho2_of(yy)})
    return out, pd.DataFrame(info)


# ----------------------------------------------------------------------------------------------
# primary statistics
def gm_re(df_long, truth, text_arm, comparator, ys, B=2000, seed=0):
    """GM over ys of RE(text | comparator) = exp(mean_y log(MSE_comp / MSE_text)), with a paired
    replicate bootstrap: (point, one-sided 95% lower bound, one-sided 95% upper bound, per-y RE)."""
    piv = df_long[df_long.estimator.isin([text_arm, comparator]) & df_long.y.isin(ys)] \
        .pivot_table(index="rep", columns=["estimator", "y"], values="est")
    se_t = np.stack([(piv[(text_arm, y)].to_numpy() - truth[y]) ** 2 for y in ys], 1)
    se_c = np.stack([(piv[(comparator, y)].to_numpy() - truth[y]) ** 2 for y in ys], 1)
    per_y = se_c.mean(0) / se_t.mean(0)
    point = float(np.exp(np.log(per_y).mean()))
    rng = np.random.default_rng(seed)
    R = len(se_t)
    bs = np.empty(B)
    for b in range(B):
        i = rng.integers(0, R, R)
        bs[b] = np.exp(np.log(se_c[i].mean(0) / se_t[i].mean(0)).mean())
    return point, float(np.quantile(bs, 0.05)), float(np.quantile(bs, 0.95)), dict(zip(ys, per_y))


def verdict(point, lo, hi):
    bar = "MET" if (point >= 1.10 and lo >= 1.05) else ("EXCLUDED" if hi < 1.10 else "OPEN")
    anyg = "YES" if lo > 1.00 else ("NO" if hi < 1.05 else "UNCLEAR")
    return bar, anyg


def shapley(logmse: dict):
    """3-factor Shapley shares. logmse: key = frozenset of active factors among {'A','B','C'}."""
    from itertools import combinations
    from math import factorial
    F = ["A", "B", "C"]
    phi = {}
    for j in F:
        others = [f for f in F if f != j]
        tot = 0.0
        for k in range(3):
            for S in combinations(others, k):
                S = frozenset(S)
                wgt = factorial(len(S)) * factorial(2 - len(S)) / factorial(3)
                tot += wgt * (logmse[S] - logmse[S | {j}])
        phi[j] = tot
    return phi


def ceiling_av(pop, design, F, cat, ynames, lams=(1e-3, 1e-2, 1e-1, 1.0, 10.0), lamcs=(5.0, 20.0, 100.0), k=5, seed=12345):
    """Same-class population ceiling (protocol C1 / §8.4): the y-type ridge fitted on the take-some
    population with 5-fold cross-fitting (D-4), oracle (lam, lam_c) per y; returns AV per y."""
    ta = (design.n_h >= design.N_h)[design.strata]
    idx = np.flatnonzero(~ta)
    folds = np.random.default_rng(seed).permutation(len(idx)) % k
    d = 1 / design.pi
    x = pop.lag_expenses.clip(lower=0).to_numpy(float) + 1
    Y = pop[ynames].to_numpy(float)
    is_bin = np.array([y.startswith("y_bin_") for y in ynames])
    fac = np.zeros(design.H)
    ok = (design.n_h > 0) & (design.n_h < design.N_h)
    fac[ok] = design.N_h[ok] ** 2 * (1 - design.n_h[ok] / design.N_h[ok]) / design.n_h[ok]
    st = design.strata[idx]
    best = np.full(len(ynames), np.inf)
    for jset, ratio in ((np.flatnonzero(~is_bin), True), (np.flatnonzero(is_bin), False)):
        if len(jset) == 0:
            continue
        T = Y[:, jset] / x[:, None] if ratio else Y[:, jset]
        w = d * x if ratio else d
        back = x if ratio else np.ones_like(x)
        lo, hi = (np.percentile(T[idx], [1, 99], axis=0) if ratio else (-np.inf, np.inf))
        grams = [RidgeGram(F[idx[folds != f]], T[idx[folds != f]], w[idx[folds != f]], cat, 0) for f in range(k)]
        for lc in (lamcs if (cat is not None and np.any(cat)) else (20.0,)):
            for l in lams:
                E = np.zeros((len(idx), len(jset)))
                for f, g in enumerate(grams):
                    te = idx[folds == f]
                    b0, Bx = g.coef(l, lc, np.arange(len(jset)))
                    E[folds == f] = (T[te] - np.clip(b0 + F[te] @ Bx, lo, hi)) * back[te][:, None]
                av = np.zeros(len(jset))
                for h in np.unique(st):
                    m = st == h
                    if m.sum() > 1:
                        av += fac[h] * E[m].var(0, ddof=1)
                best[jset] = np.minimum(best[jset], av)
    return best
