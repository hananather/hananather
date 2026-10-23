"""Frame-side constructions shared by the diagnostics and the Monte Carlo:
production-style stratification, auxiliary matrices, embedding principal components.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from survey import StratifiedDesign, cum_sqrt_f_boundaries, neyman_allocation


def production_design(pop: pd.DataFrame, n: int, size_col="lag_expenses", group_col="ntee_major10",
                      n_size_classes=3, take_all_share=0.01, min_n=10, alloc_power=0.5):
    """Business-survey style design: group x size-class strata (cum-sqrt-f on log size within the
    whole population), a take-all stratum for the largest units, Neyman allocation on size."""
    x = pop[size_col].clip(lower=0).to_numpy(float)
    z = np.log1p(x)
    b = cum_sqrt_f_boundaries(z, n_size_classes)
    size_cls = np.searchsorted(b, z)
    g = pd.factorize(pop[group_col])[0]
    lab = g * n_size_classes + size_cls
    lab = pd.factorize(lab)[0]
    n_take_all = int(round(take_all_share * len(pop)))
    ta = np.zeros(len(pop), bool)
    ta[np.argsort(-x)[:n_take_all]] = True
    lab = np.where(ta, lab.max() + 1, lab)
    lab = pd.factorize(lab)[0]
    H = lab.max() + 1
    take_all_mask = np.zeros(H, bool)
    take_all_mask[np.unique(lab[ta])] = True
    # multipurpose compromise: Neyman allocation on x**alloc_power (0.5 = sqrt), with a minimum
    # stratum sample size, so that counts and proportions are not starved in small-unit strata
    n_h = neyman_allocation(lab, x ** alloc_power, n, min_n=min_n, take_all=take_all_mask)
    return StratifiedDesign(lab, n_h), {"size_boundaries_log": b, "take_all_n": int(ta.sum()),
                                        "strata": int(H)}


def dummies(s: pd.Series, drop_first=False, prefix=None):
    return pd.get_dummies(s.astype(str), drop_first=drop_first, prefix=prefix, dtype=float)


def frame_matrix(pop: pd.DataFrame, design: StratifiedDesign, level: str):
    """Auxiliary matrices for the conventional (status-quo) estimators.
    level='weak'   : stratum indicators + NTEE10 + state + BMF income-code class counts
    level='strong' : weak + lagged revenue / expenses / assets / employees, and stratum x lagged expenses
                     (separate-ratio-by-stratum), i.e. what a register with prior-year admin data allows.
    """
    S = dummies(pd.Series(design.strata), prefix="h")
    parts = [S, dummies(pop.state, drop_first=True, prefix="st"),
             dummies(pop.ntee_major10, drop_first=True, prefix="nt"),
             dummies(pop.bmf_income_cd.fillna(-1), drop_first=True, prefix="ic")]
    if level == "strong":
        lag = pop[["lag_revenue", "lag_expenses", "lag_assets", "lag_employees"]].astype(float)
        parts.append(lag.reset_index(drop=True))
        parts.append(S.mul(pop.lag_expenses.to_numpy(float), axis=0).add_prefix("x_"))
    X = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
    X = X.loc[:, X.std() > 0]
    return X.to_numpy(float), list(X.columns)


def pcs(E: np.ndarray, k: int):
    """Population principal components (no y involved, so a legitimate frame transform)."""
    Ec = E - E.mean(0)
    _, s, Vt = np.linalg.svd(Ec, full_matrices=False)
    Z = Ec @ Vt[:k].T
    return Z / Z.std(0), s[:k] ** 2 / (s ** 2).sum()
