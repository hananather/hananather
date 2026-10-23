"""Build auxiliary matrices and estimator specs for one population + design."""
from __future__ import annotations

import numpy as np
import pandas as pd

from frame import dummies, pcs
from mc import Cell


def text_groups(pop: pd.DataFrame) -> np.ndarray:
    """Cluster ids for near-duplicate mission texts (chapters/affiliates using template missions).
    Used to keep duplicates in the same cross-fitting fold."""
    norm = (pop.text_mission.fillna("").str.lower().str.replace(r"[^a-z ]", " ", regex=True)
            .str.replace(r"\s+", " ", regex=True).str.strip())
    return pd.factorize(norm)[0]


def impute_ntee(pop: pd.DataFrame, E: np.ndarray, col="ntee_major10", missing="MISSING", seed=0):
    """Text-coded industry: multinomial logistic regression on embeddings, trained on frame units
    with a known code, predicting units whose code is missing. Uses no survey variable.
    Returns (imputed series, 5-fold CV accuracy on known units)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    known = (pop[col] != missing).to_numpy()
    clf = LogisticRegression(C=2.0, max_iter=2000)
    acc = cross_val_score(clf, E[known], pop[col][known], cv=5).mean()
    clf.fit(E[known], pop[col][known])
    out = pop[col].copy()
    out[~known] = clf.predict(E[~known])
    return out, float(acc)


def kmeans_groups(E: np.ndarray, k=10, seed=0):
    from sklearn.cluster import KMeans
    return KMeans(k, n_init=4, random_state=seed).fit_predict(E)


def build_mats(pop: pd.DataFrame, design, embeddings: dict, K_add=10, K_ratio=32):
    """embeddings: name -> (N, D) array. Returns (mats dict, size vector, emb_names)."""
    S = dummies(pd.Series(design.strata), prefix="h").to_numpy()
    x = pop.lag_expenses.clip(lower=0).to_numpy(float)
    emp = pop.lag_employees.clip(lower=0).to_numpy(float)
    lag = pop[["lag_revenue", "lag_expenses", "lag_assets", "lag_employees"]].to_numpy(float)
    cat = np.c_[dummies(pop.ntee_major10, True).to_numpy(), dummies(pop.state, True).to_numpy()]
    inc = dummies(pop.bmf_income_cd.fillna(-1), True).to_numpy()
    mats = {
        "strata": S,
        "ratio": np.c_[S, x],
        "size": np.c_[S, lag],
        "size_cat": np.c_[S, lag, cat],
        "sep": np.c_[S, S * x[:, None]],
        "weak": np.c_[S, cat, inc],
    }
    # text-free ML features: categorical x size interactions (what a modeller could do without text)
    ntee = dummies(pop.ntee_major10).to_numpy()
    mats["ml_frame"] = np.c_[S, lag, cat, ntee * x[:, None], ntee * emp[:, None], inc]
    mats["ml_frame_ratio"] = np.c_[cat, inc, np.log1p(np.clip(lag, 0, None))]  # ratio-scale features
    if "ntee_imputed" in pop:
        cat_imp = np.c_[dummies(pop.ntee_imputed, True).to_numpy(), dummies(pop.state, True).to_numpy()]
        mats["size_catimp"] = np.c_[S, lag, cat_imp]
    sx, se = x / x.std(), emp / max(emp.std(), 1e-9)
    names = []
    for nm, E in embeddings.items():
        Z, _ = pcs(np.asarray(E, float), max(K_add, K_ratio))
        Za, Zr = Z[:, :K_add], Z[:, :K_ratio]
        mats[f"size_cat+{nm}:add{K_add}"] = np.c_[mats["size_cat"], Za]
        mats[f"size_cat+{nm}:ratio{K_ratio}"] = np.c_[mats["size_cat"], Zr * x[:, None], Zr * emp[:, None]]
        mats[f"ml+{nm}"] = np.c_[mats["ml_frame"], Zr, Zr * sx[:, None] * x.std(), Zr * se[:, None] * emp.std()]
        mats[f"mlr+{nm}"] = np.c_[mats["ml_frame_ratio"], Zr]
        mats[f"weak+{nm}:add{K_ratio}"] = np.c_[mats["weak"], Zr]
        names.append(nm)
    return mats, x, names


def conventional_specs():
    return {
        "HT": {"kind": "ht"},
        "GREG_ratio": {"kind": "greg", "X": "ratio"},
        "GREG_size": {"kind": "greg", "X": "size"},
        "GREG_size_cat": {"kind": "greg", "X": "size_cat"},
        "GREG_sep": {"kind": "greg", "X": "sep"},
        "GREG_weak": {"kind": "greg", "X": "weak"},
        # text-free machine learning (fair comparator for text ML)
        "MCAL_frame_raw": {"kind": "mcal", "F": "ml_frame", "hard": 0, "X_base": "size", "cf": True},
        "MCAL_frame_ratio": {"kind": "mcal", "F": "ml_frame_ratio", "hard": 0, "X_base": "size", "cf": True, "ratio": True},
        "MCAL_frame_ratio_sep": {"kind": "mcal", "F": "ml_frame_ratio", "hard": 0, "X_base": "sep", "cf": True, "ratio": True},
    }


def embedding_specs(nm, K_add=10, K_ratio=32, full=True, n_hard_size_cat=None):
    sp = {
        f"GREG_size_cat+{nm}:add{K_add}": {"kind": "greg", "X": f"size_cat+{nm}:add{K_add}"},
        f"RIDGE_size_cat+{nm}:ratio{K_ratio}": {"kind": "ridge", "X": f"size_cat+{nm}:ratio{K_ratio}", "hard": n_hard_size_cat},
        f"MCAL_raw+{nm}": {"kind": "mcal", "F": f"ml+{nm}", "hard": 0, "X_base": "size", "cf": True},
        f"MCAL_ratio+{nm}": {"kind": "mcal", "F": f"mlr+{nm}", "hard": 0, "X_base": "size", "cf": True, "ratio": True},
        # text model calibrated on top of the strongest conventional calibration (separate ratios by stratum)
        f"MCAL_ratio_sep+{nm}": {"kind": "mcal", "F": f"mlr+{nm}", "hard": 0, "X_base": "sep", "cf": True, "ratio": True},
        f"GREG_weak+{nm}:add{K_ratio}": {"kind": "greg", "X": f"weak+{nm}:add{K_ratio}"},
    }
    if full:
        sp[f"MCAL_ratio+{nm}:nocf"] = {"kind": "mcal", "F": f"mlr+{nm}", "hard": 0, "X_base": "size", "cf": False, "ratio": True}
        sp[f"SYN_ratio+{nm}"] = {"kind": "syn", "F": f"mlr+{nm}", "hard": 0, "ratio": True}
    return sp


def nonresponse_specs(emb_names, ps_C=1.0):
    """Late filers as nonrespondents. All use the respondents only, with adjusted weights."""
    sp = {
        "NR_naive_HT": {"kind": "ht", "nr": True, "nr_adj": "none"},
        "NR_rhg_HT": {"kind": "ht", "nr": True, "nr_adj": "rhg"},
        "NR_rhg_GREG_size_cat": {"kind": "greg", "X": "size_cat", "nr": True, "nr_adj": "rhg"},
        "NR_ps_frame_GREG_size_cat": {"kind": "greg", "X": "size_cat", "nr": True, "nr_adj": "ps", "ps_F": "weak", "ps_C": ps_C},
        "NR_rhg_MCAL_frame_ratio": {"kind": "mcal", "F": "ml_frame_ratio", "hard": 0, "X_base": "size", "cf": True, "ratio": True, "nr": True, "nr_adj": "rhg"},
    }
    for nm in emb_names:
        sp[f"NR_ps_text_GREG_size_cat+{nm}"] = {"kind": "greg", "X": "size_cat", "nr": True, "nr_adj": "ps", "ps_F": f"weak+{nm}:add32", "ps_C": ps_C}
        sp[f"NR_rhg_MCAL_ratio+{nm}"] = {"kind": "mcal", "F": f"mlr+{nm}", "hard": 0, "X_base": "size", "cf": True, "ratio": True, "nr": True, "nr_adj": "rhg"}
        sp[f"NR_ps_text_MCAL_ratio+{nm}"] = {"kind": "mcal", "F": f"mlr+{nm}", "hard": 0, "X_base": "size", "cf": True, "ratio": True, "nr": True, "nr_adj": "ps", "ps_F": f"weak+{nm}:add32", "ps_C": ps_C}
    return sp


def make_cell(pop, design, ynames, embeddings, groups=None, resp=None, frozen=None, **kw):
    mats, x, names = build_mats(pop, design, embeddings, **kw)
    Y = pop[ynames].to_numpy(float)
    cell = Cell(Y, ynames, design, mats, x, groups=groups, resp=resp, frozen=frozen)
    return cell, names
