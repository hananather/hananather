"""Helpers for the results notebook: estimator families, tables with Monte Carlo SEs, figures."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

import plotstyle

Y_LABEL = {
    "y_expenses": "Total expenses", "y_revenue": "Total revenue", "y_employees": "Employees",
    "y_salaries": "Salaries", "y_gov_grants": "Government grants", "y_prog_rev": "Program service revenue",
    "y_contrib": "Contributions", "y_fundraising_exp": "Fundraising expenses", "y_officer_comp": "Officer compensation",
    "y_volunteers_cap10k": "Volunteers (capped 10k)", "y_volunteers": "Volunteers (raw)",
    "y_bin_has_gov_grants": "Share with gov. grants", "y_bin_grants_to_indiv_us": "Share granting to individuals",
    "y_bin_lobbying": "Share lobbying", "y_bin_has_volunteers": "Share with volunteers",
}
Y_TYPE = {
    "y_expenses": "register-type", "y_revenue": "register-type", "y_employees": "register-type",
    "y_salaries": "admin-in-practice", "y_gov_grants": "survey-only", "y_prog_rev": "survey-only",
    "y_contrib": "survey-only", "y_fundraising_exp": "survey-only", "y_officer_comp": "survey-only",
    "y_volunteers_cap10k": "survey-only (heavy-tailed)", "y_volunteers": "survey-only (heavy-tailed)",
    "y_bin_has_gov_grants": "proportion", "y_bin_grants_to_indiv_us": "proportion",
    "y_bin_lobbying": "proportion", "y_bin_has_volunteers": "proportion",
}
CONTROL_EMB = ("random", "PERM_")


def family(est: str) -> str:
    if any(c in est for c in CONTROL_EMB):
        return "negative control"
    if est.endswith(":nocf") or est.startswith("SYN"):
        return "pitfall demo"
    if est.startswith("NR_"):
        return "nonresponse"
    if est in ("HT", "GREG_ratio", "GREG_size", "GREG_size_cat", "GREG_sep", "GREG_weak"):
        return "status quo"
    if est.startswith("MCAL_frame"):
        return "text-free ML"
    if est.startswith("FROZEN_frame"):
        return "text-free ML"
    if "weak+" in est:
        return "text (weak register)"
    return "text"


def emb_of(est: str):
    m = re.search(r"\+([A-Za-z0-9_]+?)(?::|$|_add|_ratio)", est)
    return m.group(1) if m else None


def best_conventional(m: pd.DataFrame) -> pd.Series:
    """Per y, the status-quo or text-free-ML estimator with the smallest MC MSE (conservative comparator)."""
    c = m[m.estimator.map(family).isin(["status quo", "text-free ML"]) & (m.estimator != "GREG_weak")]
    return c.loc[c.groupby("y").rrmse_pct.idxmin()].set_index("y").estimator


def fmt_re(re, se):
    return f"{re:.2f} ± {1.96 * se:.2f}"


def re_vs(df_long: pd.DataFrame, truth: dict, base_by_y: dict | str):
    """RE of every estimator vs a per-y baseline estimator, with paired-replicate delta-method SE."""
    rows = []
    piv = df_long.pivot_table(index=["rep", "y"], columns="estimator", values="est")
    for y in piv.index.get_level_values("y").unique():
        b = base_by_y[y] if isinstance(base_by_y, dict) else base_by_y
        sub = piv.xs(y, level="y")
        eb = (sub[b] - truth[y]) ** 2
        for est in sub.columns:
            ee = (sub[est] - truth[y]) ** 2
            if ee.isna().any():
                continue
            R = len(ee)
            re_ = eb.mean() / ee.mean()
            c = np.cov(ee, eb)
            vl = c[0, 0] / (R * ee.mean() ** 2) + c[1, 1] / (R * eb.mean() ** 2) - 2 * c[0, 1] / (R * ee.mean() * eb.mean())
            rows.append({"y": y, "estimator": est, "baseline": b, "re": re_, "re_se": re_ * np.sqrt(max(vl, 0))})
    return pd.DataFrame(rows)


def cv_band(cv_pct: float) -> str:
    """Statistics Canada's commonly used CV quality letters (confirm against the program's own guideline)."""
    for lim, b in [(5, "A"), (10, "B"), (15, "C"), (25, "D"), (35, "E")]:
        if cv_pct < lim:
            return b
    return "F"


def dotplot(ax, tab: pd.DataFrame, ys, ests, labels=None, xlim=(0.5, 3.0)):
    """Dot plot of RE (log x) with 95% MC intervals; one row per y, one colour per estimator (fixed order)."""
    labels = labels or {e: e for e in ests}
    k = len(ests)
    offs = np.linspace(-0.28, 0.28, k) if k > 1 else [0]
    for j, e in enumerate(ests):
        sub = tab[tab.estimator == e].set_index("y").reindex(ys)
        yy = np.arange(len(ys)) + offs[j]
        ax.errorbar(sub.re, yy, xerr=1.96 * sub.re_se, fmt="o", ms=5, lw=1.2, capsize=0,
                    color=plotstyle.SERIES[j % len(plotstyle.SERIES)], label=labels[e])
    ax.axvline(1.0, color=plotstyle.NEUTRAL, lw=1, zorder=0)
    ax.set_xscale("log")
    ticks = [0.5, 0.75, 1, 1.25, 1.5, 2, 3]
    ax.set_xticks([t for t in ticks if xlim[0] <= t <= xlim[1]])
    ax.set_xticklabels([f"{t:g}" for t in ticks if xlim[0] <= t <= xlim[1]])
    ax.set_xlim(*xlim)
    ax.set_yticks(np.arange(len(ys)))
    ax.set_yticklabels([Y_LABEL.get(y, y) for y in ys])
    ax.invert_yaxis()
    ax.set_xlabel("relative efficiency vs comparator (MSE ratio, log scale; >1 = text better)")
    ax.legend(loc="lower right", fontsize=8)
