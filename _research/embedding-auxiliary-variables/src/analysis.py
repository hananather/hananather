"""Stage 2A analysis: validity gates, primary pairs, secondaries, attribution, planted recovery
(protocol §§4.4, 4.5, 5, 9). Works on the long-format Monte Carlo output of mc.run_cell."""
from __future__ import annotations

import numpy as np
import pandas as pd

from protocol import H_SET, H_D, H_S, R_SET, gm_re, shapley, verdict


def paired_re(df, truth, a, b, y, B=2000, seed=0):
    """RE(a | b) = MSE_b / MSE_a on paired replicates, with bootstrap 90% interval (one-sided 95% bounds)."""
    p = df[(df.y == y) & df.estimator.isin([a, b])].pivot(index="rep", columns="estimator", values="est")
    ea, eb = (p[a] - truth[y]) ** 2, (p[b] - truth[y]) ** 2
    re = eb.mean() / ea.mean()
    rng = np.random.default_rng(seed)
    R = len(p)
    bs = np.array([(eb.to_numpy()[i].mean() / ea.to_numpy()[i].mean()) for i in (rng.integers(0, R, R) for _ in range(B))])
    return re, np.quantile(bs, 0.05), np.quantile(bs, 0.95)


def cell_metrics(df, truth, z=1.959964):
    out = []
    for (e, y), g in df.groupby(["estimator", "y"]):
        T = truth[y]
        err = g.est.to_numpy() - T
        R = len(err)
        row = {"estimator": e, "y": y, "R": R, "rb_pct": 100 * err.mean() / T,
               "rb_mcse_pct": 100 * err.std(ddof=1) / np.sqrt(R) / abs(T), "rrmse_pct": 100 * np.sqrt(np.mean(err ** 2)) / abs(T)}
        v = g["var"].to_numpy()
        if np.isfinite(v).all():
            se = np.sqrt(np.maximum(v, 0))
            cov = np.abs(err) <= z * se
            row.update({"coverage_pct": 100 * cov.mean(), "var_rb_pct": 100 * (v.mean() / np.var(g.est, ddof=1) - 1),
                        "stud_sd": float(np.std(err / np.where(se > 0, se, np.nan), ddof=1))})
        out.append(row)
    return pd.DataFrame(out)


def gates(df, truth, T="F1p", E="Estar", controls=("N1random", "N2perm", "N3permh"), const="N4const"):
    """Validity gates G1-G9 (protocol §9.2). Returns a DataFrame of (gate, item, value, pass)."""
    rows = []
    cm = cell_metrics(df, truth).set_index(["estimator", "y"])
    real = [y for y in truth if not y.startswith(("y_plc", "y_plant", "y_bin_plc"))]
    # G1 design sanity
    for y in real:
        for e in ["HT", f"FROZEN_diff_bmfmax@{T}", f"FROZEN_diff_bmfmax@{T}+{E}"]:
            if (e, y) in cm.index:
                r = cm.loc[(e, y)]
                rows.append(("G1", f"{e} | {y}", round(r.rb_pct, 3), abs(r.rb_pct) <= 3 * r.rb_mcse_pct))
    # G2 negative controls: upper bound of RE vs twin <= 1.03 for each y in H; N4 exact
    twin = f"MCAL_bmfmax@{T}"
    for c in controls:
        for y in H_SET:
            re, lo, hi = paired_re(df, truth, f"MCAL_bmfmax@{T}+{c}", twin, y)
            rows.append(("G2", f"{c} | {y}", round(re, 3), hi <= 1.03))
    piv = df[df.estimator.isin([twin, f"MCAL_bmfmax@{T}+{const}"])].pivot_table(index=["rep", "y"], columns="estimator", values="est")
    rel = float((np.abs(piv.iloc[:, 0] - piv.iloc[:, 1]) / np.abs(piv[twin])).max())
    rows.append(("G2", "N4 constant = twin (max rel diff)", rel, rel <= 1e-8))
    # G3 placebo: every main arm vs B0 and text vs twin on placebo targets
    for y in [y for y in truth if y.startswith(("y_plc", "y_bin_plc"))]:
        re, lo, hi = paired_re(df, truth, f"MCAL_bmfmax@{T}+{E}", twin, y)
        rows.append(("G3", f"text vs twin | {y}", round(re, 3), hi <= 1.03))
    # G4 text-arm validity on all real y: coverage gap to comparator and to HT, variance bias
    for arm, comp in [(f"MCAL_bmfmax@{T}+{E}", twin), (f"FROZEN_calm_bmfmax@{T}+{E}", f"GREG_size_cat@{T}")]:
        for y in real:
            if (arm, y) not in cm.index or "coverage_pct" not in cm.columns:
                continue
            a, c, h = cm.loc[(arm, y)], cm.loc[(comp, y)], cm.loc[("HT", y)]
            ok = (a.coverage_pct - c.coverage_pct >= -2) and (a.coverage_pct - h.coverage_pct >= -5) and abs(a.var_rb_pct) <= 15
            rows.append(("G4", f"{arm.split('@')[0]} | {y}",
                         f"cov {a.coverage_pct:.1f} (comp {c.coverage_pct:.1f}, HT {h.coverage_pct:.1f}); varRB {a.var_rb_pct:.0f}%", ok))
    # G7 register-type y: upper bound <= 1.03; lower bound > 1.03 = leakage alarm
    for y in R_SET:
        re, lo, hi = paired_re(df, truth, f"MCAL_bmfmax@{T}+{E}", twin, y)
        rows.append(("G7", y, round(re, 3), hi <= 1.03 and lo <= 1.03))
    # G8 numbers masked: |log RE(text) - log RE(text_nonum)| <= 0.03 (GM over H)
    p1, *_ = gm_re(df, truth, f"MCAL_bmfmax@{T}+{E}", twin, H_SET, B=200)
    p2, *_ = gm_re(df, truth, f"MCAL_bmfmax@{T}+{E}_nonum", twin, H_SET, B=200)
    rows.append(("G8", "GM-RE text vs numbers-masked", f"{p1:.3f} vs {p2:.3f}", abs(np.log(p1) - np.log(p2)) <= 0.03))
    # G9 oracle: F2 twin vs F1p twin on gov grants, fundraising, officer comp: lower bound > 2
    for y in ["y_gov_grants", "y_fundraising_exp", "y_officer_comp"]:
        re, lo, hi = paired_re(df, truth, "MCAL_bmfmax@F2", twin, y)
        rows.append(("G9", y, round(re, 2), lo > 2))
    return pd.DataFrame(rows, columns=["gate", "item", "value", "pass"])


def primary(df, truth, T="F1p", E="Estar"):
    rows = []
    pairs = {
        "P1 efficacy": (f"MCAL_bmfmax@{T}+{E}", f"MCAL_bmfmax@{T}"),
        "P1' (NTEE10 twin)": (f"MCAL_frame@{T}+{E}", f"MCAL_frame@{T}"),
        "P2 production": (f"FROZEN_calm_bmfmax@{T}+{E}", f"GREG_size_cat@{T}"),
        "P2-twin (no text)": (f"FROZEN_calm_bmfmax@{T}", f"GREG_size_cat@{T}"),
        "P2' (current-sample single weight)": (f"MCALm_bmfmax@{T}+{E}", f"GREG_size_cat@{T}"),
        "P2 text increment (frozen text vs frozen twin)": (f"FROZEN_calm_bmfmax@{T}+{E}", f"FROZEN_calm_bmfmax@{T}"),
        "S-F1 (no payroll register)": (f"MCAL_bmfmax@F1+{E}", "MCAL_bmfmax@F1"),
        "S-F2 (full prior return)": (f"MCAL_bmfmax@F2+{E}", "MCAL_bmfmax@F2"),
        "secondary: E* K=64": (f"MCAL_bmfmax@{T}+Estar64", f"MCAL_bmfmax@{T}"),
        "ladder L4: MiniLM T2": (f"MCAL_bmfmax@{T}+L4", f"MCAL_bmfmax@{T}"),
    }
    for name, (a, b) in pairs.items():
        if a not in set(df.estimator) or b not in set(df.estimator):
            continue
        for label, ys in [("H", H_SET), ("H_D", H_D), ("H_S", H_S)]:
            pt, lo, hi, per = gm_re(df, truth, a, b, ys)
            bar, anyg = verdict(pt, lo, hi)
            rows.append({"pair": name, "set": label, "text_arm": a, "comparator": b, "gm_re": pt, "lo95": lo, "hi95": hi,
                         "production_bar": bar, "any_gain": anyg, **{f"re:{k}": v for k, v in per.items()}})
    return pd.DataFrame(rows)


def no_harm(df, truth, arm, comp):
    """P2 no-harm condition: min point RE over all real y >= 0.95 and no y with upper bound < 0.95."""
    real = [y for y in truth if not y.startswith(("y_plc", "y_plant", "y_bin_plc"))]
    res = [(y,) + paired_re(df, truth, arm, comp, y, B=500) for y in real]
    t = pd.DataFrame(res, columns=["y", "re", "lo95", "hi95"])
    return t, bool(t.re.min() >= 0.95 and (t.hi95 >= 0.95).all())


def factorial(df, truth, T="F1p", E="Estar", ys=H_SET):
    cells = {frozenset(): f"GREG_size_cat@{T}", frozenset("C"): f"RIDGECAL@{T}+{E}", frozenset("B"): f"RIDGECAL@{T}+C",
             frozenset("BC"): f"RIDGECAL@{T}+C+{E}", frozenset("A"): f"MCAL_frame@{T}", frozenset("AC"): f"MCAL_frame@{T}+{E}",
             frozenset("AB"): f"MCAL_bmfmax@{T}", frozenset("ABC"): f"MCAL_bmfmax@{T}+{E}"}
    rows = []
    for y in ys:
        lm = {}
        for k, e in cells.items():
            g = df[(df.estimator == e) & (df.y == y)]
            lm[k] = np.log(np.mean((g.est - truth[y]) ** 2))
        phi = shapley(lm)
        tot = lm[frozenset()] - lm[frozenset("ABC")]
        rows.append({"y": y, "log_RE_total": tot, "phi_A_learner": phi["A"], "phi_B_codes": phi["B"], "phi_C_text": phi["C"]})
    t = pd.DataFrame(rows)
    mean = t[["log_RE_total", "phi_A_learner", "phi_B_codes", "phi_C_text"]].mean()
    return t, mean


def planted(df, truth, pinfo, T="F1p", E="Estar"):
    """Realised RE of text vs twin on planted columns against theory 1/(1-rho^2); phi = log RE_MC / log RE_theory."""
    rows = []
    for _, r in pinfo[pinfo.family == "lgn"].iterrows():
        re, lo, hi = paired_re(df, truth, f"MCAL_bmfmax@{T}+{E}", f"MCAL_bmfmax@{T}", r.y, B=500)
        th = 1 / (1 - r.rho2_achieved) if r.rho2_achieved > 0 else 1.0
        rows.append({"y": r.y, "rho2": r.rho2_target, "rho2_achieved": r.rho2_achieved, "re_mc": re, "lo95": lo, "hi95": hi,
                     "re_theory": th, "phi": (np.log(re) / np.log(th)) if th > 1 else np.nan})
    return pd.DataFrame(rows)
