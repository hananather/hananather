# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 05 — Protocol v1.0, Stage 2B: does text help a weak register?
#
# **The scenario.** The register has no return-based lags: only IRS codes, state, and the BMF income and
# asset amounts and codes. This is tier F0. Real cases are new registrants, units without tax accounts,
# 990-N filers and non-charity NPOs on a business-register-only frame. The design follows the tier:
# **D1-W** stratifies NTEE10 × 3 BMF-income classes plus a take-all of the top 1% on BMF income.
#
# **Hypotheses (protocol §9.4, secondary)**
# - **S2, text helps a weak register.** GM over R ∪ H_D of RE(`MCAL_bmfmax@F0+E*` ∣ `MCAL_bmfmax@F0`) must reach point ≥ 1.20 with lower bound ≥ 1.10.
# - **S2b, text substitutes for admin data.** The gap closure G = log RE(F0 + text ∣ F0) / log RE(F1p ∣ F0) is the share of the admin-lag gain that text recovers. G ≥ 0.5 means "substitutes".
#
# **Variants (deviation D-28a)**
# - **w** is exactly as pre-registered.
# - **wprof** is a sensitivity. The 703 units with zero or missing BMF amounts (6.7% of expenses, including Harvard) would sit in the bottom size classes under w. In wprof the NSO's 206 largest enterprises are profiled into take-all, as real business registers do.
#
# **Text** is the same E* (TY2022 990 mission + programs). This is richer than BN registration text, so the setting favours text.

# %%
import os, sys, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
PROJ = os.path.abspath("..") if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
sys.path.insert(0, os.path.join(PROJ, "src"))
DATA = os.environ.get("EMBAUX_DATA", "/tmp/claude-0/data")
import plotstyle; plotstyle.use()
from protocol import H_SET, H_D, H_S, R_SET, gm_re, verdict
from analysis import cell_metrics, paired_re
from report import Y_LABEL
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
RES = os.path.join(PROJ, "results", "stage2b_ne_v3"); os.makedirs(RES, exist_ok=True)
S2_Y = R_SET + H_D
runs = {}
for v in ["w", "wprof"]:
    p = f"{DATA}/results/stage2b_ne_v3_{v}"
    if os.path.exists(f"{p}/mc.parquet"):
        runs[v] = (pd.read_parquet(f"{p}/mc.parquet"), json.load(open(f"{p}/meta.json")))
        print(v, "R =", runs[v][1]["R"], "failures =", sum(runs[v][1]["failures"].values()))

# %% [markdown]
# ## 1. The register defect behind D1-W

# %%
pop = pd.read_parquet(f"{DATA}/pop/ne_pop_v3.parquet")
z = pop.bmf_income_amt <= 0
miss = pop.bmf_income_cd.isna()
print(f"zero or missing BMF income amount: {z.sum()} units ({z.mean():.1%}); not in the 2023-08 BMF at all: {miss.sum()}")
print(f"their share of TY2023 expenses: {pop.y_expenses[z].sum() / pop.y_expenses.sum():.1%}; "
      f"of government grants: {pop.y_gov_grants[z].sum() / pop.y_gov_grants.sum():.1%}")
display(pop[z].nlargest(8, "lag_expenses")[["bmf_name", "lag_expenses", "bmf_income_amt", "bmf_income_cd", "bmf_subsection"]])
print("corr(log BMF income, log TY2023 revenue) =", round(np.corrcoef(np.log1p(pop.bmf_income_amt.clip(lower=0)), np.log1p(pop.y_revenue.clip(lower=0)))[0, 1], 3),
      "| corr(log TY2022 revenue, log TY2023 revenue) =", round(np.corrcoef(np.log1p(pop.lag_revenue.clip(lower=0)), np.log1p(pop.y_revenue.clip(lower=0)))[0, 1], 3))

# %% [markdown]
# ## 2. Validity: bias, coverage, negative control

# %%
for v, (df, meta) in runs.items():
    cm = cell_metrics(df, meta["truth"])
    arms = ["HT", "GREG_size_cat@F0", "MCAL_bmfmax@F0", "MCAL_bmfmax@F0+Estar", "MCAL_bmfmax@F0+N2perm", "GREG_size_cat@F1p", "MCAL_bmfmax@F1p"]
    sub = cm[cm.estimator.isin(arms)]
    for col, lab in [("rb_pct", "relative bias %"), ("coverage_pct", "95% CI coverage %"), ("rrmse_pct", "RRMSE %")]:
        t = sub.pivot(index="y", columns="estimator", values=col)[arms]; t.index = [Y_LABEL.get(i, i) for i in t.index]
        print(f"\n=== {v}: {lab} ==="); display(t.round(2))
    cm.to_csv(f"{RES}/cell_metrics_{v}.csv", index=False)
    re, lo, hi = zip(*[paired_re(df, meta["truth"], "MCAL_bmfmax@F0+N2perm", "MCAL_bmfmax@F0", y, B=500) for y in S2_Y + H_S])
    print(f"{v}: negative control N2perm vs twin, max upper bound over y = {max(hi):.3f} (gate: <= 1.03)")

# %% [markdown]
# ## 3. S2 and S2b

# %%
rows = []
for v, (df, meta) in runs.items():
    tr = meta["truth"]
    pairs = {"S2: F0 text vs F0 twin": ("MCAL_bmfmax@F0+Estar", "MCAL_bmfmax@F0"),
             "F0 twin vs F0 GREG (what ML on codes adds)": ("MCAL_bmfmax@F0", "GREG_size_cat@F0"),
             "F0 text vs F0 GREG": ("MCAL_bmfmax@F0+Estar", "GREG_size_cat@F0"),
             "F0 NTEE10-twin text increment": ("MCAL_frame@F0+Estar", "MCAL_frame@F0"),
             "F0 E* K=64": ("MCAL_bmfmax@F0+Estar64", "MCAL_bmfmax@F0"),
             "F0 MiniLM (L4)": ("MCAL_bmfmax@F0+L4", "MCAL_bmfmax@F0"),
             "F1p GREG vs F0 twin (what lags add)": ("GREG_size_cat@F1p", "MCAL_bmfmax@F0"),
             "F1p twin vs F0 twin (what lags add, ML)": ("MCAL_bmfmax@F1p", "MCAL_bmfmax@F0"),
             "F1p text vs F1p twin (P1 on D1-W samples)": ("MCAL_bmfmax@F1p+Estar", "MCAL_bmfmax@F1p")}
    for name, (a, b) in pairs.items():
        for lab, ys in [("R ∪ H_D", S2_Y), ("H", H_SET), ("R", R_SET), ("H_D", H_D)]:
            pt, lo, hi, per = gm_re(df, tr, a, b, ys)
            bar, anyg = verdict(pt, lo, hi)
            rows.append({"variant": v, "pair": name, "set": lab, "gm_re": pt, "lo95": lo, "hi95": hi, "any_gain": anyg,
                         "S2 bar (>=1.20, lo>=1.10)": ("MET" if (pt >= 1.20 and lo >= 1.10) else ("EXCLUDED" if hi < 1.20 else "OPEN")),
                         **{f"re:{k}": x for k, x in per.items()}})
S2 = pd.DataFrame(rows)
S2.to_csv(f"{RES}/s2_pairs.csv", index=False)
display(S2[S2.set == "R ∪ H_D"][["variant", "pair", "gm_re", "lo95", "hi95", "any_gain", "S2 bar (>=1.20, lo>=1.10)"]].round(3))

# %%
gap = []
for v in runs:
    t = S2[(S2.variant == v) & (S2.set == "R ∪ H_D")].set_index("pair")
    num = np.log(t.loc["S2: F0 text vs F0 twin", "gm_re"])
    for ref in ["F1p GREG vs F0 twin (what lags add)", "F1p twin vs F0 twin (what lags add, ML)"]:
        den = np.log(t.loc[ref, "gm_re"])
        gap.append({"variant": v, "F1p reference": ref.split(" vs")[0], "log RE text": num, "log RE lags": den, "gap closure G": num / den if den > 0 else np.nan})
gap = pd.DataFrame(gap); display(gap.round(3)); gap.to_csv(f"{RES}/s2b_gap_closure.csv", index=False)
for v in runs:
    g = gap[gap.variant == v]
    gbest = g.loc[g["log RE lags"].idxmax(), "gap closure G"]
    print(f"{v}: S2b gap closure against the better F1p reference = {gbest:.3f} -> {'substitutes (>= 0.5)' if gbest >= 0.5 else 'does not substitute (< 0.5)'}")

# %% [markdown]
# ### Per-variable RE of text in the weak register

# %%
per = S2[(S2.set == "R ∪ H_D") & S2.pair.isin(["S2: F0 text vs F0 twin", "F1p twin vs F0 twin (what lags add, ML)"])]
per = per.set_index(["variant", "pair"])[[f"re:{y}" for y in S2_Y]]
per.columns = [Y_LABEL.get(c[3:], c[3:]) for c in per.columns]
display(per.T.round(3))

# %%
if runs:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labs = [Y_LABEL.get(y, y) for y in S2_Y]
    yy = np.arange(len(S2_Y))
    for j, v in enumerate(runs):
        r = S2[(S2.variant == v) & (S2.set == "R ∪ H_D") & (S2.pair == "S2: F0 text vs F0 twin")].iloc[0]
        ax.scatter([r[f"re:{y}"] for y in S2_Y], yy + (j - 0.5) * 0.25, s=28, color=plotstyle.SERIES[j], label=f"text vs twin, F0 ({v})", zorder=3)
        r2 = S2[(S2.variant == v) & (S2.set == "R ∪ H_D") & (S2.pair == "F1p twin vs F0 twin (what lags add, ML)")].iloc[0]
        ax.scatter([r2[f"re:{y}"] for y in S2_Y], yy + (j - 0.5) * 0.25, s=28, marker="s", facecolor="none", edgecolor=plotstyle.SERIES[j],
                   label=f"admin lags vs twin ({v})", zorder=3)
    ax.axvline(1, color=plotstyle.NEUTRAL, lw=1); ax.set_xscale("log")
    ax.set_yticks(yy); ax.set_yticklabels(labs); ax.invert_yaxis()
    ax.set_xlabel("relative efficiency vs the F0 twin (MSE ratio, log scale)")
    ax.set_title("Weak register (F0): text vs what last year's admin data would add")
    ax.legend(fontsize=7.5, loc="lower right")
    plt.tight_layout(); plt.savefig(f"{RES}/s2_weak_register.png", dpi=150); plt.show()
