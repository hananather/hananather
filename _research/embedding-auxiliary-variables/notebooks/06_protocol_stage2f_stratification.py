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
# # 06 — Protocol v1.0, Stage 2F: text at the design stage (stratification)
#
# This is the "stratification" use from the claim under test. Text changes **the design**, not the estimator.
#
# | Design | Strata | Question |
# |---|---|---|
# | D1 | NTEE10 (MISSING as a level) × 3 size classes + take-all | Status quo |
# | D1-imp | D1, with the 31% MISSING NTEE10 replaced by text-coded NTEE10 (logit on E*, no survey variable) | **S4:** text as a frame coder |
# | D1-H44 | NTEE10 × 4 size classes + take-all | Equal-H twin of D2a |
# | D2a | NTEE10 × 2 size × 2 classes of a frozen text score (last year's text head's predicted gov-grant ratio) | **S5:** prediction strata |
#
# Every design has n = 1,000, a take-all of the top 1% on lagged expenses, and compromise allocation.
#
# Estimators are run under each design: HT and GREG_size_cat@F1p (R = 2,000), and MCAL_bmfmax@F1p (R = 500). The MCAL rows test **M8**: are design gains absorbed by a strong estimator?
#
# **Rules (protocol §9.4)**
# - **S4:** GM over all 15 real y of RE(D1-imp ∣ D1) ≥ 1.05, for HT and for GREG.
# - **S5:** GM over H of RE(D2a ∣ D1-H44) meets the production bar (point ≥ 1.10, lower bound ≥ 1.05).
#
# Designs differ, so samples are independent across designs. The bootstrap resamples replicates jointly, which is valid for independent arms too.

# %%
import os, sys, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
PROJ = os.path.abspath("..") if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
sys.path.insert(0, os.path.join(PROJ, "src"))
DATA = os.environ.get("EMBAUX_DATA", "/tmp/claude-0/data")
import plotstyle; plotstyle.use()
from protocol import H_SET, H_D, H_S, R_SET, gm_re, verdict
from analysis import cell_metrics
from report import Y_LABEL
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
OUT = f"{DATA}/results/stage2f_ne_v3"
RES = os.path.join(PROJ, "results", "stage2f_ne_v3"); os.makedirs(RES, exist_ok=True)
df = pd.read_parquet(f"{OUT}/mc.parquet"); truth = json.load(open(f"{OUT}/meta.json"))["truth"]
info = json.load(open(f"{OUT}/designs.json")); display(pd.DataFrame(info).T)
df["arm"] = df.estimator + "|" + df.design
ALL = list(truth)

# %% [markdown]
# ## 1. Validity under each design

# %%
cm = cell_metrics(df.drop(columns="estimator").rename(columns={"arm": "estimator"}), truth)
cm[["est", "design"]] = cm.estimator.str.split("|", expand=True)
cm.to_csv(f"{RES}/cell_metrics.csv", index=False)
for col in ["rb_pct", "coverage_pct", "rrmse_pct"]:
    t = cm[cm.est == "GREG_size_cat@F1p"].pivot(index="y", columns="design", values=col); t.index = [Y_LABEL.get(i, i) for i in t.index]
    print(f"GREG_size_cat@F1p, {col}:"); display(t.round(2))

# %% [markdown]
# ## 2. S4 and S5

# %%
rows = []
for est in ["HT", "GREG_size_cat@F1p", "MCAL_bmfmax@F1p"]:
    for name, (a, b) in {"S4: D1-imp vs D1": ("D1-imp", "D1"), "S5: D2a vs D1-H44": ("D2a", "D1-H44"),
                         "D1-H44 vs D1 (one more size class)": ("D1-H44", "D1"), "D2a vs D1": ("D2a", "D1")}.items():
        for lab, ys in [("all 15", ALL), ("H", H_SET), ("H_D", H_D), ("H_S", H_S), ("R", R_SET)]:
            pt, lo, hi, per = gm_re(df.drop(columns="estimator").rename(columns={"arm": "estimator"}), truth, f"{est}|{a}", f"{est}|{b}", ys)
            bar, anyg = verdict(pt, lo, hi)
            rows.append({"estimator": est, "comparison": name, "set": lab, "gm_re": pt, "lo95": lo, "hi95": hi,
                         "production_bar": bar, "any_gain": anyg, **{f"re:{k}": v for k, v in per.items()}})
S = pd.DataFrame(rows); S.to_csv(f"{RES}/s4_s5.csv", index=False)
show = S[((S.comparison.str.startswith("S4")) & (S.set == "all 15")) | ((S.comparison.str.startswith("S5")) & (S.set == "H"))]
display(show[["estimator", "comparison", "set", "gm_re", "lo95", "hi95", "production_bar", "any_gain"]].round(3))
s4 = S[(S.comparison.str.startswith("S4")) & (S.set == "all 15") & S.estimator.isin(["HT", "GREG_size_cat@F1p"])]
print("S4 (>= 1.05 for HT and GREG):", "PASS" if (s4.gm_re >= 1.05).all() else "FAIL")
s5 = S[(S.comparison.str.startswith("S5")) & (S.set == "H") & (S.estimator == "GREG_size_cat@F1p")].iloc[0]
print("S5 (production bar, GREG):", s5.production_bar, "| HT:", S[(S.comparison.str.startswith("S5")) & (S.set == "H") & (S.estimator == "HT")].iloc[0].production_bar)

# %% [markdown]
# ### Per-variable RE, and whether a strong estimator absorbs design gains (M8)

# %%
for comp in ["S4: D1-imp vs D1", "S5: D2a vs D1-H44"]:
    t = S[(S.comparison == comp) & (S.set == "all 15")].set_index("estimator")[[f"re:{y}" for y in ALL]].T
    t.index = [Y_LABEL.get(i[3:], i[3:]) for i in t.index]
    print(comp); display(t.round(3))

# %%
fig, ax = plt.subplots(figsize=(8.6, 3.8))
sub = S[S.set.isin(["all 15"]) & S.comparison.isin(["S4: D1-imp vs D1", "S5: D2a vs D1-H44", "D1-H44 vs D1 (one more size class)"])].reset_index(drop=True)
yy = np.arange(len(sub))
cols = {"HT": plotstyle.SERIES[0], "GREG_size_cat@F1p": plotstyle.SERIES[1], "MCAL_bmfmax@F1p": plotstyle.SERIES[2]}
for est, c in cols.items():
    m = sub.estimator == est
    ax.errorbar(sub.gm_re[m], yy[m], xerr=[sub.gm_re[m] - sub.lo95[m], sub.hi95[m] - sub.gm_re[m]], fmt="o", color=c, capsize=0, lw=1.4, label=est)
ax.axvline(1, color=plotstyle.NEUTRAL, lw=1); ax.axvline(1.05, color=plotstyle.SERIES[3], lw=1, ls="--")
ax.set_yticks(yy); ax.set_yticklabels(sub.comparison + "  (" + sub.estimator.str.split("_").str[0] + ")", fontsize=8); ax.invert_yaxis()
ax.set_xlabel("GM relative efficiency over all 15 variables (90% bootstrap interval)")
ax.set_title("Text at the design stage", loc="left"); ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
plt.tight_layout(); plt.savefig(f"{RES}/design_stage.png", dpi=150, bbox_inches="tight"); plt.show()
