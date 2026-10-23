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
# # 03 — Protocol v1.0, Stage 2A: New England proof-of-concept verdict
#
# This notebook evaluates the pre-registered protocol (`notes/benchmark-protocol.md`) on the New England
# development population. It is the **proof-of-concept verdict**. Generalization is *not* assessed here; that
# is the job of the 8 held-out Census divisions, which wait on the sign-offs listed in §14 of the protocol.
# Every departure from the protocol is in `prereg/deviations.md`.
#
# **Cell**
# - **Population and frame:** NE continuing non-profits, N = 20,626. Frame known on D = 2024-06-30, so prior-year information comes only from returns filed by then.
# - **Register tier:** F1p, the primary (lagged revenue, expenses, assets, employees, payroll). F1 and F2 arms run alongside.
# - **Design:** D1 production design, n = 1,000.
# - **Monte Carlo:** R = 2,000, common random numbers, one shared fold vector per replicate.
#
# **Embedding.** E* is selected by the pre-registered rule on the *previous* wave: TF-IDF/LSA of mission + program text, K = 16 components.
#
# **Primary pairs**
# - **P1 efficacy:** `MCAL_bmfmax@F1p+E*` vs `MCAL_bmfmax@F1p`. Same learner and folds, all IRS codes; the only difference is the text.
# - **P2 production:** `FROZEN_calm_bmfmax@F1p+E*` vs `GREG_size_cat@F1p`. One bounded weight set, with heads trained on last year's respondents, against today's production GREG.
#
# **Statistic.** GM-RE over the 10 headline variables H, with one-sided 95% bootstrap bounds.
#
# **Decision rules**
# - **Production bar:** MET if point ≥ 1.10 and lower bound ≥ 1.05; EXCLUDED if upper bound < 1.10; OPEN otherwise.
# - **Any gain:** YES if lower bound > 1.00; NO if upper bound < 1.05; UNCLEAR otherwise.

# %%
import os, sys, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
PROJ = os.path.abspath("..") if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
sys.path.insert(0, os.path.join(PROJ, "src"))
DATA = os.environ.get("EMBAUX_DATA", "/tmp/claude-0/data")
import plotstyle; plotstyle.use()
from protocol import H_SET, H_D, H_S, R_SET
from analysis import gates, primary, no_harm, factorial, planted, cell_metrics, paired_re
from report import Y_LABEL
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 60)
OUT = f"{DATA}/results/stage2_ne"
df = pd.read_parquet(f"{OUT}/mc.parquet")
meta = json.load(open(f"{OUT}/meta.json")); truth = meta["truth"]
pinfo = pd.read_csv(f"{OUT}/planted_info.csv")
print(f"R = {meta['R']}, estimators = {df.estimator.nunique()}, targets = {df.y.nunique()}, failures = {sum(meta['failures'].values())}")
RES = os.path.join(PROJ, "results", "stage2_ne"); os.makedirs(RES, exist_ok=True)

# %% [markdown]
# ## 1. Validity gates (read before any efficiency number)
# A failed gate quarantines the claims it guards.
#
# | Gate | Checks |
# |---|---|
# | G1 | Design sanity: HT and the frozen difference estimator are unbiased |
# | G2 | Random, permuted and within-stratum-permuted embeddings don't beat the twin, and a constant embedding reproduces it exactly |
# | G3 | Placebo targets gain nothing |
# | G4 | Text arms keep valid coverage relative to their comparator and to HT |
# | G7 | No gain on register-type variables (a gain there would be leakage) |
# | G8 | Masking numbers in the text changes nothing |
# | G9 | The harness can detect a known large gain: last year's value of the same variable |

# %%
G = gates(df, truth)
G.to_csv(f"{RES}/gates.csv", index=False)
summary = G.groupby("gate")["pass"].agg(["sum", "count"]).assign(all_pass=lambda d: d["sum"] == d["count"])
display(summary)
display(G[~G["pass"]])

# %% [markdown]
# ## 2. Primary pairs and secondaries

# %%
P = primary(df, truth)
P.to_csv(f"{RES}/primary_pairs.csv", index=False)
show = P[["pair", "set", "gm_re", "lo95", "hi95", "production_bar", "any_gain"]].copy()
display(show.round(3))

# %% [markdown]
# ### Per-variable RE for P1 and P2 (headline set H)

# %%
per = P[P.set == "H"].set_index("pair")[[f"re:{y}" for y in H_SET]]
per.columns = [Y_LABEL.get(c[3:], c[3:]) for c in per.columns]
display(per.T.round(3))

# %%
fig, ax = plt.subplots(figsize=(7.5, 4.6))
rows = P[(P.set == "H")].reset_index(drop=True)
yy = np.arange(len(rows))
ax.errorbar(rows.gm_re, yy, xerr=[rows.gm_re - rows.lo95, rows.hi95 - rows.gm_re], fmt="o", color=plotstyle.SERIES[0], capsize=0, lw=1.5)
ax.axvline(1.0, color=plotstyle.NEUTRAL, lw=1); ax.axvline(1.10, color=plotstyle.SERIES[1], lw=1, ls="--")
ax.text(1.101, len(rows) - 0.5, "production bar 1.10", color=plotstyle.TEXT2, fontsize=8)
ax.set_yticks(yy); ax.set_yticklabels(rows.pair); ax.invert_yaxis()
ax.set_xlabel("GM relative efficiency over the 10 headline variables (90% bootstrap interval)")
ax.set_title("Stage 2A, New England: what text adds (and what ML adds)")
plt.tight_layout(); plt.savefig(f"{RES}/primary_pairs.png", dpi=150); plt.show()

# %% [markdown]
# ### P2 no-harm condition
# Production also requires that no variable is made worse: min point RE ≥ 0.95, and no upper bound below 0.95.

# %%
nh, ok = no_harm(df, truth, "FROZEN_calm_bmfmax@F1p+Estar", "GREG_size_cat@F1p")
nh["label"] = nh.y.map(lambda y: Y_LABEL.get(y, y))
display(nh.round(3)); print("no-harm condition holds:", ok)

# %% [markdown]
# ## 3. Attribution: learner (A) × codes (B) × text (C), Shapley shares of log RE
# The 8-cell factorial runs on the same samples and folds. Shares sum to log RE(all three on | none).

# %%
F_t, F_mean = factorial(df, truth)
F_t["label"] = F_t.y.map(lambda y: Y_LABEL.get(y, y))
display(F_t.round(4)); print("mean over H:"); display(F_mean.round(4).to_frame("mean"))
print("text share of the total log gain:", round(F_mean.phi_C_text / F_mean.log_RE_total, 3) if F_mean.log_RE_total else np.nan)

# %% [markdown]
# ## 4. Planted-signal recovery: what the harness can detect
# The planted columns carry a text signal with a *known* same-class partial ρ². For each level we compare the
# realized Monte Carlo RE of text vs twin with the theoretical 1/(1 − ρ²).
#
# ρ²_min(1,000) is the smallest planted ρ² at which a single-variable MET (point ≥ 1.10, lower bound ≥ 1.05)
# passes in at least 2 of 3 plantings. A null on real data is then stated as "text partial ρ² < ρ²_min".

# %%
PL = planted(df, truth, pinfo)
display(PL.round(3))
met = PL.assign(met=(PL.re_mc >= 1.10) & (PL.lo95 >= 1.05)).groupby("rho2").met.sum()
rho2_min = next((r for r, k in met.items() if k >= 2), None)
print("plantings meeting the bar by level:", met.to_dict(), "| rho2_min(1000) =", rho2_min)
print("protocol-form planted family (diagnostic, D-3): achieved rho2 by a =",
      pinfo[pinfo.family == "pr"].groupby("a").rho2_achieved.mean().round(3).to_dict())

# %% [markdown]
# ## 5. Controls and pitfall demonstrations

# %%
cm = cell_metrics(df, truth)
ctl = ["MCAL_bmfmax@F1p+N1random", "MCAL_bmfmax@F1p+N2perm", "MCAL_bmfmax@F1p+N3permh", "MCAL_bmfmax@F1p+Estar_nonum",
       "MCAL_bmfmax@F1p+Estar:nocf", "SYN_bmfmax@F1p+Estar", "MCAL_bmfmax@F1p+Estar", "MCAL_bmfmax@F1p"]
sub = cm[cm.estimator.isin(ctl) & cm.y.isin(H_SET)]
print("Relative bias (%):"); display(sub.pivot(index="y", columns="estimator", values="rb_pct")[ctl].round(2))
print("95% CI coverage (%):"); display(sub.pivot(index="y", columns="estimator", values="coverage_pct")[[c for c in ctl if not c.startswith("SYN")]].round(1))

# %% [markdown]
# ## 6. Real-world translation of P2
# - The take-some sample saving at equal precision is σ ≈ 1 − 1/RE.
# - **The binding variable sets the saving for a multipurpose survey.**
# - CV bands use the Statistics Canada letters (A < 5%, B < 10%, C < 15%, D < 25%, E < 35%, F ≥ 35%).

# %%
from report import cv_band
rows = []
for y in H_SET + R_SET:
    b0 = cm[(cm.estimator == "GREG_size_cat@F1p") & (cm.y == y)].rrmse_pct.iat[0]
    p2 = cm[(cm.estimator == "FROZEN_calm_bmfmax@F1p+Estar") & (cm.y == y)].rrmse_pct.iat[0]
    re = (b0 / p2) ** 2
    rows.append({"variable": Y_LABEL.get(y, y), "CV B0 %": b0, "band B0": cv_band(b0), "CV P2 %": p2, "band P2": cv_band(p2),
                 "RE": re, "take-some saving %": 100 * (1 - 1 / re)})
tr = pd.DataFrame(rows).round(2); display(tr)
print("binding saving over H (minimum):", round(tr[tr.variable.isin([Y_LABEL.get(y, y) for y in H_SET])]["take-some saving %"].min(), 1), "%")
tr.to_csv(f"{RES}/real_world_translation.csv", index=False)
