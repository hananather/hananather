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
# # 02 — Do text embeddings reduce variance? Monte Carlo results
#
# **Setup recap** (details in `notes/benchmark-protocol.md` and notebook 01):
# - **Population:** New England non-profits, N = 20,626.
# - **Frame:** TY2022 register (NTEE, state, BMF codes, lagged revenue / expenses / assets / employees, mission and program text).
# - **Survey targets:** TY2023 values.
# - **Design:** production-style stratified design, NTEE10 × 3 size classes plus a take-all stratum of the 206 largest units. Compromise Neyman allocation, n = 1,000.
# - **Replicates:** R independent samples. Every estimator sees the same samples (common random numbers), so comparisons between estimators are precise.
#
# **How to read relative efficiency (RE).** RE = MSE(comparator) / MSE(estimator). Two rules of thumb:
# - With RE = 1.25, the estimator achieves the comparator's precision with about 20% fewer take-some sample units.
# - A CV shrinks only by 1 − 1/√RE, i.e. by 11% at RE = 1.25.
#
# Every RE carries a 95% Monte Carlo interval (paired-replicate delta method).

# %%
import os, sys, json, glob
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
PROJ = os.path.abspath("..") if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
sys.path.insert(0, os.path.join(PROJ, "src"))
DATA = os.environ.get("EMBAUX_DATA", "/tmp/claude-0/data")
TAG = os.environ.get("EMBAUX_MC_TAG", "v1_n1000")
import plotstyle; plotstyle.use()
from report import (Y_LABEL, Y_TYPE, family, emb_of, best_conventional, re_vs, cv_band, dotplot, fmt_re)
from mc import metrics
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)

df = pd.read_parquet(f"{DATA}/results/mc_ne_{TAG}.parquet")
meta = json.load(open(f"{DATA}/results/mc_ne_{TAG}.json"))
truth = meta["truth"]; R = meta["R"]; n = meta["n"]
YS = [y for y in Y_LABEL if y in truth]
print(f"R = {R} replicates, n = {n}, estimators = {df.estimator.nunique()}, targets = {len(YS)}")
m = metrics(df, truth, baseline="GREG_size_cat")
m["family"] = m.estimator.map(family); m["y_type"] = m.y.map(Y_TYPE)

# %% [markdown]
# ## 1. Stage 1: what the population says is possible (anticipated-variance ceilings)
# Each auxiliary set is fitted on the whole population (optimistic). The ratio AV(strong frame) / AV(frame + text) is
# the most a model-assisted estimator could gain at this design. Monte Carlo gains far above these ceilings would mean
# leakage; gains far below would mean overfitting.

# %%
dg = pd.read_csv(f"{DATA}/results/diag_ne_n{n}.csv")
keep = ["weak", "strong+lsa256_mission:pc32_ratio", "strong+lsa256_full:pc32_ratio", "strong+minilm_mission:pc32_ratio",
        "strong+minilm_full:pc32_ratio", "strong+PERMUTED_minilm_full:pc32_ratio", "strong+ORACLE_lagged_survey_vars"]
ceil = dg[dg.features.isin(keep)].pivot(index="y", columns="features", values="re_vs_baseline")[keep]
ceil.index = [Y_LABEL.get(i, i) for i in ceil.index]
display(ceil.round(2))

# %% [markdown]
# ## 2. Validity first: are the estimators approximately unbiased, and do their intervals cover?
# Text estimators are only interesting if they keep design consistency. The table shows relative bias (% of the
# total, with MC SE) and 95% CI coverage for the main estimators. Coverage MC SE ≈ 1 pp at R = 500.

# %%
MAIN = ["HT", "GREG_size_cat", "MCAL_frame_ratio", "RIDGE_size_cat+lsa256_full:ratio32", "MCAL_ratio+lsa256_full",
        "MCAL_ratio+minilm_full", "MCAL_ratio+lsa256_full:nocf", "SYN_ratio+lsa256_full"]
main = m[m.estimator.isin(MAIN)]
bias = main.pivot(index="y", columns="estimator", values="rel_bias_pct")[MAIN].reindex(YS)
cov = main.pivot(index="y", columns="estimator", values="coverage_pct")[[e for e in MAIN if not e.startswith("SYN")]].reindex(YS)
bias.index = cov.index = [Y_LABEL[y] for y in YS]
print("Relative bias (%):"); display(bias.round(2))
print("95% CI coverage (%):"); display(cov.round(1))
print("Max |MC SE of relative bias| (pp):", round(main.rel_bias_mcse_pct.max(), 2))

# %% [markdown]
# ## 3. Efficiency: text estimators against the strongest conventional comparator
# The comparator for each target is **the best conventional estimator for that target**: the status-quo GREG
# family plus text-free machine learning, chosen after the fact by Monte Carlo MSE. This choice favours the
# comparator, which makes it conservative for the text claim.

# %%
best = best_conventional(m)
display(best.rename("best conventional comparator").to_frame().assign(label=lambda d: d.index.map(Y_LABEL)))
tab = re_vs(df, truth, best.to_dict())
tab["family"] = tab.estimator.map(family)
ESTS = ["RIDGE_size_cat+lsa256_full:ratio32", "MCAL_ratio+lsa256_full", "MCAL_ratio+minilm_full", "MCAL_ratio+random384"]
LAB = {"RIDGE_size_cat+lsa256_full:ratio32": "ridge calibration, TF-IDF full text",
       "MCAL_ratio+lsa256_full": "model calibration, TF-IDF full text",
       "MCAL_ratio+minilm_full": "model calibration, MiniLM full text",
       "MCAL_ratio+random384": "negative control: random vectors"}
fig, ax = plt.subplots(figsize=(8, 6))
dotplot(ax, tab, YS, ESTS, LAB, xlim=(0.6, 2.5))
ax.set_title("Relative efficiency of text estimators vs the best conventional estimator (95% MC intervals)")
plt.tight_layout(); plt.show()
wide = tab[tab.estimator.isin(ESTS)].assign(cell=lambda d: [fmt_re(a, b) for a, b in zip(d.re, d.re_se)])
wide = wide.pivot(index="y", columns="estimator", values="cell")[ESTS].reindex(YS)
wide.index = [f"{Y_LABEL[y]} ({Y_TYPE[y]})" for y in YS]
display(wide)

# %% [markdown]
# ## 4. Attribution: is it the text, or just a more flexible model?
# Same learner, same folds, same design. The only difference is whether text PCs enter the features.

# %%
att = re_vs(df, truth, "MCAL_frame_ratio")
att = att[att.estimator.isin(["MCAL_ratio+lsa256_full", "MCAL_ratio+lsa256_mission", "MCAL_ratio+minilm_full",
                              "MCAL_ratio+minilm_mission", "MCAL_ratio+random384", "MCAL_ratio+PERM_lsa256_full"])]
att_w = att.assign(cell=lambda d: [fmt_re(a, b) for a, b in zip(d.re, d.re_se)]).pivot(index="y", columns="estimator", values="cell").reindex(YS)
att_w.index = [Y_LABEL[y] for y in YS]
display(att_w)

# %% [markdown]
# ## 5. Embedding ladder and text field
# Is a neural embedding better than TF-IDF? Is the richer program text better than the short mission statement?
# (Cohere rungs are added when the API becomes reachable; see README.)

# %%
lad = tab[tab.estimator.isin(["MCAL_ratio+lsa256_mission", "MCAL_ratio+lsa256_full", "MCAL_ratio+minilm_mission", "MCAL_ratio+minilm_full"])]
lad_w = lad.pivot(index="y", columns="estimator", values="re").reindex(YS).round(2)
lad_w.index = [Y_LABEL[y] for y in YS]
display(lad_w)

# %% [markdown]
# ## 6. Controls and pitfalls
# - **Random and row-permuted embeddings** must not beat the comparator.
# - **No cross-fitting** should show overfitting: optimistic variance estimates and under-coverage.
# - **The synthetic estimator** Σ m̂ (no design correction) shows model bias.

# %%
ctl = m[m.estimator.isin(["MCAL_ratio+random384", "MCAL_ratio+PERM_lsa256_full", "RIDGE_size_cat+random384:ratio32",
                          "MCAL_ratio+lsa256_full:nocf", "SYN_ratio+lsa256_full", "GREG_weak+lsa256_full:add32"])]
for col in ["re", "rel_bias_pct", "coverage_pct", "var_rel_bias_pct"]:
    t = ctl.pivot(index="y", columns="estimator", values=col).reindex(YS); t.index = [Y_LABEL[y] for y in YS]
    print(col, "(re is vs GREG_size_cat)" if col == "re" else ""); display(t.round(2))

# %% [markdown]
# ## 7. The weak-register scenario: no lagged financial data on the frame
# This corresponds to registers without prior-year administrative values (new units, countries without tax linkage).
# Here text competes only with industry, state and a coarse size code.

# %%
wk = re_vs(df, truth, "GREG_weak")
wk = wk[wk.estimator.isin(["GREG_weak+lsa256_full:add32", "GREG_weak+minilm_full:add32", "GREG_weak+random384:add32", "GREG_size_cat"])]
wk_w = wk.pivot(index="y", columns="estimator", values="re").reindex(YS).round(2); wk_w.index = [Y_LABEL[y] for y in YS]
display(wk_w)

# %% [markdown]
# ## 8. Real-world translation: CVs, quality bands, and sample-size savings
# - **Take-some saving:** σ ≈ 1 − 1/RE, applied to the take-some part of the sample (the take-all units are fixed).
# - **Quality letters:** Statistics Canada's commonly used CV bands are A < 5%, B < 10%, C < 15%, D < 25%, E < 35%, F ≥ 35%. Confirm these against the target program's own guideline.
# - **The binding target sets the saving.** A multipurpose survey must meet *all* its targets, so the saving is set by the target that gains least, not the one that gains most.

# %%
n_ta = 206
tr = []
for y in YS:
    b = best[y]
    cv_b = m[(m.y == y) & (m.estimator == b)].rrmse_pct.iat[0]
    for e in ["MCAL_ratio+lsa256_full", "MCAL_ratio+minilm_full"]:
        r_ = tab[(tab.y == y) & (tab.estimator == e)]
        re_, cv_t = r_.re.iat[0], m[(m.y == y) & (m.estimator == e)].rrmse_pct.iat[0]
        tr.append({"target": Y_LABEL[y], "text estimator": e, "comparator": b, "CV comparator %": cv_b, "band": cv_band(cv_b),
                   "CV text %": cv_t, "band text": cv_band(cv_t), "RE": re_,
                   "take-some sample saving %": 100 * (1 - 1 / re_) if re_ > 1 else 0.0})
trt = pd.DataFrame(tr).round(2); display(trt)
for e in ["MCAL_ratio+lsa256_full", "MCAL_ratio+minilm_full"]:
    sub = trt[trt["text estimator"] == e]
    print(f"{e}: binding (minimum) saving across ALL targets = {sub['take-some sample saving %'].min():.1f}%;"
          f" across survey-only targets = {sub[sub.target.isin([Y_LABEL[y] for y in YS if Y_TYPE[y].startswith('survey-only')])]['take-some sample saving %'].min():.1f}%")
