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
# # 04 — Protocol v1.0, Stages 3–4 (New England): nonresponse and small area estimation
#
# ## Stage 3: NR-LATE, a real nonresponse mechanism
# **Response.** A take-some unit responds if its TY2023 return was filed ≤ 300 days after fiscal year end. This is
# real filing behaviour: 61% of take-some units respond, but only 13% of take-all units, because large
# organizations file on extension.
#
# **Take-all units**, two pre-registered scenarios:
# - **TA-follow-up:** take-all units respond with probability 0.9 (MCAR), as if intensively chased.
# - **TA-late:** take-all units follow the real 300-day rule.
#
# Take-all nonrespondents are imputed by their own last-cycle value (TA-hist-own).
#
# **Take-some estimators**, all on respondents only:
# - status quo: weighting-class (stratum) adjustment plus GREG on the F1p register;
# - BMF-max twin and text model calibration, with the same weighting-class adjustment;
# - propensity-score variants: frame-only propensities, and text in the propensity.
#
# **Rule S6.** The text arm must satisfy all three, on the five variables {government grants, contributions, fundraising, volunteers, lobbying}:
# - it removes ≥ 30% of the reference method's nonresponse bias on ≥ 3 of the 5 (bias-reduction fraction, BRF);
# - its RRMSE ratio is ≤ 1.00;
# - its coverage gap is ≥ −2 pp.
#
# **Reference method (protocol §7.2).** For each y, the reference is the text-free method with the smallest limiting
# absolute bias. The candidates run here are weighting class + HT, weighting class + GREG (the status quo), the
# weighting-class BMF-max twin, and the frame-propensity twin. The limiting bias is estimated by the take-some Monte Carlo
# RB at R = 1,000 (deviation D-29). The comparison against the status quo GREG is reported as a secondary.
#
# **Pilot claim tested here (§7.2).** Calibrating to lagged size *increases* nonresponse bias for survey-only y:
# abs(B(weighting class + GREG)) > abs(B(weighting class)) for government grants, contributions and fundraising.
#
# ## Stage 4: small area estimation
# **Domains.** 66 unplanned domains, state × NTEE10, with 0–50 sampled units each.
#
# **Estimators.** Each is fitted with three models: production GREG (B0), the BMF-max twin, and the text model. The two names mirror notebook 03's arms; here they are cross-fitted on the current sample, not frozen.
# - **DIR:** direct (model-free).
# - **MA:** model-assisted, approximately design-unbiased.
# - **SYN:** synthetic.
# - **FH:** Fay–Herriot composite.
#
# **Rule S7**, for government grants and the share with government grants:
# - ARRMSE(FH text) / ARRMSE(FH twin) ≤ 0.90;
# - ARB not worse by more than 1 pp;
# - the publishable share (domain RRMSE < 25%) up by at least 5 pp;
# - fewer than 5% of domains worse by more than 10%.

# %%
import os, sys, json
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
PROJ = os.path.abspath("..") if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
sys.path.insert(0, os.path.join(PROJ, "src"))
DATA = os.environ.get("EMBAUX_DATA", "/tmp/claude-0/data")
import plotstyle; plotstyle.use()
from analysis import cell_metrics, paired_re
from report import Y_LABEL
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
OUT = f"{DATA}/results/stage34_ne_v3"
RES = os.path.join(PROJ, "results", "stage34_ne_v3"); os.makedirs(RES, exist_ok=True)
S6_Y = ["y_gov_grants", "y_contrib", "y_fundraising_exp", "y_volunteers_cap10k", "y_bin_lobbying"]

# %% [markdown]
# ## Stage 3 results

# %%
nr_rows = []
for scen in ["follow", "late"]:
    p = f"{OUT}/nr_{scen}.parquet"
    if not os.path.exists(p):
        print("missing", p); continue
    d = pd.read_parquet(p); meta = json.load(open(f"{OUT}/nr_{scen}_meta.json")); truth = meta["truth"]
    cm = cell_metrics(d, truth); cm["scenario"] = scen
    # take-some part only: remove the take-all total actually used (true or imputed) from each estimate
    ta = pd.read_parquet(f"{OUT}/nr_{scen}_ta.parquet")
    dts = d.merge(ta, on=["rep", "y"]); dts["est"] = dts.est - dts.ta_used
    truth_ts = {y: truth[y] - meta["truth_ta"][y] for y in truth}
    cts = cell_metrics(dts[["rep", "estimator", "y", "est", "var"]], truth_ts).rename(columns=lambda c: c + "_ts" if c not in ("estimator", "y") else c)
    cm = cm.merge(cts[["estimator", "y", "rb_pct_ts", "rrmse_pct_ts", "coverage_pct_ts"]], on=["estimator", "y"])
    nr_rows.append(cm)
    for col, lab in [("rb_pct", "relative bias of the total (%)"), ("rb_pct_ts", "relative bias of the take-some part (%)")]:
        t = cm.pivot(index="y", columns="estimator", values=col); t.index = [Y_LABEL.get(i, i) for i in t.index]
        print(f"\n=== TA-{scen}: {lab} ==="); display(t.round(2))
    for col, lab in [("coverage_pct", "coverage of the total (%); take-all imputation error is not in the variance"),
                     ("coverage_pct_ts", "coverage of the take-some part (%)")]:
        c = cm.pivot(index="y", columns="estimator", values=col); c.index = [Y_LABEL.get(i, i) for i in c.index]
        print(f"=== TA-{scen}: {lab} ==="); display(c.round(1))
nr = pd.concat(nr_rows) if nr_rows else pd.DataFrame()

# %%
TEXT_FREE = ["NR_rhg_HT", "NR_SQ_rhg_GREG_size_cat@F1p", "NR_rhg_MCAL_bmfmax@F1p", "NR_ps_frame_MCAL_bmfmax@F1p"]
def s6(nr, scen, text="NR_rhg_MCAL_bmfmax@F1p+Estar", ref="protocol"):
    """ref='protocol': per-y text-free method with the smallest |RB| of the take-some part; ref=<name>: that method."""
    m = nr[nr.scenario == scen].set_index(["estimator", "y"])
    out = []
    for y in S6_Y:
        r = min(TEXT_FREE, key=lambda e: abs(m.loc[(e, y), "rb_pct_ts"])) if ref == "protocol" else ref
        b_t, b_r = m.loc[(text, y), "rb_pct_ts"], m.loc[(r, y), "rb_pct_ts"]
        out.append({"y": Y_LABEL.get(y, y), "reference": r, "bias text % (take-some)": b_t, "bias ref %": b_r,
                    "BRF": 1 - abs(b_t) / abs(b_r) if b_r else np.nan,
                    "RRMSE ratio (take-some)": m.loc[(text, y), "rrmse_pct_ts"] / m.loc[(r, y), "rrmse_pct_ts"],
                    "coverage gap (pp)": m.loc[(text, y), "coverage_pct_ts"] - m.loc[(r, y), "coverage_pct_ts"]})
    t = pd.DataFrame(out)
    passed = ((t["BRF"] >= 0.30).sum() >= 3) and (t["RRMSE ratio (take-some)"] <= 1.0).all() and (t["coverage gap (pp)"] >= -2).all()
    return t, passed

if len(nr):
    for scen in ["follow", "late"]:
        if not (nr.scenario == scen).any():
            continue
        for ref, lab in [("protocol", "S6 (protocol reference)"), ("NR_SQ_rhg_GREG_size_cat@F1p", "secondary: vs status quo GREG"),
                         ("NR_rhg_MCAL_bmfmax@F1p", "secondary: vs the same learner without text")]:
            t, ok = s6(nr, scen, ref=ref)
            print(f"TA-{scen}, {lab}: {'PASS' if ok else 'FAIL'}"); display(t.round(3))
            t.to_csv(f"{RES}/s6_{scen}_{ref.split('@')[0]}.csv", index=False)
        m = nr[nr.scenario == scen].set_index(["estimator", "y"]).rb_pct_ts
        claim = pd.DataFrame([{"y": Y_LABEL.get(y, y), "|bias| weighting class + HT %": abs(m[("NR_rhg_HT", y)]),
                               "|bias| weighting class + GREG %": abs(m[("NR_SQ_rhg_GREG_size_cat@F1p", y)])}
                              for y in ["y_gov_grants", "y_contrib", "y_fundraising_exp"]])
        claim["calibration increases bias"] = claim.iloc[:, 2] > claim.iloc[:, 1]
        print(f"TA-{scen}: pilot claim (calibrating to size increases nonresponse bias)"); display(claim.round(2))

# %%
if len(nr):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
    arms = ["NR_rhg_HT", "NR_SQ_rhg_GREG_size_cat@F1p", "NR_rhg_MCAL_bmfmax@F1p", "NR_rhg_MCAL_bmfmax@F1p+Estar"]
    names = ["weighting class + HT", "+ GREG on lagged size (status quo)", "+ ML twin (codes, no text)", "+ ML with text"]
    for ax, scen in zip(axes, ["follow", "late"]):
        if not (nr.scenario == scen).any():
            continue
        m = nr[nr.scenario == scen].set_index(["estimator", "y"]).rb_pct_ts
        yy = np.arange(len(S6_Y))
        for j, (a, nm) in enumerate(zip(arms, names)):
            ax.barh(yy + (j - 1.5) * 0.2, [m[(a, y)] for y in S6_Y], height=0.18, color=plotstyle.SERIES[j], label=nm)
        ax.axvline(0, color=plotstyle.NEUTRAL, lw=1)
        ax.set_yticks(yy); ax.set_yticklabels([Y_LABEL.get(y, y) for y in S6_Y]); ax.invert_yaxis()
        ax.set_title(f"TA-{scen}"); ax.set_xlabel("relative bias of the take-some part (%)")
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("Late-filer nonresponse: calibration to lagged size adds bias; text removes part of it")
    plt.tight_layout(); plt.savefig(f"{RES}/nr_bias.png", dpi=150); plt.show()

# %% [markdown]
# ## Stage 4 results: small area estimation

# %%
per = pd.read_csv(f"{OUT}/sae_per_domain.csv")
per["publishable"] = per.rrmse_pct < 25
summ = per.groupby(["y", "estimator"]).agg(ARRMSE=("rrmse_pct", "mean"), median_RRMSE=("rrmse_pct", "median"),
                                           ARB=("arb_pct", "mean"), publishable=("publishable", "mean"),
                                           coverage=("coverage", "mean")).reset_index()
for y in summ.y.unique():
    t = summ[summ.y == y].set_index("estimator").drop(columns="y")
    print(f"\n=== {Y_LABEL.get(y, y)} ==="); display(t.round(2))

# %%
def s7(per, y):
    a = per[(per.y == y) & (per.estimator == "FH_text")].set_index("domain")
    b = per[(per.y == y) & (per.estimator == "FH_twin")].set_index("domain")
    j = a.join(b, lsuffix="_t", rsuffix="_w", how="inner")
    ratio = j.rrmse_pct_t.mean() / j.rrmse_pct_w.mean()
    worse10 = (j.rrmse_pct_t > 1.10 * j.rrmse_pct_w).mean()
    pub = (j.rrmse_pct_t < 25).mean() - (j.rrmse_pct_w < 25).mean()
    arb = j.arb_pct_t.mean() - j.arb_pct_w.mean()
    ok = (ratio <= 0.90) and (arb <= 1.0) and (pub >= 0.05) and (worse10 < 0.05)
    return {"y": Y_LABEL.get(y, y), "ARRMSE ratio text/twin": ratio, "ARB change (pp)": arb, "publishable share change": pub,
            "share of domains >10% worse": worse10, "S7 pass": ok}
s7t = pd.DataFrame([s7(per, y) for y in ["y_gov_grants", "y_bin_has_gov_grants", "y_contrib", "y_fundraising_exp", "y_volunteers_cap10k"]])
display(s7t.round(3)); s7t.to_csv(f"{RES}/s7.csv", index=False)

# %%
y = "y_gov_grants"
a = per[(per.y == y) & (per.estimator == "FH_text")].set_index("domain").rrmse_pct
b = per[(per.y == y) & (per.estimator == "FH_twin")].set_index("domain").rrmse_pct
fig, ax = plt.subplots(figsize=(5.2, 5))
ax.scatter(b, a.reindex(b.index), s=18, color=plotstyle.SERIES[0])
lim = [0, max(b.max(), a.max()) * 1.05]; ax.plot(lim, lim, color=plotstyle.NEUTRAL, lw=1)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("domain RRMSE %, FH with BMF-max twin"); ax.set_ylabel("domain RRMSE %, FH with text")
ax.set_title("Government grants by state x NTEE10 domain\n(points below the line: text helps)")
plt.tight_layout(); plt.savefig(f"{RES}/sae_gov_grants_domains.png", dpi=150); plt.show()
