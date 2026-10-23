"""Summary figure for FINDINGS.md: (a) efficiency vs interval coverage of the estimator families on the five
survey-only dollar items (Stage 2A, full response); (b) take-some RRMSE under real late-filer nonresponse,
relative to the status quo (Stage 3 + D-34)."""
import json, os, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plotstyle
from analysis import cell_metrics
from protocol import H_D, H_SET, R_SET, gm_re

plotstyle.use()
DATA = os.environ.get("EMBAUX_DATA", "/tmp/claude-0/data")
S2, S3 = f"{DATA}/results/stage2_ne_v3", f"{DATA}/results/stage34_ne_v3"

df = pd.read_parquet(f"{S2}/mc.parquet"); tr = json.load(open(f"{S2}/meta.json"))["truth"]
arms = {"GREG_size_cat@F1p": ("GREG (status quo)", 0), "MCAL_bmfmax@F1p": ("model calibration, codes", 1),
        "MCAL_bmfmax@F1p+Estar": ("model calibration, codes + text", 1), "FROZEN_calm_bmfmax@F1p+Estar": ("frozen single weight, codes + text", 1),
        "FROZEN_diff_bmfmax@F1p": ("frozen PPI, codes", 2), "FROZEN_diff_bmfmax@F1p+Estar": ("frozen PPI, codes + text", 2), "HT": ("HT", 3)}
cm = cell_metrics(df[df.estimator.isin(arms) & df.y.isin(H_D)], tr)
pts = []
for a, (lab, fam) in arms.items():
    re = 1.0 if a == "GREG_size_cat@F1p" else gm_re(df, tr, a, "GREG_size_cat@F1p", H_D, B=200)[0]
    pts.append((lab, fam, re, cm[cm.estimator == a].coverage_pct.mean()))

d = pd.read_parquet(f"{S3}/dr_late.parquet"); ta = pd.read_parquet(f"{S3}/dr_late_ta.parquet")
n0 = pd.read_parquet(f"{S3}/nr_late.parquet"); ta0 = pd.read_parquet(f"{S3}/nr_late_ta.parquet")
meta = json.load(open(f"{S3}/nr_late_meta.json")); tts = {y: meta["truth"][y] - meta["truth_ta"][y] for y in meta["truth"]}
def ts(x, t):
    x = x.merge(t, on=["rep", "y"]); x["est"] = x.est - x.ta_used; return x[["rep", "estimator", "y", "est", "var"]]
both = pd.concat([ts(d, ta), ts(n0[n0.estimator.isin(["NR_rhg_MCAL_bmfmax@F1p", "NR_rhg_MCAL_bmfmax@F1p+Estar"])], ta0)])
c = cell_metrics(both, tts).set_index(["estimator", "y"]).rrmse_pct
nr = {"NR_rhg_HT": "weighting class + HT", "NR_rhg_MCAL_bmfmax@F1p": "+ model calibration, codes",
      "NR_rhg_MCAL_bmfmax@F1p+Estar": "+ model calibration, codes + text", "NR_rhg_FZDIFF_bmfmax@F1p": "frozen PPI, codes",
      "NR_rhg_FZDIFF_bmfmax@F1p+Estar": "frozen PPI, codes + text", "NR_ps_FZDIFF_bmfmax@F1p+Estar": "frozen PPI, codes + text, text propensity"}
A = H_SET + R_SET
gm = lambda e, ys: float(np.exp(np.mean([np.log(c[(e, y)] / c[("NR_SQ_rhg_GREG_size_cat@F1p", y)]) for y in ys])))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"width_ratios": [1.05, 1]})
for lab, fam, re, cov in pts:
    ax1.scatter(re, cov, s=60, color=plotstyle.SERIES[fam], zorder=3)
    off = {"model calibration, codes": (-60, -14), "model calibration, codes + text": (-20, -16), "frozen single weight, codes + text": (6, 6),
           "GREG (status quo)": (6, -12), "frozen PPI, codes": (6, -12), "frozen PPI, codes + text": (6, 5), "HT": (6, -12)}[lab]
    ax1.annotate(lab, (re, cov), xytext=off, textcoords="offset points", fontsize=7.5, color=plotstyle.TEXT2)
ax1.axhline(95, color=plotstyle.NEUTRAL, lw=1, ls="--"); ax1.axvline(1, color=plotstyle.NEUTRAL, lw=1)
ax1.set_xlabel("efficiency vs GREG on the 5 dollar survey items (GM MSE ratio)")
ax1.set_ylabel("mean 95% CI coverage (%)")
ax1.set_title("(a) Full response: PPI is as efficient as calibration and keeps nominal coverage", loc="left", fontsize=9.5)
ax1.set_xlim(0.8, 1.32); ax1.set_ylim(86, 96.5)
labs = list(nr.values()); yy = np.arange(len(labs))
v_all = [gm(e, A) for e in nr]; v_d = [gm(e, H_D) for e in nr]
ax2.barh(yy - 0.18, v_all, height=0.34, color=plotstyle.SERIES[0], label="all 15 variables")
ax2.barh(yy + 0.18, v_d, height=0.34, color=plotstyle.SERIES[1], label="5 dollar survey items")
ax2.axvline(1, color=plotstyle.NEUTRAL, lw=1)
ax2.set_yticks(yy); ax2.set_yticklabels(labs, fontsize=8); ax2.invert_yaxis()
ax2.set_xlabel("take-some RRMSE relative to the status quo (GM ratio; < 1 is better)")
ax2.set_title("(b) Real late-filer nonresponse", loc="left", fontsize=9.5)
ax2.legend(fontsize=7.5, loc="lower right", frameon=False)
plt.tight_layout()
out = os.path.join(ROOT, "results", "summary.png"); plt.savefig(out, dpi=150, bbox_inches="tight")
print(out, [(p[0], round(p[2], 3), round(p[3], 1)) for p in pts], [round(v, 3) for v in v_all], [round(v, 3) for v in v_d])
