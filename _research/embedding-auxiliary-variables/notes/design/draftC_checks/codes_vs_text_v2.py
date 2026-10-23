"""Attribution pre-check: does a fine-grained register code capture the text gain? (Stage-1 ceilings, REG-S+)"""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/hananather/_research/embedding-auxiliary-variables/src")
from frame import production_design, frame_matrix, pcs, dummies
from diagnostics import evaluate_feature_sets
t0 = time.time()
pop = pd.read_parquet("/tmp/claude-0/data/pop/ne_pop.parquet")
des, _ = production_design(pop, 1000)
X, _ = frame_matrix(pop, des, "strong")
B = np.c_[X, pop[["lag_salaries", "lag_contrib", "lag_prog_rev"]].to_numpy(float)]
x = pop.lag_expenses.clip(lower=0).to_numpy(float); xs = x / x.std()
emp = pop.lag_employees.clip(lower=0).to_numpy(float); es = emp / emp.std()
# fine register codes: NTEE letter, NTEE code (>=20 units, else letter), subsection, ruling decade, asset code
code = pop.ntee_code.fillna("MISSING")
cnt = code.map(code.value_counts())
code = np.where(cnt >= 20, code, pop.ntee_letter.fillna("MISSING"))
C = np.c_[dummies(pop.ntee_letter.fillna("MISSING"), True).to_numpy(), dummies(pd.Series(code), True).to_numpy(),
          dummies(pop.subsection.fillna("NA"), True).to_numpy(),
          dummies((pop.ruling_year.fillna(0) // 10).astype(int), True).to_numpy(),
          dummies(pop.bmf_asset_cd.fillna(-1), True).to_numpy()]
C = C[:, C.std(0) > 0]
Cf = C
E = np.load("/tmp/claude-0/data/emb/ne/lsa256_full.npy")
Z, _ = pcs(E, 32)
T = np.c_[Z, Z * xs[:, None], Z * es[:, None]]
ys = ["y_gov_grants", "y_fundraising_exp", "y_officer_comp", "y_volunteers_cap10k", "y_bin_has_gov_grants",
      "y_bin_grants_to_indiv_us", "y_grants_paid", "y_expenses"]
Y = pop[ys].to_numpy(float)
k = B.shape[1]
fs = {"base": (B, k), "codes": (np.c_[B, Cf], k), "text": (np.c_[B, T], k), "codes+text": (np.c_[B, Cf, T], k)}
print({n: f[0].shape[1] for n, f in fs.items()}, "n_code_levels", len(set(code)), flush=True)
r = evaluate_feature_sets(Y, ys, des, fs, baseline="base", lams=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0))
out = r.pivot_table(index="y", columns="features", values="re_vs_baseline")[["codes", "text", "codes+text"]]
out["text_given_codes"] = out["codes+text"] / out["codes"]
out["share_text_gain_explained_by_codes"] = 1 - np.log(out["text_given_codes"]) / np.log(out["text"])
print(out.round(3).to_string())
gm = np.exp(np.log(out.loc[ys[:6]]).mean())
print("GM over 6 headline y:\n", gm.round(3).to_string())
out.to_csv("/tmp/claude-0/-home-user-hananather/61c19d0b-7d5d-50f6-9821-a76e0ea08cdb/scratchpad/drC/codes_vs_text_v2.csv")
print("seconds", round(time.time() - t0))
