"""Harness unit tests (run: python3 -I src/test_harness.py). Each prints PASS/FAIL."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from mc import _ridge_fit, Cell, run_cell, metrics
from survey import StratifiedDesign, neyman_allocation

ok = True
def report(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")

rng = np.random.default_rng(0)
# T-M1 penalty is effective whatever the weight scale (the ratio-weight defect)
F = rng.normal(size=(300, 20)); y = F[:, :1] * 3 + rng.normal(size=(300, 1))
for scale in (1.0, 1e11):
    w = rng.uniform(1, 5, 300) * scale
    norms = [np.linalg.norm(_ridge_fit(F, y, w, lam, 0)[1]) for lam in (1e-3, 1e-1, 10.0)]
    report(f"ridge penalty shrinks coefficients (weight scale {scale:g})", norms[0] > norms[1] > norms[2], np.round(norms, 3))

# small synthetic population for estimator tests
N = 6000
x = np.exp(rng.normal(9, 1.2, N)); z = rng.normal(size=N)
Y = np.c_[x * np.exp(0.3 * z + 0.2 * rng.normal(size=N)), (rng.uniform(size=N) < 1 / (1 + np.exp(-z))).astype(float)]
names = ["y_cont", "y_bin_flag"]
strata = np.searchsorted(np.quantile(x, [0.5, 0.9, 0.99]), x)
n_h = neyman_allocation(strata, np.sqrt(x), 400, min_n=10, take_all=np.array([0, 0, 0, 1], bool))
des = StratifiedDesign(strata, n_h)
S = np.eye(4)[strata]
const = np.ones((N, 5))
E = np.c_[z, rng.normal(size=(N, 4))]
mats = {"base": np.c_[S, x], "text": np.c_[np.log(x)[:, None], E], "const": np.c_[np.log(x)[:, None], const]}
cell = Cell(Y, names, des, mats, x)
fz = np.c_[x * 1.05, np.full(N, 0.5)]
cell.frozen = {"fz": fz[cell.ts]}
specs = {"HT": {"kind": "ht"}, "GREG": {"kind": "greg", "X": "base"},
         "MCAL_text": {"kind": "mcal", "F": "text", "hard": 0, "X_base": "base", "ratio": True},
         "MCAL_const": {"kind": "mcal", "F": "const", "hard": 0, "X_base": "base", "ratio": True},
         "FROZEN": {"kind": "frozen", "m": "fz"}}
df = run_cell(cell, specs, R=400, seed=1, verbose=False)
truth = dict(zip(names, Y.sum(0)))
m = metrics(df, truth, "GREG").set_index(["estimator", "y"])
for est in ["HT", "FROZEN"]:
    for yn in names:
        rb, se = m.loc[(est, yn), "rel_bias_pct"], m.loc[(est, yn), "rel_bias_mcse_pct"]
        report(f"{est} unbiased within 3 MC SE ({yn})", abs(rb) < 3 * se, f"rb={rb:.3f}% se={se:.3f}%")
report("text with signal beats GREG on continuous y", m.loc[("MCAL_text", "y_cont"), "re"] > 1.1, f"RE={m.loc[('MCAL_text', 'y_cont'), 're']:.2f}")
report("constant embedding gains nothing", abs(m.loc[("MCAL_const", "y_cont"), "re"] - 1) < 0.1, f"RE={m.loc[('MCAL_const', 'y_cont'), 're']:.2f}")
report("binary target not destroyed by ratio spec", 0.8 < m.loc[("MCAL_text", "y_bin_flag"), "re"], f"RE={m.loc[('MCAL_text', 'y_bin_flag'), 're']:.2f}")
# same samples whatever estimators are run (common random numbers)
df2 = run_cell(cell, {"HT": {"kind": "ht"}}, R=5, seed=1, verbose=False)
report("sample stream independent of estimator set", np.allclose(df2[df2.estimator == "HT"].est.to_numpy(), df[(df.estimator == "HT") & (df.rep < 5)].est.to_numpy()))
# N4: constant embedding reproduces the twin to machine precision (shared folds, same lambda path)
mats2 = {"base": np.c_[S, x], "twin": np.c_[np.log(x)[:, None], (strata == 0)[:, None].astype(float)],
         "twin_const": np.c_[np.log(x)[:, None], (strata == 0)[:, None].astype(float), np.ones((N, 4))]}
cats2 = {"twin": np.array([False, True]), "twin_const": np.array([False, True, False, False, False, False])}
cell2 = Cell(Y, names, des, mats2, x, cat_masks=cats2)
sp2 = {"TWIN": {"kind": "mcal", "F": "twin", "X_base": "base", "ratio": True},
       "CONST": {"kind": "mcal", "F": "twin_const", "X_base": "base", "ratio": True}}
d2 = run_cell(cell2, sp2, R=20, seed=3, verbose=False).pivot_table(index=["rep", "y"], columns="estimator", values="est")
rel = (np.abs(d2["CONST"] - d2["TWIN"]) / np.abs(d2["TWIN"])).max()
report("N4 constant embedding identical to twin (machine precision)", rel < 1e-8, f"max rel diff {rel:.2e}")
print("ALL PASS" if ok else "SOME FAILED")
