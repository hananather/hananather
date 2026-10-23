import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/hananather/_research/embedding-auxiliary-variables/src")
from mc import _ridge_fit, LAMS
from frame import production_design
pop = pd.read_parquet("/tmp/claude-0/data/pop/ne_pop.parquet")
des, info = production_design(pop, 1000)
rng = np.random.default_rng(1)
s = des.draw(rng)
ta = (des.n_h >= des.N_h)[des.strata]
s = s[~ta[s]]
d = 1/des.pi[s]
x = pop.lag_expenses.clip(lower=0).to_numpy(float)[s] + 1
y = pop.y_gov_grants.to_numpy(float)[s]
p = 60
F = rng.normal(size=(len(s), p))  # pure-noise features
for label, T, w in [("raw  w=d", y[:,None], d), ("ratio w=d*x", (y/x)[:,None], d*x), ("ratio w normalised", (y/x)[:,None], d*x/np.mean(d*x))]:
    norms = []
    for lam in LAMS:
        b0, B = _ridge_fit(F, T, w, lam, 0)
        norms.append(np.linalg.norm(B))
    print(f"{label:22s} sum(w)={w.sum():.3g}  n={len(s)}  ||B|| over lambdas {LAMS}:", np.round(np.array(norms)/norms[0], 4))
