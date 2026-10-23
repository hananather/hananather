import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/hananather/_research/embedding-auxiliary-variables/src")
pop = pd.read_parquet("/tmp/claude-0/data/pop/ne_pop.parquet")
prev = pd.read_parquet("/tmp/claude-0/data/pop/ne_wave2022_pop.parquet")
print("N", len(pop), "prev N", len(prev), "overlap", len(set(pop.ein)&set(prev.ein)))
ys = [c for c in pop.columns if c.startswith("y_")]
rows=[]
for y in ys:
    v = pop[y].astype(float)
    tot = v.sum()
    s = np.sort(v.to_numpy())[::-1]
    top1 = s[:int(0.01*len(s))].sum()/tot if tot else np.nan
    rows.append((y, round((v==0).mean(),3), round(top1,3), f"{tot:.3g}", round(v.std()/v.mean(),1) if v.mean() else np.nan))
print(pd.DataFrame(rows, columns=["y","share0","top1%share","total","cv"]).to_string())
print("lag_expenses==0:", (pop.lag_expenses<=0).mean(), " lag_expenses<=0 but y_gov_grants>0:", ((pop.lag_expenses<=0)&(pop.y_gov_grants>0)).mean())
print("NTEE letters known:", pop.ntee_letter.notna().mean(), "distinct ntee_code", pop.ntee_code.nunique())
print("cols:", [c for c in pop.columns if not c.startswith("y_")])
# domains
d = pop.groupby(["state","ntee_major10"]).size()
print("state x ntee10 domains", len(d), "median N_d", d.median(), "N_d<30", (d<30).sum())
print("states", pop.state.value_counts().to_dict())
# correlations lag vs y
for a,b in [("lag_salaries","y_salaries"),("lag_contrib","y_contrib"),("oracle_lag_gov_grants","y_gov_grants"),("oracle_lag_func_exp_fundraising","y_fundraising_exp"),("oracle_lag_volunteers","y_volunteers")]:
    print(a,b, round(np.corrcoef(pop[a],pop[b])[0,1],3), "log:", round(np.corrcoef(np.log1p(pop[a].clip(0)),np.log1p(pop[b].clip(0)))[0,1],3))
print("has website", pop.has_website.mean())
print("resp rate", (pop.days_to_file_t<=300).mean())
