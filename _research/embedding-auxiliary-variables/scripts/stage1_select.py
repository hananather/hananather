"""Stage 1 (protocol §6.5): select E* on the previous wave by same-class ceilings vs the BMF-max twin at F1p,
then compute the current-wave P1 ceilings used by gate G5. The cloud run used the v2 previous wave (identical
frame and text; v3 only changed response timing) and selected LSA full text, K = 16 (results/stage1_estar.json).

Usage: python3 -I scripts/stage1_select.py
"""
import json, os, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import pandas as pd

from embed import load
from frame import pcs, production_design
from frozen import _b5
from protocol import H_SET, ceiling_av, tier_mats

DATA = os.environ.get("EMBAUX_DATA", os.path.expanduser("~/embaux-data"))
prev = _b5(pd.read_parquet(f"{DATA}/pop/ne_wave2022_pop_v3.parquet"))     # test B5: no previous-wave target info in BMF
des, _ = production_design(prev, 1000)
mats, cats = tier_mats(prev, des, ("F1p",))
F, cat = mats["FC@F1p"], cats["FC@F1p"]
ys = [y for y in H_SET if y in prev]
av_twin = ceiling_av(prev, des, F, cat, ys)
rows, order = [], {"lsa256": 2, "minilm": 4}
for model in ["lsa256", "minilm"]:
    for field in ["mission", "full"]:
        Zall, _ = pcs(load(f"{DATA}/emb/ne_wave2022_v3", f"{model}_{field}")[0].astype(float), 64)
        for K in (16, 32, 64):
            re_ = av_twin / ceiling_av(prev, des, np.c_[F, Zall[:, :K]], np.r_[cat, np.zeros(K, bool)], ys)
            rows.append({"model": model, "field": field, "K": K, "rung": order[model],
                         "gm_ceiling": float(np.exp(np.log(re_).mean())), **dict(zip(ys, re_.tolist()))})
            print(model, field, K, round(rows[-1]["gm_ceiling"], 4), flush=True)
tab = pd.DataFrame(rows)
best = tab.gm_ceiling.max()
estar = tab[tab.gm_ceiling >= best - 0.02].sort_values(["rung", "K"]).iloc[0]    # ties within 0.02 -> simpler rung, smaller K
print("E* =", estar.model, estar.field, int(estar.K), "| GM ceiling", round(estar.gm_ceiling, 4), "| best", round(best, 4))
tab.to_csv(f"{DATA}/results/stage1_selection_local.csv", index=False)

# current-wave P1 ceilings for gate G5
pop = pd.read_parquet(f"{DATA}/pop/ne_pop_v3.parquet")
des, _ = production_design(pop, 1000)
mats, cats = tier_mats(pop, des, ("F1p",))
F, cat = mats["FC@F1p"], cats["FC@F1p"]
Z, _ = pcs(load(f"{DATA}/emb/ne_v3", "lsa256_full")[0].astype(float), 16)
c = dict(zip(H_SET, (ceiling_av(pop, des, F, cat, H_SET) / ceiling_av(pop, des, np.c_[F, Z], np.r_[cat, np.zeros(16, bool)], H_SET)).tolist()))
os.makedirs(f"{DATA}/results/stage2_ne_v3", exist_ok=True)
json.dump(c, open(f"{DATA}/results/stage2_ne_v3/ceilings_P1.json", "w"), indent=1)
print("P1 ceilings GM", round(float(np.exp(np.mean(np.log(list(c.values()))))), 4))
