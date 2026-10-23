"""Stage 2F (protocol §3.2, §7.1; S4, S5): text at the design stage, on the NE proof-of-concept population.

Designs (each n = 1,000, take-all = top 1% on lag_expenses, compromise allocation as in D1):
  D1      NTEE10 (MISSING as a level) x 3 size classes                       status quo
  D1-imp  D1 with MISSING NTEE10 replaced by text-coded NTEE10 (multinomial logit on E* = Z_16 of LSA;
          trained on frame units with a known code; no survey variable)          S4: text as frame coder
  D1-H44  NTEE10 x 4 size classes                                             equal-H twin of D2a
  D2a     NTEE10 x 2 size classes x 2 classes of a frozen text score (previous-wave text head's predicted
          gov-grant ratio, split at the frame median within each NTEE10 x size cell)   S5: prediction strata

Estimators: HT and GREG_size_cat@F1p (R = 2,000), MCAL_bmfmax@F1p (R = 500, first 500 replicates, M8).

Usage: stage2f_ne.py run | merge
"""
import json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

from embed import load
from experiments import impute_ntee
from frame import pcs, production_design
from mc import Cell, run_cell
from protocol import H_SET, R_SET, tier_mats
from survey import StratifiedDesign, cum_sqrt_f_boundaries, neyman_allocation

DATA = os.environ.get("EMBAUX_DATA", "/tmp/claude-0/data")
OUT = f"{DATA}/results/stage2f_ne_v3"
SEED, N_SAMPLE, K_STAR = 20261023, 1000, 16
REAL_Y = H_SET + R_SET + ["y_bin_has_gov_grants"]


def design_from_labels(lab, x, n, take_all_share=0.01, min_n=10, alloc_power=0.5):
    """production_design's take-all and allocation rules applied to arbitrary take-some labels."""
    lab = pd.factorize(lab)[0]
    ta = np.zeros(len(x), bool)
    ta[np.argsort(-x)[: int(round(take_all_share * len(x)))]] = True
    lab = pd.factorize(np.where(ta, lab.max() + 1, lab))[0]
    H = lab.max() + 1
    mask = np.zeros(H, bool); mask[np.unique(lab[ta])] = True
    return StratifiedDesign(lab, neyman_allocation(lab, x ** alloc_power, n, min_n=min_n, take_all=mask))


def designs(pop):
    x = pop.lag_expenses.clip(lower=0).to_numpy(float)
    out, info = {}, {}
    out["D1"], _ = production_design(pop, N_SAMPLE)
    Z, _ = pcs(load(f"{DATA}/emb/ne_v3", "lsa256_full")[0], K_STAR)
    imp, acc = impute_ntee(pop, Z)
    info["D1-imp"] = {"cv_accuracy_known": acc, "n_imputed": int((pop.ntee_major10 == "MISSING").sum())}
    out["D1-imp"], _ = production_design(pop.assign(ntee_imp=imp.to_numpy()), N_SAMPLE, group_col="ntee_imp")
    out["D1-H44"], _ = production_design(pop, N_SAMPLE, n_size_classes=4)
    fz = np.load(f"{DATA}/results/stage2_ne_v3/prep.npz")["fz_text__y_gov_grants"].mean(0)[:, 0]
    score = fz / (x + 1.0)
    z = np.log1p(x)
    size2 = np.searchsorted(cum_sqrt_f_boundaries(z, 2), z)
    cell = pd.factorize(pop.ntee_major10.astype(str) + "_" + size2.astype(str))[0]
    med = pd.Series(score).groupby(cell).transform("median").to_numpy()
    out["D2a"] = design_from_labels(cell * 2 + (score > med), x, N_SAMPLE)
    for k, d in out.items():
        info.setdefault(k, {}).update({"H": int(d.H), "n": int(d.n), "take_all_n": int((d.n_h >= d.N_h)[d.strata].sum())})
    return out, info


def run():
    os.makedirs(OUT, exist_ok=True)
    pop = pd.read_parquet(f"{DATA}/pop/ne_pop_v3.parquet")
    Y = pop[REAL_Y].to_numpy(float)
    x = pop.lag_expenses.clip(lower=0).to_numpy(float)
    groups = np.load(f"{DATA}/results/stage2_ne_v3/fold_groups.npy")
    des, info = designs(pop)
    json.dump(info, open(f"{OUT}/designs.json", "w"), indent=1)
    print(json.dumps(info), flush=True)
    parts = []
    for name, d in des.items():
        t0 = time.time()
        mats, cats = tier_mats(pop, d, ("F1p",))
        cell = Cell(Y, REAL_Y, d, mats, x, groups=groups, cat_masks=cats)
        a = run_cell(cell, {"HT": {"kind": "ht"}, "GREG_size_cat@F1p": {"kind": "greg", "X": "size_cat@F1p"}},
                     R=2000, seed=SEED, verbose=False)
        b = run_cell(cell, {"MCAL_bmfmax@F1p": {"kind": "mcal", "F": "FC@F1p", "X_base": "size_cat@F1p", "ratio": True}},
                     R=500, seed=SEED, verbose=False)
        df = pd.concat([a, b], ignore_index=True).assign(design=name)
        df.to_parquet(f"{OUT}/mc_{name}.parquet")
        parts.append(df)
        print(name, "done", round(time.time() - t0), "s", flush=True)
    pd.concat(parts, ignore_index=True).to_parquet(f"{OUT}/mc.parquet")
    json.dump({"truth": dict(zip(REAL_Y, Y.sum(0).tolist()))}, open(f"{OUT}/meta.json", "w"), indent=1)


if __name__ == "__main__":
    run()
