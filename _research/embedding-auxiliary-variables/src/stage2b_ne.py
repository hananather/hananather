"""Stage 2B (protocol §11.2, S2/S2b): the weak register F0 on its own design D1-W.

F0 = BMF-max codes, state, BMF income code and BMF amounts (income, revenue, assets), no return-based
lags. D1-W = NTEE10 x 3 BMF INCOME_AMT classes + take-all on INCOME_AMT (never lagged expenses).
Ratio-scale models use x~ = BMF INCOME_AMT, the only size measure F0 knows.

For the gap closure S2b the F1p twin and text arms run on the SAME D1-W samples (common random numbers:
samples depend only on (seed, rep), folds on [seed, rep, 1]) in a second cell whose x~ = lag_expenses.

Variants (deviation D-28):
  w     D1-W exactly as specified (size and take-all on BMF INCOME_AMT). The 2023-08 BMF has zero or
        missing amounts for 722 units holding 6.7% of expenses (Harvard, Tufts, Northeastern, BCBS-MA),
        which this design places in the bottom size classes.
  wprof sensitivity: large-enterprise profiling. The 206 largest units by frame lag_expenses (TY2022
        expenses if filed by D, else BMF REVENUE_AMT: 86 of the 206) get that as their design size;
        everyone else keeps BMF INCOME_AMT. Take-all is re-ranked on this combined size (171 of the 206
        land in it, all zero-BMF giants included). x~ unchanged.

Usage: stage2b_ne.py run <variant> <k> <K> <R> | merge <variant> <K>
"""
import json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

from embed import load
from frame import production_design
from mc import Cell, run_cell
from protocol import H_SET, R_SET, _dum, add_text, bmfmax_codes, control_embeddings, tier_mats

DATA = os.environ.get("EMBAUX_DATA", "/tmp/claude-0/data")
OUT0 = f"{DATA}/results/stage2b_ne_v3"
SEED, N_SAMPLE, K_STAR = 20261023, 1000, 16
REAL_Y = H_SET + R_SET + ["y_bin_has_gov_grants"]
BMF_AMT = ["bmf_income_amt", "bmf_revenue_amt", "bmf_asset_amt"]


def f0_mats(pop, design, min_count=20):
    """F0 matrices with the same naming as protocol.tier_mats (no lags, no availability terms)."""
    S = _dum(design.strata, drop_first=False)
    nt, st, inc = _dum(pop.ntee_major10), _dum(pop.state), _dum(pop.bmf_income_cd.fillna(-1))
    C, Ccont = bmfmax_codes(pop, min_count)
    amt = pop[BMF_AMT].clip(lower=0).to_numpy(float)
    cat_part = np.c_[nt, st, inc]
    mats = {"size_cat@F0": np.c_[S, amt, nt, st],
            "F@F0": np.c_[cat_part, Ccont],
            "FC@F0": np.c_[cat_part, C, Ccont], "_C": C}
    cats = {"F@F0": np.r_[np.ones(cat_part.shape[1], bool), np.zeros(Ccont.shape[1], bool)],
            "FC@F0": np.r_[np.ones(cat_part.shape[1] + C.shape[1], bool), np.zeros(Ccont.shape[1], bool)],
            "_C": np.ones(C.shape[1], bool)}
    return mats, cats


def setup(variant="w"):
    pop = pd.read_parquet(f"{DATA}/pop/ne_pop_v3.parquet")
    if variant == "wprof":
        top = np.zeros(len(pop), bool)
        top[np.argsort(-pop.lag_expenses.clip(lower=0).to_numpy())[: int(round(0.01 * len(pop)))]] = True
        pop["size_w"] = np.where(top, pop.lag_expenses.clip(lower=0), pop.bmf_income_amt.clip(lower=0))
    else:
        pop["size_w"] = pop.bmf_income_amt.clip(lower=0)
    des, info = production_design(pop, N_SAMPLE, size_col="size_w")      # D1-W (or D1-W-prof)
    Es = load(f"{DATA}/emb/ne_v3", "lsa256_full")[0]
    L4 = load(f"{DATA}/emb/ne_v3", "minilm_full")[0]
    m0, c0 = f0_mats(pop, des)
    add_text(m0, c0, pop, ("F0",), "Estar", Es, K_STAR)
    add_text(m0, c0, pop, ("F0",), "Estar64", Es, 64)
    add_text(m0, c0, pop, ("F0",), "L4", L4, K_STAR)
    add_text(m0, c0, pop, ("F0",), "N2perm", control_embeddings(Es, des, seed=SEED)["N2perm"], K_STAR)
    m1, c1 = tier_mats(pop, des, ("F1p",))
    add_text(m1, c1, pop, ("F1p",), "Estar", Es, K_STAR)
    groups = np.load(f"{DATA}/results/stage2_ne_v3/fold_groups.npy")
    Y = pop[REAL_Y].to_numpy(float)
    cell0 = Cell(Y, REAL_Y, des, m0, pop.bmf_income_amt.clip(lower=0).to_numpy(float), groups=groups, cat_masks=c0)
    cell1 = Cell(Y, REAL_Y, des, m1, pop.lag_expenses.clip(lower=0).to_numpy(float), groups=groups, cat_masks=c1)
    mc = lambda F: {"kind": "mcal", "F": F, "X_base": "size_cat@F0", "ratio": True}
    specs0 = {"HT": {"kind": "ht"}, "GREG_size_cat@F0": {"kind": "greg", "X": "size_cat@F0"},
              "MCAL_frame@F0": mc("F@F0"), "MCAL_frame@F0+Estar": mc("F@F0+Estar"),
              "MCAL_bmfmax@F0": mc("FC@F0"), "MCAL_bmfmax@F0+Estar": mc("FC@F0+Estar"),
              "MCAL_bmfmax@F0+Estar64": mc("FC@F0+Estar64"), "MCAL_bmfmax@F0+L4": mc("FC@F0+L4"),
              "MCAL_bmfmax@F0+N2perm": mc("FC@F0+N2perm")}
    mc1 = lambda F: {"kind": "mcal", "F": F, "X_base": "size_cat@F1p", "ratio": True}
    specs1 = {"GREG_size_cat@F1p": {"kind": "greg", "X": "size_cat@F1p"},
              "MCAL_bmfmax@F1p": mc1("FC@F1p"), "MCAL_bmfmax@F1p+Estar": mc1("FC@F1p+Estar")}
    return pop, des, info, cell0, specs0, cell1, specs1


def run(variant, k, K, R):
    OUT = f"{OUT0}_{variant}"
    os.makedirs(OUT, exist_ok=True)
    pop, des, info, cell0, specs0, cell1, specs1 = setup(variant)
    a, b = round(k * R / K), round((k + 1) * R / K)
    print(f"worker {k}/{K}: reps {a}..{b - 1}; D1-W strata {info['strata']}, take-all {info['take_all_n']}", flush=True)
    t0 = time.time()
    d0 = run_cell(cell0, specs0, R=R, seed=SEED, rep_range=(a, b))
    d1 = run_cell(cell1, specs1, R=R, seed=SEED, rep_range=(a, b))
    df = pd.concat([d0, d1], ignore_index=True)
    df.to_parquet(f"{OUT}/part_{k}_of_{K}.parquet")
    fails = {**d0.attrs["failures"], **d1.attrs["failures"]}
    json.dump({"failures": fails, "secs": time.time() - t0}, open(f"{OUT}/part_{k}_of_{K}.json", "w"), indent=1)
    print("done", round(time.time() - t0), "s", flush=True)


def merge(variant, K):
    OUT = f"{OUT0}_{variant}"
    df = pd.concat([pd.read_parquet(f"{OUT}/part_{k}_of_{K}.parquet") for k in range(K)], ignore_index=True)
    df.to_parquet(f"{OUT}/mc.parquet")
    pop = pd.read_parquet(f"{DATA}/pop/ne_pop_v3.parquet")
    fails = {}
    for k in range(K):
        for e, f in json.load(open(f"{OUT}/part_{k}_of_{K}.json"))["failures"].items():
            fails[e] = fails.get(e, 0) + f
    json.dump({"truth": dict(zip(REAL_Y, pop[REAL_Y].sum().tolist())), "R": int(df.rep.nunique()), "n": N_SAMPLE,
               "failures": fails}, open(f"{OUT}/meta.json", "w"), indent=1)
    print("merged", df.shape, "R =", df.rep.nunique())


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    elif sys.argv[1] == "merge":
        merge(sys.argv[2], int(sys.argv[3]))
