"""Stage 2A (protocol §11.2): New England proof-of-concept cell. F1p primary tier (F1 and F2 alongside),
design D1, n = 1,000, common random numbers, shared folds, ~35 estimators, placebo and planted targets.

Usage:
  python3 -I src/stage2_ne.py prep                 # planted calibration, placebos, frozen heads -> cache
  python3 -I src/stage2_ne.py run <k> <K> <R>      # worker k of K runs its slice of replicates 0..R-1
  python3 -I src/stage2_ne.py merge <K>
"""
import json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

from embed import load
from experiments import text_groups
from frame import production_design
from frozen import train_frozen
from mc import Cell, run_cell
from protocol import (H_SET, R_SET, J4, add_text, control_embeddings, placebo_targets, planted_targets,
                      specs_primary, specs_tier, tier_mats)

DATA = os.environ.get("EMBAUX_DATA", "/tmp/claude-0/data")
OUT = f"{DATA}/results/stage2_ne"
SEED, N_SAMPLE, K_STAR = 20261023, 1000, 16   # E* = LSA T2, K=16 (Stage 1, tie rule)
REAL_Y = H_SET + R_SET + ["y_bin_has_gov_grants"]


def setup():
    pop = pd.read_parquet(f"{DATA}/pop/ne_pop_v2.parquet")
    des, info = production_design(pop, N_SAMPLE)
    tiers = ("F1", "F1p", "F2")
    mats, cats = tier_mats(pop, des, tiers)
    Es = load(f"{DATA}/emb/ne_v2", "lsa256_full")[0]          # E* = LSA T2, K = 64 (Stage 1)
    emb = {"Estar": Es, "Estar64": Es, "L4": load(f"{DATA}/emb/ne_v2", "minilm_full")[0],
           "Estar_nonum": load(f"{DATA}/emb/ne_v2", "lsa256_full_nonum")[0]}
    add_text(mats, cats, pop, ("F1", "F1p", "F2"), "Estar", Es, K_STAR, linear_factorial=False)
    add_text(mats, cats, pop, ("F1p",), "Estar", Es, K_STAR, linear_factorial=True)
    for nm in ("L4", "Estar_nonum"):
        add_text(mats, cats, pop, ("F1p",), nm, emb[nm], K_STAR)
    add_text(mats, cats, pop, ("F1p",), "Estar64", Es, 64)          # secondary: best-ceiling K
    ctrl = control_embeddings(Es, des, seed=SEED)
    for nm, E in ctrl.items():
        add_text(mats, cats, pop, ("F1p",), nm, E, K_STAR)
    return pop, des, info, mats, cats, list(ctrl)


def prep():
    os.makedirs(OUT, exist_ok=True)
    pop, des, info, mats, cats, ctrl = setup()
    t0 = time.time()
    plc = placebo_targets(pop, des, seed=SEED)
    planted, pinfo = planted_targets(pop, des, load(f"{DATA}/emb/ne_v2", "lsa256_full")[0], mats["FC@F1p"], cats["FC@F1p"],
                                     levels=(0.0, 0.05, 0.10, 0.20), plantings=3, K=K_STAR, seed=SEED)
    print("planted calibrated", round(time.time() - t0), "s", flush=True)
    pinfo.to_csv(f"{OUT}/planted_info.csv", index=False)
    prev = pd.read_parquet(f"{DATA}/pop/ne_wave2022_pop_v2.parquet")
    Ep = load(f"{DATA}/emb/ne_frozen", "lsa256_full_prev")[0]
    Ec = load(f"{DATA}/emb/ne_frozen", "lsa256_full_curr")[0]
    fz, flog = train_frozen(prev, pop, REAL_Y, "F1p", Ep, Ec, n=N_SAMPLE, J=20, K=K_STAR, seed=SEED)
    flog.to_csv(f"{OUT}/frozen_training_log.csv", index=False)
    print("frozen heads trained", round(time.time() - t0), "s", flush=True)
    extra = {**plc, **planted}
    np.savez_compressed(f"{OUT}/prep.npz", extra_names=np.array(list(extra)), extra=np.column_stack(list(extra.values())),
                        **{k.replace(":", "__"): v for k, v in fz.items() if ":" in k or k in ("fz_twin", "fz_text")})
    json.dump({"design": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in info.items()},
               "n_h": des.n_h.tolist(), "N_h": des.N_h.tolist()}, open(f"{OUT}/design.json", "w"), indent=1)


def build_cell():
    pop, des, info, mats, cats, ctrl = setup()
    z = np.load(f"{OUT}/prep.npz", allow_pickle=False)
    extra_names = [str(s) for s in z["extra_names"]]
    ynames = REAL_Y + extra_names
    Y = np.c_[pop[REAL_Y].to_numpy(float), z["extra"]]
    q_real = len(REAL_Y)
    frozen = {}
    for key in ("fz_twin", "fz_text"):
        a = z[key]                                           # (J, N, q_real); placebo/planted heads = 0
        frozen[key] = np.concatenate([a, np.zeros(a.shape[:2] + (len(extra_names),))], 2)
        for y in J4:
            frozen[f"{key}:{y}"] = z[f"{key}__{y}"]
    x = pop.lag_expenses.clip(lower=0).to_numpy(float)
    cell = Cell(Y, ynames, des, mats, x, groups=text_groups(pop), frozen=frozen, cat_masks=cats)
    n_sc = cell.mats["size_cat@F1p"].shape[1]
    specs = specs_primary("F1p", "Estar", controls=ctrl, extra_text=("L4", "Estar_nonum", "Estar64"), with_frozen=True, n_sc=n_sc)
    specs.update(specs_tier("F1", "Estar"))
    specs.update(specs_tier("F2", "Estar"))
    return cell, specs, ynames


def run(k, K, R):
    cell, specs, ynames = build_cell()
    a, b = round(k * R / K), round((k + 1) * R / K)
    print(f"worker {k}/{K}: reps {a}..{b - 1}; estimators {len(specs)}; targets {len(ynames)}", flush=True)
    t0 = time.time()
    df = run_cell(cell, specs, R=R, seed=SEED, rep_range=(a, b))
    df.to_parquet(f"{OUT}/part_{k}_of_{K}.parquet")
    json.dump({"timing_s": df.attrs["timing_s"], "failures": df.attrs["failures"], "secs": time.time() - t0},
              open(f"{OUT}/part_{k}_of_{K}.json", "w"), indent=1)
    print("done", round(time.time() - t0), "s", flush=True)


def merge(K):
    df = pd.concat([pd.read_parquet(f"{OUT}/part_{k}_of_{K}.parquet") for k in range(K)], ignore_index=True)
    df.to_parquet(f"{OUT}/mc.parquet")
    cell, specs, ynames = build_cell()
    truth = dict(zip(cell.ynames, cell.Y.sum(0).tolist()))
    fails = {}
    for k in range(K):
        for e, f in json.load(open(f"{OUT}/part_{k}_of_{K}.json"))["failures"].items():
            fails[e] = fails.get(e, 0) + f
    json.dump({"truth": truth, "R": int(df.rep.nunique()), "n": N_SAMPLE, "failures": fails}, open(f"{OUT}/meta.json", "w"), indent=1)
    print("merged", df.shape, "R =", df.rep.nunique())


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "prep":
        prep()
    elif cmd == "run":
        run(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "merge":
        merge(int(sys.argv[2]))
