"""Stage 2A repair (deviation D-32): the linear factorial cells, rerun on the same samples and folds.

The RIDGECAL arms (ridge-GREG on [size_cat, soft blocks] with in-sample coefficients) failed gates G3/G4:
with the 419 sparse code dummies, cross-validation on residual variance ignores the calibration term
(T_X - X_HT) B, picks a small penalty, and the estimator explodes (RRMSE 200-700%). LINCAL replaces them
with exactly the MCAL pipeline (cross-fitted predictions, fold-averaged population predictions, model
calibration on [size_cat, m]), with a *linear raw-scale* learner on the same RC@ matrices. The factorial's
factor A is then purely the learner: linear in raw columns vs ratio-scale ridge on log features.

Usage: stage2a_lincal.py run <k> <K> <R> | merge <K>
"""
import json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd

import stage2_ne as s2
from mc import run_cell

OUT = s2.OUT


def specs(cell):
    n_sc = cell.mats["size_cat@F1p"].shape[1]
    lin = lambda X: {"kind": "mcal", "F": X, "X_base": "size_cat@F1p", "ratio": False, "hard": n_sc}
    return {"LINCAL@F1p+Estar": lin("RC@F1p+Estar"), "LINCAL@F1p+C": lin("RC@F1p+C"), "LINCAL@F1p+C+Estar": lin("RC@F1p+C+Estar")}


def run(k, K, R):
    cell, _, _ = s2.build_cell()
    a, b = round(k * R / K), round((k + 1) * R / K)
    t0 = time.time()
    df = run_cell(cell, specs(cell), R=R, seed=s2.SEED, rep_range=(a, b))
    df.to_parquet(f"{OUT}/lincal_part_{k}_of_{K}.parquet")
    json.dump({"timing_s": df.attrs["timing_s"], "failures": df.attrs["failures"], "secs": time.time() - t0},
              open(f"{OUT}/lincal_part_{k}_of_{K}.json", "w"), indent=1)
    print("done", round(time.time() - t0), "s", flush=True)


def merge(K):
    df = pd.concat([pd.read_parquet(f"{OUT}/lincal_part_{k}_of_{K}.parquet") for k in range(K)], ignore_index=True)
    df.to_parquet(f"{OUT}/mc_lincal.parquet")
    print("merged", df.shape, "R =", df.rep.nunique())


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    else:
        merge(int(sys.argv[2]))
