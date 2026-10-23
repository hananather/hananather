"""Stages 3 (nonresponse) and 4 (SAE) on the NE proof-of-concept population (protocol §7.2-7.3).

Stage 3, NR-LATE: a take-some unit "responds" if its TY2023 return was filed <= 300 days after FYE
(real behaviour). Take-all units: TA-follow-up (respond w.p. 0.9, MCAR) or TA-late (the real 300-day
rule); take-all nonrespondents are imputed by their own last-cycle value (TA-hist-own; certainty units
were in last cycle's sample). Take-some estimators use respondents with weighting-class (stratum) or
propensity adjustment.

Stage 4, SAE: unplanned domains state x NTEE10 (66). DIR / MA / SYN / FH for B0 GREG, the BMF-max
twin and the text model.

Usage: stage34_ne.py nr <scenario: follow|late> <R> <k> <K> | sae <R> | merge_nr <scenario> <K>
"""
import json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

from embed import load
from experiments import text_groups
from frame import production_design
from mc import Cell, run_cell
from protocol import H_SET, R_SET, add_text, tier_mats
from sae import run_sae, sae_metrics

DATA = os.environ.get("EMBAUX_DATA", "/tmp/claude-0/data")
OUT = f"{DATA}/results/stage34_ne"
SEED, N_SAMPLE, K_STAR = 20261023, 1000, 16
YS = H_SET + R_SET + ["y_bin_has_gov_grants"]
LAST_CYCLE = {  # own last-cycle value used to impute take-all nonrespondents
    "y_revenue": "lag_revenue", "y_expenses": "lag_expenses", "y_employees": "lag_employees", "y_salaries": "lag_salaries",
    "y_contrib": "lag_contrib", "y_prog_rev": "lag_prog_rev", "y_gov_grants": "oracle_lag_gov_grants",
    "y_fundraising_exp": "oracle_lag_func_exp_fundraising", "y_officer_comp": "oracle_lag_officer_comp",
    "y_volunteers_cap10k": "oracle_lag_volunteers",
}


def setup():
    pop = pd.read_parquet(f"{DATA}/pop/ne_pop_v2.parquet")
    # last-cycle binaries for take-all imputation, from the TY2022 returns
    p22 = pd.read_parquet(f"{DATA}/parsed/ne_2022.parquet").set_index("ein")
    for b, src in [("y_bin_lobbying", "lobbying"), ("y_bin_foreign_activities", "foreign_activities"),
                   ("y_bin_grants_to_indiv_us", "grants_to_indiv_us")]:
        pop[f"last_{b}"] = p22.loc[pop.ein, src].fillna(False).astype(float).to_numpy()
    pop["last_y_bin_has_volunteers"] = (pop.oracle_lag_volunteers > 0).astype(float)
    pop["last_y_bin_has_gov_grants"] = (pop.oracle_lag_gov_grants > 0).astype(float)
    pop["oracle_lag_volunteers"] = pop.oracle_lag_volunteers.clip(upper=10_000)
    des, _ = production_design(pop, N_SAMPLE)
    mats, cats = tier_mats(pop, des, ("F1p",))
    add_text(mats, cats, pop, ("F1p",), "Estar", load(f"{DATA}/emb/ne_v2", "lsa256_full")[0], K_STAR)
    return pop, des, mats, cats


def last_cycle_matrix(pop):
    cols = []
    for y in YS:
        cols.append(pop[LAST_CYCLE[y]].to_numpy(float) if y in LAST_CYCLE else pop[f"last_{y}"].to_numpy(float))
    return np.column_stack(cols)


def nr(scenario, R, k, K):
    os.makedirs(OUT, exist_ok=True)
    pop, des, mats, cats = setup()
    resp = (pop.days_to_file_t <= 300).fillna(False).to_numpy()
    x = pop.lag_expenses.clip(lower=0).to_numpy(float)
    cell = Cell(pop[YS].to_numpy(float), YS, des, mats, x, groups=text_groups(pop), resp=resp, cat_masks=cats)
    Y, L = cell.Y, last_cycle_matrix(pop)
    ta_idx = np.flatnonzero(cell.ta)

    def ta_values(r):
        if scenario == "follow":
            rr = np.random.default_rng([SEED, r, 3]).random(len(ta_idx)) < 0.9
        else:
            rr = resp[ta_idx]
        return Y[ta_idx[rr]].sum(0) + L[ta_idx[~rr]].sum(0)

    specs = {
        "FULLRESP_GREG_size_cat@F1p": {"kind": "greg", "X": "size_cat@F1p"},
        "NR_rhg_HT": {"kind": "ht", "nr": True, "nr_adj": "rhg"},
        "NR_SQ_rhg_GREG_size_cat@F1p": {"kind": "greg", "X": "size_cat@F1p", "nr": True, "nr_adj": "rhg"},
        "NR_rhg_MCAL_bmfmax@F1p": {"kind": "mcal", "F": "FC@F1p", "X_base": "size_cat@F1p", "ratio": True, "nr": True, "nr_adj": "rhg"},
        "NR_rhg_MCAL_bmfmax@F1p+Estar": {"kind": "mcal", "F": "FC@F1p+Estar", "X_base": "size_cat@F1p", "ratio": True, "nr": True, "nr_adj": "rhg"},
        "NR_ps_frame_MCAL_bmfmax@F1p": {"kind": "mcal", "F": "FC@F1p", "X_base": "size_cat@F1p", "ratio": True, "nr": True, "nr_adj": "ps", "ps_F": "FC@F1p"},
        "NR_ps_text_MCAL_bmfmax@F1p+Estar": {"kind": "mcal", "F": "FC@F1p+Estar", "X_base": "size_cat@F1p", "ratio": True, "nr": True, "nr_adj": "ps", "ps_F": "FC@F1p+Estar"},
    }
    a, b = round(k * R / K), round((k + 1) * R / K)
    t0 = time.time()
    df = run_cell(cell, specs, R=R, seed=SEED, rep_range=(a, b), ta_values=ta_values)
    df.to_parquet(f"{OUT}/nr_{scenario}_part_{k}_of_{K}.parquet")
    json.dump({"truth": dict(zip(YS, Y.sum(0).tolist())), "resp_rate_ts": float(resp[cell.ts].mean()),
               "resp_rate_ta_real": float(resp[cell.ta].mean()), "secs": time.time() - t0},
              open(f"{OUT}/nr_{scenario}_meta.json", "w"), indent=1)
    print("done", round(time.time() - t0), "s", flush=True)


def merge_nr(scenario, K):
    df = pd.concat([pd.read_parquet(f"{OUT}/nr_{scenario}_part_{k}_of_{K}.parquet") for k in range(K)], ignore_index=True)
    df.to_parquet(f"{OUT}/nr_{scenario}.parquet")
    print("merged", df.shape)


def sae(R):
    os.makedirs(OUT, exist_ok=True)
    pop, des, mats, cats = setup()
    x = pop.lag_expenses.clip(lower=0).to_numpy(float)
    ys = ["y_gov_grants", "y_contrib", "y_fundraising_exp", "y_volunteers_cap10k", "y_bin_has_gov_grants",
          "y_bin_has_volunteers", "y_bin_lobbying", "y_expenses"]
    cell = Cell(pop[ys].to_numpy(float), ys, des, mats, x, groups=text_groups(pop), cat_masks=cats)
    models = {"B0": {"kind": "greg", "X": "size_cat@F1p"},
              "twin": {"kind": "mcal_ratio", "F": "FC@F1p", "X_base": "size_cat@F1p"},
              "text": {"kind": "mcal_ratio", "F": "FC@F1p+Estar", "X_base": "size_cat@F1p"}}
    dom = (pop.state + "_" + pop.ntee_major10).to_numpy()
    t0 = time.time()
    df, tt = run_sae(cell, dom, models, R=R, seed=SEED)
    df.to_parquet(f"{OUT}/sae.parquet"); tt.to_parquet(f"{OUT}/sae_truth.parquet")
    sae_metrics(df, tt).to_csv(f"{OUT}/sae_per_domain.csv", index=False)
    print("done", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "nr":
        nr(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    elif c == "merge_nr":
        merge_nr(sys.argv[2], int(sys.argv[3]))
    elif c == "sae":
        sae(int(sys.argv[2]))
