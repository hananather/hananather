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
from protocol import H_SET, R_SET, add_text, fold_groups, tier_mats
from sae import run_sae, sae_metrics

DATA = os.environ.get("EMBAUX_DATA", "/tmp/claude-0/data")
OUT = f"{DATA}/results/stage34_ne_v3"
SEED, N_SAMPLE, K_STAR = 20261023, 1000, 16
YS = H_SET + R_SET + ["y_bin_has_gov_grants"]
# own last-cycle value of every take-all unit (certainty units were in last cycle's sample): the
# UNMASKED TY2022 amounts (the frame's lag_* columns are masked for units unavailable at D, which
# must not leak into imputation -- adversarial-review finding)
LAST_CYCLE_SRC = {"y_revenue": "revenue_cy", "y_expenses": "expenses_cy", "y_employees": "employees_p1",
                  "y_salaries": "salaries_cy", "y_contrib": "contrib_cy", "y_prog_rev": "prog_rev_cy",
                  "y_gov_grants": "gov_grants", "y_fundraising_exp": "func_exp_fundraising",
                  "y_officer_comp": "officer_comp", "y_volunteers_cap10k": "volunteers",
                  "y_bin_lobbying": "lobbying", "y_bin_foreign_activities": "foreign_activities",
                  "y_bin_grants_to_indiv_us": "grants_to_indiv_us"}


def setup():
    pop = pd.read_parquet(f"{DATA}/pop/ne_pop_v3.parquet")
    p22 = pd.read_parquet(f"{DATA}/parsed/ne_2022.parquet").set_index("ein").reindex(pop.ein)
    for y, c in LAST_CYCLE_SRC.items():
        v = p22[c]
        v = v.astype(float) if v.dtype != object else v.map(lambda z: float(z) if z is not None else np.nan)
        if y.startswith("y_bin_"):
            v = v.fillna(0.0)
        elif y in ("y_gov_grants", "y_fundraising_exp", "y_officer_comp"):
            v = v.fillna(0.0)                 # blank Part VIII/IX lines are zeros
        pop[f"cyc_{y}"] = v.to_numpy(float)
    pop["cyc_y_volunteers_cap10k"] = np.clip(pop["cyc_y_volunteers_cap10k"], None, 10_000)
    pop["cyc_y_bin_has_volunteers"] = (pop["cyc_y_volunteers_cap10k"].fillna(0) > 0).astype(float)
    pop["cyc_y_bin_has_gov_grants"] = (pop["cyc_y_gov_grants"].fillna(0) > 0).astype(float)
    des, _ = production_design(pop, N_SAMPLE)
    mats, cats = tier_mats(pop, des, ("F1p",))
    add_text(mats, cats, pop, ("F1p",), "Estar", load(f"{DATA}/emb/ne_v3", "lsa256_full")[0], K_STAR)
    return pop, des, mats, cats


def last_cycle_matrix(pop):
    return np.column_stack([pop[f"cyc_{y}"].to_numpy(float) for y in YS])


def ta_imputer(pop, cell, Y, L, resp, scenario):
    """TA-hist-own (protocol §7.2): y_imp = own y_{t-1} x NTEE10 trend among responding take-all units
    (sum y_t / sum y_{t-1} within the group; binaries untrended). Missing own value -> TA-hist-aux:
    lag_expenses x ratio of y_t to lag_expenses among responding take-all units."""
    ta_idx = np.flatnonzero(cell.ta)
    grp = pop.ntee_major10.to_numpy()[ta_idx]
    is_bin = np.array([y.startswith("y_bin_") for y in YS])
    xlag = pop.lag_expenses.clip(lower=0).to_numpy(float)[ta_idx]
    Yta, Lta = Y[ta_idx], L[ta_idx]

    def values(r):
        if scenario == "follow":
            rr = np.random.default_rng([SEED, r, 3]).random(len(ta_idx)) < 0.9
        else:
            rr = resp[ta_idx]
        out = Yta[rr].sum(0).copy()
        nr = ~rr
        ok_all = rr & np.isfinite(Lta).all(1)
        for j in range(len(YS)):
            for g in np.unique(grp[nr]):
                m_nr = nr & (grp == g)
                if is_bin[j]:
                    trend = 1.0
                else:
                    m_r = rr & (grp == g) & np.isfinite(Lta[:, j])
                    den = Lta[m_r, j].sum()
                    trend = Yta[m_r, j].sum() / den if den > 0 else (Yta[ok_all, j].sum() / max(Lta[ok_all, j].sum(), 1e-9))
                own = Lta[m_nr, j]
                aux = xlag[m_nr] * (Yta[rr, j].sum() / max(xlag[rr].sum(), 1e-9))
                out[j] += np.where(np.isfinite(own), own * trend, aux).sum()
        return out
    return values


def nr(scenario, R, k, K):
    os.makedirs(OUT, exist_ok=True)
    pop, des, mats, cats = setup()
    resp = (pop.days_to_file_t <= 300).fillna(False).to_numpy()
    x = pop.lag_expenses.clip(lower=0).to_numpy(float)
    gpath = f"{OUT}/fold_groups.npy"
    if not os.path.exists(gpath):
        np.save(gpath, fold_groups(pop, load(f"{DATA}/emb/ne_v3", "lsa256_full")[0]))
    cell = Cell(pop[YS].to_numpy(float), YS, des, mats, x, groups=np.load(gpath), resp=resp, cat_masks=cats)
    Y, L = cell.Y, last_cycle_matrix(pop)
    ta_values = ta_imputer(pop, cell, Y, L, resp, scenario)
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
    ta_used = {}
    def ta_vals(r):
        v = ta_values(r); ta_used[r] = v; return v
    df = run_cell(cell, specs, R=R, seed=SEED, rep_range=(a, b), ta_values=ta_vals)
    df.to_parquet(f"{OUT}/nr_{scenario}_part_{k}_of_{K}.parquet")
    pd.DataFrame([(r, y, v[j]) for r, v in ta_used.items() for j, y in enumerate(YS)], columns=["rep", "y", "ta_used"]) \
        .to_parquet(f"{OUT}/nr_{scenario}_ta_part_{k}_of_{K}.parquet")
    json.dump({"truth": dict(zip(YS, Y.sum(0).tolist())), "truth_ta": dict(zip(YS, Y[cell.ta].sum(0).tolist())),
               "resp_rate_ts": float(resp[cell.ts].mean()), "resp_rate_ta_real": float(resp[cell.ta].mean()), "secs": time.time() - t0},
              open(f"{OUT}/nr_{scenario}_meta.json", "w"), indent=1)
    print("done", round(time.time() - t0), "s", flush=True)


def dr(R, k, K):
    """Deviation D-34: difference-form (PPI / AIPW) nonresponse estimators, not anchored on size calibration.
    Frozen heads come from Stage 2A's prep (trained on previous-wave respondents). Scenario TA-late; the
    take-some part is identical under TA-follow-up."""
    os.makedirs(OUT, exist_ok=True)
    pop, des, mats, cats = setup()
    resp = (pop.days_to_file_t <= 300).fillna(False).to_numpy()
    x = pop.lag_expenses.clip(lower=0).to_numpy(float)
    z = np.load(f"{DATA}/results/stage2_ne_v3/prep.npz")
    frozen = {"fz_twin": z["fz_twin"], "fz_text": z["fz_text"]}       # (J, N, 15), same y order as YS
    cell = Cell(pop[YS].to_numpy(float), YS, des, mats, x, groups=np.load(f"{OUT}/fold_groups.npy"), resp=resp,
                frozen=frozen, cat_masks=cats)
    Y, L = cell.Y, last_cycle_matrix(pop)
    ta_values = ta_imputer(pop, cell, Y, L, resp, "late")
    nr = lambda adj, **kw: {"nr": True, "nr_adj": adj, **kw}
    specs = {
        "NR_rhg_HT": {"kind": "ht", **nr("rhg")},
        "NR_SQ_rhg_GREG_size_cat@F1p": {"kind": "greg", "X": "size_cat@F1p", **nr("rhg")},
        "NR_rhg_FZDIFF_bmfmax@F1p": {"kind": "frozen", "m": "fz_twin", **nr("rhg")},
        "NR_rhg_FZDIFF_bmfmax@F1p+Estar": {"kind": "frozen", "m": "fz_text", **nr("rhg")},
        "NR_rhg_DIFF_bmfmax@F1p": {"kind": "mcal", "F": "FC@F1p", "ratio": True, "form": "diff", **nr("rhg")},
        "NR_rhg_DIFF_bmfmax@F1p+Estar": {"kind": "mcal", "F": "FC@F1p+Estar", "ratio": True, "form": "diff", **nr("rhg")},
        "NR_ps_DIFF_bmfmax@F1p": {"kind": "mcal", "F": "FC@F1p", "ratio": True, "form": "diff", **nr("ps", ps_F="FC@F1p")},
        "NR_ps_DIFF_bmfmax@F1p+Estar": {"kind": "mcal", "F": "FC@F1p+Estar", "ratio": True, "form": "diff", **nr("ps", ps_F="FC@F1p+Estar")},
        "NR_ps_FZDIFF_bmfmax@F1p+Estar": {"kind": "frozen", "m": "fz_text", **nr("ps", ps_F="FC@F1p+Estar")},
    }
    a, b = round(k * R / K), round((k + 1) * R / K)
    t0 = time.time()
    ta_used = {}
    def ta_vals(r):
        v = ta_values(r); ta_used[r] = v; return v
    df = run_cell(cell, specs, R=R, seed=SEED, rep_range=(a, b), ta_values=ta_vals)
    df.to_parquet(f"{OUT}/dr_late_part_{k}_of_{K}.parquet")
    pd.DataFrame([(r, y, v[j]) for r, v in ta_used.items() for j, y in enumerate(YS)], columns=["rep", "y", "ta_used"]) \
        .to_parquet(f"{OUT}/dr_late_ta_part_{k}_of_{K}.parquet")
    print("done", round(time.time() - t0), "s", flush=True)


def merge_dr(K):
    df = pd.concat([pd.read_parquet(f"{OUT}/dr_late_part_{k}_of_{K}.parquet") for k in range(K)], ignore_index=True)
    df.to_parquet(f"{OUT}/dr_late.parquet")
    pd.concat([pd.read_parquet(f"{OUT}/dr_late_ta_part_{k}_of_{K}.parquet") for k in range(K)], ignore_index=True) \
        .to_parquet(f"{OUT}/dr_late_ta.parquet")
    print("merged", df.shape)


def merge_nr(scenario, K):
    df = pd.concat([pd.read_parquet(f"{OUT}/nr_{scenario}_part_{k}_of_{K}.parquet") for k in range(K)], ignore_index=True)
    df.to_parquet(f"{OUT}/nr_{scenario}.parquet")
    ta = pd.concat([pd.read_parquet(f"{OUT}/nr_{scenario}_ta_part_{k}_of_{K}.parquet") for k in range(K)], ignore_index=True)
    ta.to_parquet(f"{OUT}/nr_{scenario}_ta.parquet")
    print("merged", df.shape)


def sae(R):
    os.makedirs(OUT, exist_ok=True)
    pop, des, mats, cats = setup()
    x = pop.lag_expenses.clip(lower=0).to_numpy(float)
    ys = ["y_gov_grants", "y_contrib", "y_fundraising_exp", "y_volunteers_cap10k", "y_bin_has_gov_grants",
          "y_bin_has_volunteers", "y_bin_lobbying", "y_expenses"]
    gpath = f"{OUT}/fold_groups.npy"
    if not os.path.exists(gpath):
        np.save(gpath, fold_groups(pop, load(f"{DATA}/emb/ne_v3", "lsa256_full")[0]))
    cell = Cell(pop[ys].to_numpy(float), ys, des, mats, x, groups=np.load(gpath), cat_masks=cats)
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
    elif c == "merge_dr":
        merge_dr(int(sys.argv[2]))
    elif c == "dr":
        dr(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    elif c == "sae":
        sae(int(sys.argv[2]))
