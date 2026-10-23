"""Build the benchmark population: frame at t-1 (TY2022), survey targets at t (TY2023).

Realistic timing:
  frame variables  = BMF snapshot 2023-08-16 (codes) + TY2022 return (lagged size, text)
  survey targets   = TY2023 return
  response timing  = SubmittedOn of the TY2023 return (used for the late-filer nonresponse study)
"""
import argparse, os, re

import numpy as np
import pandas as pd

NTEE_MAJOR10 = {  # NCCS 10-group collapse of NTEE letters
    **{k: "ART" for k in "A"}, **{k: "EDU" for k in "B"}, **{k: "ENV" for k in "CD"},
    **{k: "HEA" for k in "EFGH"}, **{k: "HMS" for k in "IJKLMNOP"}, **{k: "IFA" for k in "Q"},
    **{k: "PSB" for k in "RSTUVW"}, **{k: "REL" for k in "X"}, **{k: "MMB" for k in "Y"},
    **{k: "UNK" for k in "Z"},
}

NUM = lambda s: pd.to_numeric(s, errors="coerce")


def clean_text(*parts):
    seen, out = set(), []
    for p in parts:
        if isinstance(p, str):
            p = re.sub(r"\s+", " ", p).strip()
            if p and p.upper() not in seen and p.upper() not in ("NONE", "N/A", "NA", "SEE SCHEDULE O"):
                seen.add(p.upper())
                out.append(p)
    return " | ".join(out)


def build(prev, curr, bmf):
    p, c = prev.copy(), curr.copy()
    U = p.merge(c, on="ein", suffixes=("_t1", "_t"), how="inner")
    b = bmf.rename(columns=str.lower)[["ein", "ntee_cd", "subsection", "foundation", "ruling", "asset_cd",
                                        "income_cd", "affiliation", "classification", "activity"]]
    U = U.merge(b, on="ein", how="left")
    out = pd.DataFrame({"ein": U.ein})
    # ---------------- frame (design-time) variables ----------------
    out["state"] = U["legal_domicile_state_t1"].fillna(U["state_addr_t1"])
    out["zip5"] = U["zip_t1"].str[:5]
    out["ntee_letter"] = U.ntee_cd.str[0].where(U.ntee_cd.str[0].str.match(r"[A-Z]", na=False))
    out["ntee_major10"] = out.ntee_letter.map(NTEE_MAJOR10).fillna("MISSING")
    out["ntee_code"] = U.ntee_cd
    out["subsection"] = U.subsection.fillna(U.subsection_txt_t1.astype(str).str.zfill(2))
    out["is_c3"] = (out.subsection == "03").astype(int)
    out["ruling_year"] = NUM(U.ruling.str[:4])
    out["formation_year"] = U.formation_year_t1
    out["bmf_asset_cd"] = NUM(U.asset_cd)
    out["bmf_income_cd"] = NUM(U.income_cd)
    out["has_website"] = U.website_t1.notna().astype(int)
    out["website"] = U.website_t1
    for k_out, k_in in [("lag_revenue", "revenue_cy_t1"), ("lag_expenses", "expenses_cy_t1"),
                        ("lag_assets", "assets_eoy_t1"), ("lag_employees", "employees_p1_t1"),
                        ("lag_contrib", "contrib_cy_t1"), ("lag_prog_rev", "prog_rev_cy_t1"),
                        ("lag_salaries", "salaries_cy_t1")]:
        out[k_out] = U[k_in].fillna(0.0)
    # survey-only variables at t-1: NOT on a realistic frame; used only as positive controls
    for k in ["volunteers", "gov_grants", "officer_comp", "func_exp_fundraising"]:
        out[f"oracle_lag_{k}"] = U[f"{k}_t1"].fillna(0.0)
    # ---------------- text (t-1) ----------------
    out["text_mission"] = [clean_text(a, b) for a, b in zip(U.mission_p1_t1, U.mission_p3_t1)]
    out["text_programs"] = U.program_desc_t1.fillna("")
    out["text_full"] = [clean_text(a, b, c) for a, b, c in zip(U.mission_p1_t1, U.mission_p3_t1, U.program_desc_t1)]
    out["text_mission_t"] = [clean_text(a, b) for a, b in zip(U.mission_p1_t, U.mission_p3_t)]  # leakage control only
    # ---------------- survey targets (t) ----------------
    y = {
        "y_expenses": "expenses_cy_t", "y_revenue": "revenue_cy_t", "y_employees": "employees_p1_t",
        "y_volunteers": "volunteers_t", "y_gov_grants": "gov_grants_t", "y_prog_rev": "prog_rev_cy_t",
        "y_contrib": "contrib_cy_t", "y_salaries": "salaries_cy_t", "y_fundraising_exp": "func_exp_fundraising_t",
        "y_officer_comp": "officer_comp_t", "y_grants_paid": "grants_paid_cy_t",
    }
    for k_out, k_in in y.items():
        out[k_out] = U[k_in].fillna(0.0)
    for k in ["lobbying", "foreign_activities", "grants_to_indiv_us", "fundraising_activities", "gaming"]:
        out[f"y_bin_{k}"] = U[f"{k}_t"].fillna(False).astype(int)
    out["y_bin_has_gov_grants"] = (out.y_gov_grants > 0).astype(int)
    out["y_bin_has_volunteers"] = (out.y_volunteers > 0).astype(int)
    out["y_bin_has_employees"] = (out.y_employees > 0).astype(int)
    # ---------------- response timing (t return) ----------------
    fye = pd.to_datetime(U.tax_period_end_t, errors="coerce")
    sub = pd.to_datetime(U.SubmittedOn_t, errors="coerce")
    out["tax_period_end_t"] = fye
    out["submitted_on_t"] = sub
    out["days_to_file_t"] = (sub - fye).dt.days
    out["amended_t"] = (U.DocStatus_t == "AmendedReturn").astype(int)
    return out


def apply_edits(pop: pd.DataFrame):
    """Documented edit rules, as a statistical office would apply before estimation. Each rule is
    deliberately narrow (gross errors only); influential-but-plausible values are kept.
    Returns (edited population, edit log DataFrame)."""
    p = pop.copy()
    log = []
    # E1 employees: >10x prior year (and >100) while salaries cannot support even $1k per employee
    bad = (p.y_employees > 10 * p.lag_employees.clip(lower=10)) & (p.y_employees > 100) & \
          (p.y_salaries < 1000 * p.y_employees)
    for i in np.flatnonzero(bad):
        log.append(("E1_employees", p.ein.iat[i], p.y_employees.iat[i], p.lag_employees.iat[i]))
    p.loc[bad, "y_employees"] = p.loc[bad, "lag_employees"]
    # E2 negative amounts that must be non-negative (expenses, grants, salaries) -> 0
    for c in ["y_expenses", "y_gov_grants", "y_prog_rev", "y_grants_paid", "y_salaries", "y_fundraising_exp"]:
        neg = p[c] < 0
        for i in np.flatnonzero(neg):
            log.append((f"E2_negative_{c}", p.ein.iat[i], p[c].iat[i], 0.0))
        p.loc[neg, c] = 0.0
    # E3 impossible filing delays (submitted before fiscal year end, or > 3 years late) -> missing
    badd = (p.days_to_file_t < 0) | (p.days_to_file_t > 3 * 365)
    for i in np.flatnonzero(badd):
        log.append(("E3_filing_delay", p.ein.iat[i], p.days_to_file_t.iat[i], np.nan))
    p.loc[badd, "days_to_file_t"] = np.nan
    # derived edited variable: volunteers capped at 10,000 per organisation (separate estimand)
    p["y_volunteers_cap10k"] = p.y_volunteers.clip(upper=10_000)
    return p, pd.DataFrame(log, columns=["rule", "ein", "before", "after"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prev", required=True)
    ap.add_argument("--curr", required=True)
    ap.add_argument("--bmf", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    prev, curr = pd.read_parquet(a.prev), pd.read_parquet(a.curr)
    bmf = pd.read_csv(a.bmf, dtype=str, low_memory=False)
    pop, edits = apply_edits(build(prev, curr, bmf))
    edits.to_csv(a.out.replace(".parquet", "_editlog.csv"), index=False)
    print("edits:", edits.rule.value_counts().to_dict())
    births = len(set(curr.ein) - set(prev.ein))
    deaths = len(set(prev.ein) - set(curr.ein))
    print(f"frame(t-1)={len(prev)}  t={len(curr)}  continuing={len(pop)}  deaths={deaths}  births={births}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pop.to_parquet(a.out, index=False)
    print("wrote", a.out, pop.shape)


if __name__ == "__main__":
    main()
