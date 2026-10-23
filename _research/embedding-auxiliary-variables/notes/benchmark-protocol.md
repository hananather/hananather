# Benchmark protocol v1.0: text embeddings as auxiliary variables in an annual non-profit survey

**Status:** final protocol, 2026-10-23, for sign-off. It replaces drafts A (production realism), B (pre-registration and red team) and C (methods) in `notes/design/`, and the two judge reviews of those drafts. The text is binding from **FREEZE-1** (§9.7). Until then it is the proposal.

**Claim under test:** "Text embeddings of business descriptions, websites or area profiles can serve as covariates for stratification, nonresponse adjustment, calibration and small area estimation. This is variance reduction, not productivity."

**Question the benchmark answers:** in an NSO's annual survey of non-profit organizations, for which combinations of register content, target variable and use does adding text available at frame time change a production decision? And by how much, in sample size, cost and CV band?

**Conventions.**
- **[USER]** = needs the user (§14).
- **[VERIFY]** = a fact to confirm against a primary source before relying on it.
- **[CODE]** = code still to write (§13).
- **[NEW]** = in none of the drafts.

Estimator, matrix and spec-kind names match `src/` wherever the object already exists. Pilot names are mapped in §4.10.

---

## 0. Summary

### 0.1 The design in one screen

| Element | Choice |
|---|---|
| Real-world scenario | Annual survey of NPOs, reference year t = TY2023. Frame = register at t−1 as known on the design date **D = 2024-06-30**. Collection runs from fiscal-year end (FYE) to FYE + 300 days. Estimation with design weights, calibrated, with imputation for take-all units. |
| Testbed | IRS Form 990 e-file (GivingTuesday data lake). The t−1 return plus the BMF play the register; the t return plays the questionnaire. The full t file is the truth. |
| Populations | **New England (NE) = development and proof of concept** (N = 20,626). **8 held-out US Census divisions = confirmation** (14.7k–58.2k units each). Replications: CRA T3010 (Canada) and UK CCEW (multimodal PDFs). |
| Register tiers | F0 codes only · F1 summary financials · **F1p = F1 + payroll (primary)** · F1c stylized hybrid (drafts A/C's "REG-S+") · F2 = full prior return (charities) · PANEL = F1p plus last cycle's survey values for last cycle's sample [NEW]. |
| Primary pair P1 (efficacy) | `MCAL_bmfmax@F1p+E*` vs `MCAL_bmfmax@F1p`: the same learner, folds, λ grid and BMF-max codes, with and without text. |
| Primary pair P2 (production) | `FROZEN_calm_bmfmax@F1p+E*` vs `GREG_size_cat@F1p`. The text system is frozen heads trained on the **previous wave's respondents**, entered as extra calibration columns, giving one bounded weight set for all y. It is compared with today's production GREG. |
| Headline y set H | 10 y not on the F1p register: government grants, contributions, program revenue, fundraising expenses, officer compensation, volunteers (capped at 10k), has volunteers, grants to individuals, foreign activities, lobbying. |
| Metric | GM-RE = exp(mean over y in H of log(MSE_comparator / MSE_text)). Reported with bias, RRMSE, coverage, variance-estimator bias and weight diagnostics. |
| Pass rule | **MET:** point ≥ 1.10 and Holm-adjusted one-sided 95% lower bound ≥ 1.05. **EXCLUDED:** upper bound < 1.10. A second reading, "any gain?", asks whether the lower bound is > 1.00. |
| Controls | One-sided negative controls (random, permuted, within-stratum permuted and constant embeddings; three placebo y). A planted-signal family, which gives ρ²_min. Leakage controls (same-year text, masked numbers, late frame). Overfitting demos. Ceiling consistency. |
| Attribution | A 2³ factorial {linear vs ratio-scale model calibration} × {NTEE10 vs BMF-max codes} × {no text vs text}, split into Shapley shares. This separates "it's ML", "it's finer codes" and "it's text". |
| Other uses | Stratification and allocation (2 × 2 design × estimator); nonresponse (late filers and simulated mechanisms, take-all imputation); SAE (Fay–Herriot with frozen covariates; planned and ZIP3 domains); frozen previous-wave deployment; panel overlap. |
| Embedding ladder | Keywords → LSA → GloVe → MiniLM → Qwen3 [USER] → Cohere v4/v5 text [USER] → Cohere image/fused on CCEW PDFs or Common Crawl website screenshots [USER]. Compared against structured extraction of the numbers on the page (L7-ext). |
| Before any MC | Fix three verified estimator defects (§0.3). Rebuild the population: BMF-max fields, design-date availability (`avail_D`), valid URL flag, previous-wave codes recomputed from TY2021. Rerun the pilot as a regression test. |

### 0.2 Decisions and conflict resolutions

| # | Issue | Drafts said | Decision | Why |
|---|---|---|---|---|
| 1 | Base document | Judges split: C won one review, A the other | Methods layer from C; scenario and production layer from A; confirmation and red-team machinery from B | Both judges prescribed this merge |
| 2 | Primary register | A/C: REG-S+. B: F1. | **F1p** primary (summary financials + PD7A-type payroll). F1 is a pre-specified realism check. REG-S+ is renamed **F1c** and labelled a stylized hybrid (sensitivity only). F2 = charities. | REG-S+ matches no single real register: any source with lagged contributions and program revenue also has grants and compensation. F1 credits text with a salaries gain that payroll data remove. F1p ≈ Business Register + payroll. |
| 3 | Headline y | A/C: 6 y under REG-S+. B: D4 and S5. | H = H_D (5 detail amounts) ∪ H_S (5 survey-type items), fixed by register status under F1p. GM over H is primary; H_D and H_S are reported separately. `has_gov_grants` is secondary, because it is derived from gov_grants. | Consistent with the tier and multipurpose; avoids double-counting government funding |
| 4 | P1 comparator | A: NTEE10 twin. C: partial codes twin. B: BMF-max twin. | **BMF-max twin**, penalized with C's categorical rule. A's NTEE10 twin is reported as P1′. | Strongest fair text-free comparator. It addresses the skeptic's "it's just codes" objection. |
| 5 | P2 text system | A: MCALm (cross-fitted). C: FROZEN_calm. | **FROZEN_calm**, trained on previous-wave respondents (B). MCALm is P2′. | Deployable, exactly design-unbiased in difference form, and needs one extra calibration column in production software |
| 6 | Confirmation | A/C: NE. B: held-out divisions. | NE = development plus a labelled proof-of-concept verdict. **8 non-NE divisions = confirmatory.** | NE was already used by the pilot and for every threshold and ceiling |
| 7 | Pass threshold | A/C: lower bound ≥ 1.10 (no power at a 1.12 ceiling). B: point ≥ 1.10 and lower bound > 1.00 (lenient). | MET = point ≥ 1.10 **and** Holm lower bound ≥ 1.05. EXCLUDED = upper bound < 1.10. Plus the "any gain" reading. | Discriminates a true 1.10 from a true 1.05 at the confirmatory precision (§9.1) |
| 8 | Generalization inference | B: t₇ across divisions. C: random-effects PI. | t₇ on the 8 held-out divisions is primary. The random-effects prediction interval, on non-overlapping populations only, is a secondary production condition. | C pooled overlapping populations, so its τ was too small. Spatial correlation between divisions is listed as a limitation. |
| 9 | Replicates | A/C: "raise R to 5,000 once if inconclusive" | Removed. R is fixed before each freeze from a Monte Carlo SE target. | No optional stopping |
| 10 | B* (best text-free) | B: chosen ex post per y | Chosen on NE development, then frozen for the divisions. Secondary only. | Ex-post B* has the winner's curse and no valid CI |
| 11 | Coverage gate | A/B: absolute ≥ 93% | Relative: gap to comparator ≥ −2 pp, gap to HT ≥ −5 pp. Plus variance relative bias within ±15% and the studentized-error diagnosis. | HT itself under-covers rare binaries and heavy tails |
| 12 | Negative-control gate | All drafts: two-sided [0.97, 1.03] | **One-sided:** upper bound ≤ 1.03. RE < 1 is reported as the cost of dimension. | Useless dimensions legitimately cost efficiency (population ceiling 0.82 for random vectors on gov_grants) |
| 13 | Known-answer y | B: one exp-link level | C's planted family. ρ² is defined on the anticipated-variance (AV) scale with the same-class learner; 4 forms × 6 levels; gates at ρ² = 0 and 0.2; gives ρ²_min(n). | B's version mixed link mismatch with fitting loss; one level gives no power curve |
| 14 | Choice of E* | B: LSA fixed now. A/C: chosen on the previous wave, LSA barred as fallback. | Selection rule on NE's previous wave (§6.5), frozen at FREEZE-1. Fallback MiniLM T2. **No Cohere rung can become primary.** LSA is eligible if the rule picks it. | Confirmation is held out, so the A/C contamination concern is met by the divisions |
| 15 | Take-all nonresponse | B: take-all units respond. A: two scenarios. | TA-follow-up (90% MCAR) and TA-late (real) are both primary in Stage 3. Status-quo imputation is **the unit's own last-cycle value** (TA-hist-own) [NEW]. | Only 12% of take-all units file within 300 days. Certainty units always have a last-cycle value. |
| 16 | BMF snapshot | B: Dec 2023. Pilot: Aug 2023. | 2023-08-16 is primary: already built, ≤ D, equality test 0.48%. Dec 2023 is a sensitivity. | Already audited. Codes are near-static. |
| 17 | Population | A: deaths and EZ switchers as y = 0. B: P2. | Primary P1 = continuing units with `avail_D`. Survivorship sensitivity uses B's P2: EZ switchers keep their EZ values, true non-filers get y = 0. Births are reported as undercoverage, plus an exploratory births arm. | EZ switchers are alive and in scope |
| 18 | Design labels | A, B and C label differently | C's labels: D2a = score strata, D2b = text-informed allocation, D2c = k-means | — |
| 19 | CV loss | Pilot: weighted SSE on the ratio scale | C's cross-validated AV loss on the y scale, with a λ-stability check and a pre-specified fallback. λ is chosen on the whole sample, with a nested-CV check. | Targets the quantity the variance depends on. The fallback guards against domination by a few large units. |
| 20 | Common random numbers | A/B/C: "shared rng" defect | Stale. `run_cell` already gives each (seed, rep, estimator) its own stream. **The real fix is folds shared across arms** (today they are seeded by estimator name). | Verified in `mc.run_cell` |
| 21 | MCAL variance | A described it as residuals y − m̂ | Code is correct: final calibration residuals on [X_base, m̂]. Keep it. | Verified in `mc._estimate` |
| 22 | Panel overlap | Absent | **PANEL** tier: last cycle's survey y is known for last cycle's respondents [NEW] | The real source of "oracle lags". The lagged indicator removes the binaries' text gain (note 06 §4.2). |
| 23 | Website text | Blocked (no crawl) | Archived Common Crawl captures dated ≤ D, rendered offline [NEW; USER allowlist] | Correct timing, no live crawling |
| 24 | Lag–y correlation 0.99 | Noted only (B, RT-13) | Concept-mismatch sensitivity: register lags degraded toward realistic register-vs-survey agreement [NEW] | Quantifies a bias that favours the conventional baseline |
| 25 | Pilot regression rerun | Judges: rerun before freezing | The Stage 0c rerun **updates predictions and R, never thresholds**. Thresholds are set on cost grounds here (§9.8). | Prevents fitting the bar to the data |

### 0.3 Where the evidence stands (all NE, all exploratory)

**Three estimator defects make the pilot magnitudes uninterpretable.** All three were verified in `src/mc.py`; the first was reproduced independently.
1. `_ridge_fit` adds λ·n to a Gram matrix whose diagonal is Σw. With ratio weights d·x̃ (Σw ≈ 7.7 × 10¹⁰) the penalty does nothing: ‖B̂‖ is identical for every λ in `LAMS`. So every pilot `MCAL_ratio`, frozen head and SAE model was unpenalized weighted least squares on about 60 columns.
2. The `ridge` kind scores λ in CV on one penalty scale and applies a penalty about 26 times stronger. Its λ-CV also ignores grouped folds.
3. Binary y go through the ratio-scale learner in `MCAL_*` specs. In pilot v1, every binary shows RE ≈ 0.99–1.02, against population ceilings of 1.14–1.17.

**Pilot v1 before the fixes** (R = 500; frame ≈ F1, all lags assumed available; LSA on mission + programs). Each value is the ratio of the two REs against `GREG_size_cat`, which equals MSE(NTEE10 twin)/MSE(text); no paired SE:

| y | gov_grants | contrib | prog_rev | fundraising | officer_comp | salaries | volunteers_cap | binaries |
|---|---|---|---|---|---|---|---|---|
| RE | 1.15 | 1.17 | 1.12 | 1.10 | 1.03 | 1.14 | 0.97 | ≈ 1.00 |

- Random vectors give 0.92–1.03. Coverage of the amounts is 87–93% for the GREG, twin and text arms alike; HT reaches 92–96%.
- No cross-fitting (`:nocf`) gives 74–82% coverage on the amounts.

**Population ceilings with an oracle λ, which are optimistic** (notes 06 and A App. B):
- Text adds nothing on size variables (0.93–1.01).
- It adds 1.04–1.20 on composition variables and 1.03–1.17 on activity indicators against the pilot frame.
- It **adds ≤ 1.017 for every y once that y's own lag is on the frame**, while the lag alone is worth 2.4–14.6×.
- Text is not merely the industry code. Replacing it by the NTEE-3 code centroid removes the gain, while the within-code residual keeps most of it.
- Fine codes explain about 20% of the text log-gain. Has_gov_grants is the exception at 51%.

**Pre-registered prediction (refined at FREEZE-1 after the Stage 0c rerun):**

| Pair | Prediction | Derivation | Likely verdict |
|---|---|---|---|
| P1 | GM-RE ≈ **1.04** (80% interval 1.00–1.08) | Ceiling ≈ 1.095, minus about 0.02 for oracle λ, minus about 20–30% of the log-gain absorbed by BMF-max codes, times a fitting factor of about 0.97 | "Any gain" probably YES; production bar EXCLUDED or OPEN |
| P2 | ≈ **1.05** (0.98–1.12) | Ratio-scale ML about 1.03, text about 1.04, frozen and single-weight retention about 0.75–0.8 of the log-gain | OPEN |

The skeptic predicts 1.00–1.05 for both. Niche uses are plausible: weak registers, NTEE-missing domains, text-coded NTEE, and births.

---

## 1. Research questions and estimands

### 1.1 Research questions

| RQ | Question | Answered by |
|---|---|---|
| RQ0 Validity | Are text-assisted estimators design-valid at realistic n? Bias negligible relative to RMSE; coverage close to the comparator and to HT; variance estimator within ±15%. | Gates G1–G9 (§9.2) |
| RQ1 Signal | What is the population partial R² of t−1 text for each y, given each register tier and given the BMF-max codes? | Stage 1 same-class ceilings |
| RQ2 Realization | What share of that ceiling does each estimator family realize at sample size n? | φ(n) = RE_MC / RE_ceiling; planted family |
| RQ3 Attribution | How much of a gain is text, how much is learner and scale, how much is code granularity, and how much is the embedding model? | 2³ factorial; Φ; ladder |
| RQ4 Efficacy and production | Does text beat its twin (P1)? Does a deployable single-weight frozen text system beat production (P2)? | P1, P2; confirmed on held-out divisions |
| RQ5 Placement | Where does text help most: design, estimation, nonresponse or domains? Are those uses additive or substitutive? | §7; interaction M8 |
| RQ6 Transport | Do frozen previous-wave models keep the gain? Does the verdict replicate across populations, countries and modalities? | §7.4; divisions; T3010; CCEW |
| RQ7 Value | What does a gain buy in sample size, cost, burden and CV band? Under which governance constraints? | §8.7, §12 |

### 1.2 Estimands

All estimands are finite-population quantities of the TY2023 population U (§2.6). Truth is the full TY2023 file.

| Estimand | Definition | Role |
|---|---|---|
| Total | t_y = Σ_U y_k | **Primary** |
| Mean | t_y / N, with N known from the frame. Relative metrics are identical to the total's, so no separate tables. Under nonresponse, the Hájek mean t̂_y/N̂ is also reported. | — |
| Proportion | p_b = t_b / N for binary b | Primary for binaries |
| Ratio of totals | R₁ = t_gov_grants / t_revenue; R₂ = t_fundraising / t_contrib. Linearized: z_k = (a_k − R̂ b_k)/t̂_b. | Secondary; a multipurpose coherence check, because per-y weights can break ratios |
| Domain total / proportion | Planned: state × NTEE10 (66 in NE; 44–99 per division). Unplanned: ZIP3 (about 60 in NE). | SAE (§7.3) |
| Not estimated | Year-on-year change | 84.5% of missions are identical t−1 → t. Change appears only in the memorization probe. |

### 1.3 Target variables by register status

Class is decided by what a real frame would carry. It fixes the realistic comparator.
- **R** = register-type.
- **D** = detail admin item.
- **S** = survey-type.
- **H** = heavy-tail stress.
- **P** = placebo.
- **K** = planted.

| y (column) | Class | On the frame in tier | Set | TA share of total | Zero share |
|---|---|---|---|---|---|
| y_revenue, y_expenses | R | F1, F1p, F1c, F2 | Negative control (expected RE ≈ 1) | 0.68 / — | 0.01 |
| y_employees | R | same | Negative control | — | 0.37 |
| y_salaries | R under F1p; D under F1 | F1p+ | Tier check (H7) | 0.59 | 0.29 |
| **y_gov_grants** | D | F2 | **H_D** | 0.38 | 0.69 |
| **y_contrib** | D | F1c, F2 | **H_D** | 0.56 | 0.19 |
| **y_prog_rev** | D | F1c, F2 | **H_D** | 0.71 | 0.31 |
| **y_fundraising_exp** | D | F2 | **H_D** | 0.43 | 0.61 |
| **y_officer_comp** | D | F2 | **H_D** | 0.23 | 0.51 |
| **y_volunteers_cap10k** | S | PANEL only | **H_S** | 0.09 | 0.41 |
| **y_bin_has_volunteers** | S | PANEL only | **H_S** | p = 0.587 | — |
| **y_bin_grants_to_indiv_us** | S | PANEL only | **H_S** | p = 0.121 | — |
| **y_bin_foreign_activities** | S | PANEL only | **H_S** | p = 0.043 | — |
| **y_bin_lobbying** | S | PANEL only | **H_S** | p = 0.058 | — |
| y_bin_has_gov_grants | D (derived) | F2 | Secondary | p = 0.311 | — |
| y_bin_fundraising_activities, y_bin_gaming, y_bin_has_employees | S / derived | PANEL | Secondary | 0.21 / 0.027 / — | — |
| y_volunteers (raw), y_grants_paid | H | PANEL / F2 | Stress only; never in aggregates | 0.47 / — | 0.41 / 0.70 |
| PLC_perm, PLC_noise, PLC_bern | P | — | Gate G3 (§5) | — | — |
| y_plant(form, ρ²) | K | — | Gate G6 and power (§5.2) | — | — |

**Headline set H (pre-registered, F1p):** H_D ∪ H_S, 10 y.

**All-y set A:** every R, D and S y, plus the secondaries. It is used for multipurpose damage. H-, P- and K-class y are excluded.

**Tier-specific headline sets:**
- F1: H ∪ {salaries}.
- F1c: H minus {contrib, prog_rev}.
- F2: H_S only.
- PANEL: H, with the lag known only for panel units.

**Facts that bound every result:**
- Text acts only on the take-some share. That is 62% of gov_grants but 32% of revenue.
- Take-all units (top 1%, 206 in NE) hold 68% of revenue.

**Register-status sign-off.** The class of each y must be checked against the T3010, T1044, T2/GIFI and PD7A line lists before FREEZE-2 [USER U6].
- If T1044 carries grants and donations by source [VERIFY], gov_grants and contrib are register-type for T1044 filers. The F1p headline then applies to non-charity NPOs below the T1044 thresholds and to items T1044 lacks.
- Whether volunteers and the activity checkboxes are on T3010 is unverified [VERIFY]. If they are not, H_S is survey-only for every Canadian charity, which makes it the most robust use case.

---

## 2. Real-world scenario specification

### 2.1 The survey being simulated

**Annual Survey of Non-Profit Organizations, reference year TY2023.**
- **Purpose:** the NPO satellite account and sector tables. About 15 published variables: totals and shares, nationally, by state (or province) and by NTEE10.
- **Frame:** the register as known at D = 2024-06-30, with reference year t−1.
- **Collection:** from each unit's FYE to FYE + 300 days.
- **Estimation date:** about 2025-09.
- **Estimation:** design weights; calibration to register totals; imputation for take-all nonrespondents; weighting-class adjustment for take-some nonrespondents.

The Canadian analogue is the StatCan Business Register with CRA Business Number data:
- NAICS and the BN "major business activity" text;
- T2/GIFI and T1044 summaries and PD7A payroll;
- for registered charities, the prior-year T3010 including program descriptions.

The survey collects what the register lacks.

### 2.2 Information sets and timeline

Every matrix, model and tuning choice carries a `stage_available` tag ∈ {D, C, E, TRUTH}. `run_cell` asserts the tag [CODE].

| Information | Testbed source | D (design) | C (collection) | E (estimation) | Analogue |
|---|---|---|---|---|---|
| Register codes | BMF 2023-08-16. Sensitivity: 2023-12-12 (to download; reachable). | ✔ | ✔ | ✔ (+ population totals) | BR codes |
| BMF-max fields | FOUNDATION, AFFILIATION, CLASSIFICATION, ACTIVITY, DEDUCTIBILITY, ORGANIZATION, ACCT_PD, FILING_REQ_CD, GROUP, full NTEE_CD, ASSET/INCOME/REVENUE_AMT, NAME | ✔ | ✔ | ✔ | BR legal form and status fields |
| Lagged summary financials and payroll | TY2022 return, **only if its first original submission is ≤ D** (§2.3) | ✔ | ✔ | ✔ | T2/GIFI, T1044, PD7A |
| Lagged detail items (F2) | TY2022 Part VIII/IX items | ✔ (F2 only) | ✔ | ✔ | Prior-year T3010 |
| Last cycle's survey y (PANEL) | TY2022 y for **previous-wave sample respondents only** | ✔ | ✔ | ✔ | Rotation-overlap units |
| Text at frame time | Mission and programs from the latest return ≤ D (TY2022, else TY2021), else name | ✔ | ✔ | ✔ | BN text; T3010 programs |
| Website URL | `WebsiteAddressTxt`, rebuilt to a valid-URL flag | ✔ | ✔ | ✔ | BR contact data |
| Website content | Common Crawl captures with timestamp ≤ D [USER U7] | ✔ | ✔ | ✔ | ISTAT/OBEC web intelligence |
| Previous-wave model heads | Trained on the TY2021 → TY2022 sample's respondents | ✔ | | ✔ | Last cycle's model |
| y_t for respondents; response flag; filing date | TY2023 return; `SubmittedOn` | | ✔ | ✔ | Survey responses and paradata |
| y_t for nonrespondents and non-sampled units | TY2023 return | | | | **TRUTH only** |
| Same-year text; BMF snapshots after D | TY2023 mission; BMF 2026 | | | | TRUTH; leakage controls L1 and L3 only |

**Rules.**
- **R1, complete auxiliary information.** A calibration auxiliary must be D-measurable for every unit in U_TS.
- **R2, no population y.** No fitted object touches y outside the sample. Test T-M3.
- **R3, no current-wave researcher degrees of freedom.** Embedding rung, field, K, feature map, learner, λ policy and CV loss are fixed by this protocol or by the previous-wave rule, and frozen.

### 2.3 Availability at the design date (`avail_D`)

For each unit k, at D:
1. **If the first original TY2022 submission is ≤ D**, set `avail_D = 1`. Lags and text come from that return. The first original submission is the earliest `SubmittedOn` with `DocStatus ≠ AmendedReturn` in `idx_2022.csv`. If the parsed (latest) XML is a later amendment, download the original XML (reachable; small).
2. **Else, if a TY2021 return was submitted ≤ D**, set `avail_D = 0` and `lag_source = BMF`:
   - lag amounts = BMF REVENUE_AMT / INCOME_AMT / ASSET_AMT, with lag_expenses := REVENUE_AMT as proxy;
   - lag_employees and lag_salaries = 0, with an indicator;
   - text from TY2021 (`text_source = TY2021`).
3. **Else**, BMF amounts as in 2, and name-only text (`text_source = NAME`, `text_missing = 1`).

Every matrix carries `avail_D`, `text_missing`, `lag_source` dummies and their interactions with log size.

**Observed (national continuing units, lower bound):** 94.4% of units and 87.7% of revenue were available at 2024-06-30. Large units file late: 96.0% of units ≤ $200k against 90.2% of units > $25M.

### 2.4 Register tiers (each with its own design)

| Tier | Contents beyond the previous tier | Design size measure s_k | Real NSO case | Role |
|---|---|---|---|---|
| **F0** | BMF-max codes, state, BMF size codes and amounts | INCOME_CD class midpoint (or INCOME_AMT) | New registrants; units without tax accounts; 990-N filers; non-charity NPOs on a BR-only frame | Weak register (S2) |
| **F1** | + lagged revenue, expenses, assets, employees (`avail_D` rule) | lag_expenses if `avail_D`, else BMF REVENUE_AMT | BR with T2/GIFI or T1044 summary and PD7A headcount | Realism check, reported next to the primary |
| **F1p** | + lagged salaries | as F1 | BR + PD7A payroll: the generic business-register frame | **Primary** |
| F1c | + lagged contrib and prog_rev | as F1 | No single real register (stylized hybrid, = drafts A/C's REG-S+) | Sensitivity only |
| **F2** | + lagged gov_grants, fundraising_exp, officer_comp, grants_paid, has_gov_grants | as F1 | Registered charities with the prior-year T3010 at the NSO (line content [VERIFY]) | Charity frame; positive control for "lag beats text" |
| **PANEL** | F1p + `in_prev_resp` + y_{t−1}·1{k ∈ r′} for every y in H (r′ = previous-wave sample respondents) | as F1 | Annual survey with rotation overlap: last cycle's sample has its own survey values | [NEW] §7.5 |

**The design follows the tier.**
- F0 uses D1-W: size classes and take-all from INCOME_AMT, never from lagged expenses. This fixes B's RT-9 finding.
- F1, F1p, F1c and F2 share design D1, so they share samples (common random numbers) and differ only in matrices. One MC cell serves all four.
- PANEL uses the rotation design of §3.2.

### 2.5 Text sources and their real analogues

| Arm | Content | Coverage (NE) | NSO analogue | Stage |
|---|---|---|---|---|
| T0 | BMF NAME + SORT_NAME (**not yet in the population**; add it) | 100%, births included | BR legal and operating names | D |
| T1s | First 12 words of T1. A crude stand-in for BN "major business activity" text [approximation]. | as T1 | BN registration text (confidential) | D |
| T1 | Mission: Part I l.1 + Part III l.1, deduplicated, boilerplate removed (`text_mission`) | 99.9%; median 19–23 words | T3010 short description | D |
| **T2** | T1 + Part III 4a–4d program descriptions, whitelisted XPaths only (`text_full`) | 99.9%; median 71 words | T3010 Section C program descriptions | D |
| T2m | T2 with digits and "$" amounts masked | same | Leakage ablation | D |
| T1_t | TY2023 mission | — | none | TRUTH (control L1 only) |
| W | Homepage (and "about" page) text from the latest Common Crawl capture ≤ D; offline-rendered screenshot | ≤ 72.5% (valid URL); 51% under $200k, 92% over $25M | Enterprise websites | D [USER U7] |
| PDF | t−1 accounts PDF pages (CCEW) | CCEW only | Financial statements; annual reports | Replication R2 |

**Rule:** never render 990 XML to images. The page would contain y, and no NSO would do it.

### 2.6 Populations, births and missing codes

| Population | Definition | Use |
|---|---|---|
| **P1** (primary) | Units with a full 990 for TY2022 and TY2023; one unit per EIN; group returns excluded; edits E1–E3; `avail_D` frame | All primary cells |
| P2 (survivorship sensitivity) | Units whose latest return at D is a full TY2022 990. Targets:<br>• full 990 at t → its values;<br>• 990-EZ at t → the EZ values where the item exists (revenue, expenses, assets, contributions, program revenue, salaries, grants paid), other y on P1 only;<br>• nothing filed → y = 0, flagged dead.<br>Needs TY2023 990-EZ XML for about 1–2k NE EINs (reachable). | Stage 2K |
| Births | Full TY2023 filers with no TY2022 full return. NE: 1,414 units = 6.4% of units, 0.3% of expenses, 0.9% of gov_grants. | Reported as a fixed undercoverage share. Exploratory births arm (§7.6). |
| Previous wave P1′ | Frame TY2021 → y TY2022. NE: N = 20,317, 19,478 units shared with P1. **BMF income/asset codes and amounts must be recomputed from TY2021 returns:** the 2023 BMF holds TY2022 amounts, which are this wave's y (test B5). | E* selection, frozen heads, PANEL |
| Divisions (confirmatory) | P1 per non-NE Census division (8 populations, 285.6k units) | P1 and P2 confirmation |

**Other population facts:**
- **Missing NTEE:** 31.1% of units (38% in the smallest size quintile, 26% in the largest). Status quo keeps MISSING as its own level. D1-imp codes it from text.
- **Reference periods:** FYE in December 47.2%, June 32.8%, September 7.9%. The estimand is the IRS tax year. A December-FYE-only sensitivity is run (test A8).

### 2.7 Nonresponse = late filing

- **Definition:** r_k = 1{TY2023 return submitted ≤ 300 days after FYE}. Response is 60.1% of units but only 18.4% of expenses. By group: 61% of take-some units and **12% of take-all units**.
- **Inverted gradient:** late filing makes big units respond *less*, the opposite of a chased survey (B, RT-12). Results under NR-LATE are therefore always reported with simulated mechanisms, and claims need both to agree.
- **Take-all units** are imputed, never reweighted (§7.2).
- **Previous-wave response** uses the same 300-day rule on TY2022 returns (59.6% in NE). It defines r′ for frozen training and for PANEL.
- **Item nonresponse:** volunteers are blank for 19.7%, currently coded 0. Default estimand: "reported volunteers, blank = 0". Sensitivity: blank treated as item nonresponse, with ratio imputation within the stratum [USER U5 confirms].

### 2.8 Edits, outliers, winsorization (identical for every arm)

- **Edits E1–E3** (`build_population.apply_edits`) come first, with the log kept. E1 copies lag_employees into y_employees for flagged units. Count reported; a no-edit sensitivity is run.
- **Influential but plausible values stay in the truth.** Example: Knights of Columbus (CT) holds 42% of NE volunteers; one take-some unit holds 16.4% of take-some volunteers.
- **Winsorization.** Optional one-sided type-II winsorization for production variants (suffix `-WIN`):
  - HT and GREG apply it to y: y*_k = y_k if y_k ≤ K_h, else y_k/d_k + (1 − 1/d_k)K_h.
  - Model-assisted arms apply it to the residual e_k = y_k − m̂_k.
  - Cutoffs K_{h,y} are Kokic–Bell cutoffs computed on the previous-wave population and frozen. They never come from the current sample.
  - Primary endpoints are unwinsorized. A gain that exists only without winsorization is labelled an outlier artefact (S-WIN).

### 2.9 What is real, simulated and absent

| Real | Simulated | Absent (limitation, §12.4) |
|---|---|---|
| Population, skew, outliers, missing NTEE, text quality, design-date availability, late filing, near-duplicate chapter texts, previous-wave sample information | Sample draws; NR-sim mechanisms; placebo and planted y; concept-mismatch noise; rotation overlap | Survey measurement error and mode effects; real register-vs-survey concept gaps; BN-length text (approximated by T1s); French text (until R1); real PDFs (until R2) |

---

## 3. Designs

### 3.1 Common rules

- Stratified SRSWOR (`survey.StratifiedDesign`).
- Take-all stratum = top 1% of units on the tier's s_k (206 in NE).
- Take-all units enter with weight 1 and are excluded from modelling and calibration: T̂ = Σ_TA y_k + T̂_TS. This is as `mc.Cell` does now.
- Compromise allocation: Neyman on s_k^0.5, minimum n_h = 10, iterative take-all (`frame.production_design`, `alloc_power = 0.5`).
- **Equal-H rule:** a design that adds a stratifying variable is compared with a status-quo design with the same number of take-some strata.
- **2 × 2 rule:** every design result is reported as {design} × {text-free estimator, text estimator}, so a text gain is never counted twice.

### 3.2 Design arms

| Code | Strata / allocation | Question | Comparator | Status |
|---|---|---|---|---|
| D0 | SRSWOR | Scaling of φ and coverage in p/n (n can go low) | — | exists (`survey.srswor`) |
| **D1** | NTEE10 (incl. MISSING) × 3 size classes (cum-√f on log s_k) + take-all; H = 33 + TA | **Status quo** (F1/F1p/F1c/F2) | — | exists |
| D1-W | NTEE10 × 3 INCOME_AMT classes + take-all on INCOME_AMT | Status quo for F0 | — | [CODE] |
| D1-H44 | NTEE10 × 4 size classes + take-all | Equal-H twin of D2a | — | [CODE] (`n_size_classes=4`) |
| D1-imp | D1 with MISSING NTEE replaced by text-coded NTEE (`experiments.impute_ntee` on E*; no y). H = 30 + TA (fewer strata, which is conservative). Accuracy on *truly* missing units is checked against the 2026 BMF codes, used for evaluation only. | Text as frame coder | D1 | [CODE] |
| D2a | NTEE10 × 2 size × 2 classes of a frozen text score + TA; H = 44. Score = previous-wave frozen head's predicted gov-grant ratio, split at the frame median within NTEE10 × size cell. | Prediction stratification | D1-H44 | [CODE] |
| D2b | D1 strata; allocation n_h ∝ N_h Ŝ_{e,h}, averaged over y ∈ H after normalization; min 10; TA kept. Ŝ comes from a frozen previous-wave regression of log abs(y − m̂) on (z, e). | Text at the allocation step only | D2b-twin (same, no text) | [CODE] |
| D2c | k-means (K = 11) on E* in place of NTEE10, × 3 size + TA; H = 33 | Unsupervised "text industry" (expected null) | D1 | [CODE] |
| PANEL-D1 | D1 strata h, each split by membership in O_h = U_h ∩ r′ (previous-wave respondents). Draw n_hO = min(round(0.5 n_h), abs(O_h)) from O_h and the rest from U_h \ O_h. Weights N_hg/n_hg. Exactly design-unbiased **conditional on s′**. | Rotation overlap with known last-cycle y | D1 (same n) | [CODE] (`frame.panel_design`) |

**Allocation feasibility.** The sum of minimum n_h must be ≤ n_TS for every n in the grid (test D7). n = 500 in NE fails under D1, because 33 × 10 = 330 > 294 take-some units, so it is dropped.

### 3.3 Sample sizes

| Population | Designs | n | Note |
|---|---|---|---|
| NE (N = 20,626) | D1 family | **1,000** (primary; n_TS = 794, N_TS = 20,420) | f_TS ≈ 3.9% |
| NE | D1 | Cost curve: baseline at {750, 875, 1,000, 1,250, 1,500, 2,000}; text arms at {650, 750, 875, 1,000} | §8.7 |
| NE | D0 | {250, 500, 1,000, 2,000} | Scaling |
| Divisions | D1 | n_g = round(1,000 × N_g / 20,626), from 711 (East South Central) to 2,822 (South Atlantic) | Same f as NE; Σ n_g/1,000 ≈ 13.9 "NE-equivalents" |

### 3.4 Stratum jumpers

8.4% of take-some units change size class between t−1 and t. 20 take-some units reach take-all size, holding 1.3% of expenses. For every design, the share of MSE from jumpers is reported. D2a creates jumpers by construction.

---

## 4. Estimators and baselines

### 4.1 Notation

- **Sample:** s = s_TA ∪ s_TS. Design weights d_k = N_h/n_h. All fitting uses s_TS only.
- **x̃_k = s_k + 1**, where s_k is the tier size measure (§2.4).
- **Matrices** (all restricted to take-some rows, as in `mc.Cell`):

| Matrix | Columns | Block type | Exists? |
|---|---|---|---|
| `size_cat@T` | Stratum dummies + tier-T lags (raw $) + NTEE10 + state + `avail_D`/`lag_source` dummies | Calibration (exact) | `size_cat` exists; add @T and indicators [CODE] |
| `F@T` | NTEE10, state, BMF income code (categorical); log1p of tier-T lags, `avail_D`, `text_missing` (continuous or 0/1) | Learner features | `ml_frame_ratio` exists; add @T [CODE] |
| `C` (BMF-max) | Full NTEE_CD pooled to codes with ≥ 20 units (others pooled to their letter); NTEE letter (26 + MISSING); subsection; foundation; affiliation; classification; first activity code (pooled at ≥ 20); deductibility; organization; ACCT_PD; filing-requirement code; GROUP ≠ 0; ruling-year bins (≤1970, 71–90, 91–00, 01–10, 11–15, 16+); BMF asset code; valid-URL flag. Pooling never looks at y. | Categorical | [CODE] |
| `Z_K(E)` | First K population principal components of E (`frame.pcs`, standardized); K ∈ {8, 16, 32, 64} | Continuous | exists |
| `E:full` | All D dimensions (dual ridge via the SVD of the n_TS × D sample matrix) | Continuous | [CODE] |
| PANEL columns | `in_prev_resp`, `lag_y_panel_j` = y_{t−1,j}·1{k ∈ r′} for each j in H, and log1p of the amounts | Continuous / 0-1 | [CODE] |

### 4.2 Learners (fixed per y type before any result)

| y type | Learner | Target | Fit weights (normalized to mean 1) | Hyperparameters | Code |
|---|---|---|---|---|---|
| Non-negative amounts and counts | **Ratio ridge** (primary) | y/x̃; prediction × x̃; predicted ratios clipped to the sample's 1st–99th percentile (existing rule, applied to every arm) | d·x̃ | λ (continuous) ∈ 10^{−4, −3.5, …, 2} (13 values); λ_C (categorical) ∈ {1, 5, 20, 100, 500} | `mc._ridge_fit` after fix A1 |
| Zero-inflated amounts (gov_grants, fundraising, officer_comp, prog_rev) | Two-part (secondary): p̂(z) · x̃ · ĝ₊(z) | 1{y > 0} (logit); y/x̃ on y > 0 | d; d·x̃ | as above, for each part | [CODE] |
| Binaries | **Ridge-logistic** (primary); linear probability (secondary) | 0/1 | d | λ grid as above; IRLS with warm starts (numpy, not sklearn, for speed) | [CODE]; **never the ratio learner** |
| Exploratory | kNN on the ratio scale (k ∈ {5, 10, 20, 50}; cosine on E or Euclidean on [F_std, Z_K]); HistGBM (depth 3, rate 0.1, 200 iterations, min leaf 20; fixed) | y/x̃ | d·x̃ | as listed | [CODE] |

**Penalty rule** (fixes defect a; rule T2):
- Weights are normalized to mean 1.
- Continuous columns are standardized with the weighted SD, with penalty λ·n_train.
- **Categorical blocks** (NTEE10, state, income code, C, 0/1 indicators) stay on the raw 0/1 scale with penalty λ_C. λ_C acts as a pseudo-count: a code with m sampled units is shrunk by m/(m + λ_C) toward the intercept. This repairs the under-penalization of rare standardized dummies that made codes "hurt" in draft C's run 1.
- The intercept is unpenalized.
- The **same function** is used for CV and for the final fit.

### 4.3 Estimator list

Roles:
- **SQ** = status quo;
- **SQ-ML** = text-free ML (twins);
- **TXT** = text;
- **CTL** = control.

`@T` = tier. `+E` = Z_K(E*) added to the learner features. Every MCAL and FROZEN arm uses X_base = `size_cat@T`. The pilot specs used `size`; this must change [CODE].

| Name | `kind` and options | Learner features | Role | One weight? | Variance | Status |
|---|---|---|---|---|---|---|
| `HT` | ht | — | SQ (floor) | yes | v_L | exists |
| `GREG_ratio` | greg, X = strata + s_k | — | SQ | yes | v_L | exists |
| **`GREG_size_cat@T`** | greg, X = `size_cat@T` | — | **SQ production baseline (B0)** | yes | v_L, v_g | exists (add @T) |
| `GREG_weak@F0` | greg, X = `weak` | — | B0 for F0 | yes | v_L | exists |
| `GREG_size_cat@T-WIN` | greg + `win` | — | SQ with outlier treatment | yes | v_L on winsorized values | [CODE] |
| `MCAL_frame@T` | mcal, F = `F@T`, learner by y type | F | SQ-ML: NTEE10 twin (P1′ comparator) | no | v_L (final calibration residuals) | exists as `MCAL_frame_ratio` (fix A6 and X_base) |
| **`MCAL_bmfmax@T`** | mcal, F = `F@T`+C | F + C | **SQ-ML: P1 twin** | no | v_L | [CODE] |
| **`MCAL_bmfmax@T+E`** | mcal, F = `F@T`+C+Z_K | F + C + Z | **TXT: P1 text** | no | v_L; v_B | [CODE] |
| `MCAL_frame@T+E` | mcal | F + Z | TXT: P1′ text | no | v_L | exists as `MCAL_ratio+E` |
| `MCAL_bmfmax@T+E:full` | mcal, dual ridge | F + C + E | TXT (does the PC bottleneck hide signal?) | no | v_L | [CODE]; only after fix A1 |
| `MCAL_bmfmax_2part@T(+E)` | mcal, learner = twopart | as above | SQ-ML / TXT | no | v_L | [CODE] |
| `MCAL_bmfmax_knn@T(+E)`, `_gbm` | mcal, learner = knn / gbm | as above | Exploratory (R = 200) | no | v_L | [CODE] |
| `MCALm_bmfmax@T(+E)` | mcal_multi: calibrate once on [X_base, m̂_1..m̂_J], J4 = {gov_grants, contrib, fundraising_exp, grants_to_indiv_us}, chi-square distance, g ∈ [0.2, 5] (truncated linear) | as above | SQ-ML twin / **TXT: P2′** | **yes** | v_L, v_g | [CODE] |
| `FROZEN_diff_bmfmax@T(+E)` | frozen (difference) | previous-wave heads | TXT / sanity (exactly unbiased) | per y | v_frozen (exact for fixed m) | exists (`frozen`) |
| `FROZEN_ppi_bmfmax@T(+E)` | frozen, GREG on [strata, m] (PPI++ with a stratum-pooled power) | heads | TXT | per y | v_L | [CODE] |
| `FROZEN_cal_bmfmax@T(+E)` | frozen, `calibrate`, X_base = `size_cat@T` | heads | TXT | per y | v_L | exists |
| **`FROZEN_calm_bmfmax@T(+E)`** | frozen multi: [X_base, m_1..m_J], J4, bounded g | heads | **TXT: P2 text** (+E) / **frozen twin** (no E) | **yes** | v_L, v_g | [CODE] |
| `FROZEN_update…`, `FROZEN_census…` | frozen offset + cross-fitted NTEE10 ratio intercept; head trained on the whole previous population | heads | Exploratory; upper bound (realistic only where y_{t−1} is administrative) | per y | v_L | [CODE] |
| `RIDGECAL@T+E:ratio32` | ridge: hard = `size_cat@T`, soft = [Z32⊙x̃, Z32⊙ẽmp]; λ by CV of this estimator (fix A2) | — | TXT (factorial: linear) | per-y λ | v_L, v_df | exists as `RIDGE_size_cat+E:ratio32` |
| `RIDGECAL@T+C`, `RIDGECAL@T+C+E:ratio32` | ridge; soft adds C (raw 0/1, λ_C) and C⊙x̃ | — | SQ-ML / TXT (factorial) | per-y λ | v_L, v_df | [CODE] |
| `RIDGE1W@T+E` | ridge, one λ for all y (minimizes the mean normalized CV loss over H) | — | TXT, single weight | **yes** | v_L | [CODE] |
| `PCGREG_K@T+E` | greg on [size_cat@T, Z_K⊙x̃, Z_K⊙ẽmp], K ∈ {2, 4, 8} | — | TXT (transparent) | yes | v_L, v_df | [CODE] (generalizes `:add10`) |
| `PCCAL_pos@T+E` | greg on [size_cat@T, first K′ PCs of S32]; K′ = largest value with all g ≥ 0 in the replicate | — | TXT (production-style) | yes | v_L | [CODE] |
| `DIFF_cf@T+E` | Σ_U m̄ + Σ_s d(y − m̂), cross-fitted, no calibration (cross-PPI) | F + C + Z | TXT (what calibration adds) | per y | v_L | [CODE] |
| `GREG_size_cat@T+E:add10`, `GREG_weak+E:add32/add128` | greg | — | CTL (p/n demonstration) | yes | v_L | exists |
| `…:nocf` | mcal, `cf=False` | — | CTL (overfitting) | no | v_L | exists |
| `SYN_bmfmax@T+E` | syn: Σ_TA y + Σ_TS m̄ | — | CTL (model bias) | — | none | exists |
| `…+random`, `+perm`, `+permh`, `+permc`, `+const` | embedding replaced (§5) | — | CTL | — | — | random/perm exist; others [CODE] |

### 4.4 Primary and pre-registered pairs

| Pair | Text arm | Comparator | Question |
|---|---|---|---|
| **P1 efficacy** | `MCAL_bmfmax@F1p+E*` | `MCAL_bmfmax@F1p` | Does text carry information that the register's codes, lags and a flexible learner do not? |
| P1′ | `MCAL_frame@F1p+E*` | `MCAL_frame@F1p` | Same against the NTEE10-only twin (draft A's P1) |
| **P2 production** | `FROZEN_calm_bmfmax@F1p+E*` | `GREG_size_cat@F1p` | Does changing production to a frozen text-assisted single-weight system pay? |
| P2-twin | `FROZEN_calm_bmfmax@F1p` (no text) | `GREG_size_cat@F1p` | The same gain without text ("ML, not text") |
| P2′ | `MCALm_bmfmax@F1p+E*` | `GREG_size_cat@F1p` | P2 with current-sample cross-fitted models |
| B*-pair | `MCAL_bmfmax@F1p+E*` | **B*_frozen**: per y, the lowest-MSE text-free estimator on NE development among {GREG_size_cat, MCAL_frame, MCAL_bmfmax, MCAL_bmfmax_2part, FROZEN_calm_bmfmax}; frozen at FREEZE-2 | Is text better than the best text-free option? (secondary) |

P1 and P2 are read together. P2 passing with P1 failing means "the gain is ML, not text" (§9.6).

### 4.5 Attribution: the 2³ factorial

Runs at F1p, D1, n = 1,000 on common random numbers. Each cell is one estimator.

| A: learner and scale | B: codes | C: text | Estimator |
|---|---|---|---|
| linear | NTEE10 | no | `GREG_size_cat@F1p` |
| linear | NTEE10 | yes | `RIDGECAL@F1p+E*:ratio32` |
| linear | BMF-max | no | `RIDGECAL@F1p+C` |
| linear | BMF-max | yes | `RIDGECAL@F1p+C+E*:ratio32` |
| ratio-scale MCAL | NTEE10 | no | `MCAL_frame@F1p` |
| ratio-scale MCAL | NTEE10 | yes | `MCAL_frame@F1p+E*` |
| ratio-scale MCAL | BMF-max | no | `MCAL_bmfmax@F1p` |
| ratio-scale MCAL | BMF-max | yes | `MCAL_bmfmax@F1p+E*` |

- **Per y:** ℓ(S) = log MSE of the cell with factors S switched on.
- **Shapley share of factor j:** φ_j = Σ_{S ⊆ {A,B,C}∖{j}} [|S|! (2 − |S|)! / 3!] · [ℓ(S) − ℓ(S ∪ {j})].
  - The shares sum to ℓ(∅) − ℓ({A,B,C}) = log RE(111 | 000).
  - Report each share's fraction of the total.
  - GM over H uses the mean of the per-y shares.
- **Fine-code absorption (B):** Φ = 1 − log RE(MCAL_bmfmax+E | MCAL_bmfmax) / log RE(MCAL_frame+E | MCAL_frame).
- **MC uncertainty:** a replicate bootstrap (§8.3).

### 4.6 Tuning rules (non-negotiable)

| # | Rule |
|---|---|
| T1 | Every hyperparameter is chosen **inside each replicate** by 5-fold CV on s_TS. Nothing is tuned on population y. The only exception is control L5, which does so deliberately. |
| T2 | Scale-free penalty with categorical blocks penalized separately (§4.2). Unit test T-M1. |
| T3 | CV loss = **cross-validated anticipated variance on the y scale**: L(λ) = Σ_h N_h²(1 − f_h)/n_h · s²_h(e^oof), where e^oof_k = y_k − x̃_k ĝ^(−f(k))(z_k) for amounts and y_k − p̂^(−f(k)) for binaries. Tuning is coordinate-wise: λ_C first (with λ = 1), then λ given λ_C. |
| T4 | Folds: K = 5. **Grouped**: exact normalized-mission duplicates, cosine > 0.95 clusters on E*, and shared valid URLs share a fold. **Balanced**: groups are dealt round-robin within design strata. **Shared**: one fold vector per replicate, seeded by (cell_seed, r, 1), is used by every arm (twin, text and controls) and by both λ-CV and cross-fitting. |
| T5 | λ is chosen on the whole s_TS before cross-fitting (one hyperparameter pair; accepted approximation). **Check (Stage 2-dev, R = 200):** nested CV with λ chosen inside each outer training fold. If abs(Δ log RE) > 0.01 or abs(Δ coverage) > 1 pp on any y in H, nested CV becomes primary at FREEZE-2. |
| T6 | **λ stability (Stage 2-dev):** log the selected (λ, λ_C) per replicate. If more than 25% of replicates hit a grid boundary, or SD(log10 λ) > 1.0, the CV loss switches to the d·x̃-weighted SSE on the ratio scale. This is decided before FREEZE-2. |
| T7 | PCA uses the population frame embedding only (no y). Frozen heads use the previous wave's rotation. LSA vocabulary is fitted on t−1 frame text only. |
| T8 | Rung, field, K, feature map, learner, J4 and the λ policy are fixed by this protocol and §6.5. Never chosen on current-wave MC results. |
| T9 | Every fitted object (folds, λ, learner, calibration) is refitted inside every bootstrap replicate. |

### 4.7 Variance estimators and intervals

| Code | Formula | Use | Status |
|---|---|---|---|
| v_L | Σ_h N_h²(1 − f_h) s²_{ê,h}/n_h over take-some strata with n_h ≥ 2, where ê = final calibration residual y − c′B̂ (HT: y) | All; primary unless T-V below switches it | exists (`survey.strat_var_total`) |
| v_g | v_L on g_k ê_k, with g = w/d | Calibration estimators | [CODE] |
| v_df | v_L × n_TS/(n_TS − df), df = tr of the final fit's hat matrix | PCGREG, RIDGECAL, PCCAL. A heuristic: the exact Eustache / Bouhadra–Dagdoug forms were not retrieved [VERIFY]. | [CODE] |
| v_B | Rao–Wu–Yue rescaling bootstrap with m_h = n_h − 1: d*_k = d_k[1 − a_h + a_h (n_h/(n_h − 1)) m*_k], a_h = √(1 − f_h), B = 300, full re-estimation (T9) | P1/P2 arms, twins and GREG on the first 200 replicates of the NE primary cell | [CODE] |
| v_frozen | v_L with m fixed (unbiased for V_HT(y − m)) | FROZEN_diff | exists |
| v_R | v_L with respondent counts r_h | Nonresponse cells | exists (`mc._var_counts`) |
| v_R,B | Bootstrap that re-estimates propensity and outcome models in every replicate | Propensity and DR nonresponse estimators | [CODE] |
| v_d | Domain linearization on residual × domain indicator | Domains | exists (`sae._dom_var`) |

**Intervals:** Wald, θ̂ ± 1.96√v̂. Exploratory: a log-scale interval for skewed totals, and a logit interval for proportions below 0.10.

**Studentized-error diagnostic:** t_r = (θ̂_r − θ)/√v̂_r. Report:
- mean (bias);
- SD (> 1.10 means v̂ underestimates);
- skewness and tail shares (heavy tails).

Coverage is always reported as a **gap to HT** and a gap to the comparator.

**Rule T-V.** On NE development, if SD(t) > 1.10 for a primary arm while v_g or v_df brings it into [0.95, 1.10], that variance estimator becomes primary for that arm at FREEZE-2.

### 4.8 Single-weight estimators and weight diagnostics

- **Weights:** w_k = d_k g_k with Σ_s w x = T_x on [size_cat@T, m_1..m_J], g ∈ [0.2, 5] by truncated-linear iteration (`survey.bounded_calibration`) [CODE]. Non-convergence is flagged.
- **Application:** the same w applies to **every** y in A.
- **Diagnostics:** share of w < 0 and of w < 1; g quantiles and range; CV(w); Kish deff = n Σw²/(Σw)²; bound-hit rate; non-convergence rate; maximum change in any R-class total against production weights. The last should be 0 under exact calibration on `size_cat@T`.

### 4.9 Winsorized variants

`-WIN` versions exist for `GREG_size_cat@T`, `MCAL_bmfmax@T(+E*)` and `FROZEN_calm_bmfmax@T(+E*)`, with cutoffs frozen from the previous wave (§2.8). They are secondary (S-WIN, §9.4).

### 4.10 Pilot names → protocol names

| Pilot spec (`experiments.py`) | Protocol name | What changes |
|---|---|---|
| `GREG_size_cat` | `GREG_size_cat@F1` | `avail_D` indicators; tier lags |
| `MCAL_frame_ratio` | `MCAL_frame@F1` | X_base `size` → `size_cat@F1`; binaries → ridge-logit; penalty fix |
| `MCAL_ratio+{e}` | `MCAL_frame@F1+{e}` | same |
| `RIDGE_size_cat+{e}:ratio32` | `RIDGECAL@F1+{e}:ratio32` | λ-CV on its own penalty scale; grouped folds |
| `oracle_lag_*` columns | `lag_*` (F2 tier) | Renamed: under F2 it is a realistic register tier, not an oracle |

---

## 5. Controls

All gates are evaluated per cell. A failed gate quarantines that cell's efficiency claims until the cause is found, logged in `prereg/deviations.md`, and the fix applied to **every** cell. Results are then reported before and after the fix.

### 5.1 Control table

| ID | Control | Implementation | Expected | Gate rule | If violated |
|---|---|---|---|---|---|
| N1 | Random embedding | Gaussian, same dimension as E* (`embed.embed_random`), through every TXT arm | RE vs twin ≤ 1 | **Upper 95% bound ≤ 1.03** (gate). Point < 1 is reported as the cost of dimension. | Overfitting or a fold leak |
| N2 | Permuted embedding | E* rows permuted across all units | ≤ 1 | as N1 (gate) | as N1; also catches distribution-only effects |
| N3 | Within-stratum permutation | E* rows permuted within D1 strata (keeps the NTEE10 × size composition) | ≤ 1 | as N1 (gate) | Non-linear use of stratum-level text composition |
| N4 | Constant embedding | Every unit gets the same vector | **Identical to the twin to machine precision** | max abs difference ≤ 1e-8 relative (gate) | Code-path or fold-sharing bug |
| N2c | Within-code permutation | E* permuted within full-NTEE-code × size cells | RE vs NTEE10 twin = code-level part of the gain | Interpretation | — |
| N5 | Placebo y | PLC_perm (gov_grants permuted within stratum × lagged-size decile); PLC_noise (x̃·lognormal, σ = 2); PLC_bern (Bernoulli(p_h) within strata) | RE ≤ 1 for every arm | Upper bound ≤ 1.03 (gate G3) | Leakage through design or strata |
| N6 | Register-type y | Text arms on revenue, expenses, employees (and salaries under F1p) | ≈ 1 | Upper bound ≤ 1.03; **a lower bound > 1.03 is a leakage alarm** (gate G7) | Timing or numeric leakage suspected first |
| Po | Oracle / F2 lag | F2 twin vs F1p twin on gov_grants, fundraising, officer_comp | 4–10× (ceilings 5.1, 7.7, 9.7) | Lower bound > 2 (gate) | The harness cannot detect a known large gain |
| Pp | Planted signal (§5.2) | Semi-synthetic y with a known ρ² | RE tracks the theory | ρ² = 0: upper bound ≤ 1.03. ρ² = 0.2 (lin-same): φ ≥ 0.8 (gate G6). | Estimator or harness defect |
| L1 | Same-year text | T1_t in place of T1 | Small difference (missions 84.5% identical) | Report | Text change tracks y change; timing matters |
| L2 | Numbers masked | T2m vs T2 | abs(Δ log RE) ≤ 0.03 | Gate G8; on failure T2m becomes the primary field | The embedding reads amounts |
| L3 | Late frame | BMF 2026 snapshot as frame | Inflated size gains | Teaching control | Shows the size of a realistic timing error |
| L4 | Memorization probe | Predict log(y_t/y_{t−1}) from each rung, beyond the frame, vs shuffled names | Out-of-fold ΔR² ≤ 0.01 | FLAG | An API model may have seen post-2023 financials |
| L5 | Population-tuned λ and K | Oracle tuning on population AV | MC RE above in-sample-tuned RE | Report the optimism | Quantifies a common simulation error |
| L6 | Ungrouped folds | Random folds ignoring clusters | Small optimistic shift | Report | Near-duplicates leak across folds |
| O1 | No cross-fitting | `MCAL_bmfmax@F1p+E*:nocf` | Coverage loss (pilot 74–82%) | Must stay bad (harness check) | — |
| O2 | OLS with many PCs | `GREG_weak+E:add32/add128`; `PCGREG_K`, K up to 64 | Coverage collapse; RE < 1 | Must stay bad | — |
| O3 | Unpenalized vs penalized at large K | λ → 0 vs CV λ, K ∈ {64, 128, full} | Unpenalized loses as K grows | Report | Shows what defect (a) cost |
| O4 | Small-n penalty | n ∈ {650, 1,000, 2,000} | Gain grows with n | A gain that grows as n falls is suspicious | — |
| S1 | Synthetic estimator | `SYN_bmfmax@F1p+E*` | Bias (pilot: +15% on binaries) | Must show bias | Why the design correction is needed |
| C1 | Ceiling consistency | MC RE vs same-class ceiling (§8.4) | MC ≤ ceiling | MC RE ≤ ceiling × 1.05 + 2 MC SE (gate G5) | Optimistic tuning, leakage or a ceiling bug |
| SG | Per-cell sanity | HT abs(RB) ≤ 3 MC SE; FROZEN_diff abs(RB) ≤ 3 MC SE; failure rate ≤ 0.5% | — | Gate G1 | Harness defect |

### 5.2 Planted-signal family (power calibration and positive control)

**Null base (real distributions, no within-cell text link):**
- **Amounts:** y⁰_k = x̃_k · (y/x̃)_{π(k)}, with π a random permutation within NTEE10 × size class × state cells. This keeps the zeros, the skew, cell means and the ratio link to size, and breaks any unit-level link to text. Base y = gov_grants.
- **Binary:** p⁰_k = frame-only population logit fit of has_gov_grants.

**Planting:**
- **Amount:** y_plant = y⁰ · exp(a·u_k − a²/2), where u is standardized.
- **Binary:** b_plant = 1{U_k < logit⁻¹(logit p⁰_k + a·u_k)}, with U ~ Uniform(0, 1) drawn once.

**Signal forms u:**

| Form | Construction | Question |
|---|---|---|
| lin-same | w′Z_K(E*), w fixed random, orthogonalized against F + C | Recover a linear signal in its own representation? |
| lin-other | Same built on MiniLM, estimated with LSA (and the reverse) | Does the signal survive a representation mismatch? |
| cluster | Effects of 50 k-means clusters of E*, drawn N(0, 1) | Non-linear or local signal (ridge vs kNN) |
| tail | w′ on PCs 65–128 | Is signal beyond the top K lost? |

**Calibrating a.** a is set so that the **same-class population partial ρ² on the AV scale** equals the target:
- ρ² ∈ {0, 0.02, 0.05, 0.10, 0.20, 0.40}, where ρ² = 1 − AV(m*_{F+C+E}) / AV(m*_{F+C});
- 5 independent plantings (w, U) per level.

**Outputs:**
- realized RE against the theoretical 1/(1 − ρ²) for each family and n;
- φ = log RE_MC / log RE_theory;
- **ρ²_min(n)** = the smallest ρ² at which P1's MET rule passes in ≥ 4 of 5 plantings.

A null on real y is then reported as "text partial ρ² < ρ²_min(n)".

**Cost.** The MC is vectorized over y, so the planted columns are extra columns in the same cells: 4 forms × 6 levels × 5 plantings × 2 types = 240 columns in the NE planted cell. Divisions carry lin-same only at {0, 0.05, 0.10, 0.20}.

---

## 6. Embedding ladder and text-field arms

### 6.1 Held fixed on every rung

- Same field (T2 unless the comparison is about the field).
- Same frame features, learners, folds, designs and replicates (common random numbers).
- Same K after projection: every rung is reduced to K ∈ {8, 16, 32, 64} population PCs, plus the `:full` arm.
- MiniLM truncation (> 256 tokens) is logged by size class (test E2).

### 6.2 Ladder

| Rung | Representation | Status | Answers |
|---|---|---|---|
| L0 | None (twin) | ready | Baseline |
| L0-codes | BMF-max codes C | [CODE] (§4.1) | Can the register's own codes do it? |
| L0-neg | random, perm, permh, const | random/perm ready; others [CODE] | Does the machinery invent gains? |
| L1 | About 30 keyword flags (hospital, university, school, church, housing, foundation, PTA, VFW, Lions, Rotary, union, local, chapter, …) + log length + chapter-pattern flag | [CODE] | Can a methodologist without ML get the gain? |
| L2 | TF-IDF (1–2-grams) → SVD-256 (LSA) | ready (NE T1, T2); previous wave [CODE: minutes] | Lexical signal; deterministic |
| L3 | Averaged GloVe / spaCy static vectors | [CODE] (wheels reachable) | Semantics without context |
| L4 | all-MiniLM-L6-v2 (384-d, ONNX) | ready (NE and previous wave, T1/T2) | Small contextual model, on-prem |
| L5 | Qwen3-Embedding-0.6B (with and without instruction); BGE-M3 for EN/FR | [USER U4] | The on-prem alternative to Cohere |
| L6 | Cohere embed-v4.0 text: `classification`, 1536, float (primary). Factors: `clustering`, `search_document`; Matryoshka 256/512/1024; int8. | [USER U1] | Does a frontier API model add anything? |
| L6b | Cohere embed-v5.0-pro text | [USER U1] | Vendor-version bridging |
| L-W | Website text (Common Crawl ≤ D) through L2/L4/L6 | [USER U7] | Does estimation-time web text add beyond register text? |
| L7 | Cohere image / fused text+image: CCEW accounts pages (R2) or website screenshots (L-W) | [USER U1, U3/U7] | Do layout and visual content add beyond text? |
| L7-open | Qwen3-VL-Embedding-2B on the same images | [USER U4] | Does multimodal need Cohere? |
| **L7-ext** | Structured extraction of the lagged numbers on the same pages (text layer / OCR / table parse), added to the frame | with L7 | **The real competitor of a page embedding** |
| L8 | Small model fine-tuned on previous-wave labels, used frozen | optional (GPU) | Distance from what labels allow |

### 6.3 Comparisons and what each answers

| Comparison | Question | Statistic and decision use |
|---|---|---|
| L0-neg vs L0 | Is the harness honest? | Gates N1–N4 |
| L2 vs L0 (P1 with E = LSA) | Any text signal beyond the register? | RE |
| L1 vs L2 | Is it just keywords? | If RE(L2)/RE(L1) ≤ 1.03, recommend keywords and no embedding |
| L4 vs L2 | Contextual vs lexical (the skeptic bets no difference) | Δ log RE |
| L6 vs max(L2, L4, L5) | Is an external API justified? | Δ log RE ≥ log 1.05 (S9) |
| L6 vs best of L2/L4/L5 | Can production stay on-prem? | Δ log RE ≤ log 1.05 (S10; production condition) |
| T2 vs T1; T1s vs T1 | Program text vs mission; register-length text | Which register text to acquire |
| T0 vs none | Value of names (births) | Births arm |
| T2m vs T2 | Numeric leakage | L2 |
| PCs vs `:full` | Signal beyond the top K (B's RT-7) | Δ log RE |
| ridge vs kNN | Linear vs local signal | M10 |
| L6 input_type and dimension factors | Sensitivity to analyst choices | A spread > 5% calls for pre-registering the setting |
| L-W vs T2 | Web text vs register text | Δ log RE |
| L7 vs L6 on the same units; L7 vs L7-ext | Multimodal increment; page embedding vs extracted numbers | S11 |

**Field × model factorial** {L2, L4} × {T1, T2}: main effects and interaction on log MSE.

**Proxy validity:** Spearman correlation between the Stage-1 same-class ceiling, NTEE-probe accuracy and MC RE across rungs × fields (M9).

### 6.4 Scaling

| Factor | Grid | What it shows |
|---|---|---|
| n | D0 {250, 500, 1,000, 2,000}; D1 {650 … 2,000} | φ(n); coverage against p_eff/n |
| K | {2, 4, 8, 16, 32, 64, 128, full} at n = 1,000 | Dimension trade-off; overfitting onset for OLS vs ridge vs MCAL |
| Dimension within a model | Cohere 256 → 1,536 (Matryoshka) | Dimension vs semantics |
| Text length | By size class | Is the gain concentrated where text is long? |

### 6.5 Selection of E* (frozen at FREEZE-1)

1. **On the NE previous wave** (frame TY2021 → y TY2022; BMF codes recomputed per test B5), compute same-class Stage-1 ceilings for every local rung {L1, L2, L3, L4} × field {T1, T2} × K ∈ {16, 32, 64}. The comparison is against the BMF-max twin at F1p.
2. **E\*** = argmax of the GM over H of the ceiling RE. Ties within 0.02 go to the simpler choice: L1 < L2 < L3 < L4, then smaller K.
3. **Fallback**, if the previous-wave ceilings cannot be computed before FREEZE-1: MiniLM T2, K = 32.
4. **API and open-weight rungs** (L5, L6, L7) that become available later are **secondary arms only**. They never replace E*.
5. **The same E\*** (model, field, K) is used in the divisions. Each division gets its own PCA, and LSA is refitted on its own frame text.

### 6.6 Cohere specifics (note 04)

- **Settings:** `embed-v4.0`, `input_type = classification`, `output_dimension = 1536`, float.
- **Caching:** embed once per population and field. Cache content-addressed by (provider, model, sdk, api_version, input_type, dimension, embedding type, preprocessing version, input hash) via `embed.build` manifests. **The API is never called inside the MC.**
- **Pre-flight (< $0.50):** repeatability (200 texts twice), batch invariance, Matryoshka equivalence (`output_dimension = 256` against a truncated 1536), int8 vs float, image-token count on 100 pages at two DPIs, and a 1,000-item canary set.
- **Cost and time:**

  | Job | Cost | Time |
  |---|---|---|
  | NE text pass | < $0.50 | about 10 min |
  | National text pass (about 330k) | about $2–4 | about 2.5 h at 2,000 inputs/min |
  | CCEW images (about 25k pages) | about $14 | about 1 h at 400 images/min |

- **Key:** a trial key (1,000 calls/month, 5 images/min) cannot complete one national pass.

---

## 7. Uses beyond calibration

### 7.1 Stratification and allocation

- **Designs:** D1-imp, D2a, D2b (with its twin), D2c (§3.2).
- **Estimators:** `GREG_size_cat@F1p`, `MCAL_bmfmax@F1p` and `FROZEN_calm_bmfmax@F1p+E*`.
- **Reporting:** the 2 × 2 {D1 or equal-H twin, D2x} × {text-free, text estimator}.
- **Interaction (M8):** (ℓ₁₁ − ℓ₁₀) − (ℓ₀₁ − ℓ₀₀) on log MSE. A positive value means design and estimation uses are substitutive.
- **Metrics:**
  - RE per y, and GM over H and over A;
  - **minimum over A**, the no-harm criterion;
  - jumper MSE share;
  - realized vs predicted RE, 1 − ρ²(1 − 1/L²), with the score–y correlation within size classes;
  - for D2b, anticipated vs realized S_h.
- **Stage 1** computes the AV of every design × estimator pair, which predicts the MC.
- **Production reading:** if D1-imp alone captures the gain, the deployment is register maintenance (autocoding the activity code). That is a frame change, not an estimation change, and the easiest to govern.

### 7.2 Nonresponse adjustment and imputation

**Mechanisms (take-some).** Simulated response is redrawn in every replicate from stream (cell_seed, r, 2), with rates calibrated to 65%. δ and γ are set so that the respondent/nonrespondent ratio of mean gov_grants ≈ 0.35, matching NR-LATE.

| Code | Mechanism | Text should help? | Role |
|---|---|---|---|
| NR0 | MCAR | no | Sanity |
| NR1 | logit p = a + 0.5·z(log1p s_k) + NTEE10 effects (large units respond *more*) | no | **Negative control:** text adjustment's abs(RB) ≤ twin's + 1 pp, and abs(BRF) ≤ 10% |
| NR2 | logit p = a + 0.5·z(log1p s_k) + 0.8·z(u_k), with u = frozen text-predicted gov-grant ratio | yes, by construction | **Positive control:** BRF ≥ 70% |
| NR2′ | as NR2 with v_k, a text index orthogonal to every y in H | no; variance should rise | S6b |
| NR3 | logit p = a + 0.5·z(log1p s_k) + 0.8·z(residual of log1p y_gov_grants given F1p) (MNAR) | partly | Damage limitation |
| NR4 | p depends on valid-URL presence | — | MNAR auxiliary |
| **NR-LATE** | Real: filed ≤ 300 days after FYE (sensitivity: 180 and 365 days) | unknown | **The real test** |

**Take-all nonresponse** (two scenarios, both reported):
- **TA-follow-up:** take-all units respond MCAR at 90%.
- **TA-late:** the real indicator (12% respond).

Imputation methods for take-all nonrespondents:

| Code | Method | Role |
|---|---|---|
| TA-hist-own | Own last-cycle value × NTEE10 trend among responding take-all units: y_k = y_{t−1,k} · Σ_r y_t / Σ_r y_{t−1}. Certainty units were in last cycle's sample. | **SQ** [NEW] |
| TA-hist-aux | lag_expenses (or tier lag) × ratio | SQ fallback when the own value is missing |
| TA-nn-frame | Nearest responding donor on [log size, NTEE10, state] | SQ-ML |
| TA-nn-text | Nearest donor on [log size, Z_K(E*)] (Banff-style) | TXT |

**Take-some adjustment families.** Outcome models are cross-fitted on respondents, propensities cross-fitted on the sample, propensities clipped at 0.02. Each TXT method has a twin without E.

| Code | Method | Role | Status |
|---|---|---|---|
| NR_naive_HT | none | CTL | exists |
| NR_rhg_HT / NR_rhg_GREG_size_cat@T | Weighting class = design strata (then calibration) | **SQ** | exists |
| NR_psc(z) / (z, e) | 5 quantile classes of a ridge-logit p̂ (Haziza–Beaumont score method) | SQ-ML / TXT | [CODE] |
| NR_pmc(z, e) | 5 classes of cross-fitted m̂ for gov_grants (predictive-mean classes) | TXT | [CODE] |
| NR_joint(z, e) | 3 × 3 p̂ × m̂ classes, collapsed to ≥ 10 respondents | TXT | [CODE] |
| NR_2step(z, e) | d/p̂, then calibrate to (T_X, Σ_U m̄) (Haziza–Lesage) | TXT | exists as `ps` + `mcal` |
| NR_cal(z, e) | Calibration only on [X_base, m̂] | TXT | exists (`NR_rhg_MCAL_ratio+E`) |
| NR_DR(z, e) | AIPW: Σ_U m̄ + Σ_{s_r}(d/p̂)(y − m̂), cross-fitted nuisances | TXT | [CODE] |
| NR_imp_knn(z, e) | Donor imputation within NTEE10 on [log x̃, Z_K] | TXT (production imputation) | [CODE] |

**Stage-1 exact bias (no MC).** Response is deterministic per unit, so each method's limiting bias is a population quantity, computed on U_TS with d = 1:
- **Weighting class:** B = Σ_c N_c(ȳ_{r,c} − ȳ_c).
- **Calibration:** B = Σ_{U_TS}(r_k g*_k − 1) y_k.
- **DR:** B = Σ_{U_TS}(r_k/π*_k − 1)(y_k − m*_k).

The **reference text-free method** for each y is the one with the smallest Stage-1 limiting abs(bias) among {NR_rhg_HT, NR_rhg_GREG_size_cat, NR_cal(z), NR_DR(z)}. It is fixed in addendum v1.3 before the MC. This replaces B's ex-post choice.

The pilot's finding that calibrating to size *increases* bias for survey-only y becomes a testable claim: abs(B(NR_rhg + GREG)) > abs(B(NR_rhg)) for gov_grants, contrib and fundraising.

**Metrics:**
- RB, variance, RRMSE, coverage (v_R; v_R,B for propensity and DR);
- **BRF** = 1 − abs(RB_text)/abs(RB_ref), with an MC CI;
- Kish factor 1 + CV²(adjustment factors);
- AUC of p̂ with and without text;
- partial correlation of m̂_text with y among respondents;
- the **Little–Vartivarian 2 × 2** from NR0–NR3 and NR2′.

### 7.3 Small area estimation

**Domains.**
- **Planned:** state × NTEE10. NE has 66 domains: median N_d = 145, 17 with N_d < 30. At n = 1,000 the median expected n_d is 8, and 41% of domains expect fewer than 5. Divisions have 44–99 each.
- **Unplanned:** ZIP3 (about 60 in NE, mean n_d ≈ 16).

**Estimands:** domain totals of gov_grants, contrib and fundraising_exp; domain proportions of has_gov_grants and grants_to_indiv_us.

**Estimators:**

| Code | Estimator | Role | Status |
|---|---|---|---|
| DIR_HT | Direct HT | Floor | exists |
| DIR_GREG | Domain sums of the production `GREG_size_cat@F1p` weights | **SQ** | [CODE] (weights output) |
| DIR_FROZEN_calm | Domain sums of the P2 single weight | TXT | [CODE] |
| MA_dom(+E) | Σ_{U_d} m̂ + Σ_{s_d} d(y − m̂), national cross-fitted model | TXT | exists (`sae.run_sae` MA_) |
| SYN(+E) | Σ_{U_d} m̂ | CTL (bias) | exists |
| FH_frame | Area-level Fay–Herriot on DIR_GREG means. x_d = (1, domain mean of the frozen **frame-only** head). ψ_d smoothed by a GVF, log ψ̂_d = a + b log n_d + c log N_d, fitted per replicate on n_d ≥ 2. σ²_u by REML (truncated at 0). EBLUP; synthetic x_d′β̂ for n_d < 2. MSE by Prasad–Rao / Datta–Lahiri. | **SQ-SAE** | [CODE] (replaces `sae._fh`, which has one covariate, no intercept, raw ψ) |
| FH_text | FH_frame + the domain mean of the frozen **text** head (p = 3 ≪ m; no sampling error; independent of the current sample) | TXT | [CODE] |
| FH_textPC | FH_frame + domain means of the first 3 PCs of E* | TXT (stress) | [CODE] |
| NER_text (optional) | Unit-level nested-error EBLUP on y/x̃ with F + Z_8; parametric-bootstrap MSE (B = 100, 200 replicates) | TXT | optional |

**Metrics:**
- per-domain RB_d and RRMSE_d;
- AARB = mean_d abs(RB_d); ARRMSE = mean_d RRMSE_d; median and 90th percentile; domains with θ_d = 0 excluded and counted;
- distribution of per-domain RE;
- **the distribution of per-domain FH coverage**, not just its mean;
- MSE-estimator relative bias;
- number of domains with true CV ≤ 25% (publishable) and band migration (§8.8).

All are reported by n_d class {0, 1–4, 5–19, ≥ 20}. Non-sampled and NTEE-MISSING domains are reported separately.

**Heterogeneity check:** across domains, correlate RB_d(SYN_text) with the domain's median text length and boilerplate share. Text models may be biased exactly where text is poor.

**Replicates:** R = 500, reusing the Stage-2 seeds.

### 7.4 Frozen previous-wave deployment (the production pattern)

1. **Training samples.** Draw J = 20 independent D1 samples (n = 1,000) from the previous-wave population P1′, with BMF codes recomputed from TY2021 per test B5. **Keep the respondents only** (300-day rule at the previous wave), with RHG-adjusted weights. An NSO trains on last cycle's respondents, not on the full previous population.
2. **Fitting.** On each sample, fit the y-type learner (§4.2) on F@T + C + Z_K(E*), with E*'s PCA rotation fitted on the previous-wave frame. λ is chosen by grouped CV inside the sample. Text-free twin heads use F@T + C.
3. **Application.** Apply each head j to the current frame (TY2022 frame and text, rotated with the old PCA), giving m_j(k) for all of U_TS.
4. **Cycling.** Current replicate r uses head r mod 20. The spread of RE across heads is reported as **training-sample risk**.
5. **Variants:**
   - `FROZEN_diff`, `_ppi`, `_cal`, `_calm`;
   - `FROZEN_update` (offset plus a cross-fitted NTEE10 ratio intercept, a light drift correction);
   - `FROZEN_census` (upper bound);
   - λ frozen at the previous wave vs re-tuned;
   - a sensitivity with heads trained on the full previous sample (no nonresponse).
6. **Model ageing.** Report each head's ceiling R² on its training wave against the next wave. A drop of more than 20% flags drift.
7. **Divisions.** Each division repeats steps 1–6 on its own previous wave. This needs the national TY2021 build (reachable; §11).

### 7.5 PANEL: rotation overlap with known last-cycle values [NEW]

**Why.** In a real annual survey, the previous cycle's respondents carry y_{t−1} for every surveyed item. It is a frame variable known for all of U (zero outside r′). Note 06 §4.2 shows that the lagged indicator removes the binaries' text gain (RE 1.00–1.02), so H_S needs this test.

**Implementation.**
- s′ comes from the J = 20 previous-wave draws of §7.4, and r′ is their 300-day respondents.
- The current sample uses PANEL-D1 (§3.2), which keeps about 50% of each stratum's allocation from O_h.
- Matrices add the PANEL columns (§4.1) to `size_cat@F1p` and `F@F1p`. Their population totals are known.
- The cell runs as 20 sub-cells × 100 replicates, because the design depends on r′.

**Estimators:** `GREG_size_cat@PANEL`, `MCAL_bmfmax@PANEL(+E*)`, `FROZEN_calm_bmfmax@PANEL(+E*)`.

**Readout:**
- RE of text vs twin overall, on the overlap stratum and on the fresh stratum;
- the share of F1p's log text gain retained (S-PANEL);
- the expected pattern: retained share ≈ the fresh stratum's share of take-some residual variance.

### 7.6 Births arm (exploratory)

- **Units:** births (§2.6) with a BMF record ruled ≤ D. They are F0 units with T0 name text only.
- **Design:** an SRS of births at f ≈ 4.85% (NE about 70 units, so national births are preferred if available).
- **Comparison:** `MCAL_bmfmax@F0+E(T0)` vs `MCAL_bmfmax@F0` on the births subtotal.
- **Skeptic's prediction:** about 1.10 within births and about 1.01 on the overall total.

---

## 8. Metrics and Monte Carlo protocol

### 8.1 Per-cell metrics

For estimator a, variable y, replicates r = 1..R, estimate θ̂_r, variance estimate v̂_r and truth θ:

| Metric | Formula | MC SE |
|---|---|---|
| Relative bias | RB = (mean θ̂ − θ)/θ | sd(θ̂)/(√R abs(θ)) |
| MSE, RRMSE | MSE = R⁻¹Σ(θ̂_r − θ)²; RRMSE = √MSE/abs(θ) | sd((θ̂ − θ)²)/√R; delta method |
| Relative efficiency | RE(a ∣ b) = MSE_b/MSE_a on paired replicates | Var(log RE) ≈ R⁻¹[Var(e_a²)/MSE_a² + Var(e_b²)/MSE_b² − 2Cov(e_a², e_b²)/(MSE_a MSE_b)] (`mc.metrics`); cross-checked by a replicate bootstrap |
| Coverage | C = R⁻¹Σ 1{abs(θ̂_r − θ) ≤ 1.96√v̂_r}; lower and upper non-coverage separately | √(C(1 − C)/R) |
| Coverage gaps | C_a − C_HT; C_a − C_comparator | Paired binomial (McNemar-type) |
| Variance relative bias | (mean v̂ − MSE)/MSE; also against the empirical variance | Delta method |
| Studentized error | Mean, SD, skewness and tail shares of t_r | Bootstrap over r |
| CI relative width | 2 × 1.96 × mean √v̂ / abs(θ) | — |
| Estimated CV | √v̂_r/θ̂_r; band shares (§8.8) | — |
| Fitting efficiency | φ = log RE_MC / log RE_ceiling (same class) | Delta method (ceiling fixed) |
| Gap closure (F0) | G = log RE(F0 text ∣ F0 twin) / log RE(F1p twin ∣ F0 twin) | Unpaired across designs; bootstrap |
| Robust companions (H-class) | Median abs relative error; MSE with the top 0.5% of abs(errors) trimmed | Reported, never decisive |
| Failure rate | NaN, non-convergence, bound violations | > 0.5% flagged; > 2% excludes the estimator from that cell's claims. RE uses jointly successful replicates. |

### 8.2 Aggregation

- **GM-RE over a set Y** = exp(abs(Y)⁻¹ Σ log RE_y). Its CI comes from a replicate bootstrap (B = 2,000 over r, which keeps the correlation across y within a replicate).
- **Worst case over A:** the minimum point RE, plus a list of any y with harm (upper bound < 0.95).
- **Win/loss count:** wins have a lower bound > 1.05; losses have an upper bound < 0.95.
- **By size class:** each class's take-some MSE contribution under twin and text, showing where the gain lives.
- H-, P- and K-class y are never pooled into scores.

### 8.3 Replicates, seeds and common random numbers

**Seeds.** cell_seed = crc32(f"{pop_id}|{design_id}|{n}"). Streams:

| Stream | Seed |
|---|---|
| Sample r | `default_rng([cell_seed, r, 0])` (replaces today's sequential `default_rng(seed)` stream, so replicates can be run in any order or in parallel) |
| Shared folds | `[cell_seed, r, 1]` |
| Simulated response | `[cell_seed, r, 2]` |
| Previous-wave draw | `[cell_seed, r mod 20, 3]` |
| Estimator-internal | `[cell_seed, r, crc32(name)]` |
| Bootstrap | `[cell_seed, r, crc32(name), 4]` |

**Rules.**
- Tiers that share a design share samples.
- Every rung and arm in any later run sees identical samples and folds. T-M5 checks that adding an estimator leaves the others bit-identical.
- `reps.parquet` stores a hash of s_r.

**Replicates (fixed; no optional stopping):**

| Cell type | R |
|---|---|
| Development cells (NE) | 500 |
| **NE proof-of-concept primary cell (2A)** | **2,000** (per-y log-RE MC SE about 0.01–0.035; GM about 0.01) |
| Planted cell | 1,000 |
| Bootstrap validation | 200 replicates × B = 300 |
| SAE | 500 |
| **Confirmatory division cells** | **R_conf** = the smallest value in {1,000, 2,000} giving a per-division MC SE of log GM-RE over H ≤ 0.01, estimated from NE development variance scaled by n_g. Fixed at FREEZE-2. |

**Reproducibility.** Rerunning 10 replicates of any cell from its manifest must match bit for bit (H2). Each manifest records the git commit, prereg hash, data and embedding hashes, seeds, package versions, runtime and failure counts.

### 8.4 Ceilings and the consistency gate

**Same-class AV ceiling** (Stage 1). For each MC estimator, use the same features, target scale, weights, learner family and penalty rule. Fit on all of U_TS with population 5-fold cross-fitting.
- AV(m*) = Σ_h N_h²(1 − f_h)/n_h · S²_{y−m*,h}.
- RE_ceiling(a ∣ b) = AV(m*_b)/AV(m*_a).
- Two versions: **oracle λ** (labelled optimistic) and **fixed λ** (the median λ chosen in development).

Today's `diagnostics.py` uses a raw-scale ridge with weights d − 1, which is not the same class as the ratio MCAL arms. That partly explains the pilot violation (MC 1.43 against a ceiling of 1.13).

**Learning-curve ceiling.** Draw B = 50 samples at n. Fit the learner as the MC would, with no calibration or variance. Evaluate AV(m̂_s) on the population with m̂_s fixed. This predicts φ(n) at about 1/20 of the MC cost.

**Shrinkage κ** (B) = median over H of log RE_MC / log RE_ceiling for P1 on NE development. Pre-registered prediction for each division g and y: log RE_pred = κ · log RE_ceiling(g). M9 tests median abs(log RE_MC − log RE_pred) ≤ 0.05. If it holds, an NSO can screen variables with cheap diagnostics.

**Gate C1:** MC RE ≤ same-class ceiling × 1.05 + 2 MC SE, for every TXT arm and every y in H.

### 8.5 The p/n arithmetic behind the predictions

A cross-fitted learner with p_eff effective parameters on n_TS units inflates residual variance by about (1 + p_eff/n_TS). So

  RE_MC(text ∣ twin) ≈ RE_ceiling × (1 + p_twin/n_TS)/(1 + p_text/n_TS).

With unpenalized p ≈ 60 against 28 at n_TS = 794, the factor is 0.96. Under ridge, p_eff = df < p, so the factor is closer to 1. The planted cell measures it directly.

### 8.6 Between-population inference

- **Per division:** L_g = log GM-RE_g over H.
- **Confirmatory statistic:** L̄ = mean over the 8 held-out divisions, with SE = sd(L_g)/√8 and t₇. MC error is inside each L_g and so is counted.
- **Holm:** across P1 and P2 at one-sided α = 0.05 (§9.5).
- **Random-effects meta-analysis** (REML, within-variance = MC SE²) gives the pooled log RE, τ and a **95% prediction interval for a new population**.
  - It uses only non-overlapping populations: the 8 divisions, plus CRA T3010 when available.
  - NE is excluded from the confirmatory PI (it is development), and the NE previous wave is never pooled (it shares 19,478 units). A sensitivity includes NE.
- **Consistency:** the count of divisions with GM_g ≥ 1.05 is descriptive only.
- **Limitation:** divisions are spatially correlated and not random draws, so t₇ precision is optimistic (§12.4).

### 8.7 Equivalent sample size, binding constraint and cost

**Analytic, take-some part.** The text arm reaches the baseline's precision at n_TS with

  1/n_t − 1/N_TS = RE · (1/n_TS − 1/N_TS),

saving 1 − n_t/n_TS. Example: RE 1.20, n_TS = 794, N_TS = 20,420 gives n_t ≈ 666, a 16% saving.

**Empirical (preferred: it includes the fitting penalty at smaller n).**
- The baseline runs at n ∈ {750, 875, 1,000, 1,250, 1,500, 2,000}; the text arm at {650, 750, 875, 1,000}.
- Interpolate log MSE_y(n) monotonically in log n.
- n_t,y solves MSE_text,y(n_t) = MSE_base,y(1,000).

**Binding constraint.** The hypothetical survey's CV targets [USER U5; defaults are assumptions]:
- national totals CV ≤ 5% for gov_grants, contrib and fundraising, ≤ 10% for volunteers_cap10k;
- national proportions: absolute SE ≤ 1.0 pp;
- state totals CV ≤ 15% for gov_grants and contrib.

The minimum n meeting **all** targets is computed under each system from the Stage-1 AV functions rescaled by the MC/ceiling ratio φ. **The saving is the difference in that minimum n**, not the best single-y RE.

**Cost.**

  ΔC = Δn_TS × (c_u + b × c_b) − C_text.

| Parameter | Default [ASSUMPTION, USER U5] |
|---|---|
| c_u (collection, follow-up, editing per sampled unit) | CAD 200 |
| b (respondent burden) | 2 hours per questionnaire |
| c_b (burden cost) | CAD 50 per hour |
| C_text (annual pipeline cost) | about 0.3 FTE (≈ CAD 40k) plus compute (on-prem LSA/MiniLM ≈ 0; Cohere API about $2–5 per national pass) |

Report the break-even Δn and how it scales.
- **NE:** RE 1.10 saves about 70 questionnaires, about CAD 14k + burden, below the pipeline cost.
- **National, n ≈ 10,000 (n_TS ≈ 6,700):** about 600 questionnaires, about CAD 120k + burden.

### 8.8 CV quality bands

Placeholder bands are StatCan's common letters: A < 5%, B 5–< 10%, C 10–< 15%, D 15–< 25%, E 25–< 35% (use with caution), F ≥ 35% (suppressed). Confirm against the target program's guideline [VERIFY]. `report.cv_band` exists.

Report band distributions of the **true** CV (√MSE/θ) and of the **estimated** CV for national, state and state × NTEE10 estimates, under the status quo and the text system.

Printed on every table: **RE r shrinks CVs by 1 − 1/√r**: 1.10 → 4.7%, 1.20 → 8.7%, 1.50 → 18.4%. Band changes are rare at national level; the publishing case lives in the domains.

### 8.9 Standard result tables (every cell)

1. RE vs the BMF-max twin, the NTEE10 twin and production, per y, with MC CI.
2. RB, RRMSE, coverage, coverage gaps (HT and comparator), variance relative bias, studentized-error moments.
3. GM-RE over H, H_D and H_S; worst case over A.
4. φ and the ceiling-consistency ratio.
5. Weight diagnostics (single-weight arms).
6. Failure counts and timing.

Each table carries the cell manifest: population hash, tier, design, n, R, seeds, prereg hash, git commit and embedding hashes.

---

## 9. Pre-registered hypotheses, thresholds and decision rules

### 9.1 Decision vocabulary

Decisions use effect sizes with bounds, not p-values. Bounds are one-sided 95%, Holm-adjusted for P1 and P2.

| Reading | Rule | Meaning |
|---|---|---|
| **Production bar** | **MET**: point ≥ 1.10 and lower bound ≥ 1.05. **EXCLUDED**: upper bound < 1.10. **OPEN**: otherwise. | 1.10 ≈ 9% take-some saving (§9.8) |
| **Any gain** | **YES**: lower bound > 1.00. **NO**: upper bound < 1.05 (a 5% gain is excluded). **UNCLEAR**: otherwise. | Separates "real but small" from "nothing" |
| Harm | Upper bound < 0.97 | — |

**Power check.**
- Divisions: with between-division SD of L_g ≈ 0.03, SE(L̄) ≈ 0.011 and the t₇ critical value 1.895. A true RE of 1.10 then gives a lower bound ≈ 1.078 (MET). A true 1.07 gives an upper bound ≈ 1.092 (EXCLUDED).
- NE proof of concept: at R = 2,000 the SE of log GM-RE is about 0.01.

### 9.2 Validity gates (per cell, before any efficiency result is read)

| Gate | Criterion |
|---|---|
| G0 (Stage 0) | All blocking red-team and harness tests pass (§10), or a logged deviation accepts the failure |
| G1 Design sanity | HT abs(RB) ≤ 3 MC SE; FROZEN_diff abs(RB) ≤ 3 MC SE; failure rate ≤ 0.5% |
| G2 Negative controls | N1, N2 and N3: one-sided upper bound of RE vs twin ≤ 1.03 for every y in H. N4: equality to machine precision. |
| G3 Placebo | N5: upper bound ≤ 1.03 for every arm |
| G4 Text-arm validity | For each y in A (excluding H-class), for P1/P2 text arms:<br>• coverage gap to comparator ≥ −2 pp;<br>• gap to HT ≥ −5 pp;<br>• abs(var RB) ≤ 15%;<br>• abs(RB) ≤ 0.1 × RRMSE or within 2 MC SE.<br>If the HT gap fails, diagnose with the studentized error: SD > 1.10 means v̂ underestimates (rule T-V); skewness shared with HT means a heavy tail, not a text defect. |
| G5 Ceiling | C1 for every TXT arm and every y in H |
| G6 Planted | ρ² = 0: upper bound ≤ 1.03. ρ² = 0.2 lin-same: φ ≥ 0.8. |
| G7 Register y | N6: upper bound ≤ 1.03; a lower bound > 1.03 is a leakage alarm |
| G8 Leakage | L2 abs(Δ log RE) ≤ 0.03; B1 < 1%; B5 passed |
| G9 Oracle | Po lower bound > 2 |

### 9.3 Primary endpoints

| ID | Population and cell | Statistic | Pass (supports) | Falsifies (production bar) | Skeptic predicts | Pre-registered prediction |
|---|---|---|---|---|---|---|
| **P1 efficacy** | **Confirmatory:** 8 divisions, P1 population, F1p, D1, f ≈ 4.85%, R_conf. **Proof of concept:** NE, n = 1,000, R = 2,000. | GM over H of RE(`MCAL_bmfmax@F1p+E*` ∣ `MCAL_bmfmax@F1p`) | Production bar MET | EXCLUDED | 1.00–1.05 | **1.04** (80%: 1.00–1.08). Any gain likely YES. |
| **P2 production** | same | GM over H of RE(`FROZEN_calm_bmfmax@F1p+E*` ∣ `GREG_size_cat@F1p`); **and** no harm (min point RE over A ≥ 0.95, no y with upper bound < 0.95); **and** G4 holds for every y in A | MET + no harm + valid | EXCLUDED | 1.00–1.06 (if it passes, mostly via ML) | **1.05** (0.98–1.12) |

**Rules.**
- Predictions are updated once, at FREEZE-1, using the Stage 0c rerun and the Stage-1 same-class ceilings, and are then frozen. **Thresholds are not updated.**
- P1′, P2-twin and P2′ are reported beside the primaries as secondaries.
- The NE proof-of-concept verdict uses the same rules on MC bounds and is labelled "development population; generalization not assessed". It is the first deliverable and does not wait for the national build.

### 9.4 Secondary hypotheses (95% bounds; labelled secondary)

| ID | Hypothesis | Statistic | Pass threshold | Skeptic predicts |
|---|---|---|---|---|
| S-F1 | Realism check under F1 (no payroll) | P1 and P2 at F1 | Reported (no threshold) | Higher than F1p, through salaries |
| H7 | Salaries gain disappears with payroll | RE_inc for salaries at F1p | Upper bound < 1.05 | Yes (pilot 1.14 at F1) |
| S-F2 | Text adds with the full prior return (charities) | GM over H_S at F2; RE_inc on H_D at F2 | ≥ 1.05 | ≤ 1.02 (note 06: 0.93–1.017) |
| S-PANEL | Gain survives rotation overlap | Share of F1p log gain retained under PANEL | ≥ 0.5 | ≤ 0.3 for H_S |
| S-F1c | Hybrid register | P1 at F1c | Reported | ≈ 1.03 |
| S1 | No gain on R-class y | RE_inc on revenue, expenses, employees | Upper bound ≤ 1.03 (gate G7) | 1.00 |
| S2 | Text helps a weak register | GM over R ∪ H_D of RE(`MCAL_bmfmax@F0+E*` ∣ `MCAL_bmfmax@F0`) under D1-W | Point ≥ 1.20 and lower bound ≥ 1.10 | Yes (about 1.5–2×) |
| S2b | Text substitutes for admin data | Gap closure G | ≥ 0.5 means "substitutes" | ≤ 0.5 |
| S-B* | Beats the best text-free option | GM over H of RE vs B*_frozen | Lower bound > 1.00 and point ≥ 1.05 | 0.97–1.03 |
| M1 | Text is the main source of the best gain | Shapley share of C in the factorial | ≥ 0.5 with lower bound ≥ 0.25 | < 0.4 |
| M1b | Codes absorb text | Φ over H | Φ < 0.3 means text carries more than codes | Φ ≥ 0.5 |
| M2 | Text information is within code | RE(E*)/RE(N2c) vs NTEE10 twin, GM over H | ≥ 1.05 | ≤ 1.05 (note 06 says it survives; prior updated) |
| M3 | Validity across arms | G4 for every cross-fitted or frozen TXT arm; v_B coverage within ±3 pp of 95% | All pass | Holds with cross-fitting |
| M4 | p/n behaves as predicted | OLS PCGREG coverage non-increasing in K; ridge/MCAL coverage gap ≥ −2 pp at K ≤ 64 | Holds | Holds |
| M5 | Planted recovery | ρ²_min(1,000) | Reported; ≤ 0.10 is required for falsification | 0.08–0.12 |
| M6 / S8 | Frozen keeps the gain | log RE(FROZEN_cal_bmfmax+E* ∣ frozen twin) / log RE_inc(MCAL) | ≥ 0.75 retained; FROZEN_diff coverage gap to HT ≥ −2 pp | 0.5–0.8 retained |
| M7 | Ceilings honest | C1 for all arms | Pass | — |
| M8 | Design and estimation substitutive | Interaction contrast > 0, CI excluding 0 | Yes | Substitutive |
| M9 | Diagnostics predict MC | Median abs(log RE_MC − κ · log RE_ceiling) ≤ 0.05; Spearman ≥ 0.8 across rungs | Pass | Yes |
| M10 | Learner matters more than representation | Δ log RE across learners vs across rungs | Reported | Learner > rung |
| H14 | Single weight keeps the gain | log RE(MCALm+E* ∣ MCALm twin) / log RE_inc(per y) | ≥ 0.75 and no harm | < 0.6 |
| S-WIN | Not an outlier artefact | Share of log gain retained in `-WIN` variants | ≥ 0.75 | Holds |
| S-CM | Concept-mismatch direction | RE_inc when lag–y agreement is degraded (§11, 2I) | Increases (direction check) | Increases |
| S4 | Text-coded NTEE helps the design | GM over A, D1-imp vs D1 (HT and GREG) | ≥ 1.05 | ≤ 1.03 nationally; larger for NTEE10 domains |
| S5 | Prediction strata help | GM over H, D2a vs D1-H44 | MET | ≤ 1.05 |
| S5b | Text-informed allocation helps | GM over H, D2b vs D2b-twin | ≥ 1.05 | ≤ 1.03 |
| S6 | Text reduces nonresponse bias (NR-LATE, both take-all scenarios) | BRF vs the Stage-1 reference on {gov_grants, contrib, fundraising, volunteers_cap10k, lobbying} | BRF ≥ 30% on ≥ 3 of 5 y, RRMSE ratio ≤ 1.00, coverage gap ≥ −2 pp; **and** NR1/NR2 gates pass | < 30% |
| S6b | Response-only text adds variance | NR2′ variance ratio | > 1 | Yes |
| S7 | SAE | ARRMSE(FH_text)/ARRMSE(FH_frame), planned domains, gov_grants and has_gov_grants | ≤ 0.90, ARB not worse by > 1 pp, publishable share +5 pp, < 5% of domains with RRMSE +10% | ≥ 0.95 overall; ≤ 0.90 only in NTEE-MISSING domains |
| S9 | A modern embedding is needed | max(L4, L6) − L2 in log GM-RE | ≥ log 1.05 | < log 1.05 |
| S10 | On-prem suffices | L6 − best(L2, L4, L5) | ≤ log 1.05 | Yes |
| S11 | Multimodal adds (CCEW, websites) | L7 vs L6 on the same units, numbers masked; L7 vs L7-ext | ≥ 1.05 and L7 ≥ L7-ext | L7-ext ≫ L7; L7 ≤ 1.02 |
| S12 | Replication | P1/P2 verdict on CRA T3010 | Same verdict category | Weaker or null |
| S-births | Names help births | Births subtotal RE | ≥ 1.20 | About 1.10 |

### 9.5 Multiplicity

- **Holm** over P1 and P2 (one-sided α = 0.05) on the MET test, in the confirmatory divisions.
- The NE proof-of-concept verdict is development and carries no α claim.
- Secondaries carry unadjusted 95% bounds and BH q-values for per-y "RE > 1". **No secondary result can rescue a failed primary.**
- Everything else is exploratory: other estimators, y, n, K, learners and arms. "The best of about 45 estimators per y" is never reported as a finding.
- Gates are not hypotheses and spend no α.

### 9.6 Verdict rules

| Verdict | Conditions | Production action |
|---|---|---|
| **HARNESS NOT VALIDATED** | Any of G1–G9 fails in more than one division after diagnosis, or fails in the NE proof-of-concept cell | No claims either way |
| **FALSIFIED** (tax-linked register) | P1 production bar EXCLUDED (confirmatory); ρ²_min(n) ≤ 0.10; gates clean; NE proof of concept in the same direction | Stop. Publish the null with ρ²_min: "under a register with lagged size and payroll, t−1 text explains less than ρ²_min of residual variance of survey-only y; no production-worthy gain." |
| **SUBSTITUTE FOR A MISSING LAG** | Gain at F1/F1p, but S-F2 and S-PANEL show ≤ 1.02 once the lag exists | Acquire or link the administrative or panel lag of y. It was worth 2.4–14.6× here. |
| **REDUCED TO FINER CODES** | Shapley share of C < 0.25 and Φ ≥ 0.5; P1 any-gain NO or UNCLEAR | Use the full NTEE code (and text-coded NTEE where missing) in strata and calibration |
| **REDUCED TO ML, NOT TEXT** | P2 MET via P2-twin, P1 not MET; Shapley share of A ≥ 0.5 and of C < 0.25 | Ratio-scale model calibration on register variables, with no text |
| **NICHE** | P1/P2 not MET, but S2, S4, S6, S7 or S-births passes in a scenario that maps to a real program (non-charity NPOs on a BR-only frame; births; NTEE10 domain tables) | Scoped deployment only |
| **SUPPORTED FOR A PRODUCTION PILOT** | All of:<br>• P1 and P2 MET (confirmatory);<br>• M3, M6 and S-WIN pass;<br>• the random-effects PI lower bound for P2 > 1.00;<br>• the binding-constraint saving ≥ 10% of the take-some sample;<br>• an on-prem rung within log 1.05 of the best (S10);<br>• ΔC > 0 at the program's scale;<br>• the domain no-harm part of S7 holds;<br>• gates pass in ≥ 7 of 8 divisions. | Frozen text-score columns in existing calibration, run in parallel for one cycle |
| **ADOPTION** | The above, plus:<br>• GM-RE ≥ 1.20 on the program's own binding variables;<br>• one parallel production cycle with frozen heads;<br>• S12 (CRA) gives the same verdict. | Reduce the take-some sample, or keep n and publish more domains |

Anything between these verdicts is reported as **INCONCLUSIVE**, naming the binding failure.

### 9.7 Freeze procedure and researcher degrees of freedom

| Freeze | When | Contents | File |
|---|---|---|---|
| **FREEZE-1** | After Stage 1, before any Stage-2 MC | y sets; tiers; designs; estimator list; learners; λ grids; folds; E* (§6.5); K; J4; thresholds (§9.1–9.6); R for NE cells; updated predictions; cells | `config/prereg_v1.yaml` + `.sha256`, committed [USER U9 sign-off] |
| **FREEZE-2** | After Stage 2-dev, before any division MC | R_conf; κ; B*_frozen; variance-estimator choice (T-V); CV-loss fallback (T6); nested-CV choice (T5); deviations | `config/prereg_v1.2.yaml` |
| Addenda | Before each later confirmatory run | v1.3 NR (reference methods from the exact-bias table); v1.4 SAE; v1.5 ladder/Cohere; v1.6 replications | `config/prereg_v1.k.yaml` |

**Enforcement.**
- `scripts/run_cell.py` **refuses to start** a confirmatory cell unless the yaml hash and the git commit match the manifest.
- Division Stage-1 diagnostics run in the same automated job as their MC, after FREEZE-2, so nobody tunes against them.
- Deviations go in `prereg/deviations.md` (date, reason, impact), and results are reported both ways.

### 9.8 Why these thresholds

- **1.10** is about the smallest variance reduction that justifies re-specifying a production estimation system.
  - It saves about 9% of the take-some sample (about 70 questionnaires in NE, about 600 at national n ≈ 10,000).
  - It shrinks CVs by only 4.7%.
- **1.05** as the lower bound guards against MC and between-population noise without making a true 1.10 unpassable (§9.1).
- **GM over H** is used because the survey is multipurpose. A single-y win is reported but never triggers a recommendation.
- **1.20 for adoption** (17% saving, 8.7% CV) covers system change, documentation, QA, governance and bridging costs.

---

## 10. Dataset red-team checklist and replication datasets

### 10.1 Data tests (notebook `01_data_and_red_team`)

- Each test prints a number and **PASS / FLAG / FAIL**.
- **Blocking** means Stage 2 does not start until the test passes or a deviation is logged.
- Origin column: A = draft A (T-numbers), B = draft B, C = draft C, N6 = note 06, NEW = new here.
- Observed values are for NE unless noted.

**A. Units and population**

| ID | Test | Rule | Blocking | Observed | Origin |
|---|---|---|---|---|---|
| A1 | One row per EIN; drop log by reason (form type, group return, duplicate, multiple periods, territory) | Drop log sums exactly | yes | pass; 8,369 national TY2023 EINs with > 1 return | A T01, B A1 |
| A2 | Amended truth: share of totals from amendments; abs(Δy)/y | FLAG if > 1% of any total changes | no | 1.5% of units, 0.9% of expenses | A T02 |
| A3 | Group returns excluded; their revenue share | Report | no | 5.5% (national) | A T03 |
| A4 | Survivorship: deaths, EZ switchers, births as shares of t−1 and t totals by y; P2 built | Report | no | 1,115 deaths; 1,414 births (0.3% of expenses, 0.9% of gov_grants) | A T04, B RT-4 |
| A5 | 990-EZ threshold crossers entering the 990 counted as births | Report | no | — | A T05 |
| A6 | State: BMF vs XML ≥ 98%; t−1 vs t | ≥ 98% | no | 99.9% (t−1 vs t) | A T06 |
| A7 | Schema versions: null rate of every y and text XPath by return version | FAIL if any version > 2× the median | yes | TY2023 mixes 4 versions | A T07, B D4 |
| A8 | FYE mix; short or long periods; December-only sensitivity | Report | no | Dec 47.2%, Jun 32.8%, Sep 7.9% (corrects B's 59/25) | A T08 |
| A9 | Division assignment; per-division N | Matches expected | yes (divisions) | 14.7k–58.2k | B A7 |
| A10 | Totals vs IRS SOI extract (processing-year aligned) | FLAG > 2%, FAIL > 5% | no | — | A T31, B D5 |

**B. Timing and leakage**

| ID | Test | Rule | Blocking | Observed | Origin |
|---|---|---|---|---|---|
| B1 | BMF snapshot ≤ D; share with REVENUE_AMT = y_revenue,t | < 1% | yes | 0.48% (2023-08); 22.8% (2026) | A T12 |
| B2 | `avail_D` built from the first original TY2022 submission ≤ D; coverage by size class | Built; reported | yes | 94.4% of units / 87.7% of revenue (national, lower bound) | B RT-2 |
| B3 | Lags come from the first original filing, not a later amendment | 100% | yes | — | B B3 |
| B4 | Every embedded text's ObjectId is a return with TY ≤ t−1 submitted ≤ D | 100% | yes | TY2022 only (pilot); fallback TY2021 to add | A T10–11 |
| B5 | Previous-wave BMF income/asset codes and amounts recomputed from TY2021 returns (the 2023 BMF holds the previous wave's y) | Done; equality test < 1% | **yes** (frozen, selection, PANEL) | **Not done**: `ne_wave2022_pop` uses 2023-08 codes | A T13 |
| B6 | Numbers in text: exact y_t or y_{t−1}; digits; "$" | Exact share < 1%; T2m built | no | Exact 0.000–0.005%; digits 44.1%; "$" 3.8% | A T14 |
| B7 | XPath whitelist: no amount siblings in text | 0 violations | yes | — | A T15 |
| B8 | Staleness: identical text t−1 vs t | Report | no | Mission 84.5% | A T16 |
| B9 | LSA vocabulary fitted on frame t−1 text only; PCA on U only; no y | Asserted in code | yes | — | B B9 |
| B10 | Placebo y gains nothing (population level) | ≤ 1.00 + 0.01 | yes | −0.004 (pass) | A T18 |
| B11 | Memorization probe: change-R² beyond frame, per rung vs shuffled names | FLAG if > 0.01 | no | LSA/MiniLM −0.003 | A T17, B G2 |
| B12 | Boilerplate ("see Schedule O", empty, < 3 words) after `clean_text` | < 1% or handled | no | 2.1% | B B11 |
| B13 | Same-year text never enters a non-control arm (code audit) | 0 | yes | — | NEW |

**C. Frame completeness and baseline strength**

| ID | Test | Rule | Blocking | Observed | Origin |
|---|---|---|---|---|---|
| C1 | NTEE missing by size, age, subsection | Report | no | 31.1%; 38% → 26% across size quintiles | A T28 |
| C2 | **Baseline starvation:** BMF-max fields and NAME present and parsed | Present | **yes** | **FAIL**: `ne_pop.parquet` has 53 columns and none of them | B RT-1 |
| C3 | **Valid-URL flag** rebuilt (contains a dot; not N/A, NONE, NA or "see schedule o") | Rebuilt | **yes** | **FAIL**: `has_website` says 92.8%, valid is 72.5% (51% under $200k, 92% over $25M) | N6 |
| C4 | URLs shared by ≥ 2 EINs | Fold groups built | yes (L-W) | 8.4% of valid URLs | N6 |
| C5 | Text–code redundancy: R² of 32 PCs on BMF-max dummies; between-code variance share; NTEE probe accuracy | Report (> 0.8 = code proxy) | no | NTEE-3 between-share 14–16% (all dimensions), 20–25% (32 PCs); probe 67–71% vs 24–35% majority class | B C3, N6 |
| C6 | Foundation code × revenue-mix cross-tab [VERIFY code meanings] | Report | no | — | B C4 |
| C7 | Register-status class of each y checked against T3010, T1044, T2/GIFI and PD7A line lists | Signed off | yes (FREEZE-2) [USER U6] | — | B C5 |
| C8 | lag_expenses ≤ 0: share and y share | Report | no | 0.67% of units; 0.02% with gov_grants | C T-M12 |
| C9 | Tier-consistent designs: F0 never uses lagged amounts | Asserted | yes | Pilot weak frame used the lag design (B RT-9) | B RT-9 |

**D. y quality**

| ID | Test | Rule | Blocking | Observed | Origin |
|---|---|---|---|---|---|
| D1 | Negatives, zeros, top-k shares overall and in take-some; largest single take-some unit share per y | FLAG > 5% | no | Volunteers 16.4%; gov_grants 1.5% | A T26 |
| D2 | Prior-year column on the t return vs current-year column on the t−1 return | ≥ 98% | no | 97.9% (FLAG: restatements) | B D2 |
| D3 | Volunteers: blank vs zero; heaping; Knights of Columbus-type outlier | Estimand fixed at G0 | yes [USER U5] | 19.7% blank; 32% on multiples of 10; one unit = 42% of the total | A T27 |
| D4 | Edit log counts per rule; E1 copies lag into y_employees; no-edit sensitivity | Report | no | E1–E3 logged | A T30, B RT-14 |
| D5 | Parsed vs index revenue | 100% | no | 100% | pilot |
| D6 | Stratum jumpers and their MSE share | Report | no | 8.4%; 20 units = 1.3% of expenses | A T24 |
| D7 | Allocation feasibility for every n in every grid | Sum of minimum n_h ≤ n_TS | yes | n = 500 fails under D1, so it is dropped | A T25 |

**E. Text quality**

| ID | Test | Rule | Blocking | Observed |
|---|---|---|---|---|
| E1 | Length by size class, NTEE and `text_source` (TY2022 / TY2021 / name) | Report | no | Words: 50 (< $200k) → 184 (> $25M) |
| E2 | MiniLM truncation (> 256 tokens) by size class | Report | no | — |
| E3 | Non-English share | Report | no | — |
| E4 | Near-duplicate clusters (exact; cosine > 0.95), including those straddling take-all and take-some | Groups built | yes | 989 units (4.8%); largest cluster = empty mission (112) |
| E5 | Text and URL missingness by size, NTEE and state; correlation with y and with response | Report (feeds NR4) | no | URL 51% → 92% by size |

**F. Response mechanism**

| ID | Test | Rule | Blocking | Observed |
|---|---|---|---|---|
| F1 | Response by size, NTEE, state; unit and expense-weighted | Report | no | 60.1% / 18.4%; take-some 61%, take-all 12% |
| F2 | Respondent vs nonrespondent means within strata | Report | no | gov_grants 0.35×; expenses 0.12× |
| F3 | Late-filing gradient vs survey gradient | Stated | no | Inverted |
| F4 | Previous-wave response (300-day rule) built for frozen training and PANEL | Built | yes | 59.6% |

### 10.2 Harness and methods tests (`tests/`, pytest; all blocking before Stage 2)

| ID | Test | Current status |
|---|---|---|
| T-M1 | Penalty invariance: predictions unchanged under w → c·w; ‖B̂_λ‖ strictly decreasing in λ for every weight vector the harness uses | **FAIL** (defect a) |
| T-M2 | `ridge` kind: the CV-scored penalty equals the applied penalty (same function) | **FAIL** (defect b) |
| T-M3 | No population y: permuting y over U ∖ s leaves every estimate unchanged | To write |
| T-M4 | Folds: no group split; per-stratum balance ±1; **one fold vector shared by all arms in a replicate** | Groups yes; balance no; sharing no |
| T-M5 | Common random numbers: adding an estimator leaves every other estimate bit-identical | Likely PASS for samples and per-estimator streams (exists); re-run after T-M4 |
| T-M6 | HT and GREG unbiased, coverage ≈ 95% on a synthetic population | PASS (working log) |
| T-M7 | FROZEN_diff exact unbiasedness; v_frozen within MC error | To run |
| T-M8 | No binary y reaches a ratio-scale learner | **FAIL** (defect c) |
| T-M9 | Bounded calibration converges; g within bounds; failure rate < 0.5% | To add |
| T-M10 | A same-class ceiling exists for every MC estimator | Missing for ratio arms |
| T-M11 | PCA and LSA use no y; frozen rotation comes from the previous wave | `frozen.py` correct; add an assertion |
| T-M12 | Embedding manifest hash = hash of the texts actually used | Exists (`embed.build`) |
| T-M13 | Constant embedding reproduces the twin to machine precision (N4) | To write |
| T-M14 | **Pilot regression rerun**: v1 primary cell rerun after fixes A1–A3 and A6; per-y change in RE and coverage logged in `working-log.md` | After the fixes |
| T-M15 | Grouped λ-CV in `ridge`, `frozen.train_frozen` and `sae.run_sae` | **FAIL** (`groups=None` in `ridge` and `frozen`) |
| T-M16 | `stage_available` assertion: no TRUTH-tagged column in a non-control arm | To write |
| T-M17 | `run_cell.py` refuses a confirmatory cell on a prereg or commit mismatch | To write |

### 10.3 Datasets

| ID | Dataset | Role | State | User action |
|---|---|---|---|---|
| R0-dev | NE 990, TY2022 → TY2023 (P1) | Development and proof of concept | Built; must be rebuilt (C2, C3, B2) | none |
| R0-prev | NE 990, TY2021 → TY2022 | E* selection, frozen heads, PANEL | Built; must be rebuilt (B5) | none |
| **R0-conf** | 8 held-out Census divisions | **Confirmation** | TY2022 parsed (342k); TY2023 downloaded, parsing; **TY2021 national not downloaded** (about 3 GB gzipped, about 1 h + 25 min parse; reachable) | none (compute and disk, about 4 GB) |
| R0-EZ | TY2023 990-EZ XML for P2 | Survivorship sensitivity | Index local; XML reachable | none |
| R1 | **CRA T3010** (Canada), two consecutive fiscal periods: IDENT, NEW_ONGOING_PROGRAMS, FINANCIAL_SECTION_A_B_AND_C, FINANCIAL D_AND _SCHEDULE_6, SCHEDULE_3_COMPENSATION, program-area tables (dataset `2dc3c94d-f98c-4144-98c8-3c7da3d844d5`) | F2 realism for charities; bilingual text (French subset reported separately); government revenue by level | Blocked | **U2**; multilingual model (U4) |
| R2 | **UK CCEW**: register extract + t−1 accounts PDFs for about 5,000 main charities (`linked_charity_number = 0`) with income £25k–£10M in 2–3 regions; first 5 pages | **The genuine multimodal test**: L7 vs L6 vs L7-ext. `charitable_objects` = register text. The frame **must include the lagged numbers printed on the page**. | Blocked | **U3** + U1 |
| R2-alt | ACNC (Australia) AIS + financial reports | Fallback for R2 | Not checked [VERIFY] | data.gov.au access |
| W | Common Crawl captures ≤ D for 990 units with a valid URL | L-W text and L7 screenshots | Blocked (proxy refuses `index.commoncrawl.org`, `data.commoncrawl.org`; checked 2026-10-23) | **U7** |

**Replication red team.**
- **T3010:**
  - names are backfilled (1.4% of BNs change name over 2020–24), so names are not frame-time text;
  - 12.8% of BN-years have arithmetic violations;
  - Section D vs Schedule 6 structural missingness below $100k revenue;
  - form-version changes.
- **CCEW:**
  - `charity_activities` is a current snapshot (leaky), so use `charitable_objects` or an archived extract;
  - only the t−1 PDF (the t PDF *is* y);
  - scans vs text-layer PDFs;
  - trustee names are personal data, never published;
  - number-redacted images (boxes over digit tokens) are the visual analogue of L2.
- **W:**
  - captures must be timestamped ≤ D;
  - shared URLs go into fold groups;
  - third-party pages quoting 990 financials are excluded (own domain only);
  - mask numbers;
  - coverage reported by size;
  - screenshot rendering from WARC without live assets is approximate [VERIFY tooling: a headless browser may need binaries from a blocked CDN].

---

## 11. Notebook and results layout; staged execution plan

### 11.1 Layout

```
embedding-auxiliary-variables/
  config/
    prereg_v1.yaml (+ .sha256)     # FREEZE-1
    prereg_v1.2.yaml               # FREEZE-2 (confirmatory)
    prereg_v1.k.yaml               # stage addenda
    cells.yaml                     # every cell: pop, tier(s), design, n, R, seed, estimators, y columns
  prereg/deviations.md
  src/                             # library (§13)
  scripts/
    run_cell.py                    # one cell id -> results/...; hash and commit checks; background runs with logs
    build_population.py            # (moved CLI) P1/P2/previous wave/divisions
    build_embeddings.py            # all model/API calls; never called from notebooks
    fetch_commoncrawl.py           # after U7
  tests/test_harness.py            # T-M1..T-M17
  results/<stage>/<pop>/<cell_id>/
    reps.parquet                   # rep, estimator, y, est, var, failed, sample_hash
    metrics.parquet                # mc.metrics + gaps, studentized moments, failure counts
    weights.parquet                # single-weight diagnostics per replicate
    diag.parquet                   # Stage-1 ceilings and predictions for the cell
    manifest.json                  # git commit, prereg hash, data and embedding hashes, seeds, versions, runtime
  notebooks/                       # jupytext .py + .ipynb; read results/ only; < 2 min each
    00_protocol_and_config         # renders config, checks hashes, cell status
    01_data_and_red_team           # (existing) extended to all of §10.1; population cards
    02_benchmark_results           # (existing) frozen as the pilot archive + T-M14 regression comparison
    03_stage1_diagnostics          # previous-wave E* selection; same-class and learning-curve ceilings;
                                   #   codes-vs-text pre-check; NR exact bias; design AVs; planted calibration
    04_mc_calibration_ne_poc       # Stage 2A-2J: P1/P2 NE verdict, factorial, gates, cost curve
    05_designs_stratification      # 2F: S4, S5, S5b, M8
    06_nonresponse                 # Stage 3
    07_sae                         # Stage 4
    08_frozen_and_panel            # 2H/2P: M6, S-PANEL, ageing
    09_embedding_ladder_cohere     # Stage 5
    10_confirmatory_divisions      # FREEZE-2 check; P1/P2 confirmatory; forest plots; PI
    11_replication_cra_t3010       # Stage 6
    12_replication_ccew_multimodal # Stage 6
    13_decision_report             # verdict table (§9.6), attribution, cost and CV-band translation
```

**Rules.**
- Every notebook ends with a **decision cell**: what was learned, what was decided, and a `working-log.md` entry.
- Notebooks 04–09 open with "DEVELOPMENT (NE)". Notebook 10 is the only confirmatory notebook for P1/P2.
- **Data root:** `EMBAUX_DATA`, default `/tmp/claude-0/data`, which is session-scoped. Persist derived populations, embeddings and results with sha256 manifests [USER U8].

### 11.2 Staged plan

**Compute basis (measured):**
- About 2 s per replicate for the 40-estimator pilot v1 cell (NE, n = 1,000) on the shared 4-core CPU under load, i.e. about 0.05 s per estimator-replicate.
- MiniLM on T2: 33 min per 20k units under load, so about 9 h CPU nationally. LSA takes minutes.
- New learners (ridge-logit, two-part, DR) are **timed in Stage 0d before any budget is committed**.
- At most 3 worker processes.

| Stage | Content | Gate | CPU estimate | Needs |
|---|---|---|---|---|
| **0a Fixes** | §13 blocking items A1–A15; tests T-M1–T-M17 | All T-M pass | coding + < 1 h | — |
| **0b Data rebuild + red team** | Rebuild P1 (BMF-max fields, NAME, `avail_D`, TY2021 text fallback, valid URL, `lag_*` for all y including binaries, volunteers blank flag). Rebuild P1′ with B5. Embed T0, T1s, T2m (MiniLM about 35 min each; LSA minutes) and previous-wave LSA. Notebook 01 runs all of §10.1. | **G0** (blocking tests; volunteers estimand) | about 3 h | — |
| **0c Pilot regression** | Rerun the v1 cell (F1, D1, n = 1,000, R = 500) with fixes; log per-y changes (T-M14) | Logged | about 0.5 h | — |
| **0d Timing pilot** | 20 replicates of every estimator family at NE n = 1,000 and at the largest division n = 2,822. Project each stage. If a stage exceeds its budget, drop exploratory learners in this order: GBM, kNN, `:full`, two-part. **R of primary cells is never reduced.** | Budget table in `working-log.md` | < 0.5 h | — |
| **1 Diagnostics** | Previous wave: same-class ceilings for L1–L4 × T1/T2 × K vs BMF-max twin → E*. Current wave: ceilings for every MC estimator (oracle and fixed λ); learning-curve ceilings (n ∈ {650, 1,000, 2,000}); codes-vs-text pre-check rerun same-class; NR exact-bias table; design AVs; planted a-calibration; updated predictions. | **G1 = FREEZE-1** [USER U9] | 4–6 h | — |
| **2A NE primary** | F1p/F1/F1c/F2 matrices on shared D1 samples. P1, P1′, P2, P2-twin, P2′, factorial (8), frozen variants, controls (N1–N4, N5 placebo, O1, O3, S1), B* candidates. About 45 estimators, R = 2,000. **= NE proof-of-concept verdict.** | Gates G1–G9 | 3–5 h | — |
| 2B F0 | D1-W, F0 twins and text, R = 1,000 | — | about 1 h | — |
| 2C Planted | 240 planted columns, 10 estimators, R = 1,000; D0 n ∈ {500, 2,000} | G6 | 2–3 h | — |
| 2D Scaling | D0 n grid; K grid; `:full` (R = 500) | — | 2–3 h | — |
| 2E Bootstrap | v_B on 200 replicates × B = 300, 6 estimators | — | 2–4 h | — |
| 2F Designs | D1-H44, D1-imp, D2a, D2b, D2b-twin, D2c × 3 estimators, R = 1,000 | — | 1–2 h | — |
| 2G Cost curve | Baseline at 6 n; text arms at 4 n, R = 1,000 | — | 1–2 h | — |
| 2H Frozen + PANEL | J = 20 heads (twin and text); PANEL 20 × 100 | — | 1–2 h | — |
| 2I Sensitivities | Concept mismatch (lags × exp(η), η ~ N(0, σ²) fixed per unit, σ such that corr(log1p lag_revenue, log1p y_revenue) = 0.90); `-WIN`; Dec-2023 BMF; December-FYE only; no-edit; T5 nested-CV and T6 λ-stability checks (R = 200) | — | 2–3 h | Dec BMF download |
| 2K Populations | P2 (EZ XML); births arm | — | about 1 h | EZ XML |
| **G2 = FREEZE-2** | Write the proof-of-concept verdict; set R_conf, κ, B*_frozen, T-V/T5/T6 choices | [USER U9] | — | — |
| **2-conf Divisions** | Build 8 divisions (TY2023 parse in progress; TY2021 download + parse; B5 per division); embed E* nationally (LSA minutes; MiniLM about 9 h); previous-wave heads per division. Cells: F1p/F1/F2 on D1, about 18 estimators + controls + planted lin-same, R_conf. | **G-conf**: P1/P2 confirmatory verdict | Build 4–12 h. MC: about 18 estimators × 3 tiers ≈ 3 s per NE-equivalent replicate, × 13.9 × R_conf (1,000) ≈ 12 h CPU (about 4 h wall); double if R_conf = 2,000 | — |
| **3 Nonresponse** | NR0–NR4, NR2′, NR-LATE (180/300/365) × TA-follow-up / TA-late; about 25 estimators; NE R = 1,000; NR-LATE in the divisions | Addendum v1.3 | 5–8 h | — |
| **4 SAE** | Planned + ZIP3; NE and divisions; R = 500 | v1.4 | 2–4 h | — |
| **5 Ladder** | L1/L3 (local); L5 [U4]; L6/L6b [U1] on NE and previous wave; Stage-1 ceilings + MC ladder cells (common random numbers), R = 1,000 | v1.5 | 2–4 h + API < $1 (NE) | U1, U4 |
| **6 Replications** | CRA T3010 [U2]; CCEW multimodal [U3]; Common Crawl W [U7] | v1.6 | 3–8 h + about $14 images | U2, U3, U7 |
| **7 Decision report** | Notebook 13 | Final verdict | — | — |

**Totals.**
- Through the NE proof-of-concept verdict (Stages 0–2): about 25–40 CPU-hours, about 10–15 h of wall time at 3 workers, plus coding.
- Through confirmation: a further 20–35 CPU-hours.
- All stages: about 60–90 CPU-hours, i.e. 1–2 weeks of wall time on the shared machine.
- The binding constraints are coding time and user actions, not CPU.

---

## 12. Real-world translation and governance

### 12.1 How a result would be used

| Result pattern | Production action | What it buys |
|---|---|---|
| P1 and P2 MET at F1p | Add frozen text-score columns (FROZEN_calm) to existing calibration. Retrain the heads each cycle on last cycle's respondents. Run in parallel for one cycle, then reduce the take-some sample. | σ ≈ 1 − 1/RE_binding of the take-some sample (1.2 → 17%), or tighter domain CVs at the same n. The variance software is unchanged, because m is just another auxiliary with a known total. |
| P2 MET via P2-twin; P1 not MET | Ratio-scale model calibration on register variables, with no text | The gain without a text pipeline or its governance |
| Codes dominate (M1, Φ) | Full NTEE code + text-coded NTEE in strata and calibration | Register-coding change; StatCan already codes BN text |
| Substitute for a missing lag | Link the admin lag (T3010 line, payroll) or use panel values | 2.4–14.6× vs text's ≤ 1.4× |
| Only S2 (F0) | Text only for frame segments without admin data: non-charity NPOs, births, stale-lag units | Precision where the register is thin. Check the segment's share of output first. |
| S4 / D1-imp | Register maintenance: autocode missing activity codes | Better strata and domain classification; no estimation change |
| S6 | Text predictive-mean classes or donor matching in nonresponse/imputation (Banff) | Lower nonresponse bias. Unlike variance, bias is not fixed by a larger n. |
| S7 | One frozen text covariate per domain in Fay–Herriot | More publishable domain cells (band D or better) |
| Nothing passes | Stop; publish the null with ρ²_min | Settles a recurring proposal |

**Rules printed in the decision report:**
- RE r saves 1 − 1/r of the take-some sample, **only for the binding variable**.
- CVs shrink by 1 − 1/√r.
- ΔC = Δn × (c_u + b·c_b) − C_text. It is negative at NE scale for RE 1.10 and positive at national scale.

### 12.2 Which NSO data sources would play the role of text

| Source | At design time? | Coverage | Confidentiality | Modality | Testbed analogue | Best role |
|---|---|---|---|---|---|---|
| BN registration "major business activity" + main products (CRA RC1); SS-4 l.16–17 (US) | yes | Near-universal for BN holders; stale for old units | Confidential (Statistics Act) | Short text | **T1s / T1** | Frame coding, prediction strata, calibration score |
| T3010 program descriptions | yes (t−1) | About 84k registered charities | Public (EN/FR) | Text | T2; R1 | Charity frames (F2), where only H_S is survey-only |
| BR legal and operating names | yes | Universal, births included | Confidential | Very short text | T0 | Births and stale units |
| Organization websites | Crawl at D, or estimation time | Size-graded: 51% small → 92% large (NE valid URLs) | Public, under the NSO web-scraping policy | Text + layout | W (Common Crawl) | Estimation-time auxiliary; NR covariate; L7 |
| Annual reports / financial-statement PDFs (CCEW, ACNC; T3010 attachments [VERIFY publication]) | t−1 | Mostly larger units | Public (UK/AU) | PDF pages | R2 | **Extract the numbers first (L7-ext)**; embed narrative only if residual signal remains |
| Admin tax and payroll (T2/GIFI, T1044, T3010 lines, PD7A) | yes | Near-universal | Confidential | Numeric | Lags (F1–F2) | **The baseline text must beat** |
| Last cycle's survey responses | yes (panel units) | Previous sample | Confidential | Numeric | PANEL | Kills text gains where present |

### 12.3 Governance (deployment criteria, not afterthoughts)

1. **Confidentiality.** BN text and BR content may not go to a SaaS API, and none of Cohere's managed options processes in Canada (note 04). Permissible routes:
   - on-prem open weights (LSA, MiniLM, Qwen3/Qwen3-VL, Apache-2.0); this is why S10 is a production condition;
   - a Cohere private deployment or Model Vault with written residency.

   Public-data experiments (990, T3010, CCEW) may use the API.
2. **Reproducibility.**
   - Content-addressed embeddings, frozen with each cycle's estimates. Published estimates are never revised because a model changed.
   - On deprecation (Cohere Embed v2 was shut down on 2026-04-04), run a bridging study: re-embed the last cycle, re-estimate, and publish the break like a classification revision.
   - Re-embed a 1,000-item canary set each cycle; alarm at cosine < 0.999.
3. **Documentation** (UNECE handbook on uncertainty quantification in ML-enhanced inference). Record:
   - the model-assisted design-consistency argument (a bad head costs efficiency, not bias);
   - the variance method and its bootstrap evidence;
   - coverage results from this benchmark;
   - a model card (training data = last cycle's respondents; features; λ policy; population R²; ageing check; weight diagnostics).
4. **Explainability.** Publish LSA top terms for weighted directions. Show which unit groups have g-weights moved by text, and by how much.
5. **Fairness across domains.** Text is size- and language-graded. Report per-domain bias (S7 heterogeneity check), gains by size class, and the French subset (R1).
6. **Stability.** Compare year-to-year volatility of weights and domain estimates between frozen and re-trained heads. A model that reshuffles weights creates artificial series breaks.
7. **Responsible-ML review.** Apply StatCan's *Framework for Responsible Machine Learning Processes* [VERIFY current title and version]: purpose, data, model card, validation (this benchmark) and monitoring.

### 12.4 Known limitations and the direction of their bias

| Limitation | Direction for text |
|---|---|
| Lags and y come from the same instrument (corr 0.99) (RT-13) | Conservative (understates text). Quantified by S-CM. |
| E1 copies lag_employees into y_employees (RT-14) | Slightly favours lag baselines on a negative-control y |
| 990 mission and program text is richer than BN registration text | Favourable; T1s quantifies it |
| Primary F1p ignores panel values | Favourable for H_S; S-PANEL quantifies it |
| Oracle-λ ceilings | Favourable in Stage-1 predictions only; the MC tunes in-sample |
| Previous-wave codes from a BMF (2023-08) that post-dates D′ = 2023-06-30 | Slightly favourable to frozen heads |
| P1 conditions on survival (RT-4) | Favours lag baselines; P2 checks it |
| 2023-08 BMF rather than a snapshot at D | Slightly weaker register for `avail_D = 0` units: slightly favourable |
| Late filing as nonresponse (inverted gradient) | Unknown; triangulated with NR-sim |
| No survey measurement error | Neutral to slightly conservative |
| US NPO sector only until R1/R2 | Unknown transfer |
| Divisions spatially correlated, not random | t₇ precision optimistic in either direction; the PI is reported |
| NE reused for development | Guarded by held-out divisions |

---

## 13. Changes required in code

**Blocking before Stage 2 (A-items).**
- **A1** `src/mc.py::_ridge_fit`:
  - normalize w to mean 1;
  - standardize continuous columns with the weighted SD and penalize with λ·n_train;
  - new arguments `cat_idx` and `lam_c`, which penalize categorical columns on the raw 0/1 scale with λ_C;
  - intercept unpenalized.
  - Also `src/mc.py::LAMS`: 13 values 10^{−4..2}, plus `LAMS_C = (1, 5, 20, 100, 500)`.
  - Tests T-M1.
- **A2** `src/mc.py::_estimate` (kind `ridge`):
  - choose λ by CV of the penalized-calibration estimator itself, using the same `pen` formula as the final `greg_total` call;
  - grouped, shared folds (T-M2, T-M15).
- **A3** `src/mc.py::_ridge_cv_lambda`:
  - AV-scale CV loss on the y scale (pass stratum ids and N_h²(1 − f_h)/n_h factors, plus the x̃ back-transform);
  - coordinate search over (λ_C, λ);
  - accept a precomputed fold vector;
  - return the CV curve for λ-stability logging.
- **A4** `src/mc.py::_folds` → `make_folds(groups, strata, rng)`: groups dealt round-robin within strata. Groups = `experiments.text_groups` ∪ cosine > 0.95 clusters ∪ shared valid URLs.
- **A5** `src/mc.py::run_cell`:
  - sample stream `default_rng([seed, r, 0])`;
  - one shared fold vector per replicate (`[seed, r, 1]`) passed to every spec;
  - failure flags and sample hash in the output;
  - weight summaries for single-weight kinds;
  - sub-cells with replicate-specific designs (PANEL) and replicate-specific frozen heads;
  - `stage_available` assertion (T-M16).
- **A6** `src/mc.py::_estimate` (kinds `mcal`, `syn`): `learner ∈ {ratio, logit, twopart, knn, gbm, dual}` dispatched by y type. Binaries never go through `ratio` (T-M8). Ridge-logit is a numpy IRLS with warm starts.
- **A7** `src/experiments.py::conventional_specs`, `embedding_specs`, `nonresponse_specs`: X_base `size` → `size_cat@T` for every MCAL/FROZEN arm; protocol names (§4.3) and the factorial specs; learner by y type.
- **A8** `src/frozen.py::train_frozen`:
  - train on previous-wave **respondents** (300-day rule) with RHG-adjusted weights;
  - J = 20;
  - grouped λ-CV through the fixed `_ridge_fit`;
  - features `F@T` + C + Z_K with the previous-wave PCA (kept);
  - previous-wave codes from TY2021 (B5);
  - options for λ frozen vs re-tuned and for the full-sample sensitivity.
- **A9** `src/build_population.py::build`:
  - keep FOUNDATION, AFFILIATION, CLASSIFICATION, ACTIVITY, DEDUCTIBILITY, ORGANIZATION, ACCT_PD, FILING_REQ_CD, GROUP, ASSET/INCOME/REVENUE_AMT, NAME, SORT_NAME and the full NTEE_CD;
  - `has_valid_url` (C3);
  - `lag_*` for every y including binaries (rename `oracle_lag_*` → `lag_*`);
  - `volunteers_blank` flag;
  - `avail_D`, `lag_source`, `text_source` from `idx_2022.csv` first original submission ≤ D, with BMF and TY2021 fallbacks;
  - previous-wave mode recomputing BMF income/asset codes and amounts from TY2021 (B5);
  - division label;
  - E1 count flag.
- **A10** `src/experiments.py::build_mats`:
  - tiered matrices `size_cat@T` and `F@T` with `avail_D` / `text_missing` / `lag_source` indicators and their size interactions;
  - BMF-max block C with the ≥ 20 pooling rule, flagged categorical;
  - Z_K for K ∈ {8, 16, 32, 64};
  - `:full` features;
  - control embeddings (perm, permh within strata, permc within code × size, const);
  - PANEL columns;
  - concept-mismatch lags.
- **A11** `src/diagnostics.py::evaluate_feature_sets`: same-class ceilings (ratio target, normalized d·x̃ weights, y-type learner, categorical penalty, oracle and fixed λ). New functions: `learning_curve_ceiling`, `nr_exact_bias`, `design_av`.
- **A12** `src/survey.py`:
  - `bounded_calibration(X_s, d_s, T_X, L=0.2, U=5)` (truncated linear, convergence flag);
  - `var_g`;
  - `var_df`;
  - `rwy_bootstrap(estimate_fn, design, s, B, rng)` with full re-estimation;
  - `winsorize_type2(y, d, K)`.
- **A13** `src/mc.py::metrics`:
  - failure counts and RE on jointly successful replicates;
  - coverage gaps to HT and to the comparator (paired);
  - studentized-error mean, SD, skewness and tails;
  - variance relative bias against the empirical variance as well;
  - robust companions.
- **A14** `tests/test_harness.py`: T-M1–T-M17.
- **A15** `scripts/run_cell.py`: reads `config/cells.yaml`; checks the prereg sha256 and git commit for confirmatory cells (T-M17); writes `results/<stage>/<pop>/<cell>/` with the manifest.

**Needed by later stages.**
- `src/mc.py::_estimate`, new kinds or options:
  - `mcal_multi` (J4, bounded, one weight);
  - `ridge1w`;
  - `pccal_pos`;
  - `diff_cf`;
  - `frozen` options `ppi`, `multi` (calm), `update`, `census`;
  - `win` option on greg, mcal and frozen.
- `src/mc.py::_nr_weights`:
  - propensity-score classes;
  - predictive-mean and joint classes;
  - DR/AIPW with cross-fitted nuisances;
  - kNN donor imputation for take-some;
  - take-all nonresponse with TA-hist-own, TA-hist-aux, TA-nn-frame and TA-nn-text;
  - NR-sim mechanisms NR0–NR4 and NR2′ on stream `[seed, r, 2]`;
  - `v_R,B`.
- `src/frame.py`:
  - `size_measure(pop, tier)`;
  - `design_D1W`;
  - `production_design(n_size_classes=4)` for D1-H44;
  - `design_D1imp`;
  - `design_D2a_score`;
  - `design_D2b_allocation` (and its twin);
  - `design_D2c_kmeans`;
  - `panel_design(base_design, prev_resp_mask, keep=0.5)`;
  - `assert_equal_H`;
  - `bmfmax_codes(pop, min_count=20)`.
- `src/experiments.py`:
  - `planted_y(pop, E, form, rho2, seed)` (null base by within-cell permutation of y/x̃; a calibrated on the AV scale with the same-class learner);
  - `placebo_y(pop, design)` (PLC_perm, PLC_noise, PLC_bern);
  - `impute_ntee` with evaluation against 2026 BMF codes for truly missing units;
  - `text_groups` extended (cosine clusters, shared URLs).
- `src/sae.py`:
  - replace `_fh` with a Fay–Herriot with intercept, multiple frozen covariates, GVF-smoothed ψ, REML σ²_u and Prasad–Rao / Datta–Lahiri MSE;
  - `run_sae` gains DIR_GREG / DIR_FROZEN_calm from single-weight outputs, ZIP3 domains, shared seeds, the fixed learners and per-domain coverage distributions;
  - optional NER EBLUP.
- `src/embed.py`:
  - backends `keywords` (L1), `glove` / `spacy` (L3), `qwen3` (L5), Cohere v5 and Cohere image/fused (`inputs`);
  - preprocessing `mask_numbers` (T2m) and `t1s`;
  - MiniLM truncation-rate logging;
  - a cache key including the preprocessing version;
  - canary-set tools;
  - pre-flight tests.
- New `src/parse990ez_min.py`: TY2023 990-EZ items for P2.
- New `src/metrics_ext.py`:
  - replicate-bootstrap GM-RE;
  - Shapley shares and factorial contrasts; Φ; gap closure G; κ and predictions;
  - between-division t₇;
  - REML random-effects meta-analysis with PI;
  - equivalent n, cost curve, binding-constraint n, cost ΔC;
  - CV-band tables.
- `src/report.py::family`, `Y_TYPE`: protocol names and classes (§1.3, §4.3).
- New `scripts/build_embeddings.py` (all embedding calls, logged) and `scripts/fetch_commoncrawl.py` (CDX query ≤ D, WARC fetch, boilerplate removal, offline rendering) after U7.
- `src/fetch990.py` / `src/parse_all.py`:
  - national TY2021 fetch and parse (parse-and-delete XML if disk is short);
  - original-submission XML fetch for amended cases (B3);
  - Dec-2023 BMF download (sensitivity).

---

## 14. User actions required

| ID | Action | Unblocks | Needed by |
|---|---|---|---|
| U1 | Allowlist `api.cohere.com`; add a **production** `COHERE_API_KEY` as an environment secret (a trial key's 1,000 calls/month and 5 images/min cannot finish one national pass) | L6, L6b, L7 (Stage 5), CCEW images | Stage 5 |
| U2 | Allowlist `open.canada.ca`, or upload two consecutive years of the T3010 CSVs listed in §10.3 | R1 | Stage 6 |
| U3 | Allowlist the CCEW register and extract hosts (e.g. `ccewuksprdoneregsadata1.blob.core.windows.net` plus the register site; confirm hostnames), and confirm the terms for per-charity PDF retrieval; or upload the extract plus a PDF sample | R2, S11 | Stage 6 |
| U4 | Allowlist `huggingface.co`, or provide Qwen3-Embedding-0.6B, BGE-M3 and Qwen3-VL-Embedding-2B weights. A GPU is optional. | L5, L7-open; French text in R1 | Stage 5–6 |
| U5 | Decide or confirm before FREEZE-1 (defaults in §8.7 apply otherwise):<br>• c_u, burden b, c_b;<br>• the survey's CV targets and binding set;<br>• the volunteers estimand (blank = 0 vs item nonresponse) | §8.7; G0 | FREEZE-1 |
| U6 | Sign off the register-status class of every y against the T3010, T1044, T2/GIFI and PD7A line lists (test C7). In particular: does T1044 carry grants and donations? Are volunteers and activity checkboxes on T3010? | Headline set realism; F1p vs F2 framing | FREEZE-2 |
| U7 | Allowlist `index.commoncrawl.org` and `data.commoncrawl.org` (refused on 2026-10-23); confirm that NSO web-data policy permits archived captures | Website text (L-W) and screenshots (L7) for 990 units | Stage 6 |
| U8 | Provide persistent storage for derived data (populations, embeddings, results); `/tmp/claude-0/data` is session-scoped | Reproducibility across sessions | Stage 0b |
| U9 | Sign off FREEZE-1, FREEZE-2 and the addenda. This includes this protocol's three departures from the earlier drafts:<br>• primary tier F1p;<br>• P1 comparator = BMF-max twin;<br>• P2 = frozen single-weight system. | Confirmatory runs | Stage 1, G2 |
| U10 | Confirm the GivingTuesday data-lake terms for redistributing derived tables | Publication | Before publishing |
| U11 | Confirm the [VERIFY] facts where you have institutional access:<br>• StatCan CV-band guideline;<br>• Responsible-ML framework title;<br>• T1044 thresholds and content;<br>• whether T3010 financial-statement attachments are accessible. | §8.8, §12 | Before the decision report |

**No user action is needed for:** the national TY2021 download (reachable S3, about 3 GB gzipped), 990-EZ XML, the Dec-2023 BMF, local embeddings (LSA, MiniLM, GloVe wheels) and all Stage 0–4 compute.

---

## Appendix A. `config/prereg_v1.yaml` skeleton

```yaml
version: 1.0
frozen_on: null                       # set at FREEZE-1; sha256 in prereg_v1.sha256
populations: {dev: NE_P1, prev_wave: NE_P1prev, confirmatory: [MidAtl, ENC, WNC, SAtl, ESC, WSC, Mountain, Pacific],
              sensitivity: [NE_P2], definition: P1}
design_date: 2024-06-30
bmf_snapshot: 2023-08-16              # sensitivity 2023-12-12
response_rule_days: 300               # sensitivity [180, 365]; take-all scenarios [TA_followup_0.9, TA_late]
tiers: {primary: F1p, realism_check: F1, sensitivity: [F1c, F2, PANEL], weak: F0}
y:
  R:  [y_revenue, y_expenses, y_employees, y_salaries]
  H_D: [y_gov_grants, y_contrib, y_prog_rev, y_fundraising_exp, y_officer_comp]
  H_S: [y_volunteers_cap10k, y_bin_has_volunteers, y_bin_grants_to_indiv_us, y_bin_foreign_activities, y_bin_lobbying]
  secondary: [y_bin_has_gov_grants, y_bin_fundraising_activities, y_bin_gaming, y_bin_has_employees]
  stress: [y_volunteers, y_grants_paid]
  placebo: [PLC_perm, PLC_noise, PLC_bern]
  planted: {forms: [lin_same, lin_other, cluster, tail], rho2: [0, 0.02, 0.05, 0.10, 0.20, 0.40], plantings: 5}
designs: {primary: D1, f: 0.0485, take_all_share: 0.01, min_n: 10, alloc_power: 0.5,
          others: [D0, D1W, D1H44, D1imp, D2a, D2b, D2b_twin, D2c, PANEL_D1]}
estimators:
  P1: {text: "MCAL_bmfmax@F1p+E*", twin: "MCAL_bmfmax@F1p"}
  P2: {text: "FROZEN_calm_bmfmax@F1p+E*", comparator: "GREG_size_cat@F1p", J: [y_gov_grants, y_contrib, y_fundraising_exp, y_bin_grants_to_indiv_us], g_bounds: [0.2, 5]}
  secondary_pairs: [P1prime, P2twin, P2prime, Bstar_frozen]
learners: {amounts: ratio_ridge, zero_inflated_secondary: two_part, binary: ridge_logit}
tuning: {lams: "10^[-4:2:0.5]", lams_c: [1, 5, 20, 100, 500], cv_loss: av_yscale, folds: 5, grouped: true,
         stratum_balanced: true, shared_across_arms: true, lambda_on_full_sample: true}
embedding: {E_star: null, K: null, field: null, rule: "section 6.5", fallback: {model: minilm, field: T2, K: 32}}
R: {dev: 500, ne_primary: 2000, planted: 1000, bootstrap: [200, 300], sae: 500, confirmatory: null}   # set at FREEZE-2
seeds: "default_rng([crc32(pop|design|n), r, stream])"
gates: {neg_ub: 1.03, const_tol: 1e-8, placebo_ub: 1.03, cov_gap_comp_pp: -2, cov_gap_ht_pp: -5, var_rb: 0.15,
        ceiling_slack: 1.05, planted_phi_at_0.2: 0.8, leak_masked: 0.03, bmf_equal_max: 0.01, oracle_lb: 2.0, fail_rate: 0.005}
decision: {met: {point: 1.10, lb: 1.05}, excluded: {ub: 1.10}, any_gain: {yes_lb: 1.00, no_ub: 1.05},
           harm_ub: 0.97, holm: [P1, P2], alpha_one_sided: 0.05, falsify_requires_rho2_min_le: 0.10}
predictions: {P1: 1.04, P2: 1.05}     # updated once at FREEZE-1
verdict_rules: "section 9.6"
```
